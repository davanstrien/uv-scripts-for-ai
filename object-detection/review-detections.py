#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gradio>=6,<7",
#   "datasets>=4.5.0",
#   "pillow",
# ]
# ///
"""Human triage for a detection dataset -> an accepted/rejected verdict per image or per box.

A minimal keyboard-first review UI for the datasets the other scripts in this
directory produce (zero-shot teacher labels are suggestions, not ground truth).
Runs locally, opens your browser, writes every decision to a crash-safe journal,
and pushes the reviewed dataset back to the Hub when you finish.

    # quick first pass: accept/reject whole images (A / R keys) on a random sample
    uv run review-detections.py you/plates-illustrations --limit 200 \
        --out you/plates-illustrations-reviewed

    # detail pass: click boxes to reject them, M flags a missed instance,
    # hardest (least rectangular) images first
    uv run review-detections.py you/plates-illustrations --mode boxes \
        --out you/plates-illustrations-reviewed

Modes:
  quick  whole-image verdict. A=accept  R=reject  M=missed  arrows=skip/back.
         Defaults to RANDOM order so the printed acceptance rate is an unbiased
         sample statistic you can quote ("N% human-accepted, n=200").
  boxes  click a box to toggle it rejected; same keys advance. Defaults to
         rectangularity-ASCENDING order (irregular instances first), which is
         where review effort pays best -- but is a biased sample: do not quote
         this mode's acceptance rate.

The journal (./review-journal.jsonl) is appended per decision; re-running with
the same --journal resumes where you stopped. --out pushes rows that received a
verdict, with a `review` column ({verdict, box_keep, missed}) alongside the
original schema, so diff/retrain steps consume it unchanged.

Needs an `image` column. Bucket outputs (falcon-perception-bucket.py) carry no
images -- publish them joined to their source images first.
"""

import argparse
import io
import json
import os
import random
import threading

from fastapi.responses import HTMLResponse, Response

DISPLAY_W = 980

PAGE = """<!doctype html>
<title>review-detections</title>
<style>
  body { margin:0; background:#181818; color:#ddd; font:14px system-ui; }
  #bar { padding:8px 14px; display:flex; gap:18px; align-items:center; }
  #bar b { color:#fff; } #keys { color:#888; margin-left:auto; }
  #stage { position:relative; margin:0 auto; width:max-content; }
  #img { display:block; }
  .box { position:absolute; border:3px solid #ffd200; cursor:pointer; }
  .box.rej { border-color:#f33; border-style:dashed; }
  #flash { position:fixed; inset:0; display:none; align-items:center; justify-content:center;
           font-size:80px; pointer-events:none; }
</style>
<div id=bar><b id=pos></b><span id=stats></span><span id=verdict></span><span id=keys></span></div>
<div id=stage><img id=img><div id=boxes></div></div>
<div id=flash></div>
<script>
const MODE = "__MODE__";  // substituted by the server
document.getElementById("keys").textContent =
  MODE === "quick" ? "A accept · R reject · M missed · ←/→ move · F finish"
                   : "click box = toggle reject · A accept rest · R reject all · M missed · ←/→ · F finish";
let idx = 0, total = 0, meta = null, rejected = new Set();

async function load(i) {
  const r = await fetch(`/meta/${i}`);
  if (!r.ok) return;
  meta = await r.json();
  idx = meta.idx; total = meta.total; rejected = new Set(meta.rejected_boxes);
  document.getElementById("img").src = `/img/${idx}`;
  document.getElementById("pos").textContent = `${idx + 1} / ${total}`;
  document.getElementById("stats").textContent = meta.stats;
  document.getElementById("verdict").textContent = meta.verdict ? `decided: ${meta.verdict}` : "";
  const holder = document.getElementById("boxes");
  holder.innerHTML = "";
  meta.boxes.forEach(([x0, y0, x1, y1], j) => {
    const d = document.createElement("div");
    d.className = "box" + (rejected.has(j) ? " rej" : "");
    Object.assign(d.style, {left: x0 + "px", top: y0 + "px",
                            width: (x1 - x0) + "px", height: (y1 - y0) + "px"});
    if (MODE === "boxes") d.onclick = () => { rejected.has(j) ? rejected.delete(j) : rejected.add(j);
                                              d.classList.toggle("rej"); };
    holder.appendChild(d);
  });
}
function flash(t, c) {
  const f = document.getElementById("flash");
  f.textContent = t; f.style.color = c; f.style.display = "flex";
  setTimeout(() => f.style.display = "none", 180);
}
async function decide(verdict, missed) {
  const box_keep = meta.boxes.map((_, j) => !rejected.has(j));
  await fetch("/decide", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({idx, verdict, missed, box_keep, mode: MODE})});
  flash(verdict === "accept" ? "✓" : verdict === "reject" ? "✗" : "?",
        verdict === "accept" ? "#3c3" : "#f33");
  load(idx + 1);
}
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") load(idx + 1);
  else if (e.key === "ArrowLeft") load(idx - 1);
  else if (e.key === "a" || e.key === "A") decide("accept", false);
  else if (e.key === "r" || e.key === "R") decide("reject", false);
  else if (e.key === "m" || e.key === "M") decide("accept", true);
  else if (e.key === "f" || e.key === "F") {
    fetch("/finish", {method: "POST"});
    document.getElementById("keys").textContent = "finished — see the terminal; you can close this tab";
  }
});
load(0);
</script>
"""


def to_display_boxes(objects, width, height, scale):
    out = []
    for cx, cy, w, h in objects["bbox"]:
        x0 = (cx - w / 2) * width * scale
        y0 = (cy - h / 2) * height * scale
        out.append([round(x0), round(y0), round(x0 + w * width * scale), round(y0 + h * height * scale)])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--mode", default="quick", choices=["quick", "boxes"])
    p.add_argument("--order", default=None, choices=["random", "rect"],
                   help="default: random in quick mode (unbiased rate), rect in boxes mode")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--journal", default="./review-journal.jsonl")
    p.add_argument("--out", default=None, help="Hub repo id for the reviewed dataset")
    p.add_argument("--private", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    order = args.order or ("random" if args.mode == "quick" else "rect")

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)

    def min_rect(i):
        r = ds[i]["objects"]["rectangularity"]
        return min(r) if r else 2.0  # boxless images last

    ids = list(range(len(ds)))
    if order == "random":
        random.Random(args.seed).shuffle(ids)
    else:
        ids.sort(key=min_rect)
    if args.limit:
        ids = ids[: args.limit]

    decisions = {}  # image_id -> journal record
    if os.path.exists(args.journal):
        with open(args.journal) as f:
            for line in f:
                rec = json.loads(line)
                decisions[rec["image_id"]] = rec
        print(f"resumed {len(decisions)} decisions from {args.journal}", flush=True)

    img_cache = {}

    def render(i):
        if i not in img_cache:
            im = ds[ids[i]]["image"].convert("RGB")
            scale = min(DISPLAY_W / im.width, 1.0)
            if scale < 1.0:
                im = im.resize((round(im.width * scale), round(im.height * scale)))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88)
            img_cache[i] = (buf.getvalue(), scale)
            if len(img_cache) > 32:
                img_cache.pop(next(iter(img_cache)))
        return img_cache[i]

    import gradio as gr

    app = gr.Server(title="review-detections")
    done = threading.Event()

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.replace("__MODE__", args.mode)

    @app.get("/img/{i}")
    def img(i: int) -> Response:
        return Response(content=render(i)[0], media_type="image/jpeg")

    @app.get("/meta/{i}")
    def meta(i: int) -> dict:
        if not 0 <= i < len(ids):
            return Response(status_code=404)
        row = ds[ids[i]]
        _, scale = render(i)
        prior = decisions.get(row["image_id"])
        n = len(decisions)
        acc = sum(1 for d in decisions.values() if d["verdict"] == "accept")
        return {
            "idx": i, "total": len(ids),
            "boxes": to_display_boxes(row["objects"], row["width"], row["height"], scale),
            "verdict": prior["verdict"] if prior else None,
            "rejected_boxes": [j for j, k in enumerate(prior["box_keep"]) if not k] if prior else [],
            "stats": f"{n} decided · {acc / n:.0%} accepted" if n else "",
        }

    @app.post("/decide")
    def decide(body: dict) -> dict:
        row = ds[ids[body["idx"]]]
        rec = {
            "image_id": row["image_id"],
            "mode": body["mode"], "verdict": body["verdict"],
            "missed": bool(body.get("missed")), "box_keep": body.get("box_keep", []),
        }
        decisions[rec["image_id"]] = rec
        with open(args.journal, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return {"n": len(decisions)}

    @app.post("/finish")
    def finish() -> dict:
        done.set()
        return {"ok": True}

    print(f"open http://127.0.0.1:{args.port}/   (F in the browser, or Ctrl-C here, to finish)", flush=True)
    app.launch(server_port=args.port, inbrowser=True, quiet=True, prevent_thread_lock=True)
    import signal

    signal.signal(signal.SIGINT, lambda *_: done.set())  # gradio installs its own handler; override AFTER launch
    done.wait()  # review happens in the browser

    n = len(decisions)
    if not n:
        print("no decisions made", flush=True)
        return
    acc = sum(1 for d in decisions.values() if d["verdict"] == "accept")
    print(f"\n{n} decided · {acc} accepted ({acc / n:.0%})"
          + (" -- random-order sample, quotable" if order == "random" else " -- rect-ordered, biased sample"),
          flush=True)

    if args.out:
        reviewed = ds.filter(lambda r: r["image_id"] in decisions)
        reviewed = reviewed.map(lambda r: {"review": decisions[r["image_id"]]})
        reviewed.push_to_hub(args.out, private=args.private)
        print(f"{len(reviewed)} reviewed rows -> {args.out}", flush=True)


main()
