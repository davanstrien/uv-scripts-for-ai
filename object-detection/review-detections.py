#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gradio>=6,<7",
#   "fastapi",
#   "datasets>=4.5.0",
#   "pillow",
# ]
# ///
"""Human triage for a detection dataset -> an accept/reject verdict per image or per box.

A minimal keyboard-first review UI for datasets in THIS DIRECTORY'S schema
(yolo-normalized `objects.bbox` + `image`, `image_id`, `width`, `height` -- what
falcon-perception.py pushes; run convert-hf-dataset.py first for anything else).
Zero-shot teacher labels are suggestions, not ground truth: this is where a
human turns them into something you can quote. Runs locally, opens your
browser, journals every decision, pushes the reviewed rows back to the Hub.

    # quick first pass: accept/reject whole images (A / R keys) on a random sample
    uv run review-detections.py you/plates-illustrations --limit 200 \
        --out you/plates-illustrations-reviewed

    # detail pass: click boxes to reject them individually
    uv run review-detections.py you/plates-illustrations --mode boxes \
        --out you/plates-illustrations-reviewed

Modes:
  quick  whole-image verdict. A=accept  R=reject  M=accept-but-teacher-missed-something
         F=finish  arrows=skip/back. Defaults to RANDOM order, so the summary's
         acceptance rate is an unbiased sample statistic you can quote.
  boxes  click a box to toggle it rejected; A keeps the rest, R rejects the whole
         image (all boxes). Defaults to rectangularity-ASCENDING order (irregular
         instances first) -- best use of effort, but a biased sample: the summary
         says so and its rate should not be quoted.

Two numbers come out, measuring two different things: the ACCEPTANCE rate (are
the boxes that were drawn correct?) and the MISSED rate (how often did the
teacher skip an instance? -- the M key). Quote them separately; neither implies
the other.

The journal (default ./review-<dataset>-<split>.jsonl) is appended per decision
and tolerates a torn final line; re-running resumes at the first undecided image.
--out pushes decided rows with a `review` column ({verdict, missed, box_keep,
mode}) alongside the original schema.
"""

import argparse
import io
import json
import os
import random
import signal
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
  #err { display:none; padding:6px 14px; background:#611; color:#fbb; }
</style>
<div id=bar><b id=pos></b><span id=stats></span><span id=verdict></span><span id=keys></span></div>
<div id=err></div>
<div id=stage><img id=img><div id=boxes></div></div>
<div id=flash></div>
<script>
const MODE = "__MODE__";  // substituted by the server
document.getElementById("keys").textContent =
  MODE === "quick" ? "A accept · R reject · M missed · ←/→ move · F finish"
                   : "click box = reject it · A accept rest · R reject all · M missed · ←/→ · F finish";
let idx = 0, meta = null, rejected = new Set();

async function load(i) {
  const r = await fetch(`/meta/${i}`);
  if (!r.ok) return;
  meta = await r.json();
  idx = meta.idx; rejected = new Set(meta.rejected_boxes);
  document.getElementById("img").src = `/img/${idx}`;
  document.getElementById("pos").textContent = `${idx + 1} / ${meta.total}`;
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
  if (!meta) return;
  const box_keep = verdict === "reject" ? meta.boxes.map(() => false)
                                        : meta.boxes.map((_, j) => !rejected.has(j));
  const r = await fetch("/decide", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({idx, verdict, missed, box_keep, mode: MODE})});
  if (!r.ok) {  // do NOT advance on failure -- the journal write did not happen
    const e = document.getElementById("err");
    e.textContent = `decision NOT saved (server error ${r.status}) — fix the problem and retry`;
    e.style.display = "block";
    return;
  }
  document.getElementById("err").style.display = "none";
  flash(verdict === "accept" ? (missed ? "＋?" : "✓") : "✗",
        verdict === "accept" ? (missed ? "#fa3" : "#3c3") : "#f33");
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
fetch("/start").then(r => r.json()).then(d => load(d.start));
</script>
"""


def to_display_boxes(objects, width, height, scale):
    out = []
    for cx, cy, w, h in objects["bbox"]:
        x0 = (cx - w / 2) * width * scale
        y0 = (cy - h / 2) * height * scale
        out.append(
            [
                round(x0),
                round(y0),
                round(x0 + w * width * scale),
                round(y0 + h * height * scale),
            ]
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--mode", default="quick", choices=["quick", "boxes"])
    p.add_argument(
        "--order",
        default=None,
        choices=["random", "rect"],
        help="default: random in quick mode (unbiased rate), rect in boxes mode",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--journal",
        default=None,
        help="default: ./review-<dataset>-<split>.jsonl (scoped so runs don't mix)",
    )
    p.add_argument("--out", default=None, help="Hub repo id for the reviewed dataset")
    p.add_argument("--private", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    order = args.order or ("random" if args.mode == "quick" else "rect")
    journal_path = (
        args.journal or f"./review-{args.dataset.replace('/', '--')}-{args.split}.jsonl"
    )

    from datasets import Sequence, Value, load_dataset

    ds = load_dataset(args.dataset, split=args.split)

    missing = [
        c
        for c in ("image", "image_id", "width", "height", "objects")
        if c not in ds.column_names
    ]
    if missing:
        raise SystemExit(
            f"dataset is missing column(s) {missing} -- this tool reads the schema "
            "falcon-perception.py pushes; see the docstring."
        )

    # a lightweight view for sorting and sniffing that never decodes the image column
    meta_rows = ds.select_columns(["objects"])[:]["objects"]
    for objects in meta_rows[: min(50, len(meta_rows))]:
        if any(not (0 <= v <= 1.5) for box in objects["bbox"] for v in box):
            raise SystemExit(
                "objects.bbox does not look yolo-normalized (values outside [0,1]) -- "
                "run convert-hf-dataset.py --to yolo first."
            )

    ids = list(range(len(ds)))
    if order == "random":
        random.Random(args.seed).shuffle(ids)
    elif "rectangularity" not in meta_rows[0]:
        print("no rectangularity column -- falling back to random order", flush=True)
        random.Random(args.seed).shuffle(ids)
    else:
        ids.sort(
            key=lambda i: (
                min(meta_rows[i]["rectangularity"])
                if meta_rows[i]["rectangularity"]
                else 2.0
            )
        )
    if args.limit:
        ids = ids[: args.limit]

    # decisions are keyed by DATASET ROW INDEX -- image_id repeats across
    # concatenated per-class runs, so it cannot key a decision
    decisions = {}
    if os.path.exists(journal_path):
        with open(journal_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # torn final line from a crash mid-append
                    print("journal: skipped one torn line (crash recovery)", flush=True)
                    continue
                decisions[rec["row"]] = rec
        print(f"resumed {len(decisions)} decisions from {journal_path}", flush=True)

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

    def stats_line():
        n = len(decisions)
        if not n:
            return ""
        acc = sum(1 for d in decisions.values() if d["verdict"] == "accept")
        mis = sum(1 for d in decisions.values() if d["missed"])
        return f"{n} decided · {acc / n:.0%} accepted · {mis} missed-flagged"

    import gradio as gr

    app = gr.Server(title="review-detections")
    done = threading.Event()

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.replace("__MODE__", args.mode)

    @app.get("/start")
    def start() -> dict:
        first = next((i for i in range(len(ids)) if ids[i] not in decisions), 0)
        return {"start": first}

    @app.get("/img/{i}")
    def img(i: int) -> Response:
        if not 0 <= i < len(ids):
            return Response(status_code=404)
        return Response(content=render(i)[0], media_type="image/jpeg")

    @app.get("/meta/{i}")
    def meta(i: int) -> Response:
        if not 0 <= i < len(ids):
            return Response(status_code=404)
        row = ds[ids[i]]
        _, scale = render(i)
        prior = decisions.get(ids[i])
        payload = {
            "idx": i,
            "total": len(ids),
            "boxes": to_display_boxes(
                row["objects"], row["width"], row["height"], scale
            ),
            "verdict": prior["verdict"] if prior else None,
            "rejected_boxes": [j for j, k in enumerate(prior["box_keep"]) if not k]
            if prior
            else [],
            "stats": stats_line(),
        }
        return Response(content=json.dumps(payload), media_type="application/json")

    @app.post("/decide")
    def decide(body: dict) -> dict:
        if not 0 <= body.get("idx", -1) < len(ids):
            return Response(status_code=400)
        row_idx = ids[body["idx"]]
        rec = {
            "row": row_idx,
            "image_id": ds[row_idx]["image_id"],
            "dataset": args.dataset,
            "split": args.split,
            "mode": body["mode"],
            "order": order,
            "verdict": body["verdict"],
            "missed": bool(body.get("missed")),
            "box_keep": [bool(b) for b in body.get("box_keep", [])],
        }
        with open(
            journal_path, "a"
        ) as f:  # journal FIRST -- only report saved if it is
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        decisions[row_idx] = rec
        return {"n": len(decisions)}

    @app.post("/finish")
    def finish() -> dict:
        done.set()
        return {"ok": True}

    print(
        f"open http://127.0.0.1:{args.port}/   (F in the browser, or Ctrl-C here, to finish)",
        flush=True,
    )
    app.launch(
        server_port=args.port, inbrowser=True, quiet=True, prevent_thread_lock=True
    )
    signal.signal(
        signal.SIGINT, lambda *_: done.set()
    )  # gradio installs its own handler; override AFTER launch
    done.wait()  # review happens in the browser
    signal.signal(
        signal.SIGINT, signal.default_int_handler
    )  # Ctrl-C must work again (e.g. to abort the push)

    n = len(decisions)
    if not n:
        print("no decisions made", flush=True)
        return
    acc = sum(1 for d in decisions.values() if d["verdict"] == "accept")
    mis = sum(1 for d in decisions.values() if d["missed"])
    quotable = order == "random" and all(
        d["order"] == "random" for d in decisions.values()
    )
    print(
        f"\n{n} decided · {acc} accepted ({acc / n:.0%}) · {mis} with missed instances ({mis / n:.0%})",
        flush=True,
    )
    print(
        "acceptance rate is "
        + (
            "an unbiased random-order sample -- quotable"
            if quotable
            else "from a non-random or mixed-order queue -- NOT quotable"
        ),
        flush=True,
    )

    if args.out:
        rows = sorted(decisions)
        reviewed = ds.select(rows)
        feats = reviewed.features.copy()
        feats["review"] = {
            "verdict": Value("string"),
            "missed": Value("bool"),
            "mode": Value("string"),
            "box_keep": Sequence(Value("bool")),
        }
        reviewed = reviewed.map(
            lambda r, i: {
                "review": {
                    k: decisions[rows[i]][k]
                    for k in ("verdict", "missed", "mode", "box_keep")
                }
            },
            with_indices=True,
            features=feats,
        )
        reviewed.push_to_hub(args.out, private=args.private)
        print(f"{len(reviewed)} reviewed rows -> {args.out}", flush=True)


main()
