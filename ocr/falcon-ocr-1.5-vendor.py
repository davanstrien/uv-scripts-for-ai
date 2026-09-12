# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]>=0.2.0",
#     "pillow>=10",
# ]
#
# [tool.hf-jobs]
# image   = "ghcr.io/tiiuae/falcon-ocr@sha256:d0b4120bfd7b6fafdd9ae95124d32e8abdbb710ada6b4a303b33fc7636b5e1d7"
# flavor  = "a10g-small"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown using Falcon OCR 1.5 through TII's own Docker image, via saturate.

The other two Falcon recipes (falcon-ocr.py, falcon-ocr-1.5.py) drive the weights with the
`falcon-perception` package on the default uv image. This one runs the *vendor's deployable
product*: `ghcr.io/tiiuae/falcon-ocr` boots its own vLLM fork (float32, max_model_len 16k,
their chat template) plus a FastAPI pipeline (PP-DocLayoutV3 layout -> per-region OCR ->
markdown assembly). The script boots the image's entrypoint as a subprocess, waits for both
health endpoints, then hands the driver half (concurrency, retries, output, resume) to the
`saturate` package, like the other `-saturate.py` recipes:

- **Adaptive concurrency** with a low ceiling: the pipeline->vLLM hop inside the image
  returned HTTP 200 + an error body ("Can not write request body") at 8 in flight, so the
  window starts at 2 and is capped at `--max-inflight` (default 8). Those rows land as
  error rows and `--retry-errors` re-admits only them.
- **Crash-safe, resumable output**: results stream to the output repo as parquet parts;
  re-running the same command skips everything already done (anti-join on id).
- **Durable error rows** instead of `[OCR ERROR]` strings in the text column.

Two modes, per the model card:
- `e2e` (default): `skip_layout=true` - the whole page goes to the VLM in one shot. The v1.5
  card recommends this; on BHL scans it had the best content CER of every Falcon path but
  can loop on sparse/blank pages (nothing tells it "no regions").
- `--layout`: the layout+OCR pipeline. Guards blank pages, drops footnotes/marginalia more.
  Use it for newspapers and other very large pages: the image serves with max_model_len 16k
  above the model's 8k positions, and e2e on 4.5-7k px pages came back EMPTY and killed vLLM
  (CUDA device-side assert), while layout mode read all of them. Keep `--max-edge` at 2048.

Identity and reproducibility (the reason this recipe is fussy):
- The image bakes the weights into /models/Falcon-OCR at BUILD time. ghcr publishes only a
  moving `latest` tag; it moved twice on 2026-09-11/12 and the previous digest now 404s. So
  the header pins a DIGEST, and the script refuses to run unless the baked
  model.safetensors hashes to the expected v1.5 LFS sha256 (`--expect-weights` to change it,
  `--revision` to stage a different Hub commit over the baked dir, hash-checked the same way).
- Every output row carries image digest, weights sha256 and resolved Hub commit.

Run on HF Jobs. The vendor image has NO `uv` and a Python 3.10 without ensurepip, so
`hf jobs uv run` cannot start on it (the CLI execs `uv run`). Bootstrap uv with pip first:

    hf jobs run --flavor a10g-small -s HF_TOKEN --timeout 2h \\
        ghcr.io/tiiuae/falcon-ocr@sha256:d0b4120bfd7b6fafdd9ae95124d32e8abdbb710ada6b4a303b33fc7636b5e1d7 \\
        -- bash -lc 'pip install -q uv && uv run \\
            https://huggingface.co/datasets/uv-scripts/ocr/raw/main/falcon-ocr-1.5-vendor.py \\
            input-dataset output-dataset --limit 10'

The `[tool.hf-jobs]` header records the same launch config for the day the image ships `uv`
(or `hf jobs uv run` learns to bootstrap it); until then it is documentation, not a launcher.
Plain `uv run` on a machine without the image fails fast in the preflight below.

Output layout (the saturate shape, not input+column): the output repo holds
`data/part-*.parquet` with rows `{id, markdown, regions_json, model, model_commit,
weights_sha256, image, mode, output_tokens, processing_ms, error}` keyed by the input row id
(`--id-column`, or `<split>-<index>` by default). Read it with
`datasets.load_dataset(<output>, data_dir="data")` or `saturate.read_output`; join back to
the input on id. Run metadata lands in `data/completions/`.

Model: tiiuae/Falcon-OCR v1.5 (0.3B, Apache 2.0) as baked into the image digest above
Backend: vendor image (vLLM fork + falcon_ocr_sdk pipeline), saturate driver
"""

import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL_ID = "tiiuae/Falcon-OCR"
MODEL_LABEL = "Falcon-OCR-1.5 (vendor image)"

# Per-value provenance:
# - image: digest of ghcr.io/tiiuae/falcon-ocr that bakes the v1.5 weights (built
#   2026-09-12 02:20 +04). `latest` moves and old digests get pruned: the pin can rot, the
#   weights hash below cannot.
# - weights_sha256: LFS sha256 of model.safetensors at the v1.5 release head fe757d59
#   (v1 = 6e7f73a5...).
# - entrypoint / ports: the image's own single-GPU CMD (vLLM :8000 + pipeline :5002).
# - max_edge 2048: payload-shrink only; the pipeline's layout detector resizes internally.
#   (4096 worked in layout mode; e2e at 4096 on 7k px pages overflowed the 8k positions.)
# - window: initial 2 / cap 8 — the pipeline->vLLM hop errored at 8 in flight on 09-11.
SERVING = {
    "image": "ghcr.io/tiiuae/falcon-ocr@sha256:d0b4120bfd7b6fafdd9ae95124d32e8abdbb710ada6b4a303b33fc7636b5e1d7",
    "weights_sha256": "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16",
    "v15_commit": "fe757d59ecd79d4d68760162306a70a015761ad9",
    "model_dir": "/models/Falcon-OCR",
    "entrypoint": "/app/entrypoint_single.sh",
    "vllm_port": 8000,
    "pipeline_port": 5002,
    "parse_route": "/falconocr/parse",
    "max_edge": 2048,
    "window_initial": 2,
    "window_max": 8,
}

BOOT_TIMEOUT_S = 1800


def preflight() -> None:
    """Fail fast with the exact launch line when not inside the vendor image."""
    if os.path.isfile(SERVING["entrypoint"]) and os.path.isdir(SERVING["model_dir"]):
        return
    print("This recipe only runs inside the TII Falcon-OCR image; entrypoint or model dir missing.\n"
          "Launch it with:\n"
          f"  hf jobs run --flavor a10g-small -s HF_TOKEN --timeout 2h {SERVING['image']} -- "
          "bash -lc 'pip install -q uv && uv run "
          "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/falcon-ocr-1.5-vendor.py IN OUT'",
          file=sys.stderr)
    sys.exit(1)


def to_pil(value):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    raise ValueError(f"unsupported image value: {type(value)}")


def encode_jpeg(value, max_edge: int) -> str:
    """RGB-convert, downscale so the longest edge is max_edge, return a base64 JPEG (q95)."""
    from PIL import Image

    img = to_pil(value).convert("RGB")
    w, h = img.size
    if max_edge and max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def check_or_stage_weights(revision: str | None, expect_sha: str) -> dict:
    """Verify the baked weights, or stage a Hub revision over them; either way hash-check.

    Fails loud on any mismatch: a silent v1/v1.5 mix-up is exactly the failure this recipe
    exists to prevent.
    """
    weights = os.path.join(SERVING["model_dir"], "model.safetensors")
    resolved_commit = None
    if revision:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi()
        resolved_commit = api.model_info(MODEL_ID, revision=revision).sha
        info = api.model_info(MODEL_ID, revision=resolved_commit, files_metadata=True)
        lfs = {s.rfilename: s.lfs for s in info.siblings if s.lfs}
        if "model.safetensors" not in lfs:
            raise RuntimeError(f"{MODEL_ID}@{resolved_commit} has no LFS model.safetensors")
        expect_sha = lfs["model.safetensors"].sha256
        print(f"staging {MODEL_ID}@{resolved_commit} over {SERVING['model_dir']}", flush=True)
        snapshot_download(
            MODEL_ID, revision=resolved_commit, local_dir=SERVING["model_dir"],
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.jinja"],
        )
    got = sha256_of(weights)
    if got != expect_sha:
        raise RuntimeError(
            f"model.safetensors sha256 {got} != expected {expect_sha}. "
            "The image digest or the staged revision does not hold the weights this recipe claims."
        )
    print(f"weights verified: sha256 {got[:12]}... ({'staged' if revision else 'baked in image'})", flush=True)
    return {"weights_sha256": got, "model_commit": resolved_commit or SERVING["v15_commit"]}


def wait_healthy(url: str, deadline: float, proc: subprocess.Popen) -> None:
    last = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vendor entrypoint exited with code {proc.returncode} before {url} was healthy")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 - boot polling: every failure kind is "not yet"
            last = e
        time.sleep(5)
    raise RuntimeError(f"{url} not healthy after {BOOT_TIMEOUT_S}s: {last}")


def boot_services() -> subprocess.Popen:
    """Start the image's single-GPU entrypoint (vLLM :8000 + pipeline :5002); gate on both."""
    env = dict(os.environ)
    # The services must run on the IMAGE's python and packages, not on whatever uv resolved for
    # this driver (a different interpreter and a different transformers). `uv run` prepends its
    # venv bin to PATH, so the entrypoint's bare `python -m vllm...` would otherwise resolve to
    # the driver venv and fail with "No module named 'vllm'" (observed).
    for key in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    venv_bin = os.path.join(sys.prefix, "bin")
    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep) if p and p != venv_bin and "/.cache/uv/" not in p
    )
    env.setdefault("VLLM_GPU", "0")
    env.setdefault("PIPELINE_GPU", "0")
    proc = subprocess.Popen(["/bin/bash", SERVING["entrypoint"]], env=env)
    deadline = time.time() + BOOT_TIMEOUT_S
    wait_healthy(f"http://127.0.0.1:{SERVING['vllm_port']}/health", deadline, proc)
    wait_healthy(f"http://127.0.0.1:{SERVING['pipeline_port']}/health", deadline, proc)
    print("vendor services healthy (vLLM + pipeline)", flush=True)
    return proc


def parse_shard(spec: str) -> tuple[int, int]:
    rank, _, world = spec.partition("/")
    rank, world = int(rank), int(world or 1)
    if world < 1 or not 0 <= rank < world:
        raise argparse.ArgumentTypeError(f"--shard must be rank/world with 0 <= rank < world, got {spec!r}")
    return rank, world


def main():
    ap = argparse.ArgumentParser(description="Falcon OCR 1.5 via TII's vendor image, driven by saturate")
    ap.add_argument("input_dataset", help="Input dataset repo id (rows with an image column)")
    ap.add_argument("output_dataset", help="Output dataset repo id (created if missing)")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--config", default=None, help="Dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--input-revision", default=None,
                    help="Pin the input dataset revision (index ids are only stable per revision)")
    ap.add_argument("--id-column", default=None, help="Column to use as row id (default: split-index ids)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="Strided fan-out across jobs, e.g. 2/8 (default: 0/1)")
    ap.add_argument("--layout", action="store_true",
                    help="Use the layout+OCR pipeline instead of the card-recommended e2e mode")
    ap.add_argument("--max-edge", type=int, default=SERVING["max_edge"],
                    help="Downscale so the longest edge is at most this many px (default 2048)")
    ap.add_argument("--max-inflight", type=int, default=SERVING["window_max"],
                    help="Ceiling for the adaptive window (default 8; the in-image pipeline->vLLM hop "
                         "errored above that)")
    ap.add_argument("--revision", default=None,
                    help="Stage this Hub commit/branch of tiiuae/Falcon-OCR over the baked weights "
                         "(hash-checked against the Hub LFS sha256). Default: use the baked weights and "
                         "require them to hash to the v1.5 release.")
    ap.add_argument("--expect-weights", default=SERVING["weights_sha256"],
                    help="Expected sha256 of the baked model.safetensors (default: v1.5)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    args = ap.parse_args()

    preflight()
    from saturate import Auto, dataset_rows, pump, shard_select

    rank, world = args.shard
    skip_layout = not args.layout
    mode = "layout" if args.layout else "e2e"

    rows = dataset_rows(
        args.input_dataset, config=args.config, split=args.split,
        columns=[args.image_column], ids=args.id_column or "index",
        revision=args.input_revision, limit=args.limit,
    )
    if world > 1:
        rows = shard_select(rows, rank=rank, world=world)

    provenance = check_or_stage_weights(args.revision, args.expect_weights)

    def to_request(row):
        b64 = encode_jpeg(row[args.image_column], args.max_edge)
        return {"images": [f"data:image/jpeg;base64,{b64}"], "skip_layout": skip_layout}

    def parse(row, body):
        # /falconocr/parse: markdown_result is the assembled page; json_result carries the layout
        # regions (label/bbox/score/content) — kept verbatim so a run can be re-assembled under a
        # different reading-order or region policy without GPUs. The pipeline answers 200 with an
        # error in json_result when ITS call to vLLM fails under load; raising here makes that a
        # durable error row (re-admitted by --retry-errors) instead of a silent empty page.
        md = body.get("markdown_result")
        jr = body.get("json_result")
        if md is None and isinstance(jr, dict) and jr.get("error"):
            raise RuntimeError(f"pipeline error: {jr['error']}")
        return {
            "markdown": md or "",
            "regions_json": json.dumps(jr),
            "model": MODEL_ID,
            "model_commit": provenance["model_commit"],
            "weights_sha256": provenance["weights_sha256"],
            "image": SERVING["image"],
            "mode": mode,
            "output_tokens": body.get("total_output_tokens"),
            "processing_ms": body.get("processing_time_ms"),
        }

    output = f"hf://datasets/{args.output_dataset}/data"
    proc = boot_services()
    try:
        stats = pump(
            rows, to_request, parse, f"http://127.0.0.1:{SERVING['pipeline_port']}", output,
            route=SERVING["parse_route"],
            window=Auto(initial=SERVING["window_initial"], target_waiting=4,
                        max_limit=args.max_inflight, step=1),
            shard=(rank, world),
            retry_errors=args.retry_errors,
        )
    finally:
        proc.terminate()

    print(f"https://huggingface.co/datasets/{args.output_dataset} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows, mode={mode})", file=sys.stderr)
    print("FALCON_OCR_15_VENDOR_SATURATE " + stats.to_json(), flush=True)
    # Interpreter finalisation segfaults in this image (pyarrow/torch atexit clash, exit 139
    # AFTER a clean run), which would mark a successful job ERROR. Parts and stats are flushed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
