# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "requests",
# ]
#
# [tool.hf-jobs]
# image   = "ghcr.io/tiiuae/falcon-ocr@sha256:d0b4120bfd7b6fafdd9ae95124d32e8abdbb710ada6b4a303b33fc7636b5e1d7"
# flavor  = "a10g-small"
# timeout = "2h"
# secrets = ["HF_TOKEN"]
# ///

"""
Convert document images to markdown using Falcon OCR 1.5 through TII's own Docker image.

The other two Falcon recipes (falcon-ocr.py, falcon-ocr-1.5.py) drive the weights with the
`falcon-perception` package on the default uv image. This one runs the *vendor's deployable
product*: `ghcr.io/tiiuae/falcon-ocr` boots its own vLLM fork (float32, max_model_len 16k,
their chat template) plus a FastAPI pipeline (PP-DocLayoutV3 layout -> per-region OCR ->
markdown assembly). The script is the driver half: it boots the image's entrypoint as a
subprocess, waits for both health endpoints, posts pages to `/falconocr/parse` concurrently
and pushes the result dataset.

Two modes, per the model card:
- `e2e` (default): `skip_layout=true` — the whole page goes to the VLM in one shot. The v1.5
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
- Recorded in inference_info: image digest, weights sha256, resolved Hub commit (when staged).

Run on HF Jobs. The vendor image has NO `uv` and a Python 3.10 without ensurepip, so
`hf jobs uv run` cannot start on it (the CLI execs `uv run`). Bootstrap uv with pip first:

    hf jobs run --flavor a10g-small -s HF_TOKEN --timeout 2h \\
        ghcr.io/tiiuae/falcon-ocr@sha256:d0b4120bfd7b6fafdd9ae95124d32e8abdbb710ada6b4a303b33fc7636b5e1d7 \\
        -- bash -lc 'pip install -q uv && uv run \\
            https://huggingface.co/datasets/uv-scripts/ocr/raw/main/falcon-ocr-1.5-vendor.py \\
            input-dataset output-dataset --max-samples 10'

The `[tool.hf-jobs]` header records the same launch config for the day the image ships `uv`
(or `hf jobs uv run` learns to bootstrap it); until then it is documentation, not a launcher.
Plain `uv run` on a machine without the image fails fast in the preflight below.

Model: tiiuae/Falcon-OCR v1.5 (0.3B, Apache 2.0) as baked into the image digest above
Backend: vendor image (vLLM fork + falcon_ocr_sdk pipeline), HTTP driver
"""

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, Union

import requests
from datasets import load_dataset
from huggingface_hub import DatasetCard, HfApi, login
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "tiiuae/Falcon-OCR"
MODEL_LABEL = "Falcon-OCR-1.5 (vendor image)"

SERVING = {
    # Digest of ghcr.io/tiiuae/falcon-ocr that bakes the v1.5 weights (built 2026-09-12 02:20 +04).
    # `latest` is a moving target and old digests get pruned: this pin can rot, the hash cannot.
    "image": "ghcr.io/tiiuae/falcon-ocr@sha256:d0b4120bfd7b6fafdd9ae95124d32e8abdbb710ada6b4a303b33fc7636b5e1d7",
    # LFS sha256 of model.safetensors at the v1.5 release head fe757d59 (v1 = 6e7f73a5...).
    "weights_sha256": "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16",
    "v15_commit": "fe757d59ecd79d4d68760162306a70a015761ad9",
    "model_dir": "/models/Falcon-OCR",
    "entrypoint": "/app/entrypoint_single.sh",
    "vllm_port": 8000,
    "pipeline_port": 5002,
    "parse_route": "/falconocr/parse",
    # Payload-shrink only; the pipeline's layout detector resizes internally as it needs.
    "max_edge": 2048,
}

BOOT_TIMEOUT_S = 1800


def preflight() -> None:
    """Fail fast with the exact launch line when not inside the vendor image."""
    if os.path.isfile(SERVING["entrypoint"]) and os.path.isdir(SERVING["model_dir"]):
        return
    logger.error("This recipe only runs inside the TII Falcon-OCR image; entrypoint or model dir missing.")
    logger.error("Launch it with:")
    logger.error(
        "  hf jobs run --flavor a10g-small -s HF_TOKEN --timeout 2h %s -- "
        "bash -lc 'pip install -q uv && uv run "
        "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/falcon-ocr-1.5-vendor.py IN OUT'",
        SERVING["image"],
    )
    sys.exit(1)


def ensure_output_columns_free(dataset, columns, overwrite=False):
    """Fail fast if an output column would collide with an existing input column."""
    clash = [c for c in columns if c in dataset.column_names]
    if not clash:
        return dataset
    if overwrite:
        logger.warning(f"--overwrite: replacing existing column(s) {clash}")
        return dataset.remove_columns(clash)
    logger.error(
        f"Output column(s) {clash} already exist in the input dataset "
        f"(columns: {dataset.column_names})."
    )
    logger.error("Choose a different --output-column, or pass --overwrite to replace them.")
    sys.exit(1)


def to_pil_image(image: Union[Image.Image, Dict[str, Any], str]) -> Image.Image:
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return pil_img.convert("RGB")


def encode_jpeg(image, max_edge: int) -> str:
    """RGB-convert, downscale so the longest edge is max_edge, return a base64 JPEG (q95)."""
    img = to_pil_image(image)
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

    Returns provenance for inference_info. Fails loud on any mismatch: a silent v1/v1.5 mix-up
    is exactly the failure this recipe exists to prevent.
    """
    weights = os.path.join(SERVING["model_dir"], "model.safetensors")
    resolved_commit = None
    if revision:
        api = HfApi()
        resolved_commit = api.model_info(MODEL_ID, revision=revision).sha
        info = api.model_info(MODEL_ID, revision=resolved_commit, files_metadata=True)
        lfs = {s.rfilename: s.lfs for s in info.siblings if s.lfs}
        if "model.safetensors" not in lfs:
            raise RuntimeError(f"{MODEL_ID}@{resolved_commit} has no LFS model.safetensors")
        expect_sha = lfs["model.safetensors"].sha256
        logger.info(f"Staging {MODEL_ID}@{resolved_commit} over {SERVING['model_dir']}")
        from huggingface_hub import snapshot_download

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
    logger.info(f"Weights verified: sha256 {got[:12]}… ({'staged' if revision else 'baked in image'})")
    return {"weights_sha256": got, "model_commit": resolved_commit or SERVING["v15_commit"],
            "weights_source": "staged" if revision else "baked"}


def wait_healthy(url: str, deadline: float, proc: subprocess.Popen) -> None:
    last = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vendor entrypoint exited with code {proc.returncode} before {url} was healthy")
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.RequestException as e:
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
    logger.info(f"Service PATH: {env['PATH']}")
    env.setdefault("VLLM_GPU", "0")
    env.setdefault("PIPELINE_GPU", "0")
    proc = subprocess.Popen(["/bin/bash", SERVING["entrypoint"]], env=env)
    deadline = time.time() + BOOT_TIMEOUT_S
    wait_healthy(f"http://127.0.0.1:{SERVING['vllm_port']}/health", deadline, proc)
    wait_healthy(f"http://127.0.0.1:{SERVING['pipeline_port']}/health", deadline, proc)
    logger.info("Vendor services healthy (vLLM + pipeline)")
    return proc


def ocr_one(session: requests.Session, url: str, image, skip_layout: bool, max_edge: int,
            retries: int = 3) -> dict:
    """One page -> {markdown, regions_json, output_tokens, processing_ms, error}."""
    payload = {"images": [f"data:image/jpeg;base64,{encode_jpeg(image, max_edge)}"],
               "skip_layout": skip_layout}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.post(url, json=payload, timeout=600)
            r.raise_for_status()
            body = r.json()
            # The pipeline answers 200 with an error string in json_result when its own call to
            # vLLM fails under load ("Can not write request body"); treat that as retryable.
            md = body.get("markdown_result")
            jr = body.get("json_result")
            if md is None and isinstance(jr, dict) and jr.get("error"):
                raise RuntimeError(f"pipeline error: {jr['error']}")
            return {"markdown": md or "", "regions_json": json.dumps(jr),
                    "output_tokens": body.get("total_output_tokens"),
                    "processing_ms": body.get("processing_time_ms"), "error": None}
        except Exception as e:  # noqa: BLE001 - per-page retry; the class of error is logged
            last_err = e
            logger.warning(f"attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    return {"markdown": f"[OCR ERROR: {str(last_err)[:200]}]", "regions_json": None,
            "output_tokens": None, "processing_ms": None, "error": str(last_err)[:500]}


def create_dataset_card(source_dataset, num_samples, processing_time, image_column, split,
                        mode, provenance, concurrency) -> str:
    return f"""---
tags:
- ocr
- document-processing
- falcon-ocr
- falcon-ocr-1.5
- vendor-image
- uv-script
- generated
---

# Document Processing using Falcon OCR 1.5 (vendor image, {mode} mode)

OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using
[Falcon OCR 1.5](https://huggingface.co/tiiuae/Falcon-OCR) served by TII's own Docker image
(vLLM fork + layout/OCR pipeline).

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) — commit `{provenance["model_commit"]}` ({provenance["weights_source"]} weights, sha256 `{provenance["weights_sha256"][:12]}…`)
- **Image**: `{SERVING["image"]}`
- **Mode**: `{mode}` ({"whole page to the VLM, skip_layout=true" if mode == "e2e" else "layout detection + per-region OCR + markdown assembly"})
- **Concurrency**: {concurrency} requests in flight
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

## Reproduction

```bash
hf jobs run --flavor a10g-small -s HF_TOKEN --timeout 2h \\
    {SERVING["image"]} \\
    -- bash -lc 'pip install -q uv && uv run \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/falcon-ocr-1.5-vendor.py \\
        {source_dataset} <output-dataset> --image-column {image_column} --split {split}{"" if mode == "e2e" else " --layout"}'
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(input_dataset, output_dataset, image_column="image", split="train", max_samples=None,
         shuffle=False, seed=42, output_column="markdown", overwrite=False, private=False,
         hf_token=None, layout=False, concurrency=4, max_edge=SERVING["max_edge"],
         revision=None, expect_weights=SERVING["weights_sha256"], keep_regions=False):
    preflight()
    start_time = datetime.now()
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    mode = "layout" if layout else "e2e"
    logger.info(f"Loading dataset: {input_dataset} (split {split})")
    dataset = load_dataset(input_dataset, split=split)
    if image_column not in dataset.column_names:
        raise ValueError(f"Column '{image_column}' not found. Available: {dataset.column_names}")
    out_cols = [output_column] + ([f"{output_column}_regions"] if keep_regions else [])
    dataset = ensure_output_columns_free(dataset, out_cols, overwrite=overwrite)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"{len(dataset)} images, mode={mode}, concurrency={concurrency}")

    provenance = check_or_stage_weights(revision, expect_weights)
    proc = boot_services()
    url = f"http://127.0.0.1:{SERVING['pipeline_port']}{SERVING['parse_route']}"
    results: list[dict | None] = [None] * len(dataset)
    try:
        with requests.Session() as session, concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
            futures = {
                pool.submit(ocr_one, session, url, dataset[i][image_column], not layout, max_edge): i
                for i in range(len(dataset))
            }
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()
                done += 1
                if done % 10 == 0 or done == len(dataset):
                    logger.info(f"{done}/{len(dataset)} pages")
    finally:
        proc.terminate()

    errors = sum(1 for r in results if r["error"])
    elapsed = datetime.now() - start_time
    processing_time_str = f"{elapsed.total_seconds() / 60:.1f} min"
    logger.info(f"Done: {len(results)} pages, {errors} errors, {processing_time_str}")

    dataset = dataset.add_column(output_column, [r["markdown"] for r in results])
    if keep_regions:
        dataset = dataset.add_column(f"{output_column}_regions", [r["regions_json"] for r in results])

    inference_entry = {
        "model_id": MODEL_ID,
        "model_name": MODEL_LABEL,
        "model_size": "0.3B",
        "revision": revision or "baked",
        "model_commit": provenance["model_commit"],
        "weights_sha256": provenance["weights_sha256"],
        "image": SERVING["image"],
        "mode": mode,
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "backend": "vendor-image (vllm fork + falcon_ocr_sdk pipeline)",
        "errors": errors,
    }
    if "inference_info" in dataset.column_names:
        def update_inference_info(example):
            try:
                existing = json.loads(example["inference_info"]) if example["inference_info"] else []
            except (json.JSONDecodeError, TypeError):
                existing = []
            existing.append(inference_entry)
            return {"inference_info": json.dumps(existing)}
        dataset = dataset.map(update_inference_info)
    else:
        dataset = dataset.add_column("inference_info", [json.dumps([inference_entry])] * len(dataset))

    logger.info(f"Pushing to {output_dataset}")
    dataset.push_to_hub(output_dataset, private=private, token=token,
                        commit_message=f"Add {MODEL_LABEL} OCR results ({len(dataset)} samples)")
    card = DatasetCard(create_dataset_card(input_dataset, len(dataset), processing_time_str,
                                           image_column, split, mode, provenance, concurrency))
    card.push_to_hub(output_dataset, token=token)
    logger.info(f"Done: https://huggingface.co/datasets/{output_dataset}")
    # Interpreter finalisation segfaults in this image (pyarrow/torch atexit clash, exit 139
    # AFTER a clean push), which would mark a successful job ERROR. Everything is flushed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Falcon OCR 1.5 via TII's vendor image (vLLM fork + layout pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    parser.add_argument("input_dataset")
    parser.add_argument("output_dataset")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-column", default="markdown")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output column if it already exists in the input dataset")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--hf-token")
    parser.add_argument("--layout", action="store_true",
                        help="Use the layout+OCR pipeline instead of the card-recommended e2e mode")
    parser.add_argument("--keep-regions", action="store_true",
                        help="Also store the pipeline's json_result (layout regions) in <output-column>_regions")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Requests in flight (default 4; 8 produced pipeline->vLLM transport errors)")
    parser.add_argument("--max-edge", type=int, default=SERVING["max_edge"],
                        help="Downscale so the longest edge is at most this many px (default 2048)")
    parser.add_argument("--revision", default=None,
                        help="Stage this Hub commit/branch of tiiuae/Falcon-OCR over the baked weights "
                             "(hash-checked against the Hub LFS sha256). Default: use the baked weights "
                             "and require them to hash to the v1.5 release.")
    parser.add_argument("--expect-weights", default=SERVING["weights_sha256"],
                        help="Expected sha256 of the baked model.safetensors (default: v1.5)")
    args = parser.parse_args()
    main(input_dataset=args.input_dataset, output_dataset=args.output_dataset,
         image_column=args.image_column, split=args.split, max_samples=args.max_samples,
         shuffle=args.shuffle, seed=args.seed, output_column=args.output_column,
         overwrite=args.overwrite, private=args.private, hf_token=args.hf_token,
         layout=args.layout, concurrency=args.concurrency, max_edge=args.max_edge,
         revision=args.revision, expect_weights=args.expect_weights, keep_regions=args.keep_regions)
