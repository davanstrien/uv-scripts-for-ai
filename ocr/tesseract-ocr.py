# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytesseract",
#     "datasets>=3.1.0",
#     "huggingface-hub[hf_transfer]",
#     "pillow",
#     "tqdm",
# ]
# ///
"""
Plain-text OCR with **Tesseract** — the classical CPU OCR engine, as a baseline.

This is the odd one out in this collection: no GPU, no VLM. It runs Google's
Tesseract (v5, LSTM) over an image dataset and writes the recognised text to a
column, so you can put a cheap, fast, no-GPU baseline next to the VLM recipes
(e.g. in a per-collection leaderboard like ocr-bench). Output is plain text, not
markdown — Tesseract has no notion of tables/formulas/layout-as-markdown.

CPU-ONLY: this recipe deliberately does not require a GPU. Run it on a
`cpu-basic` / `cpu-upgrade` flavor. `--num-workers` fans OCR out across cores.

SYSTEM DEPENDENCY: the `tesseract` binary is NOT in the default Jobs image. On
startup this script installs it via apt (`tesseract-ocr`; Jobs containers run as
root). For non-English languages it also tries to install the matching
`tesseract-ocr-<lang>` data pack. If your Jobs image blocks apt, bake Tesseract
into a custom `--image` instead.

HF Jobs (CPU — no GPU needed):

    hf jobs uv run --flavor cpu-upgrade -s HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/tesseract-ocr.py \\
        input-dataset output-dataset \\
        --max-samples 100 --shuffle

Language packs (apt codes are 3-letter ISO 639-2, same as Tesseract's `--lang`):

    --lang eng            English (default; ships with the base package)
    --lang fra            French   (installs tesseract-ocr-fra)
    --lang eng+fra        multiple languages, '+'-joined

Engine: Tesseract OCR (https://github.com/tesseract-ocr/tesseract), Apache-2.0.
"""

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stable leaderboard label. Tesseract is not on the Hub, so this is deliberately
# NOT an org/name Hub id (faking one would produce broken links in dataset cards
# and a misleading leaderboard row). The precise installed version is recorded
# separately in inference_info as `tesseract_version`.
MODEL_ID = "tesseract-5"
MODEL_NAME = "Tesseract"
PROJECT_URL = "https://github.com/tesseract-ocr/tesseract"

# Error sentinel written to the output column when a single image fails, so one
# bad page never sinks the whole run (matches the other recipes in this repo).
OCR_ERROR = "[OCR ERROR]"


def ensure_tesseract_installed(lang: str) -> None:
    """Install the `tesseract` binary (and language data) if it's missing.

    The default Jobs image has no `tesseract`. Containers run as root, so apt
    works. The base `tesseract-ocr` package ships English; other languages need
    their own `tesseract-ocr-<code>` data pack. Base install is fatal on
    failure (nothing to OCR with); language-pack installs are best-effort (a
    missing pack surfaces as a clear Tesseract error at OCR time anyway).
    """
    requested = [c.strip() for c in lang.split("+") if c.strip()]

    if shutil.which("tesseract") is None:
        logger.info("`tesseract` not found — installing via apt (Jobs runs as root)...")
        try:
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(
                ["apt-get", "install", "-y", "-qq", "tesseract-ocr"], check=True
            )
        except Exception as e:
            logger.error(f"Failed to apt-install tesseract-ocr: {e}")
            logger.error(
                "If this Jobs image blocks apt, run with a custom --image that has "
                "Tesseract preinstalled (apt package `tesseract-ocr`)."
            )
            sys.exit(1)
        if shutil.which("tesseract") is None:
            logger.error("tesseract still not on PATH after install — aborting.")
            sys.exit(1)
        logger.info("Installed tesseract.")

    # Install any missing non-English language data packs (best-effort).
    try:
        import pytesseract

        available = set(pytesseract.get_languages(config=""))
    except Exception:
        available = set()
    missing = [c for c in requested if c not in available and c not in ("osd",)]
    if missing:
        pkgs = [f"tesseract-ocr-{c}" for c in missing]
        logger.info(f"Installing language data pack(s): {pkgs}")
        try:
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", *pkgs], check=True)
        except Exception as e:
            logger.warning(
                f"Could not install language pack(s) {pkgs}: {e}. "
                f"OCR will fail for those languages if the data is absent."
            )


def detect_tesseract_version() -> str:
    """Return the installed Tesseract version string (best-effort)."""
    try:
        import pytesseract

        return str(pytesseract.get_tesseract_version())
    except Exception:
        return "unknown"


def ensure_output_columns_free(dataset, columns, overwrite=False):
    """Fail fast if an output column would collide with an existing input column.

    Adding a column that already exists silently overwrites it (e.g. a
    ground-truth `text`/`markdown` column) or crashes on push with a
    duplicate-column error only *after* OCR has run. Catch it up front. With
    overwrite=True, drop the clashing column(s) here instead (logged).
    """
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
    logger.error(
        "Choose a different --output-column, or pass --overwrite to replace them."
    )
    sys.exit(1)


def to_pil(image: Union[Image.Image, Dict[str, Any], str]) -> Image.Image:
    """Coerce a datasets image cell to an RGB PIL image.

    Handles the three shapes a HF image column yields: a decoded PIL image, a
    `{"bytes": ...}` dict, or a file path string.
    """
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return pil_img.convert("RGB")


def ocr_image(
    image: Union[Image.Image, Dict[str, Any], str],
    lang: str,
    config: str,
) -> str:
    """Run Tesseract on a single image, returning stripped text (or the error sentinel)."""
    import pytesseract

    try:
        pil_img = to_pil(image)
        return pytesseract.image_to_string(pil_img, lang=lang, config=config).strip()
    except Exception as e:  # noqa: BLE001 — one bad page shouldn't sink the run
        logger.error(f"Error OCR'ing image: {e}")
        return OCR_ERROR


def run_ocr(
    dataset,
    image_column: str,
    lang: str,
    config: str,
    num_workers: int,
) -> List[str]:
    """OCR every row's image, preserving dataset order.

    Tesseract shells out to a subprocess, so ThreadPoolExecutor gives real
    parallelism (the GIL is released during the call). We process in chunks so
    only a chunk's worth of decoded images is held in memory at once — the full
    dataset stays on disk.
    """
    n = len(dataset)
    results: List[str] = []
    chunk = max(num_workers * 4, 16)

    with tqdm(total=n, desc="OCR", unit="img") as pbar:
        if num_workers <= 1:
            for i in range(n):
                results.append(ocr_image(dataset[i][image_column], lang, config))
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for start in range(0, n, chunk):
                    stop = min(start + chunk, n)
                    images = [dataset[i][image_column] for i in range(start, stop)]
                    for text in pool.map(
                        lambda img: ocr_image(img, lang, config), images
                    ):
                        results.append(text)
                        pbar.update(1)
    return results


def create_dataset_card(
    source_dataset: str,
    num_samples: int,
    processing_time: str,
    tesseract_version: str,
    lang: str,
    psm: int,
    oem: int,
    num_workers: int,
    output_column: str,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the Tesseract OCR run."""
    return f"""---
tags:
- ocr
- text-recognition
- tesseract
- uv-script
- generated
---

# Document OCR using Tesseract

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using [Tesseract]({PROJECT_URL}), the classical open-source CPU OCR engine — a cheap, no-GPU baseline alongside the VLM OCR recipes.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Engine**: [Tesseract]({PROJECT_URL}) `{tesseract_version}`
- **Language(s)**: `{lang}`
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `{output_column}`
- **Dataset Split**: `{split}`
- **Page Segmentation Mode (psm)**: {psm}
- **OCR Engine Mode (oem)**: {oem}
- **Workers**: {num_workers}

## Model Information

Tesseract is a classical (non-VLM) OCR engine:
- Runs on CPU — no GPU required
- v4+ uses an LSTM-based recognition engine
- 100+ languages via installable data packs
- Plain-text output (no markdown / table / formula structure)
- Apache-2.0 licensed

## Dataset Structure

The dataset contains all original columns plus:
- `{output_column}`: The recognised text (plain text)
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Reproduction

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/tesseract-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --lang {lang} \\
    --psm {psm}
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    lang: str = "eng",
    psm: int = 3,
    oem: int = 3,
    num_workers: int = 0,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    config: str = None,
    create_pr: bool = False,
):
    """Process images from an HF dataset through Tesseract OCR."""

    start_time = datetime.now(timezone.utc)

    # Fan-out defaults to all cores. When running >1 worker, cap Tesseract's own
    # OpenMP threads to 1 so N processes don't oversubscribe the CPU (per the
    # Tesseract FAQ). Must be set before the binary is invoked.
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1
    if num_workers > 1:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")

    ensure_tesseract_installed(lang)
    tesseract_version = detect_tesseract_version()
    logger.info(
        f"Using Tesseract {tesseract_version} (lang={lang}, psm={psm}, oem={oem})"
    )
    logger.info(f"CPU-only run with {num_workers} worker(s)")

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN and not dry_run:
        login(token=HF_TOKEN)

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Fail fast if the output column would collide with an existing input column
    dataset = ensure_output_columns_free(dataset, [output_column], overwrite=overwrite)

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    tess_config = f"--psm {psm} --oem {oem}"
    logger.info(f"Processing {len(dataset)} images -> column '{output_column}'")

    all_outputs = run_ocr(dataset, image_column, lang, tess_config, num_workers)

    n_errors = sum(1 for t in all_outputs if t == OCR_ERROR)
    if n_errors:
        logger.warning(f"{n_errors}/{len(all_outputs)} images failed OCR ({OCR_ERROR})")

    processing_duration = datetime.now(timezone.utc) - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Inference info tracking. `model_id` + `column_name` are the fields consumers
    # (e.g. ocr-bench) read to find the text column and the leaderboard label.
    inference_entry = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "column_name": output_column,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": "tesseract",
        "tesseract_version": tesseract_version,
        "lang": lang,
        "psm": psm,
        "oem": oem,
    }

    if "inference_info" in dataset.column_names:
        logger.info("Updating existing inference_info column")

        def update_inference_info(example):
            try:
                existing_info = (
                    json.loads(example["inference_info"])
                    if example["inference_info"]
                    else []
                )
            except (json.JSONDecodeError, TypeError):
                existing_info = []
            existing_info.append(inference_entry)
            return {"inference_info": json.dumps(existing_info)}

        dataset = dataset.map(update_inference_info)
    else:
        logger.info("Creating new inference_info column")
        inference_list = [json.dumps([inference_entry])] * len(dataset)
        dataset = dataset.add_column("inference_info", inference_list)

    if dry_run:
        logger.info("--dry-run: skipping push to Hub.")
        preview = next((t for t in all_outputs if t and t != OCR_ERROR), "")
        logger.info(f"Sample OCR output (first non-empty, truncated):\n{preview[:500]}")
        logger.info(
            f"Done (dry run). {len(dataset)} rows OCR'd in {processing_time_str}."
        )
        return

    # Push to hub with retry and XET fallback
    logger.info(f"Pushing to {output_dataset}")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            dataset.push_to_hub(
                output_dataset,
                private=private,
                token=HF_TOKEN,
                max_shard_size="500MB",
                **({"config_name": config} if config else {}),
                create_pr=create_pr,
                commit_message=f"Add {MODEL_ID} OCR results ({len(dataset)} samples)"
                + (f" [{config}]" if config else ""),
            )
            break
        except Exception as e:
            logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = 30 * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("All upload attempts failed. OCR results are lost.")
                sys.exit(1)

    # Create and push dataset card
    logger.info("Creating dataset card")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        num_samples=len(dataset),
        processing_time=processing_time_str,
        tesseract_version=tesseract_version,
        lang=lang,
        psm=psm,
        oem=oem,
        num_workers=num_workers,
        output_column=output_column,
        image_column=image_column,
        split=split,
    )
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Done! Tesseract OCR processing complete.")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")
    if processing_duration.total_seconds() > 0:
        logger.info(
            f"Processing speed: {len(dataset) / processing_duration.total_seconds():.2f} images/sec"
        )

    if verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in ["pytesseract", "datasets", "pyarrow", "pillow"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info(f"  tesseract=={tesseract_version}")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 70)
        print("Tesseract OCR — CPU baseline")
        print("=" * 70)
        print("\nClassical LSTM OCR engine. No GPU. Plain-text output.")
        print("\nExamples:")
        print("\n1. Basic OCR (English):")
        print("   uv run tesseract-ocr.py input-dataset output-dataset")
        print("\n2. Another language (installs the data pack on Jobs):")
        print("   uv run tesseract-ocr.py docs results --lang fra")
        print("\n3. Test with a small sample, no push:")
        print("   uv run tesseract-ocr.py large-dataset out --max-samples 5 --dry-run")
        print("\n4. Running on HF Jobs (CPU flavor):")
        print("   hf jobs uv run --flavor cpu-upgrade \\")
        print("     -s HF_TOKEN \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/tesseract-ocr.py \\"
        )
        print("       input-dataset output-dataset --max-samples 100 --shuffle")
        print("\nFor full help: uv run tesseract-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="CPU OCR baseline using Tesseract (classical LSTM engine, plain text)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run tesseract-ocr.py my-docs analyzed-docs
  uv run tesseract-ocr.py docs results --lang eng+fra --psm 4
  uv run tesseract-ocr.py large-dataset test --max-samples 50 --shuffle --dry-run

Page segmentation modes (--psm), common ones:
  3  Fully automatic page segmentation, no OSD (default; good for full pages)
  4  Assume a single column of text of variable sizes
  6  Assume a single uniform block of text
  1  Automatic page segmentation with OSD
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID from Hugging Face Hub")
    parser.add_argument("output_dataset", help="Output dataset ID for Hugging Face Hub")
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (default: image)",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language code(s), '+'-joined (default: eng). "
        "Non-English packs are apt-installed on Jobs.",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=3,
        help="Page segmentation mode (default: 3, full-page auto)",
    )
    parser.add_argument(
        "--oem",
        type=int,
        default=3,
        help="OCR engine mode (default: 3, based on what's available; 1=LSTM only)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel OCR workers (default: 0 = all CPU cores)",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token")
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--config",
        help="Config/subset name when pushing to Hub (for benchmarking multiple models in one repo)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of pushing directly (for parallel benchmarking)",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Column name for output text (default: markdown, for cross-model consistency)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run OCR but do NOT push to the Hub (local smoke testing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after processing (useful for pinning deps)",
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        num_workers=args.num_workers,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verbose=args.verbose,
        config=args.config,
        create_pr=args.create_pr,
    )
