# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "paddlepaddle-gpu>=3.0.0",
#     "paddleocr>=3.7.0",
#     "paddlex[ocr]>=3.7.0",
#     "opencv-contrib-python-headless",
#     "datasets>=3.1.0",
#     "huggingface-hub",
#     "pillow",
#     "numpy",
#     "tqdm",
# ]
#
# [tool.uv]
# # PaddleOCR/PaddleX pull in opencv-contrib-python (full) which needs system
# # libGL.so.1 — not present in the slim uv-on-bookworm image used by HF Jobs.
# # Swap to the headless cv2 variant (same `import cv2`, no GUI deps). A matching
# # importlib.metadata patch in main() makes paddlex recognise the headless name.
# override-dependencies = [
#     "opencv-contrib-python ; python_version < '0'",
#     "opencv-python ; python_version < '0'",
# ]
#
# [[tool.uv.index]]
# name = "paddle"
# url = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
# explicit = true
#
# [tool.uv.sources]
# paddlepaddle-gpu = { index = "paddle" }
# ///
"""
OCR images with PP-OCRv6 — a lightweight detection+recognition pipeline from
PaddlePaddle. Three tiers from **1.5M to 34.5M parameters**.

Unlike the VLM-based OCR recipes here, PP-OCRv6 is a **classical det+rec pipeline**
that outputs **plain text** (not markdown). At 1.5M-34.5M params it's far smaller
than the VLM OCRs and runs on a cheap t4-small GPU.

Model tiers (pick with `--model-tier`):
  tiny    1.5M params  (0.4M det + 1.1M rec)  49 languages, ~73% recognition
  small   7.7M params  (2.5M det + 5.3M rec)  50 languages, ~81% recognition
  medium  34.5M params (22M det + 19M rec)     50 languages, ~83% recognition

All tiers are Apache 2.0 licensed. Runs via PaddleOCR's default Paddle engine
(`paddle_static`) — same proven header pattern as `pp-doclayout.py`.

HF Jobs examples:

    # Tiny on a cheap GPU
    hf jobs uv run --flavor t4-small -s HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-ocrv6.py \\
        INPUT_DATASET OUTPUT_DATASET \\
        --model-tier tiny --max-samples 5

    # Medium on a small GPU (recommended for quality)
    hf jobs uv run --flavor t4-small -s HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-ocrv6.py \\
        INPUT_DATASET OUTPUT_DATASET \\
        --model-tier medium --max-samples 10

Models: PaddlePaddle/PP-OCRv6_<tier>_det + PP-OCRv6_<tier>_rec
Blog: https://huggingface.co/blog/PaddlePaddle/pp-ocrv6
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_MODELS = {
    "tiny":   ("PP-OCRv6_tiny_det",   "PP-OCRv6_tiny_rec"),
    "small":  ("PP-OCRv6_small_det",  "PP-OCRv6_small_rec"),
    "medium": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}

TIER_PARAMS = {
    "tiny":   "1.5M (0.4M det + 1.1M rec)",
    "small":  "7.7M (2.5M det + 5.3M rec)",
    "medium": "34.5M (22M det + 19M rec)",
}

TIER_LANGUAGES = {
    "tiny":   "49 languages (zh, zh-Hant, en + 46 Latin-script — no Japanese)",
    "small":  "50 languages (zh, zh-Hant, en, ja + 46 Latin-script)",
    "medium": "50 languages (zh, zh-Hant, en, ja + 46 Latin-script)",
}

TIER_REC = {
    "tiny":   73.5,
    "small":  81.3,
    "medium": 83.2,
}

BUCKET_PREFIX = "hf://buckets/"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".jp2", ".j2k",
}


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def is_bucket_url(s: str) -> bool:
    return s.startswith(BUCKET_PREFIX)


def parse_bucket_url(url: str) -> Tuple[str, str]:
    if not is_bucket_url(url):
        raise ValueError(f"Not a bucket URL: {url}")
    rest = url[len(BUCKET_PREFIX):].strip("/")
    parts = rest.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Bucket URL must include namespace and bucket name: {url}")
    bucket_id = f"{parts[0]}/{parts[1]}"
    prefix = parts[2] if len(parts) > 2 else ""
    return bucket_id, prefix


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_pil(image: Union[Image.Image, Dict[str, Any], str, bytes]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image)}")


def pil_to_array(pil_img: Image.Image) -> np.ndarray:
    return np.asarray(pil_img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def extract_text(result: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Pull text and per-line details from a PaddleOCR predict result.

    Returns (concatenated_text, per_line_details) where per_line_details is
    a list of dicts with keys: text, score, bbox (4-point detection polygon as
    [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in input-image pixel coordinates).
    """
    payload = result.json if hasattr(result, "json") else result
    res = payload.get("res", payload) if isinstance(payload, dict) else {}
    rec_texts = res.get("rec_texts", []) or []
    rec_scores = res.get("rec_scores", []) or []
    dt_polys = res.get("dt_polys", []) or []

    # Concatenate reading-order text lines (PaddleOCR returns them in order)
    text = "\n".join(rec_texts)

    per_line = []
    for i, t in enumerate(rec_texts):
        entry = {"text": t}
        if i < len(rec_scores):
            entry["score"] = float(rec_scores[i])
        if i < len(dt_polys):
            entry["bbox"] = [[float(c) for c in point] for point in dt_polys[i]]
        per_line.append(entry)

    return text, per_line


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@dataclass
class SourceItem:
    key: str
    image: Optional[Image.Image]
    extras: Dict[str, Any]


def iter_dataset_images(
    dataset_id: str,
    image_column: str,
    split: str,
    shuffle: bool,
    seed: int,
    max_samples: Optional[int],
):
    from datasets import load_dataset

    logger.info(f"Loading dataset: {dataset_id} (split={split})")
    ds = load_dataset(dataset_id, split=split)

    if image_column not in ds.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {ds.column_names}"
        )

    if shuffle:
        logger.info(f"Shuffling with seed {seed}")
        ds = ds.shuffle(seed=seed)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
        logger.info(f"Limited to {len(ds)} samples")

    total = len(ds)

    def gen() -> Iterator[SourceItem]:
        failed = 0
        for i in range(total):
            try:
                row = ds[i]
                image = to_pil(row[image_column])
            except (UnidentifiedImageError, OSError) as e:
                # Still yield a placeholder so the output row stays aligned with
                # the source row (the dataset sink writes results positionally).
                failed += 1
                logger.warning(
                    f"Unreadable image at row {i}: {type(e).__name__}: {e} "
                    f"— writing empty result"
                )
                yield SourceItem(key=f"row-{i:08d}", image=None, extras={"failed": True})
                continue
            yield SourceItem(key=f"row-{i:08d}", image=image, extras={})
        if failed:
            logger.info(f"{failed} unreadable image(s) written as empty results")

    return gen(), total, ds


SOURCE_PATHS_SNAPSHOT = "_source_paths.json"


def _bucket_snapshot_path(output_url: str) -> Tuple[str, str]:
    out_bucket_id, out_prefix = parse_bucket_url(output_url)
    snapshot_key = (
        f"{out_prefix}/{SOURCE_PATHS_SNAPSHOT}".lstrip("/")
        if out_prefix
        else SOURCE_PATHS_SNAPSHOT
    )
    return out_bucket_id, snapshot_key


def iter_bucket_images(
    bucket_url: str,
    shuffle: bool,
    seed: int,
    max_samples: Optional[int],
    hf_token: Optional[str],
    output_url: Optional[str] = None,
) -> Tuple[Iterator[SourceItem], int]:
    from huggingface_hub import HfApi, HfFileSystem

    bucket_id, prefix = parse_bucket_url(bucket_url)
    fs = HfFileSystem(token=hf_token)
    base = f"{BUCKET_PREFIX}{bucket_id}/{prefix}".rstrip("/")

    snapshot_bucket_id: Optional[str] = None
    snapshot_key: Optional[str] = None
    cached_paths: Optional[List[str]] = None

    if output_url and is_bucket_url(output_url):
        snapshot_bucket_id, snapshot_key = _bucket_snapshot_path(output_url)
        snapshot_url = f"{BUCKET_PREFIX}{snapshot_bucket_id}/{snapshot_key}"
        try:
            with fs.open(snapshot_url, "rb") as f:
                snapshot = json.load(f)
            mismatches = []
            if snapshot.get("source_url") != bucket_url:
                mismatches.append(
                    f"source_url ({snapshot.get('source_url')!r} vs {bucket_url!r})"
                )
            if snapshot.get("shuffle") != shuffle:
                mismatches.append(f"shuffle ({snapshot.get('shuffle')} vs {shuffle})")
            if shuffle and snapshot.get("seed") != seed:
                mismatches.append(f"seed ({snapshot.get('seed')} vs {seed})")
            if snapshot.get("max_samples") != max_samples:
                mismatches.append(
                    f"max_samples ({snapshot.get('max_samples')} vs {max_samples})"
                )
            if mismatches:
                logger.warning(
                    "Existing snapshot params differ from this run ("
                    + "; ".join(mismatches)
                    + "); ignoring snapshot and re-listing."
                )
            else:
                cached_paths = snapshot["paths"]
                logger.info(
                    f"Reusing existing snapshot of {len(cached_paths)} source paths "
                    f"(written {snapshot.get('created_at', 'unknown')})"
                )
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not read existing snapshot ({e}); re-listing.")

    if cached_paths is not None:
        all_paths = cached_paths
    else:
        logger.info(f"Listing images under {base}")
        all_paths = []
        try:
            for entry in fs.find(base, detail=False):
                ext = Path(entry).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    all_paths.append(entry)
        except FileNotFoundError as e:
            raise ValueError(f"Bucket prefix not found: {base}") from e

        if not all_paths:
            raise ValueError(
                f"No image files (any of {sorted(IMAGE_EXTENSIONS)}) under {base}"
            )

        all_paths.sort()
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(all_paths)
        if max_samples:
            all_paths = all_paths[:max_samples]

        if snapshot_bucket_id is not None and snapshot_key is not None:
            api = HfApi(token=hf_token)
            payload = {
                "source_url": bucket_url,
                "shuffle": shuffle,
                "seed": seed,
                "max_samples": max_samples,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "paths": all_paths,
            }
            api.batch_bucket_files(
                snapshot_bucket_id,
                add=[(json.dumps(payload).encode(), snapshot_key)],
                token=hf_token,
            )
            logger.info(
                f"Wrote source-path snapshot ({len(all_paths)} paths) to "
                f"hf://buckets/{snapshot_bucket_id}/{snapshot_key}"
            )

    total = len(all_paths)
    logger.info(f"Found {total} images in bucket")

    def key_for(path: str) -> str:
        return path

    def gen() -> Iterator[SourceItem]:
        skipped = 0
        for path in all_paths:
            try:
                with fs.open(path, "rb") as f:
                    data = f.read()
                image = to_pil(data)
            except (UnidentifiedImageError, OSError) as e:
                skipped += 1
                logger.warning(
                    f"Skipping unreadable image {path}: {type(e).__name__}: {e}"
                )
                continue
            yield SourceItem(key=key_for(path), image=image, extras={})
        if skipped:
            logger.info(f"Skipped {skipped} unreadable image(s) total")

    return gen(), total


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

class DatasetRepoSink:
    def __init__(
        self,
        repo_id: str,
        *,
        hf_token: Optional[str],
        private: bool,
        config: Optional[str],
        create_pr: bool,
        source_id: str,
        original_dataset=None,
        output_column: str = "markdown",
        overwrite: bool = False,
    ):
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.private = private
        self.config = config
        self.create_pr = create_pr
        self.source_id = source_id
        self.original_dataset = original_dataset
        self.output_column = output_column
        self.overwrite = overwrite
        self._texts: List[str] = []
        self._blocks: List[str] = []

    @property
    def kind(self) -> str:
        return "dataset"

    def already_done(self) -> set:
        return set()

    def write(self, key: str, text: str, blocks: List[Dict[str, Any]]) -> None:
        self._texts.append(text)
        self._blocks.append(json.dumps(blocks, ensure_ascii=False))

    def finalize(self, tier: str, det_model: str, rec_model: str, args_dict: Dict[str, Any]) -> None:
        from datasets import Dataset

        if self.original_dataset is not None:
            if len(self._texts) != len(self.original_dataset):
                logger.warning(
                    f"Text count ({len(self._texts)}) != dataset rows "
                    f"({len(self.original_dataset)}); padding with empty strings."
                )
                while len(self._texts) < len(self.original_dataset):
                    self._texts.append("")
                    self._blocks.append("[]")
            # Guard again at save time in case the input column set changed under us.
            base = self.original_dataset
            clash = [c for c in (self.output_column, "pp_ocr_blocks") if c in base.column_names]
            if clash:
                if not self.overwrite:
                    raise ValueError(
                        f"Output column(s) {clash} already exist in the input dataset; "
                        f"pass a different --output-column, or --overwrite to replace them."
                    )
                logger.warning(f"--overwrite: replacing existing column(s) {clash}")
                base = base.remove_columns(clash)
            ds = base.add_column(self.output_column, self._texts)
            ds = ds.add_column("pp_ocr_blocks", self._blocks)
        else:
            if not self._texts:
                logger.warning("No rows produced; nothing to push.")
                return
            ds = Dataset.from_list([
                {"source_path": None, self.output_column: t, "pp_ocr_blocks": b}
                for t, b in zip(self._texts, self._blocks)
            ])

        inference_entry = build_inference_entry(tier, det_model, rec_model, args_dict)

        if "inference_info" in ds.column_names:
            logger.info("Updating existing inference_info column")

            def _update(example):
                try:
                    existing = (
                        json.loads(example["inference_info"])
                        if example["inference_info"]
                        else []
                    )
                except (json.JSONDecodeError, TypeError):
                    existing = []
                existing.append(inference_entry)
                return {"inference_info": json.dumps(existing)}

            ds = ds.map(_update)
        else:
            ds = ds.add_column(
                "inference_info", [json.dumps([inference_entry])] * len(ds)
            )

        logger.info(f"Pushing {len(ds)} rows to {self.repo_id}")
        push_kwargs = {
            "private": self.private,
            "token": self.hf_token,
            "max_shard_size": "500MB",
            "create_pr": self.create_pr,
            "commit_message": f"Add PP-OCRv6-{tier} OCR results ({len(ds)} samples)"
            + (f" [{self.config}]" if self.config else ""),
        }
        if self.config:
            push_kwargs["config_name"] = self.config

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    logger.warning("Disabling XET (fallback to HTTP upload)")
                    os.environ["HF_HUB_DISABLE_XET"] = "1"
                ds.push_to_hub(self.repo_id, **push_kwargs)
                break
            except Exception as e:
                logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    logger.error("All upload attempts failed.")
                    raise
                time.sleep(30 * (2 ** (attempt - 1)))

        from huggingface_hub import DatasetCard

        card = DatasetCard(
            create_dataset_card(
                source=self.source_id,
                tier=tier,
                det_model=det_model,
                rec_model=rec_model,
                num_samples=len(ds),
                processing_time=args_dict["processing_time"],
                engine=args_dict.get("engine", "paddle_static"),
                output_id=self.repo_id,
                output_column=self.output_column,
            )
        )
        card.push_to_hub(self.repo_id, token=self.hf_token)
        logger.info(f"Done: https://huggingface.co/datasets/{self.repo_id}")


class BucketShardSink:
    METADATA_FILE = "_metadata.json"
    SHARD_PATTERN = "shard-{:05d}.parquet"

    def __init__(
        self,
        bucket_url: str,
        *,
        hf_token: Optional[str],
        shard_size: int,
        resume: bool,
        source_id: str,
    ):
        from huggingface_hub import HfApi, HfFileSystem, create_bucket

        self.bucket_url = bucket_url
        self.bucket_id, self.prefix = parse_bucket_url(bucket_url)
        self.hf_token = hf_token
        self.shard_size = shard_size
        self.resume = resume
        self.source_id = source_id

        self._api = HfApi(token=hf_token)
        self._fs = HfFileSystem(token=hf_token)

        try:
            create_bucket(self.bucket_id, exist_ok=True, token=hf_token)
        except Exception as e:
            logger.warning(f"create_bucket('{self.bucket_id}') warning: {e}")

        self._buffer: List[Dict[str, Any]] = []
        self._next_shard_idx = self._discover_next_shard_idx()
        self._completed_keys = self._discover_completed_keys() if resume else set()
        if self._completed_keys:
            logger.info(
                f"Resume: found {len(self._completed_keys)} already-processed keys, will skip them"
            )

    @property
    def kind(self) -> str:
        return "bucket"

    def already_done(self) -> set:
        return self._completed_keys

    def _shard_path(self, idx: int) -> str:
        return self._join(self.SHARD_PATTERN.format(idx))

    def _join(self, name: str) -> str:
        return f"{self.prefix}/{name}".lstrip("/") if self.prefix else name

    def _list_existing_shards(self) -> List[str]:
        try:
            tree = self._api.list_bucket_tree(
                self.bucket_id, prefix=self.prefix or None, recursive=True
            )
        except Exception:
            return []
        shards: List[str] = []
        for item in tree:
            path = getattr(item, "path", None)
            ftype = getattr(item, "type", None)
            if not path or ftype not in (None, "file"):
                continue
            base = Path(path).name
            if base.startswith("shard-") and base.endswith(".parquet"):
                shards.append(path)
        return sorted(shards)

    def _discover_next_shard_idx(self) -> int:
        shards = self._list_existing_shards()
        max_idx = -1
        for s in shards:
            stem = Path(s).stem
            try:
                max_idx = max(max_idx, int(stem.split("-")[-1]))
            except ValueError:
                continue
        return max_idx + 1

    def _discover_completed_keys(self) -> set:
        import pyarrow.parquet as pq

        keys: set = set()
        for shard_path in self._list_existing_shards():
            full = f"{BUCKET_PREFIX}{self.bucket_id}/{shard_path}"
            try:
                with self._fs.open(full, "rb") as f:
                    table = pq.read_table(f, columns=["__source_key"])
                keys.update(table.column("__source_key").to_pylist())
            except Exception as e:
                logger.warning(f"Could not read keys from {shard_path}: {e}")
        return keys

    def _flush(self) -> None:
        if not self._buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = ["__source_key", "text", "pp_ocr_blocks"]
        table_dict = {c: [row.get(c) for row in self._buffer] for c in columns}
        table = pa.Table.from_pydict(table_dict)

        buf = io.BytesIO()
        pq.write_table(table, buf, compression="zstd")
        data = buf.getvalue()

        shard_remote = self._shard_path(self._next_shard_idx)
        logger.info(
            f"Writing shard {self._next_shard_idx} ({len(self._buffer)} rows, "
            f"{len(data) / 1024 / 1024:.1f} MiB) to {shard_remote}"
        )
        self._api.batch_bucket_files(
            self.bucket_id, add=[(data, shard_remote)], token=self.hf_token
        )
        self._next_shard_idx += 1
        self._buffer.clear()

    def write(self, key: str, text: str, blocks: List[Dict[str, Any]]) -> None:
        row: Dict[str, Any] = {
            "__source_key": key,
            "text": text,
            "pp_ocr_blocks": json.dumps(blocks, ensure_ascii=False),
        }
        self._buffer.append(row)
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def finalize(self, tier: str, det_model: str, rec_model: str, args_dict: Dict[str, Any]) -> None:
        self._flush()
        meta = {
            "model": f"PP-OCRv6_{tier}",
            "det_model": det_model,
            "rec_model": rec_model,
            "tier": tier,
            "engine": "paddle_static",
            "source": self.source_id,
            "shard_size": args_dict["shard_size"],
            "last_run_at": datetime.now(timezone.utc).isoformat(),
            "processing_time": args_dict.get("processing_time"),
        }
        meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
        meta_path = self._join(self.METADATA_FILE)
        self._api.batch_bucket_files(
            self.bucket_id, add=[(meta_bytes, meta_path)], token=self.hf_token
        )
        logger.info(
            f"Done: https://huggingface.co/buckets/{self.bucket_id}"
            + (f"/{self.prefix}" if self.prefix else "")
        )


# ---------------------------------------------------------------------------
# inference_info + dataset card
# ---------------------------------------------------------------------------

def build_inference_entry(tier: str, det_model: str, rec_model: str, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_id": f"PaddlePaddle/PP-OCRv6_{tier}",
        "det_model": det_model,
        "rec_model": rec_model,
        "tier": tier,
        "params": TIER_PARAMS.get(tier, "unknown"),
        "rec_accuracy_pct": TIER_REC.get(tier),
        "languages": TIER_LANGUAGES.get(tier, ""),
        "engine": "paddle_static",
        # column_name is the key ocr-bench's column discovery reads; keep
        # output_column too for backward compat with existing outputs.
        "column_name": args_dict.get("output_column", "markdown"),
        "output_column": args_dict.get("output_column", "markdown"),
        "blocks_column": "pp_ocr_blocks",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def create_dataset_card(
    source: str,
    tier: str,
    det_model: str,
    rec_model: str,
    num_samples: int,
    processing_time: str,
    engine: str,
    output_id: str,
    output_column: str = "markdown",
) -> str:
    tier_display = tier.upper() if tier == "tiny" else tier.capitalize()
    if is_bucket_url(source):
        source_link = f"[{source}]({source})"
    else:
        source_link = f"[{source}](https://huggingface.co/datasets/{source})"

    return f"""---
tags:
- ocr
- text-recognition
- paddleocr
- pp-ocrv6
- uv-script
- generated
---

# OCR with PP-OCRv6 {tier_display}

Plain-text OCR results for images from {source_link}, produced by
PaddlePaddle's [PP-OCRv6](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6)
{tier} pipeline ({TIER_PARAMS.get(tier, "unknown")}).

## Processing details

- **Source**: {source_link}
- **Model**: PP-OCRv6_{tier} ({det_model} + {rec_model})
- **Tier**: {tier} ({TIER_PARAMS.get(tier, "unknown")})
- **Recognition accuracy**: {TIER_REC.get(tier, "?"):.1f}%
- **Languages**: {TIER_LANGUAGES.get(tier, "")}
- **Engine**: {engine}
- **Samples**: {num_samples:,}
- **Processing time**: {processing_time}
- **Processing date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
- **License**: Apache 2.0 (models)

## Schema

Each row contains the original columns plus:

- `{output_column}`: Plain text extracted from the image (reading-order concatenation of
  detected text lines, newline-separated).
- `pp_ocr_blocks`: JSON list, one dict per detected text line:
  ```json
  [
    {{
      "text": "recognized text",
      "score": 0.987,
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }}
  ]
  ```
  `score` is the recognition confidence and `bbox` is the detection polygon
  (4-point quadrilateral in input-image pixel coordinates).
- `inference_info`: JSON list tracking every model applied to this dataset.

> **Note:** PP-OCRv6 is a classical detection+recognition pipeline, not a VLM.
> It outputs **plain text** rather than markdown. Per-line bounding boxes and
> confidence scores are available in `pp_ocr_blocks`.

## Usage

```python
import json
from datasets import load_dataset

ds = load_dataset("{output_id}", split="train")
print(ds[0]["{output_column}"])
for block in json.loads(ds[0]["pp_ocr_blocks"]):
    print(block["text"], block["score"])
```

## Reproduction

```bash
hf jobs uv run --flavor t4-small -s HF_TOKEN \\
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-ocrv6.py \\
    {source} <output> --model-tier {tier}
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts).
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    from huggingface_hub import login

    start_time = datetime.now()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # ---------- tier → model names ----------
    if args.model_tier not in TIER_MODELS:
        raise ValueError(
            f"Invalid tier {args.model_tier!r}. Choose from: {list(TIER_MODELS)}"
        )
    det_model, rec_model = TIER_MODELS[args.model_tier]
    tier = args.model_tier
    logger.info(f"PP-OCRv6 {tier}: {det_model} + {rec_model}")

    # ---------- source ----------
    original_dataset = None
    if is_bucket_url(args.input_source):
        src_iter, total = iter_bucket_images(
            args.input_source,
            shuffle=args.shuffle,
            seed=args.seed,
            max_samples=args.max_samples,
            hf_token=hf_token,
            output_url=args.output_target,
        )
    else:
        src_iter, total, original_dataset = iter_dataset_images(
            args.input_source,
            image_column=args.image_column,
            split=args.split,
            shuffle=args.shuffle,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        # Fail fast, before minutes of inference, if the output column would collide
        # with an existing input column (e.g. a 'text' ground-truth column). Writing
        # into it would either crash on push or silently overwrite the input data.
        # --overwrite opts in to replacing the existing column(s) instead of erroring.
        if original_dataset is not None:
            clash = [
                col
                for col in (args.output_column, "pp_ocr_blocks")
                if col in original_dataset.column_names
            ]
            if clash and not args.overwrite:
                logger.error(
                    f"Output column(s) {clash} already exist in the input dataset "
                    f"(columns: {original_dataset.column_names})."
                )
                logger.error(
                    "Choose a different --output-column, or pass --overwrite to replace them."
                )
                sys.exit(1)
            if clash:
                logger.warning(f"--overwrite: will replace existing column(s) {clash}")

    # ---------- sink ----------
    if is_bucket_url(args.output_target):
        sink: Union[BucketShardSink, DatasetRepoSink] = BucketShardSink(
            args.output_target,
            hf_token=hf_token,
            shard_size=args.shard_size,
            resume=not args.no_resume,
            source_id=args.input_source,
        )
    else:
        sink = DatasetRepoSink(
            args.output_target,
            hf_token=hf_token,
            private=args.private,
            config=args.config,
            create_pr=args.create_pr,
            source_id=args.input_source,
            original_dataset=original_dataset,
            output_column=args.output_column,
            overwrite=args.overwrite,
        )

    completed = sink.already_done()

    # ---------- model ----------
    # PaddleX gates `import cv2` at module load time on
    # `is_dep_available("opencv-contrib-python")`, which checks
    # `importlib.metadata.version(...)`. We ship `opencv-contrib-python-headless`
    # (same `cv2`, no system libGL.so.1 needed) — but that's a different
    # distribution name, so the gate fails and the OCR pipeline's `ocr` extra
    # check returns False. Patch the metadata lookup to alias the GUI cv2 distros
    # to the headless variant before importing paddleocr; this lets paddlex's own
    # `import cv2` succeed and `is_extra_available('ocr')` return True.
    import importlib.metadata as _metadata

    _orig_metadata_version = _metadata.version

    def _patched_metadata_version(dep_name):
        if dep_name in ("opencv-contrib-python", "opencv-python"):
            for headless_alias in (
                "opencv-contrib-python-headless",
                "opencv-python-headless",
            ):
                try:
                    return _orig_metadata_version(headless_alias)
                except _metadata.PackageNotFoundError:
                    continue
        return _orig_metadata_version(dep_name)

    _metadata.version = _patched_metadata_version

    # Silence the connectivity check for speed (not needed in a Job)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # ---------- loop ----------
    processed = 0
    skipped = 0
    errors = 0
    pbar = tqdm(src_iter, total=total, desc=f"PP-OCRv6 {tier}")
    for item in pbar:
        if item.key in completed:
            skipped += 1
            continue
        if item.extras.get("failed") or item.image is None:
            # Unreadable source image — write an empty result in position so the
            # output stays row-aligned with the source dataset.
            sink.write(item.key, "", [])
            errors += 1
            processed += 1
            continue
        try:
            arr = pil_to_array(item.image)
            result = ocr.predict(arr)
            if result:
                text, blocks = extract_text(result[0])
            else:
                text, blocks = "", []
        except Exception as e:
            logger.error(f"Error on {item.key}: {e}")
            text, blocks = "", []
            errors += 1

        sink.write(item.key, text, blocks)
        processed += 1

    duration = datetime.now() - start_time
    processing_time_str = f"{duration.total_seconds() / 60:.2f} min"
    logger.info(
        f"Processed {processed} (skipped {skipped}, errors {errors}) in {processing_time_str}"
    )

    args_dict = {
        "tier": tier,
        "det_model": det_model,
        "rec_model": rec_model,
        "engine": "paddle_static",
        "shard_size": args.shard_size,
        "processing_time": processing_time_str,
        "output_column": args.output_column,
    }
    sink.finalize(
        tier=tier,
        det_model=det_model,
        rec_model=rec_model,
        args_dict=args_dict,
    )

    if args.verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in [
            "paddleocr",
            "paddlex",
            "paddlepaddle-gpu",
            "huggingface-hub",
            "datasets",
            "pillow",
            "numpy",
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PP-OCRv6 OCR over an HF dataset or bucket of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input_source",
        help="HF dataset id (namespace/dataset) OR hf://buckets/ns/bucket[/prefix]",
    )
    p.add_argument(
        "output_target",
        help="HF dataset id (namespace/dataset) OR hf://buckets/ns/bucket/run-name",
    )
    p.add_argument(
        "--model-tier",
        default="medium",
        choices=list(TIER_MODELS),
        help="PP-OCRv6 model tier: tiny (1.5M), small (7.7M), medium (34.5M). Default: medium.",
    )
    # Dataset-source-specific
    p.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (dataset-repo source only, default: image)",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split (dataset-repo source only, default: train)",
    )
    p.add_argument(
        "--max-samples", type=int, help="Limit number of samples (for testing)"
    )
    p.add_argument(
        "--shuffle", action="store_true", help="Shuffle source before processing"
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffle (default: 42)"
    )
    # Dataset-sink-specific
    p.add_argument(
        "--private", action="store_true", help="Private dataset output (dataset sink only)"
    )
    p.add_argument(
        "--config",
        help="Config/subset name when pushing to Hub (dataset sink only)",
    )
    p.add_argument(
        "--create-pr",
        action="store_true",
        help="Create PR instead of direct push (dataset sink only)",
    )
    p.add_argument(
        "--output-column",
        default="markdown",
        help=(
            "Column name for the recognized text (dataset sink only, default: markdown). "
            "Must not collide with an existing input column — many corpora already ship a "
            "'text' ground-truth column, so 'text' would fail on push. Blocks always go to "
            "'pp_ocr_blocks'."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column(s) if they already exist in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    # Bucket-sink-specific
    p.add_argument(
        "--shard-size",
        type=int,
        default=256,
        help="Rows per parquet shard for bucket sink (default: 256)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume scan when writing to a bucket sink",
    )
    # Auth + diagnostics
    p.add_argument("--hf-token", help="Hugging Face API token (else uses HF_TOKEN env)")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions at the end",
    )
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
