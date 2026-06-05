---
viewer: false
tags: [uv-script, object-detection]
---

# Object Detection Dataset Scripts

5 scripts to convert, validate, inspect, diff, and sample object detection datasets on the Hub. Supports 6 bbox formats — no setup required.
This repository is inspired by [panlabel](https://github.com/strickvl/panlabel)

## Quick Start

Convert bounding box formats without cloning anything:

```bash
# Convert COCO-style bboxes to YOLO normalized format
uv run convert-hf-dataset.py merve/coco-dataset merve/coco-yolo \
    --from coco_xywh --to yolo --max-samples 100
```

That's it! The script will:

- Load the dataset from the Hub
- Convert all bounding boxes in-place
- Push the result to a new dataset repo
- View results at: `https://huggingface.co/datasets/merve/coco-yolo`

## Scripts

| Script | Description |
|--------|-------------|
| `convert-hf-dataset.py` | Convert between 6 bbox formats and push to Hub |
| `validate-hf-dataset.py` | Check annotations for errors (invalid bboxes, duplicates, bounds) |
| `stats-hf-dataset.py` | Compute statistics (counts, label histogram, area, co-occurrence) |
| `diff-hf-datasets.py` | Compare two datasets semantically (IoU-based annotation matching) |
| `sample-hf-dataset.py` | Create subsets (random or stratified) and push to Hub |

## Supported Bbox Formats

All scripts support these 6 bounding box formats, matching the [panlabel](https://github.com/strickvl/panlabel) Rust CLI:

| Format | Encoding | Coordinate Space |
|--------|----------|------------------|
| `coco_xywh` | `[x, y, width, height]` | Pixels |
| `xyxy` | `[xmin, ymin, xmax, ymax]` | Pixels |
| `voc` | `[xmin, ymin, xmax, ymax]` | Pixels (alias for `xyxy`) |
| `yolo` | `[center_x, center_y, width, height]` | Normalized 0–1 |
| `tfod` | `[xmin, ymin, xmax, ymax]` | Normalized 0–1 |
| `label_studio` | `[x, y, width, height]` | Percentage 0–100 |

Conversions go through XYXY pixel-space as the intermediate representation, so any format can be converted to any other format.

## Common Options

All scripts accept flexible column mapping. Datasets can store annotations as flat columns or nested under an `objects` dict — both layouts are handled automatically.

| Option | Description |
|--------|-------------|
| `--bbox-column` | Column containing bboxes (default: `bbox`) |
| `--category-column` | Column containing category labels (default: `category`) |
| `--width-column` | Column for image width (default: `width`) |
| `--height-column` | Column for image height (default: `height`) |
| `--split` | Dataset split (default: `train`) |
| `--max-samples` | Limit number of samples (useful for testing) |
| `--hf-token` | HF API token (or set `HF_TOKEN` env var) |
| `--private` | Make output dataset private |

Every script supports `--help` to see all available options:

```bash
uv run convert-hf-dataset.py --help
```

## Convert (`convert-hf-dataset.py`)

Convert bounding boxes between any of the 6 supported formats:

```bash
# COCO -> XYXY
uv run convert-hf-dataset.py merve/license-plates merve/license-plates-voc \
    --from coco_xywh --to voc

# YOLO -> COCO
uv run convert-hf-dataset.py merve/license-plates merve/license-plates-yolo \
    --from coco_xywh --to yolo

# TFOD (normalized xyxy) -> COCO
uv run convert-hf-dataset.py merve/license-plates-tfod merve/license-plates-coco \
    --from tfod --to coco_xywh

# Label Studio (percentage xywh) -> XYXY
uv run convert-hf-dataset.py merve/ls-dataset merve/ls-xyxy \
    --from label_studio --to xyxy

# Test on 10 samples first
uv run convert-hf-dataset.py merve/dataset merve/converted \
    --from xyxy --to yolo --max-samples 10

# Shuffle before converting a subset
uv run convert-hf-dataset.py merve/dataset merve/converted \
    --from coco_xywh --to tfod --max-samples 500 --shuffle
```

| Option | Description |
|--------|-------------|
| `--from` | Source bbox format (required) |
| `--to` | Target bbox format (required) |
| `--batch-size` | Batch size for map (default: 1000) |
| `--create-pr` | Push as PR instead of direct commit |
| `--shuffle` | Shuffle dataset before processing |
| `--seed` | Random seed for shuffling (default: 42) |

## Validate (`validate-hf-dataset.py`)

Check annotations for common issues:

```bash
# Basic validation
uv run validate-hf-dataset.py merve/coco-dataset

# Validate YOLO-format dataset
uv run validate-hf-dataset.py merve/yolo-dataset --bbox-format yolo

# Validate TFOD-format dataset
uv run validate-hf-dataset.py merve/tfod-dataset --bbox-format tfod

# Strict mode (warnings become errors)
uv run validate-hf-dataset.py merve/dataset --strict

# JSON report
uv run validate-hf-dataset.py merve/dataset --report json

# Stream large datasets without full download
uv run validate-hf-dataset.py merve/huge-dataset --streaming --max-samples 5000

# Push validation report to Hub
uv run validate-hf-dataset.py merve/dataset --output-dataset merve/validation-report
```

**Issue Codes:**

| Code | Level | Description |
|------|-------|-------------|
| E001 | Error | Bbox/category count mismatch |
| E002 | Error | Invalid bbox (missing values) |
| E003 | Error | Non-finite coordinates (NaN/Inf) |
| E004 | Error | xmin > xmax |
| E005 | Error | ymin > ymax |
| W001 | Warning | No annotations in example |
| W002 | Warning | Zero or negative area |
| W003 | Warning | Bbox before image origin |
| W004 | Warning | Bbox beyond image bounds |
| W005 | Warning | Empty category label |
| W006 | Warning | Duplicate file name |

## Stats (`stats-hf-dataset.py`)

Compute rich statistics for a dataset:

```bash
# Basic stats
uv run stats-hf-dataset.py merve/coco-dataset

# Top 20 label histogram, JSON output
uv run stats-hf-dataset.py merve/dataset --top 20 --report json

# Stats for TFOD-format dataset
uv run stats-hf-dataset.py merve/dataset --bbox-format tfod

# Stream large datasets
uv run stats-hf-dataset.py merve/huge-dataset --streaming --max-samples 10000

# Push stats report to Hub
uv run stats-hf-dataset.py merve/dataset --output-dataset merve/stats-report
```

Reports include: summary counts, label distribution, annotation density, bbox area/aspect ratio distributions, per-category area stats, category co-occurrence pairs, and image resolution distribution.

## Diff (`diff-hf-datasets.py`)

Compare two datasets semantically using IoU-based annotation matching:

```bash
# Basic diff
uv run diff-hf-datasets.py merve/dataset-v1 merve/dataset-v2

# Stricter matching
uv run diff-hf-datasets.py merve/old merve/new --iou-threshold 0.7

# Per-annotation change details
uv run diff-hf-datasets.py merve/old merve/new --detail

# JSON report
uv run diff-hf-datasets.py merve/old merve/new --report json
```

Reports include: shared/unique images, shared/unique categories, matched/added/removed/modified annotations.

## Sample (`sample-hf-dataset.py`)

Create random or stratified subsets:

```bash
# Random 500 samples
uv run sample-hf-dataset.py merve/dataset merve/subset -n 500

# 10% fraction
uv run sample-hf-dataset.py merve/dataset merve/subset --fraction 0.1

# Stratified sampling (preserves class distribution)
uv run sample-hf-dataset.py merve/dataset merve/subset \
    -n 200 --strategy stratified

# Filter by categories
uv run sample-hf-dataset.py merve/dataset merve/subset \
    -n 100 --categories "cat,dog,bird"

# Reproducible sampling
uv run sample-hf-dataset.py merve/dataset merve/subset \
    -n 500 --seed 42
```

| Option | Description |
|--------|-------------|
| `-n` | Number of samples to select |
| `--fraction` | Fraction of dataset (0.0–1.0) |
| `--strategy` | `random` (default) or `stratified` |
| `--categories` | Comma-separated list of categories to filter by |
| `--category-mode` | `images` (default) or `annotations` |

## Run Locally

```bash
# Clone and run
git clone https://huggingface.co/datasets/uv-scripts/panlabel
cd panlabel
uv run convert-hf-dataset.py input-dataset output-dataset --from coco_xywh --to yolo

# Or run directly from URL
uv run https://huggingface.co/datasets/uv-scripts/panlabel/raw/main/convert-hf-dataset.py \
    input-dataset output-dataset --from coco_xywh --to yolo
```

Works with any Hugging Face dataset containing object detection annotations — COCO, YOLO, VOC, TFOD, or Label Studio format.
