---
viewer: false
tags: [uv-script, deduplication, semantic-similarity, data-processing, hf-jobs]
---

# Semantic Deduplication UV Script

> Part of [uv-scripts](https://huggingface.co/uv-scripts) — self-contained UV scripts you run locally or on Hugging Face Jobs in one command.

Remove duplicate / near-duplicate text samples from a Hugging Face dataset by semantic similarity — clean training data and prevent train/test leakage. Uses [SemHash](https://github.com/MinishLab/semhash) with Model2Vec embeddings: **CPU-optimized, no GPU required.**

## Quick start

```bash
# CPU is enough — run on Hugging Face Jobs
hf jobs uv run --flavor cpu-upgrade --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/deduplication/raw/main/semantic-dedupe.py \
    your-input-dataset text your-output-dataset

# or locally
uv run https://huggingface.co/datasets/uv-scripts/deduplication/raw/main/semantic-dedupe.py \
    your-input-dataset text your-output-dataset
```

Arguments are `INPUT_DATASET TEXT_COLUMN OUTPUT_DATASET`. Tune with `--threshold` (similarity cutoff) and `--max-samples` (for testing); run with `--help` for all options.
