# uv-scripts-for-ai

> **UV scripts for working with data on the Hugging Face Hub — run any of them in one command as a Job. No install, no setup.**

Each recipe here is a single, self-contained [UV script](https://docs.astral.sh/uv/guides/scripts/) (Python with its dependencies declared inline). You don't clone anything or set up an environment — you point Hugging Face Jobs at the script's URL and it runs on managed CPU/GPU infrastructure, reading and writing straight from the Hub.

## Quickstart

OCR an image dataset and push the text back to the Hub — one command:

```bash
hf jobs uv run --flavor l4x1 \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  your-username/my-images your-username/my-images-text
```

That's it — no GPU of your own, no `pip install`. (You need a Hub account with a positive [credit balance](https://huggingface.co/settings/billing); a small CPU job costs ~$0.01/hr. Run `hf jobs hardware` for current flavors + prices.)

## What's a UV script?

A normal Python file with a metadata block at the top declaring its dependencies:

```python
# /// script
# dependencies = ["datasets", "transformers", "torch"]
# ///
```

`uv` — and `hf jobs uv run` — reads that block, builds the environment, and runs the file. One file, no `requirements.txt`, no setup.

This is the standard [PEP 723](https://peps.python.org/pep-0723/) inline-script-metadata format — see the [uv scripts guide](https://docs.astral.sh/uv/guides/scripts/) to learn more.

## Recipes

| Domain | What it does | Status |
|---|---|---|
| **[ocr/](ocr/)** | OCR / document → text & structured data (model zoo: GLM, PaddleOCR-VL, Nanonets, olmOCR, …) | ⭐ flagship |
| data-processing/ | Filter / dedup / stats over large datasets (DuckDB, Polars) | _coming_ |
| vision/ | Zero-shot detection & segmentation over datasets (SAM3, VLMs) | _coming_ |
| embeddings/ | Embed a dataset; build an interactive atlas | _coming_ |
| dataset-creation/ | Turn raw files (PDFs, image URLs) into Hub datasets | _coming_ |
| synthetic-data/ | Generate datasets with LLMs | _coming_ |
| audio/ | Transcription & speech translation | _coming_ |

## Why Hugging Face Jobs?

- **Cheapest managed serverless GPU** — pay by the minute, only while the job runs.
- **No infra** — `hf jobs uv run <url>` and you're done.
- **Hub-native** — mount datasets/models/buckets with `-v hf://…`; write results straight back to the Hub.

## Run from anywhere

Every recipe runs the same way — pass the script URL to `hf jobs uv run`:

```bash
hf jobs uv run <script-url> [args]
```

---

*Recipes mirror to the [`uv-scripts`](https://huggingface.co/uv-scripts) Hugging Face org via GitHub Actions — GitHub is the source of truth. See [CONTRIBUTING.md](CONTRIBUTING.md) to add one.*
