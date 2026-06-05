---
title: README
emoji: 📚
colorFrom: red
colorTo: indigo
sdk: static
pinned: false
---

# UV Scripts

**Run a data or ML task over a Hugging Face dataset in one command — for humans and agents.**

Each recipe is a single self-contained [UV script](https://docs.astral.sh/uv/guides/scripts/): dependencies are declared inline, so you run it straight from a URL with no clone, no virtualenv, no `pip install`. Run it locally with `uv run` where you have the hardware, or hand it to [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) for a managed GPU. Most recipes read a Hub dataset and write a new one, so they chain into pipelines.

## See every recipe (no GPU, no token)

The lowest-friction start — just [uv](https://docs.astral.sh/uv/getting-started/installation/), runs locally in seconds:

```bash
uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/list-recipes.py
```

It prints a runnable URL for every recipe in the org. Run any of them the same way: `uv run <url>` locally, or `hf jobs uv run <url>` on a GPU.

## Run one for real

The flagship is **[OCR](https://huggingface.co/datasets/uv-scripts/ocr)** — turn an image dataset into text & structured data, 30+ models. On a managed GPU (no hardware of your own; pay-per-second):

```bash
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  davanstrien/ufo-ColPali your-username/ufo-ocr --max-samples 10
```

One command → a new dataset with a `markdown` column.

## For your coding agent

Recipes are built to be agent-driven — same `input output` arg order, runnable from a URL, self-describing headers. Two prompts to paste into Claude Code, Cursor, or similar:

**Try it now** — runs a real OCR job and hands back a dataset:

```
Using uv-scripts, OCR a sample dataset on Hugging Face Jobs:
  hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    davanstrien/ufo-ColPali $MY_HF_USERNAME/ufo-ocr-test --max-samples 10
Then open the output dataset and show me the `markdown` column.
```

**Put it to work** — when you need data for a task:

```
I need a dataset for <my task>. uv-scripts has recipes that create, OCR,
transcribe, classify, deduplicate, and embed datasets on Hugging Face. List them:
  uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/list-recipes.py
Pick the one that fits, read its script header for the arguments, and run it with:
  hf jobs uv run --flavor l4x1 --secrets HF_TOKEN <script-url> INPUT_DATASET OUTPUT_DATASET
Each recipe reads a Hub dataset and writes a new one, so chain them as needed.
```

Prefer a packaged setup? The cookbook ships an **agent skill** for discovering and running recipes — see the [GitHub repo](https://github.com/davanstrien/uv-scripts-for-ai). Hugging Face also ships an [`hf` CLI skill for agents](https://huggingface.co/docs/hub/agents-cli). _(We'll refine these prompts over time.)_

## More

Every other recipe is in the list below — detection & segmentation, audio transcription, NER & classification, embeddings & atlas maps, batch LLM/VLM inference, synthetic data, and dataset creation. Or browse on **[GitHub](https://github.com/davanstrien/uv-scripts-for-ai)** · run `hf jobs hardware` for GPU flavors & pricing.
