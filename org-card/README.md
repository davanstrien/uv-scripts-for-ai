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

Each recipe is a single self-contained [UV script](https://docs.astral.sh/uv/guides/scripts/): dependencies are declared inline, so you run it straight from a URL — no clone, no virtualenv, no `pip install`. Run it locally with `uv run`, or hand it to [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) for a managed GPU. Most recipes read a Hub dataset and write a new one, so they chain into pipelines.

## Quickstart

**See every recipe** — locally, no GPU or token:

```bash
uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/list-recipes.py
```

**Run one on a GPU** — the flagship, OCR an image dataset to text:

```bash
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  davanstrien/ufo-ColPali your-username/ufo-ocr --max-samples 10
```

One command → a new dataset with a `markdown` column. Pay-per-second, no hardware of your own.

<details>
<summary><b>Drive it with your coding agent →</b></summary>

Recipes take their arguments in the same `input output` order and run from a URL, so an agent can pick one and run it with no setup. Paste into Claude Code, Cursor, or similar:

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

The cookbook also ships a ready-made **agent skill** for discovering and running recipes — see the [GitHub repo](https://github.com/davanstrien/uv-scripts-for-ai), and Hugging Face's own [`hf` CLI skill for agents](https://huggingface.co/docs/hub/agents-cli). _(We'll refine these prompts over time.)_

</details>

## Browse

Every recipe is in the list below — OCR, detection & segmentation, audio transcription, NER & classification, embeddings & atlas maps, batch LLM/VLM inference, synthetic data, and dataset creation. Or browse on **[GitHub](https://github.com/davanstrien/uv-scripts-for-ai)** · run `hf jobs hardware` for GPU flavors & pricing.
