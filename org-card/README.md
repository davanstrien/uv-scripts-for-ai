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

For GPU runs, [install the `hf` CLI and sign in](https://huggingface.co/docs/hub/jobs-quickstart). Jobs requires pay-as-you-go credit.

**See every recipe** — locally, no GPU or token:

```bash
uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/list-recipes.py
```

**Run one on a GPU** — extract text from [seven scanned NASA booklet pages](https://huggingface.co/datasets/uv-scripts/ocr-demo):

```bash
hf jobs uv run --flavor a10g-small --timeout 15m --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  uv-scripts/ocr-demo your-username/ocr-demo-results
```

Replace `your-username` with your Hugging Face username. The Job saves a new dataset with a `markdown` column. Follow the [OCR walkthrough](https://huggingface.co/datasets/uv-scripts/ocr#use-your-own-documents) to use your own scans or PDFs and retrieve and check the results.

## Drive it with your coding agent

Recipes take their arguments in the same `input output` order and run from a URL, so an agent (Claude Code, Cursor, …) can pick one and run it with no setup. The simplest start — paste this so it discovers what's available:

```
List the uv-scripts recipes and tell me which fit my task:
uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/list-recipes.py
For context on how these work, read the org page https://huggingface.co/uv-scripts
and the GitHub repo https://github.com/davanstrien/uv-scripts-for-ai.
```

<details>
<summary><b>More prompts — run a job, build a dataset →</b></summary>

**Try it now** — runs a real OCR job and hands back a dataset:

```
Using uv-scripts, OCR a sample dataset on Hugging Face Jobs:
  hf jobs uv run --flavor a10g-small --timeout 15m --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    uv-scripts/ocr-demo $MY_HF_USERNAME/ocr-demo-results
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
Background: https://huggingface.co/uv-scripts and https://github.com/davanstrien/uv-scripts-for-ai
```

The cookbook also ships a ready-made **agent skill** for discovering and running recipes — see the [GitHub repo](https://github.com/davanstrien/uv-scripts-for-ai), and Hugging Face's own [`hf` CLI skill for agents](https://huggingface.co/docs/hub/agents-cli). _(We'll refine these prompts over time.)_

</details>

## Browse

Every recipe is in the list below — OCR, detection & segmentation, audio transcription, NER & classification, embeddings & atlas maps, batch LLM/VLM inference, synthetic data, and dataset creation. Or browse on **[GitHub](https://github.com/davanstrien/uv-scripts-for-ai)** · run `hf jobs hardware` for GPU flavors & pricing.
