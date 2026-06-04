# uv-scripts-for-ai

> **A UV script is the smallest reliable unit of data & ML work: one file, with its dependencies pinned inside it, that runs the same on your laptop or on a managed GPU. Run one in a command — `uv run` locally or `hf jobs uv run` on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) — and chain many into a pipeline.**

Because each script is self-contained and self-describing, it's easy for **people and AI agents alike** to run and compose. No clone, no environment to set up, no `requirements.txt` — the script *is* the unit.

A **recipe** here is one such UV script: a single Python file that declares its own dependencies inline. Most read and write the [Hugging Face Hub](https://huggingface.co/datasets), so the Hub is the substrate that passes data from one step to the next.

## Quickstart

**First, install [uv](https://docs.astral.sh/uv/getting-started/installation/)** — it's the only thing you install; every script brings its own Python dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Kick the tires** — run a recipe straight from its URL; `uv` reads its dependency block and builds the environment for you (a few seconds, no account needed):

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py --help
```

**Run it for real on a GPU** — point Hugging Face Jobs at the same URL. Here `davanstrien/ufo-ColPali` is a small *public* image dataset you can use as-is; the output lands in your namespace:

```bash
hf jobs uv run --flavor l4x1 \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  davanstrien/ufo-ColPali your-username/ufo-ocr
```

No GPU of your own, no `pip install`. (Jobs needs the `hf` CLI — `uv tool install huggingface_hub` — and a Hub account with a positive [credit balance](https://huggingface.co/settings/billing); a small CPU job costs ~$0.01/hr. Run `hf jobs hardware` for current flavors and prices.)

## What's a UV script?

A normal Python file with a metadata block at the top declaring its dependencies:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "transformers", "torch"]
# ///
```

`uv` — and `hf jobs uv run` — reads that block, builds the environment, and runs the file. One file, no `requirements.txt`, no setup. This is the standard [PEP 723](https://peps.python.org/pep-0723/) inline-script-metadata format — see the [uv scripts guide](https://docs.astral.sh/uv/guides/scripts/) to learn more.

## Why UV scripts

A self-contained, pinned script is the smallest *reliable* unit of work — which is what makes it a good building block for both people and agents:

- **Discrete & single-purpose** — one script, one job. That job can be a two-second transform or a multi-hour fine-tune; either way it's one self-contained unit you pick by reading a header, not a codebase.
- **Self-describing** — the [PEP 723](https://peps.python.org/pep-0723/) dependency block, the docstring, and `--help` together declare what it needs and how to call it.
- **Reproducible by construction** — dependencies are pinned *in the file*. No env drift, nothing to debug later, no "works on my machine."
- **Composable** — recipes hand off through the Hub (usually a dataset in, a dataset or model out), so you can chain them into a pipeline.
- **Runs anywhere** — `uv run` locally, `hf jobs uv run` for GPU, or anywhere `uv` is installed. Same file, same result.

**Built for agents, too.** Every recipe takes the same `input output` shape and is callable from a URL, so an AI agent can pick a tool from its header and run it with no setup. And on Jobs the agent runs **disposable, Hub-scoped managed compute** — ephemeral disk, I/O bounded by the token's repo permissions, a per-job cost ceiling — never arbitrary code on your machine. That sandboxing is the difference between "an agent *could* call this" and "I'd let an agent run this unattended."

## Recipes

| Domain | What it does | On the Hub |
|---|---|---|
| **[ocr/](ocr/)** ⭐ | OCR / document → text & structured data — GLM, PaddleOCR-VL, Nanonets, olmOCR, dots, … (30+ models) | [`uv-scripts/ocr`](https://huggingface.co/datasets/uv-scripts/ocr) |
| **vision** | Zero-shot detection & segmentation over image datasets | [`sam3`](https://huggingface.co/datasets/uv-scripts/sam3) · [`object-detection`](https://huggingface.co/datasets/uv-scripts/object-detection) · [`vlm-object-detection`](https://huggingface.co/datasets/uv-scripts/vlm-object-detection) |
| **audio** | Transcription & speech translation | [`transcription`](https://huggingface.co/datasets/uv-scripts/transcription) |
| **embeddings & atlas** | Embed a dataset; build an interactive map | [`build-atlas`](https://huggingface.co/datasets/uv-scripts/build-atlas) |
| **data processing** | Filter / dedup / stats over large datasets | [`dataset-stats`](https://huggingface.co/datasets/uv-scripts/dataset-stats) · [`deduplication`](https://huggingface.co/datasets/uv-scripts/deduplication) · [`classification`](https://huggingface.co/datasets/uv-scripts/classification) |
| **dataset creation** | Turn PDFs / image URLs into Hub datasets | [`dataset-creation`](https://huggingface.co/datasets/uv-scripts/dataset-creation) · [`iiif-tiles`](https://huggingface.co/datasets/uv-scripts/iiif-tiles) |
| **synthetic data** | Generate datasets with LLMs | [`synthetic-data`](https://huggingface.co/datasets/uv-scripts/synthetic-data) |
| **inference** | Run any open LLM / VLM over a dataset | [`vllm`](https://huggingface.co/datasets/uv-scripts/vllm) · [`openai-oss`](https://huggingface.co/datasets/uv-scripts/openai-oss) · [`transformers-inference`](https://huggingface.co/datasets/uv-scripts/transformers-inference) |
| **entity extraction** | NER / structured extraction over text | [`gliner`](https://huggingface.co/datasets/uv-scripts/gliner) |
| ***…and more*** | *Training, evaluation, RAG indexing — migrating as they mature* | [`training`](https://huggingface.co/datasets/uv-scripts/training) · [`transformers-training`](https://huggingface.co/datasets/uv-scripts/transformers-training) |

Only **[ocr/](ocr/)** lives in this repo so far — the others link to the [`uv-scripts`](https://huggingface.co/uv-scripts) Hugging Face org where they run today, and migrate here over time. (GitHub is the source of truth; each folder mirrors to its Hub dataset.)

**What fits here:** any self-contained UV script for data or ML work on the Hub. OCR and dataset work are the current focus, but inference, evaluation, RAG indexing, and **training** (fine-tuning with TRL / `transformers`, producing a model) are all in scope. If it's one pinned script that reads from or writes to the Hub, it belongs.

## Compose a pipeline

Because recipes hand off through the Hub, you can chain them — each step's output dataset is the next step's input. A document-collection pipeline, end to end:

```
PDFs / scans          →   OCR to markdown      →   dedup + stats        →   embed + visualise
dataset-creation          ocr/glm-ocr.py           deduplication            build-atlas
```

Each arrow is a Hub dataset; each box is one `hf jobs uv run` (or `uv run`) — and every box runs today from its Hub URL, even before it's migrated into this repo. A pipeline can just as well end in a *trained model* instead of another dataset. A human writes the chain as a shell script; an agent writes it as a plan — same scripts either way.

## Run anywhere — and reach for Jobs when you need a GPU

Every recipe runs the same way wherever `uv` is installed:

```bash
uv run <script-url> [args]          # local — your CPU/GPU
hf jobs uv run <script-url> [args]  # managed — HF Jobs is the easy GPU button
```

Why reach for Jobs:

- **Cheapest managed serverless GPU** — pay by the minute, only while the job runs.
- **No infra** — `hf jobs uv run <url>` and you're done.
- **Hub-native** — mount datasets/models/buckets with `-v hf://…`; write results straight back to the Hub. Running from the `https://huggingface.co/datasets/uv-scripts/…` URL also attributes usage to the recipe.

---

*Recipes mirror to the [`uv-scripts`](https://huggingface.co/uv-scripts) Hugging Face org via GitHub Actions — GitHub is the source of truth. See [CONTRIBUTING.md](CONTRIBUTING.md) to add one.*
