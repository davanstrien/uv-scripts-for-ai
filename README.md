# uv-scripts-for-ai

> **A UV script is a single Python file that declares its own dependencies inline, so it runs the same way on your laptop or on a managed GPU. Run one with `uv run` locally or `hf jobs uv run` on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), and chain several into a pipeline.**

Each script carries its own dependencies, so people and agents can run one without cloning a repo, making a virtualenv, or installing a `requirements.txt` first.

A **recipe** here is one such script. Most read and write the [Hugging Face Hub](https://huggingface.co/datasets), so one script's output dataset becomes the next one's input.

## Quickstart

**First, install [uv](https://docs.astral.sh/uv/getting-started/installation/)** — it's the only thing you install; every script brings its own Python dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Try it locally** — run a recipe straight from its URL. `uv` reads its dependency block and builds the environment for you (a few seconds, no account needed):

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py --help
```

**Run it on a GPU** — point Hugging Face Jobs at the same URL. Here `davanstrien/ufo-ColPali` is a small *public* image dataset you can use as-is; the output lands in your namespace:

```bash
hf jobs uv run --flavor l4x1 \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  davanstrien/ufo-ColPali your-username/ufo-ocr
```

No GPU of your own, no `pip install`. (Jobs needs the `hf` CLI — `uv tool install huggingface_hub` — and a [Pro, Team, or Enterprise](https://huggingface.co/pricing) account; it's pay-as-you-go, billed by the second, and a small CPU job costs ~$0.01/hr. Run `hf jobs hardware` for current flavors and prices.)

## What's a UV script?

A normal Python file with a metadata block at the top that lists its dependencies:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "transformers", "torch"]
# ///
```

Normally, running someone's Python script means cloning their repo, making a virtual environment, and `pip install`-ing a `requirements.txt` first — and if your versions don't match theirs, it can still break. Here the dependencies live inside the file, in that comment block, so `uv` (and `hf jobs uv run`) reads them, installs exactly those versions into a throwaway environment, and runs the file — straight from a URL, with nothing to set up. This is the standard [PEP 723](https://peps.python.org/pep-0723/) inline-script-metadata format; see the [uv scripts guide](https://docs.astral.sh/uv/guides/scripts/) to learn more.

## Why UV scripts

A self-contained, pinned script is easy to run and reuse, for a few reasons:

- **Discrete & single-purpose** — one script, one job. That job can be a two-second transform or a multi-hour fine-tune; either way it's one self-contained unit you pick by reading a header instead of a whole codebase.
- **Self-describing** — the [PEP 723](https://peps.python.org/pep-0723/) dependency block, the docstring, and `--help` tell you what it needs and how to call it.
- **Reproducible** — dependencies are pinned *in the file*, so there's no env drift and no "works on my machine."
- **Composable** — recipes hand off through the Hub (usually a dataset in, a dataset or model out), so you can chain them into a pipeline.
- **Runs anywhere** — `uv run` locally, `hf jobs uv run` for GPU, or anywhere `uv` is installed.

**Built for agents, too.** Every recipe takes its arguments in the same `input output` order and runs from a URL, so an AI agent can pick a tool from its header and run it with no setup. On Jobs the agent runs in a sandbox: a throwaway disk, access limited to what the token's repo permissions allow, and a cost cap per job — not arbitrary code on your machine. (Hugging Face also ships an [`hf` CLI skill for agents](https://huggingface.co/docs/hub/agents-cli) for driving Jobs from an editor.)

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

Each arrow is a Hub dataset; each box is one `hf jobs uv run` (or `uv run`), and every box runs today from its Hub URL, even before it's migrated into this repo. A pipeline can also end in a *trained model* instead of another dataset. You can write the chain as a shell script, or an agent can generate it — the scripts are the same.

## Run anywhere; use Jobs for a GPU

Every recipe runs the same way wherever `uv` is installed — locally, or on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) for a managed GPU. Same file, same arguments:

```bash
SCRIPT=https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py

# locally — on your own machine
uv run $SCRIPT davanstrien/ufo-ColPali your-username/ufo-ocr

# on a managed GPU — pick hardware with --flavor
hf jobs uv run --flavor l4x1 $SCRIPT davanstrien/ufo-ColPali your-username/ufo-ocr
```

Why reach for [Jobs](https://huggingface.co/docs/hub/jobs):

- **Pay by the second** — billed only while the job runs. Run `hf jobs hardware`, or see the [flavors](https://huggingface.co/docs/huggingface_hub/guides/jobs#select-the-hardware) and [pricing](https://huggingface.co/docs/hub/jobs).
- **No infra** — `hf jobs uv run <url>` and you're done. See the [`hf jobs` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs).
- **Hub-native** — read and write datasets, models, and [storage buckets](https://huggingface.co/docs/hub/storage-buckets) directly. Running from the `https://huggingface.co/datasets/uv-scripts/…` URL also attributes usage to the recipe.

## Model licenses

These scripts are orchestration code: they download third-party models from the Hugging Face Hub at runtime and run inference. **This repo does not redistribute any model weights.** Each model you run carries its own license (MIT, Apache-2.0, OpenRAIL-M, and some with non-commercial or other use-based terms); those terms govern your use of the *model*, not this repo's code. **You are responsible for checking each model's license** — on its Hugging Face model card — before using it, especially in production.

## License

The code and documentation in this repository are licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

---

*Recipes mirror to the [`uv-scripts`](https://huggingface.co/uv-scripts) Hugging Face org via GitHub Actions — GitHub is the source of truth. See [CONTRIBUTING.md](CONTRIBUTING.md) to add one.*
