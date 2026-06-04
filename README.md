# uv-scripts-for-ai

> **Single-purpose UV scripts for data & ML work — self-contained, version-pinned, and self-describing, so both you and your agents can run one in a command and chain many into a pipeline. Run them locally with `uv run`, or on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) for zero-setup GPU.**

Each recipe is a single [UV script](https://docs.astral.sh/uv/guides/scripts/): one Python file that declares its own dependencies inline. Nothing to clone, no environment to set up, no `requirements.txt`. The script *is* the unit — and because its deps are pinned in the file, it runs the same today as in six months, on your laptop or on managed GPU.

Every recipe reads and writes the [Hugging Face Hub](https://huggingface.co/datasets): a dataset goes in, a dataset comes out.

## Quickstart

**Test any recipe locally in two seconds** — the script declares its own deps, so `uv` builds the environment for you:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py --help
```

**Run it for real on a GPU** — point Hugging Face Jobs at the same URL; it runs on managed infrastructure, reading and writing straight from the Hub:

```bash
hf jobs uv run --flavor l4x1 \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  your-username/my-images your-username/my-images-text
```

No GPU of your own, no `pip install`. (Jobs needs a Hub account with a positive [credit balance](https://huggingface.co/settings/billing); a small CPU job costs ~$0.01/hr — run `hf jobs hardware` for current flavors and prices.)

## Why UV scripts

A self-contained, pinned script is the smallest *reliable* unit of work — which is exactly what makes it a good building block for both people and agents:

- **Discrete & single-purpose** — one script does one task. You pick the right tool by reading a header, not a codebase.
- **Self-describing** — the [PEP 723](https://peps.python.org/pep-0723/) dependency block, the docstring, and `--help` together declare everything needed to run it.
- **Reproducible by construction** — dependencies are pinned *in the file*. No env drift, nothing to debug later, no "works on my machine."
- **Composable** — each recipe is dataset-in → dataset-out, so the Hub is the glue. Chain several into a pipeline.
- **Runs anywhere** — `uv run` locally, `hf jobs uv run` for GPU, or anywhere `uv` is installed. Same file, same result.

### Good tools for agents

The same properties make these scripts natural tools for AI agents: an agent can **discover** a recipe from its header, **run** it from a URL with no setup, and trust the result is **reproducible** because the environment is pinned. And on Jobs, the agent runs sandboxed managed compute — not arbitrary code on your machine.

## What's a UV script?

A normal Python file with a metadata block at the top declaring its dependencies:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "transformers", "torch"]
# ///
```

`uv` — and `hf jobs uv run` — reads that block, builds the environment, and runs the file. One file, no `requirements.txt`, no setup. This is the standard [PEP 723](https://peps.python.org/pep-0723/) inline-script-metadata format — see the [uv scripts guide](https://docs.astral.sh/uv/guides/scripts/) to learn more.

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

Only **[ocr/](ocr/)** lives in this repo so far — the others link to the [`uv-scripts`](https://huggingface.co/uv-scripts) Hugging Face org where they run today, and migrate here over time. (GitHub is the source of truth; each folder mirrors to its Hub dataset.)

## Compose a pipeline

Because every recipe is dataset-in → dataset-out, you can chain them — the Hub passes data from one step to the next. A document-collection pipeline, end to end:

```
PDFs / scans          →   OCR to markdown      →   dedup + stats        →   embed + visualise
dataset-creation          ocr/glm-ocr.py           deduplication            build-atlas
```

Each arrow is a Hub dataset; each box is one `hf jobs uv run` (or `uv run`). A human writes that as a shell script; an agent writes it as a plan — same scripts either way.

## Run anywhere — Jobs is the easy GPU button

Every recipe runs the same way wherever `uv` is:

```bash
uv run <script-url> [args]          # local — your CPU/GPU
hf jobs uv run <script-url> [args]  # managed — HF Jobs picks up the GPU
```

Why reach for Jobs:

- **Cheapest managed serverless GPU** — pay by the minute, only while the job runs.
- **No infra** — `hf jobs uv run <url>` and you're done.
- **Hub-native** — mount datasets/models/buckets with `-v hf://…`; write results straight back to the Hub. Running from the `https://huggingface.co/datasets/uv-scripts/…` URL also attributes usage to the recipe.

---

*Recipes mirror to the [`uv-scripts`](https://huggingface.co/uv-scripts) Hugging Face org via GitHub Actions — GitHub is the source of truth. See [CONTRIBUTING.md](CONTRIBUTING.md) to add one.*
