---
name: uv-recipes
description: "Run self-contained UV-script recipes over Hugging Face datasets on Hugging Face Jobs (`hf jobs uv run <url>`): OCR & document extraction, audio transcription, object detection & segmentation, NER/entity extraction, text & image classification, embeddings & dataset maps, synthetic data, and batch LLM/VLM inference. Each recipe reads a Hub dataset and writes a new one, so recipes chain into pipelines. Use when the user wants to batch-process a dataset at scale, pre-label data for human review, run a model over many images/audio/documents/rows, or build a data pipeline on Hugging Face — locally with `uv run` or on a managed GPU with `hf jobs uv run`. Recipes live at huggingface.co/uv-scripts (one dataset repo per task family); discover them and read a recipe's header before running."
---

# uv-recipes

A recipe is one self-contained Python file (PEP 723 UV script) that reads a Hugging Face dataset and writes a new one. Recipes take arguments in the same `INPUT_DATASET OUTPUT_DATASET` order and run straight from a URL — no clone, no venv, no install. They're self-describing: the dependency block, docstring, and `--help` say what each needs.

## Requires

- **`uv`** — `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **The `hf` CLI** — `curl -LsSf https://hf.co/cli/install.sh | bash -s` (or `uv tool install "huggingface_hub[cli]"`). Authenticate with `hf auth login`, or set `HF_TOKEN`. Jobs needs a Pro/Team/Enterprise account.
- **Strongly recommended — install the Hugging Face skills:** `hf skills add`. This installs the **`hf-cli`** skill (the full `hf` surface: jobs, auth, repos, datasets, buckets, webhooks) and stays current via `hf skills update`. The two compose: `uv-recipes` picks and runs a recipe, `hf-cli` drives everything else on the Hub.

## Run a recipe (prefer Jobs)

Default to **Hugging Face Jobs** — managed GPU, no local CUDA, pay-per-second:

```bash
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  INPUT_DATASET OUTPUT_DATASET --max-samples 10
```

- `--secrets HF_TOKEN` forwards the user's token so the job can write the output dataset back.
- `--flavor` picks hardware: `cpu-basic`, `t4-small`, `l4x1` (good default for ≤3B vision models), `a10g-large`, `a100-large`. Run `hf jobs hardware` for live prices.
- **Default timeout is 30 min.** For training or large batches add `--timeout 2h`.
- Default image is `astral-sh/uv:python3.12-bookworm`. Some vLLM recipes need `--image vllm/vllm-openai` — **the recipe's docstring has the exact command; read it first.**
- Track a run: `hf jobs logs <job-id>`, `hf jobs ps`, `hf jobs inspect <job-id>`.

**Local alternative** (only if the machine has the hardware — most recipes need a CUDA GPU):
`uv run <url> INPUT OUTPUT`. Inspect without running: `uv run <url> --help`.

## Discover recipes (no fixed list — read it live)

```bash
# task families — one dataset repo per task
curl -s "https://huggingface.co/api/datasets?author=uv-scripts" | jq -r '.[].id'

# recipes inside a family
curl -s "https://huggingface.co/api/datasets/uv-scripts/ocr/tree/main" \
  | jq -r '.[].path | select(endswith(".py"))'

# what a specific recipe needs: args, required image, output column
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py --help
```

Repo name is the task family (e.g. `ocr`, `transcription`, `sam3`, `gliner`, `classification`, `build-atlas`); a few repos are utilities, not recipes. Prefer a smaller model first (0.3–1B on `l4x1`); scale up only if quality demands it.

## Compose a pipeline

Each recipe's output dataset is the next recipe's input (handoff through the Hub): e.g. `dataset-creation` (PDFs→dataset) → `ocr/glm-ocr.py` (→`markdown`) → `deduplication` → `build-atlas` (embed + map). A pipeline can also end in a trained model.

## Pre-label, then review

A recipe's prediction column is a set of **suggestions, not final labels**. To put a human in the loop, **build a small, task-specific review UI on the spot** — a single-file Gradio app or static HTML that shows each input beside its predicted label with an accept / edit control, writing corrections back to a Hub dataset. Tailor it to the task: image + markdown side-by-side for OCR, image with box overlay for detection, text with highlighted spans for NER. Keep it minimal.

To spend review effort well (active learning), order the queue by informativeness first — by model **uncertainty** (low confidence from a `classification` recipe) or by **diversity** (embed with `build-atlas`, sample to cover the space). Review the most-informative first; optionally fine-tune on the verified labels and re-predict, repeating until predictions stop changing. A **scheduled job** or **webhook** can re-run pre-labelling as new data lands.

## Rules that keep jobs working

- Jobs disk is ephemeral — recipes write to the **Hub** (`push_to_hub`) or a **bucket** (`-v hf://…`), never local paths.
- GPU recipes check `torch.cuda.is_available()` and exit clearly if missing — match `--flavor` to the recipe.
- Test on `--max-samples 10` first.
