---
viewer: false
tags: [uv-script, ocr, extraction, vision-language-model, document-processing, hf-jobs]
---

# OCR UV Scripts

> Part of [uv-scripts](https://huggingface.co/uv-scripts) — self-contained UV scripts you run on Hugging Face Jobs in one command.

A model zoo of OCR scripts — one per model — that add a `markdown` column to an image dataset. Pick a model from the table below, point it at your dataset, and run it on a GPU with one command. A few recipes do **structured extraction** instead — image *or* text → JSON given a schema (see [Structured extraction](#structured-extraction-image-or-text--json) below). Two more companions sit alongside: `pp-doclayout.py` detects layout regions (bboxes for text/title/table/figure/…) instead of text, and `ocr-vllm-judge.py` compares model outputs head-to-head.

## Quick Start

Run OCR on any dataset without needing your own GPU:

```bash
# Quick test with 10 samples
hf jobs uv run --flavor l4x1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    your-input-dataset your-output-dataset \
    --max-samples 10
```

This will:

- Process the first 10 images from your dataset
- Add OCR results as a new `markdown` column
- Push the results to a new dataset
- View results at: `https://huggingface.co/datasets/[your-output-dataset]`

## Models at a glance

**Start here:** for a quick first run, try **`lighton-ocr2.py`** (1B, very fast) or **`paddleocr-vl-1.6.py`** (0.9B, current OmniDocBench SOTA); for the smallest footprint, **`falcon-ocr.py`** (0.3B, strong on tables). Reach for a 7–8B model only when quality demands it. Several of these models sit on the public [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) — pull the live ranking from your terminal in one command:

```bash
hf datasets leaderboard allenai/olmOCR-bench
```

But which model wins on *your* documents is still document-dependent — so [ocr-bench](https://github.com/davanstrien/ocr-bench) builds a **per-collection leaderboard** for your own data (pairwise VLM-as-judge, optionally human-validated), using these scripts under the hood.

_Sorted by model size:_

| Script | Model | Size | Backend | Notes |
|--------|-------|------|---------|-------|
| `falcon-ocr.py` | [Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR) | 0.3B | falcon-perception | Smallest in collection. #1 on multi-column docs and tables (olmOCR), Apache 2.0 |
| `smoldocling-ocr.py` | [SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview) | 256M | Transformers | DocTags structured output |
| `glm-ocr.py` | [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | vLLM | 94.62% OmniDocBench V1.5 |
| `paddleocr-vl.py` | [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) | 0.9B | Transformers | 4 task modes (ocr/table/formula/chart) |
| `paddleocr-vl-1.5.py` | [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | Transformers | 94.5% OmniDocBench, 6 task modes |
| `paddleocr-vl-1.6.py` | [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 0.9B | vLLM | **96.33% OmniDocBench v1.6** (SOTA), drop-in upgrade of 1.5 |
| `lighton-ocr.py` | [LightOnOCR-1B](https://huggingface.co/lightonai/LightOnOCR-1B-1025) | 1B | vLLM | Fast, 3 vocab sizes |
| `lighton-ocr2.py` | [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 1B | vLLM | 7× faster than v1, RLVR trained |
| `hunyuan-ocr.py` | [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) | 1B | vLLM | Lightweight VLM |
| `dots-ocr.py` | [DoTS.ocr](https://huggingface.co/Tencent/DoTS.ocr) | 1.7B | vLLM | 100+ languages |
| `firered-ocr.py` | [FireRed-OCR](https://huggingface.co/FireRedTeam/FireRed-OCR) | 2.1B | vLLM | Qwen3-VL fine-tune, Apache 2.0 |
| `abot-ocr.py` | [ABot-OCR](https://huggingface.co/acvlab/ABot-OCR) | 2B | vLLM | Qwen3-VL based, doc→Markdown (text/LaTeX/HTML tables). Needs `vllm/vllm-openai` image. [paper](https://arxiv.org/abs/2605.27978) |
| `nanonets-ocr.py` | [Nanonets-OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s) | 2B | vLLM | LaTeX, tables, forms |
| `dots-mocr.py` | [dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr) | 3B | vLLM | 8 prompt modes incl. SVG generation, layout + bbox, 100+ languages |
| `nanonets-ocr2.py` | [Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-s) | 3B | vLLM | Next-gen, Qwen2.5-VL base |
| `deepseek-ocr-vllm.py` | [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | 4B | vLLM | 5 resolution + 5 prompt modes |
| `deepseek-ocr.py` | [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | 4B | Transformers | Same model, Transformers backend |
| `deepseek-ocr2-vllm.py` | [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | 3B | vLLM | Newer, requires nightly vLLM |
| `nuextract3.py` | [NuExtract3](https://huggingface.co/numind/NuExtract3) | 4B | vLLM | Markdown OCR **+ schema-guided JSON extraction** (template/Pydantic). Needs `vllm/vllm-openai` image |
| `qianfan-ocr.py` | [Qianfan-OCR](https://huggingface.co/baidu/Qianfan-OCR) | 4.7B | vLLM | #1 OmniDocBench v1.5 (93.12), Layout-as-Thought, 192 languages |
| `olmocr2-vllm.py` | [olmOCR-2-7B](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8) | 7B | vLLM | 82.4% olmOCR-Bench |
| `rolm-ocr.py` | [RolmOCR](https://huggingface.co/reducto/RolmOCR) | 7B | vLLM | Qwen2.5-VL based, general-purpose |
| `numarkdown-ocr.py` | [NuMarkdown-8B](https://huggingface.co/numind/NuMarkdown-8B-Thinking) | 8B | vLLM | Reasoning-based OCR |

**Variants & tools** (same models, different I/O): `glm-ocr-v2.py` adds checkpoint/resume for very large jobs · `glm-ocr-bucket.py` and `falcon-ocr-bucket.py` read images/PDFs from a mounted bucket and write one `.md` per page · `ocr-vllm-judge.py` runs pairwise OCR-quality comparisons.

## Structured extraction (image or text → JSON)

Most scripts here output markdown. These take a **schema** and return **structured data** instead — give them the fields you want, they fill them in:

| Script | Model | Size | Input | Output |
|--------|-------|------|-------|--------|
| `lfm2-vl-extract.py` | [LFM2.5-VL-1.6B-Extract](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-Extract) | 1.6B | image | JSON |
| `nuextract3.py` | [NuExtract3](https://huggingface.co/numind/NuExtract3) | 4B | image | markdown **or** JSON |
| `lfm2-extract.py` | [LFM2-1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract) | 1.2B | **text** | JSON / XML / YAML |

Pass `--schema` (inline JSON, a URL, or a file path). The LFM models are small and fast; run them on the `vllm/vllm-openai` image so the CUDA toolkit is present (each script's docstring has the exact command). Because `lfm2-extract.py` works on a **text** column, you can **chain it after OCR**: a recipe above turns a page into `markdown`, then `lfm2-extract.py` turns that markdown into fields.

```bash
# image → JSON directly
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
    --image vllm/vllm-openai --python /usr/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lfm2-vl-extract.py \
    my-images my-fields --schema '{"title": "the document title", "date": "any date shown"}'
```

## Layout detection (not OCR)

`pp-doclayout.py` runs PaddleOCR's [PP-DocLayout-L](https://huggingface.co/PaddlePaddle/PP-DocLayout-L) (or M / S / plus-L) and emits per-image **bounding boxes + region classes** (text, title, table, figure, formula, list, header, footer, ...) — it does NOT extract text. Useful for filtering pages, cropping regions for downstream OCR, dataset analysis, and training-data prep.

| Script | Model | Size | Backend | Notes |
|--------|-------|------|---------|-------|
| `pp-doclayout.py` | [PP-DocLayout-L](https://huggingface.co/PaddlePaddle/PP-DocLayout-L) | 123M | paddleocr | Layout bboxes (no text). Bucket support: incremental parquet shards, resumable. |

```bash
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-doclayout.py \
    your-dataset your-layout-output --max-samples 10
```

Source/sink can be either an HF dataset repo OR an `hf://buckets/...` URL (auto-detected). Bucket output writes incremental zstd parquet shards via the buckets API — resumable across runs (snapshot-backed source listing) and no git/commit overhead. See the script's `--help` for all flags.

## Common Options

The scripts aim to expose a **consistent interface**: every OCR model script takes `input-dataset output-dataset` as positional arguments, accepts the shared core flags below, and writes a `markdown` column — so switching models is usually just swapping the script URL. Models differ where they need to, though: some add their own flags (task modes, resolution presets, `--think`, vocab sizes), a few need a specific Docker image, and per-model defaults (batch size, context length, temperature) are tuned to each model card. Always check a script's `--help` for its specifics.

| Option | Description |
|--------|-------------|
| `--image-column` | Column containing images (default: `image`) |
| `--output-column` | Output column name (default: `markdown`) |
| `--split` | Dataset split (default: `train`) |
| `--max-samples` | Limit number of samples (useful for testing) |
| `--private` | Make output dataset private |
| `--shuffle` | Shuffle dataset before processing |
| `--seed` | Random seed for shuffling (default: `42`) |
| `--batch-size` | Images per batch (default varies per model) |
| `--max-model-len` | Max context length (default varies per model) |
| `--max-tokens` | Max output tokens (default varies per model) |
| `--gpu-memory-utilization` | GPU memory fraction (default: `0.8`) |
| `--config` | Config name for Hub push (for benchmarking) |
| `--create-pr` | Push as PR instead of direct commit |
| `--verbose` | Log resolved package versions after run |

Every script supports `--help` to see all available options:

```bash
uv run glm-ocr.py --help
```

## NuExtract3: markdown OCR + structured extraction

[NuExtract3](https://huggingface.co/numind/NuExtract3) (4B, Apache-2.0) is the one script here that does both document-to-markdown OCR *and* schema-guided JSON extraction. Give it a template (or a JSON Schema / Pydantic model) and it returns JSON shaped to match.

> **Run it with the `vllm/vllm-openai` image.** NuExtract3's Qwen3.5 architecture needs the image's prebuilt CUDA kernels — the default uv-script image lacks `nvcc`, so flashinfer's JIT compile fails at engine warmup. Use `--image vllm/vllm-openai:latest --python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages` on `a100-large`.

```bash
# Markdown OCR (default mode)
hf jobs uv run --flavor a100-large \
    --image vllm/vllm-openai:latest \
    --python /usr/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nuextract3.py \
    my-documents my-markdown --max-samples 10

# Structured extraction with an inline template
hf jobs uv run --flavor a100-large \
    --image vllm/vllm-openai:latest \
    --python /usr/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nuextract3.py \
    receipts extracted \
    --template '{"store": "verbatim-string", "date": "date", "total": "number"}'
```

**Templates** (`--template`) and **JSON Schemas** (`--schema`) each accept **inline JSON, a URL, or a file path**, so a schema can be hosted once and reused. Add `--enable-thinking` for harder layouts (slower; reasoning trace stored in a `<output-column>_reasoning` column). Template field names act as the model's extraction instructions, so name them descriptively — overly leading names can prompt over-generation, so verify against a few examples.

## Model-specific modes & flags

Beyond the shared flags, some models add their own. Run `--help` on any script for the full list; the common ones:

| Script | Extra options |
|--------|---------------|
| `glm-ocr.py` | `--task ocr\|formula\|table` |
| `paddleocr-vl.py` | `--task-mode ocr\|table\|formula\|chart` |
| `paddleocr-vl-1.5.py` | `--task-mode ocr\|table\|formula\|chart\|spotting\|seal` |
| `paddleocr-vl-1.6.py` | `--task-mode ocr\|table\|formula` |
| `lighton-ocr.py` | `--vocab-size 151k\|32k\|16k` (smaller = faster on European languages) |
| `deepseek-ocr-vllm.py` | `--resolution-mode tiny\|small\|base\|large\|gundam`, `--prompt-mode document\|image\|free\|figure\|describe`; pass `-e UV_TORCH_BACKEND=auto` |
| `dots-ocr.py` | `--prompt-mode ocr\|layout-all\|layout-only` |
| `dots-mocr.py` | `--prompt-mode` (8: ocr, layout-all, layout-only, web-parsing, scene-spotting, grounding-ocr, svg, general); SVG: `--model rednote-hilab/dots.mocr-svg --prompt-mode svg` |
| `qianfan-ocr.py` | `--prompt-mode ocr\|table\|formula\|chart\|scene\|kie`, `--think` (Layout-as-Thought); `kie` needs `--custom-prompt` |
| `numarkdown-ocr.py` | `--include-thinking` (store the reasoning trace) |
| `nuextract3.py` | `--template` / `--schema` / `--enable-thinking` — see the NuExtract3 section above |

**Image-mode models** — `abot-ocr.py` and `nuextract3.py` (Qwen3.5 architecture) need the `vllm/vllm-openai` image because the default uv-script image lacks `nvcc`. Add `--image vllm/vllm-openai:latest --python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages` (see the NuExtract3 example above for the full command).

## Output & features

- **Markdown column** — each run adds an `--output-column` (default `markdown`) with the OCR result.
- **Multi-model comparison** — every script records `inference_info`, so you can run several models into the *same* dataset and compare. Point a second model at the same output repo:
  ```bash
  uv run rolm-ocr.py     my-dataset my-dataset --max-samples 100
  uv run nanonets-ocr.py my-dataset my-dataset --max-samples 100   # appends
  ```
- **Reproducible sampling** — `--shuffle` (with `--seed`, default 42) draws a representative sample instead of the first N rows.
- **Automatic dataset cards** — every run writes a card with the model config, processing stats, column descriptions, and a reproduction command.

## More examples

```bash
# DeepSeek-OCR on historical scans, large resolution mode
hf jobs uv run --flavor a100-large -s HF_TOKEN -e UV_TORCH_BACKEND=auto \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \
    NationalLibraryOfScotland/Britain-and-UK-Handbooks-Dataset out \
    --max-samples 100 --shuffle --resolution-mode large

# dots.mocr — SVG generation from charts/figures
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-mocr.py \
    your-charts svg-output --prompt-mode svg --model rednote-hilab/dots.mocr-svg

# Qianfan — key-information extraction
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/qianfan-ocr.py \
    invoices extracted-fields \
    --prompt-mode kie --custom-prompt "Extract: name, date, total. Output as JSON."
```

**Python API:**

```python
from huggingface_hub import run_uv_job

job = run_uv_job(
    "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py",
    args=["input-dataset", "output-dataset", "--batch-size", "16"],
    flavor="l4x1",
)
```

**Run locally** (needs your own GPU) — same scripts, run directly from the URL:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    input-dataset output-dataset
```

---

Works with any Hugging Face dataset containing images — documents, forms, receipts, books, handwriting.
