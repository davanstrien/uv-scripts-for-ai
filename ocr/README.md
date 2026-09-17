---
viewer: false
tags: [uv-script, ocr, extraction, vision-language-model, document-processing, hf-jobs]
---

# OCR UV Scripts

<a href="https://huggingface.co/uv-scripts"><picture><source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md.svg" alt="Follow uv-scripts on Hugging Face"></picture></a>

> Part of [uv-scripts](https://huggingface.co/uv-scripts) — self-contained UV scripts you run on Hugging Face Jobs in one command.

A model zoo of OCR scripts — one per model — that add a `markdown` column to an image dataset. Pick a model from the table below, point it at your dataset, and run it on a GPU with one command. A few recipes do **structured extraction** instead — image *or* text → JSON given a schema (see [Structured extraction](#structured-extraction-image-or-text--json) below). Two more companions sit alongside: `pp-doclayout.py` detects layout regions (bboxes for text/title/table/figure/…) instead of text, and `ocr-vllm-judge.py` compares model outputs head-to-head.

## Quick Start

First, [install the `hf` CLI and sign in](https://huggingface.co/docs/hub/jobs-quickstart). Jobs requires pay-as-you-go credit.

Try [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) on seven scanned pages from [NASA’s *Food for Space Flight* booklet](https://huggingface.co/datasets/uv-scripts/ocr-demo). Replace `your-username` with your Hugging Face username:

```bash
hf jobs uv run --flavor a10g-small --timeout 15m --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    uv-scripts/ocr-demo your-username/ocr-demo-results
```

The Job adds a `markdown` column to all seven rows and saves them in `your-username/ocr-demo-results`. Dependency installation and model loading can take a few minutes before OCR starts. The [dataset card](https://huggingface.co/datasets/uv-scripts/ocr-demo) documents the source and licence. Check the extracted text against the originals, especially tables and reading order.

### Try the same pages as a PDF

The [OCR demo Bucket](https://huggingface.co/buckets/uv-scripts/ocr-demo) holds the
original PDF, a seven-page extract matching the dataset, and the page images.
Mount the `demo/` prefix to process just the extract, and create your own Bucket
for the results:

```bash
hf buckets create your-username/ocr-output --private
hf jobs uv run --flavor a10g-small --timeout 15m --secrets HF_TOKEN \
    -v hf://buckets/uv-scripts/ocr-demo/demo:/input:ro \
    -v hf://buckets/your-username/ocr-output/pdf:/output:rw \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr-bucket.py \
    /input /output
```

This writes `food-for-space-flight/page_001.md` through `page_007.md` under your
output Bucket's `pdf/` prefix. See [Get and check your results](#get-and-check-your-results)
for how to download them.

## Use your own documents

**Images in a Hub dataset:** replace the input dataset ID in the [Quick Start](#quick-start) command and choose a new output dataset ID. Start with `--max-samples 10` to limit OCR processing; loading the input dataset may still download more rows. The defaults expect a `train` split and an `image` column; use `--split` and `--image-column` if yours differ. If the input already has a `markdown` column, choose a different `--output-column`, such as `glm_markdown`. Add `--private` to create a private output dataset.

**Scans or PDFs on your machine:** put a few images or a short PDF in `./my-scans` for the first run. This recipe processes every supported file in that folder, including subfolders, and every page of each PDF. Create the output folder before launching:

```bash
mkdir -p ./ocr-output
hf jobs uv run --flavor a10g-small --timeout 15m --secrets HF_TOKEN \
    -v ./my-scans:/input -v ./ocr-output:/output:rw \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr-bucket.py \
    /input /output
```

The CLI uploads the local folders to a private bucket and makes them available inside the Job. `:rw` lets the Job write output. The script saves one `.md` file per image, or per PDF page. See [mounting local data](https://huggingface.co/docs/huggingface_hub/en/guides/jobs#mount-local-data) for more detail.

## Get and check your results

Open the Job page linked by the CLI to see its status and logs. You can also check from your terminal:

```bash
hf jobs inspect JOB_ID
hf jobs logs JOB_ID
```

Once the Job has completed:

- **Dataset output:** open `https://huggingface.co/datasets/your-username/ocr-demo-results` and inspect the images alongside their `markdown` results. Use your chosen dataset and column names if you changed them.
- **Bucket output:** browse your output Bucket or download the files with `hf buckets sync hf://buckets/your-username/ocr-output/pdf ./ocr-output`.
- **Local-folder output:** run the `hf buckets sync` command printed by the CLI at launch. It downloads the results into `./ocr-output`; they are not synced back automatically. Images produce files such as `page.md`; a PDF produces files such as `report/page_001.md`.

Check for empty results or `[OCR ERROR]` markers and compare a few outputs with their source pages before scaling up. The GLM recipes can finish with failed batches, so a completed Job does not guarantee that every page was processed successfully.

## Models at a glance

**Other models to try after the GLM-OCR example:** **`lighton-ocr2.py`** (1B, very fast), **`paddleocr-vl-1.6.py`** (0.9B, 96.33 OmniDocBench) or **`ovis-ocr2.py`** (0.9B, 96.58 OmniDocBench — current SOTA); for the smallest footprint, **`falcon-ocr.py`** (0.3B, strong on tables). Reach for a 7–8B model only when quality demands it. Several of these models sit on the public [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench) — pull the live ranking from your terminal in one command:

```bash
hf datasets leaderboard allenai/olmOCR-bench
```

But which model wins on *your* documents is still document-dependent — so [ocr-bench](https://github.com/davanstrien/ocr-bench) builds a **per-collection leaderboard** for your own data (pairwise VLM-as-judge, optionally human-validated), using these scripts under the hood.

**Language coverage:** [LANGUAGES.md](LANGUAGES.md) lists what each model's card claims (and how much evidence backs it). Machine-readable catalog for agents — script → model, params, backend, image pins, languages: [`models.json`](models.json).

_Sorted by model size:_

| Script | Model | Size | Backend | Notes |
|--------|-------|------|---------|-------|
| [`tesseract-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/tesseract-ocr.py) | [Tesseract 5](https://github.com/tesseract-ocr/tesseract) | n/a (classical) | pytesseract (CPU) | **The legacy baseline** — no GPU at all, runs on `cpu-upgrade`. Plain-text output, `--lang`/`--psm`/`--oem` exposed, 100+ language packs via apt. Apache 2.0 |
| [`pp-ocrv6.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/pp-ocrv6.py) | [PP-OCRv6](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6) | 1.5–34.5M | PaddleOCR (paddle) | **Smallest neural** — classical det+rec pipeline, not a VLM. Three tiers (`--model-tier tiny\|small\|medium`), plain-text output (not markdown). 48 langs. Runs on `t4-small`. Apache 2.0 |
| [`falcon-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/falcon-ocr.py) | [Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR) | 0.3B | falcon-perception | Smallest VLM in collection. #1 on multi-column docs and tables (olmOCR), Apache 2.0 |
| [`smoldocling-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/smoldocling-ocr.py) | [SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview) | 256M | Transformers | DocTags structured output |
| [`surya-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/surya-ocr.py) | [Surya OCR 2](https://huggingface.co/datalab-to/surya-ocr-2) | 0.65B | vLLM | **Structured** OCR + `--task layout\|table`: per-block HTML with bboxes & reading order in an extra `surya_blocks` column. 91 langs, top-under-3B on olmOCR-Bench. Modified OpenRAIL-M license. Needs the **pinned** `vllm/vllm-openai:v0.20.1` image |
| [`glm-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/glm-ocr.py) | [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | vLLM | 94.62% OmniDocBench V1.5 |
| [`paddleocr-vl.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/paddleocr-vl.py) | [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) | 0.9B | vLLM | 4 task modes (ocr/table/formula/chart) |
| [`paddleocr-vl-1.5.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/paddleocr-vl-1.5.py) | [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | Transformers | 94.5% OmniDocBench, 6 task modes |
| [`paddleocr-vl-1.6.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/paddleocr-vl-1.6.py) | [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 0.9B | vLLM | **96.33% OmniDocBench v1.6**, drop-in upgrade of 1.5 |
| [`ovis-ocr2.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/ovis-ocr2.py) | [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | 0.9B | vLLM | **96.58 OmniDocBench v1.6** (SOTA; first end-to-end model to top it). Qwen3.5 base; markdown + LaTeX + HTML tables. Apache 2.0 |
| [`ovis-ocr2-server.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/ovis-ocr2-server.py) | [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | 0.9B | vLLM server | **Server-mode sibling** of `ovis-ocr2.py`: in-job `vllm serve` + concurrent driver — ~1.7× its throughput, per-image failure isolation. See [SERVING.md](SERVING.md) |
| [`lighton-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lighton-ocr.py) | [LightOnOCR-1B](https://huggingface.co/lightonai/LightOnOCR-1B-1025) | 1B | vLLM | Fast, 3 vocab sizes |
| [`lighton-ocr2.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lighton-ocr2.py) | [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 1B | vLLM | 7× faster than v1, RLVR trained |
| [`lighton-ocr2-server.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lighton-ocr2-server.py) | [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 1B | vLLM server | **Server-mode sibling** of `lighton-ocr2.py` (the card's own documented path) — ~1.8× its throughput. See [SERVING.md](SERVING.md) |
| [`hunyuan-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/hunyuan-ocr.py) | [HunyuanOCR 1.0](https://huggingface.co/tencent/HunyuanOCR/tree/f6af82ee007fe6091b29fb3bb287b491ead41c82) | 1B | vLLM | Lightweight VLM. Pinned to the last 1.0 revision (repo root became 1.5 in-place on 2026-07-06). [Hunyuan Community License](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) (excludes EU/UK/KR) |
| [`hunyuan-ocr-1.5.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/hunyuan-ocr-1.5.py) | [HunyuanOCR-1.5](https://huggingface.co/tencent/HunyuanOCR) | 1B | vLLM | 128K context, 4K images, 12 task types, ancient scripts. ~4-5× faster/page than dots.ocr & DeepSeek-OCR-2 (tech report). [Hunyuan Community License](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) (excludes EU/UK/KR) |
| [`dots-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/dots-ocr.py) | [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) | 1.7B | vLLM | 100 languages (in-house bench), explicit low-resource claim |
| [`firered-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/firered-ocr.py) | [FireRed-OCR](https://huggingface.co/FireRedTeam/FireRed-OCR) | 2.1B | vLLM | Qwen3-VL fine-tune, Apache 2.0 |
| [`abot-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/abot-ocr.py) | [ABot-OCR](https://huggingface.co/acvlab/ABot-OCR) | 2B | vLLM | Qwen3-VL based, doc→Markdown (text/LaTeX/HTML tables). Needs `vllm/vllm-openai` image. [paper](https://arxiv.org/abs/2605.27978) |
| [`nanonets-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/nanonets-ocr.py) | [Nanonets-OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s) | 2B | vLLM | LaTeX, tables, forms |
| [`dots-mocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/dots-mocr.py) | [dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr) | 3B | vLLM | 8 prompt modes incl. SVG generation, layout + bbox, 100+ languages |
| [`nanonets-ocr2.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/nanonets-ocr2.py) | [Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) | 3B | vLLM | Next-gen, Qwen2.5-VL base. **Pin `--image vllm/vllm-openai:v0.10.2`** (vLLM ≥0.11 breaks Qwen2.5-VL → all `!`) |
| [`deepseek-ocr-vllm.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/deepseek-ocr-vllm.py) | [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | 4B | vLLM | 5 resolution + 5 prompt modes |
| [`jina-ocr-v1.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/jina-ocr-v1.py) | [jina-ocr-v1](https://huggingface.co/jinaai/jina-ocr-v1) | 3.4B MoE (~570M active) | vLLM | DeepSeek-OCR fine-tune + FastMTP speculative decoding (`--spec 0` to disable) and an n-gram repetition stop. Card: 91.14 OmniDocBench v1.6, **83.4 olmOCR-Bench**. Offline vLLM only. **CC-BY-NC-4.0** |
| [`deepseek-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/deepseek-ocr.py) | [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | 4B | Transformers | Same model, Transformers backend |
| [`deepseek-ocr2-vllm.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/deepseek-ocr2-vllm.py) | [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | 3B | vLLM | Newer; needs nightly vLLM **+ the `vllm/vllm-openai` image** ([why](#if-a-vllm-script-crashes-at-startup-the-nvcc--nvrtc-error)) |
| [`unlimited-ocr-vllm.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/unlimited-ocr-vllm.py) | [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | 3.3B | vLLM | DeepSeek-OCR-based; layout-grounded markdown (`--strip-grounding` for clean text). Single-image batch — needs Baidu's **dedicated `vllm/vllm-openai:unlimited-ocr` image** (`-cu129` on Hopper). Multi-page "long-horizon" parsing → serve it ([doc](serving-unlimited-ocr.md)); both engines do clean docs, **SGLang more robust** on hard scans. MIT |
| [`nuextract3.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/nuextract3.py) | [NuExtract3](https://huggingface.co/numind/NuExtract3) | 4B | vLLM | Markdown OCR **+ schema-guided JSON extraction** (template/Pydantic). Needs `vllm/vllm-openai` image |
| [`qianfan-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/qianfan-ocr.py) | [Qianfan-OCR](https://huggingface.co/baidu/Qianfan-OCR) | 4.7B | vLLM | #1 OmniDocBench v1.5 (93.12), Layout-as-Thought, 192 languages |
| [`olmocr2-vllm.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/olmocr2-vllm.py) | [olmOCR-2-7B](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8) | 7B | vLLM | 82.4% olmOCR-Bench |
| [`rolm-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/rolm-ocr.py) | [RolmOCR](https://huggingface.co/reducto/RolmOCR) | 7B | vLLM | Qwen2.5-VL based, general-purpose |
| [`numarkdown-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/numarkdown-ocr.py) | [NuMarkdown-8B](https://huggingface.co/numind/NuMarkdown-8B-Thinking) | 8B | vLLM | Reasoning-based OCR |

**Variants & tools** (same models, different I/O): `glm-ocr-v2.py` adds checkpoint/resume for very large jobs · `glm-ocr-bucket.py` and `falcon-ocr-bucket.py` read images/PDFs from a mounted bucket and write one `.md` per page · `surya-ocr-bucket.py` is the structured bucket recipe — OCR a bucket of files (no dataset round-trip) via either a FUSE mount **or** `huggingface_hub` batch-copy (`--io-mode mount|copy`), writing per-page `.md` + `.json` (`surya_blocks`) back to a bucket (resumable) and/or a pushed dataset · `ocr-vllm-judge.py` runs pairwise OCR-quality comparisons.

`surya-ocr.py` is the structured outlier: besides the flattened text column it writes a `surya_blocks` JSON column (per-block HTML + bounding boxes + reading order), and `--task` switches between OCR, `layout`, and `table`. It runs as **offline vLLM batch** (no server) and must use the **pinned** `vllm/vllm-openai:v0.20.1` image — its `qwen3_5` architecture is recent and version-sensitive, and that image puts vLLM at `/usr/local/lib/python3.12/site-packages` (use `--python /usr/local/bin/python3`; the exact command is in the script's docstring). Weights are **modified OpenRAIL-M**.

## Structured extraction (image or text → JSON)

Most scripts here output markdown. These take a **schema** and return **structured data** instead — give them the fields you want, they fill them in:

| Script | Model | Size | Input | Output |
|--------|-------|------|-------|--------|
| [`lfm2-vl-extract.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lfm2-vl-extract.py) | [LFM2.5-VL-1.6B-Extract](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-Extract) | 1.6B | image | JSON |
| [`nuextract3.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/nuextract3.py) | [NuExtract3](https://huggingface.co/numind/NuExtract3) | 4B | image | markdown **or** JSON |
| [`lfm2-extract.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lfm2-extract.py) | [LFM2-1.2B-Extract](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract) | 1.2B | **text** | JSON / XML / YAML |
| [`lift-extract.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lift-extract.py) | [lift](https://huggingface.co/datalab-to/lift) | 9B | image **or** PDF | JSON |

Pass `--schema` (inline JSON, a URL, or a file path). The LFM models are small and fast; run them on the `vllm/vllm-openai` image so the CUDA toolkit is present (each script's docstring has the exact command). Because `lfm2-extract.py` works on a **text** column, you can **chain it after OCR**: a recipe above turns a page into `markdown`, then `lfm2-extract.py` turns that markdown into fields.

`lift-extract.py` is the one outlier: a 9B model that also reads **multi-page PDFs** (`--pdf-column`, `--page-range`) and runs on either Transformers (`--method hf`) or vLLM (`--method vllm`). Its weights are **modified OpenRAIL-M** (free for research, personal use, and startups under $5M; no competitive use against Datalab's API) — the only non-permissive license here, so check the terms.

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
| [`pp-doclayout.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/pp-doclayout.py) | [PP-DocLayout-L](https://huggingface.co/PaddlePaddle/PP-DocLayout-L) | 123M | paddleocr | Layout bboxes (no text). Bucket support: incremental parquet shards, resumable. |

```bash
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/pp-doclayout.py \
    your-dataset your-layout-output --max-samples 10
```

Source/sink can be either an HF dataset repo OR an `hf://buckets/...` URL (auto-detected). Bucket output writes incremental zstd parquet shards via the buckets API — resumable across runs (snapshot-backed source listing) and no git/commit overhead. See the script's `--help` for all flags.

## If a vLLM script crashes at startup (the `nvcc` / `nvrtc` error)

The vLLM recipes run on the **default** Jobs image and carry a guard (`VLLM_USE_FLASHINFER_SAMPLER=0`) so they work there with the plain command. But some — especially nightly-vLLM ones — JIT-compile a CUDA kernel at engine init and crash on the default image with one of:

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
nvrtc: error: failed to open libnvrtc-builtins.so...
```

Run those on the **`vllm/vllm-openai` image**, which ships the full CUDA toolkit. Add these flags to any recipe — they point `import vllm` at the image's CUDA-matched build:

```bash
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
    --image vllm/vllm-openai --python /usr/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/<script>.py \
    INPUT OUTPUT --max-samples 10
```

This is **required** for a few scripts (e.g. `deepseek-ocr2-vllm.py`, `abot-ocr.py`, `nuextract3.py`) and a safe fallback for any vLLM recipe that crashes at startup. (It's also the more robust way to run any vLLM recipe — full CUDA toolkit, ABI-matched build. It isn't a speed-up: uv still reinstalls the script's deps either way.)

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

Open the [script source](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/glm-ocr.py) to inspect its arguments without installing dependencies. On a machine with compatible dependencies, `uv run <script-url> --help` shows its CLI options; `uv` resolves dependencies even for `--help`.

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
| [`surya-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/surya-ocr.py) | `--task ocr\|layout\|table`, `--table-mode full\|simple`, `--pdf-column`/`--page-range`, `--blocks-column` |
| [`pp-ocrv6.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/pp-ocrv6.py) | `--model-tier tiny\|small\|medium` (1.5M–34.5M params) |
| [`glm-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/glm-ocr.py) | `--task ocr\|formula\|table` |
| [`ovis-ocr2.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/ovis-ocr2.py) | `--keep-image-tags` (retain visual-region `<img>` bbox tags, filtered by default), `--min-pixels`/`--max-pixels` (processor bounds, card defaults 448²/2880²) |
| [`paddleocr-vl.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/paddleocr-vl.py) | `--task-mode ocr\|table\|formula\|chart` |
| [`paddleocr-vl-1.5.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/paddleocr-vl-1.5.py) | `--task-mode ocr\|table\|formula\|chart\|spotting\|seal` |
| [`paddleocr-vl-1.6.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/paddleocr-vl-1.6.py) | `--task-mode ocr\|table\|formula` |
| [`lighton-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lighton-ocr.py) | `--vocab-size 151k\|32k\|16k` (smaller = faster on European languages) |
| [`deepseek-ocr-vllm.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/deepseek-ocr-vllm.py) | `--resolution-mode tiny\|small\|base\|large\|gundam`, `--prompt-mode document\|image\|free\|figure\|describe`; pass `-e UV_TORCH_BACKEND=auto` |
| [`dots-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/dots-ocr.py) | `--prompt-mode ocr\|layout-all\|layout-only` |
| [`dots-mocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/dots-mocr.py) | `--prompt-mode` (8: ocr, layout-all, layout-only, web-parsing, scene-spotting, grounding-ocr, svg, general); SVG: `--model rednote-hilab/dots.mocr-svg --prompt-mode svg` |
| [`qianfan-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/qianfan-ocr.py) | `--prompt-mode ocr\|table\|formula\|chart\|scene\|kie`, `--think` (Layout-as-Thought); `kie` needs `--custom-prompt` |
| [`unlimited-ocr-vllm.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/unlimited-ocr-vllm.py) | `--strip-grounding` (drop `<\|det\|>`/`<\|ref\|>` grounding tags); needs the **`vllm/vllm-openai:unlimited-ocr`** image |
| [`numarkdown-ocr.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/numarkdown-ocr.py) | `--include-thinking` (store the reasoning trace) |
| [`nuextract3.py`](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/nuextract3.py) | `--template` / `--schema` / `--enable-thinking` — see the NuExtract3 section above |

**Image-mode models** — `abot-ocr.py` and `nuextract3.py` (Qwen3.5 architecture) need the `vllm/vllm-openai` image because the default uv-script image lacks `nvcc`. Add `--image vllm/vllm-openai:latest --python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages` (see the NuExtract3 example above for the full command). `unlimited-ocr-vllm.py` is a special case — its architecture isn't in any stable vLLM wheel, so it needs Baidu's **dedicated** `vllm/vllm-openai:unlimited-ocr` image (tag `:unlimited-ocr-cu129` on Hopper), e.g. `--image vllm/vllm-openai:unlimited-ocr --python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages` (its docstring has the full command).

## Output & features

- **Markdown column** — each run adds an `--output-column` (default `markdown`) with the OCR result.
- **Multi-model comparison** — every script records `inference_info`, so you can run several models into the *same* dataset and compare. Point a second model at the same output repo:
  ```bash
  uv run rolm-ocr.py     my-dataset my-dataset --max-samples 100
  uv run nanonets-ocr.py my-dataset my-dataset --max-samples 100   # appends
  ```
- **Reproducible sampling** — `--shuffle` (with `--seed`, default 42) draws a representative sample instead of the first N rows.
- **Automatic dataset cards** — every run writes a card with the model config, processing stats, column descriptions, and a reproduction command.

## Batch processing and live endpoints

Start with the batch examples above to process a collection of documents. For
concurrent processing or an API for your application:

- **Process a dataset:** `-server.py` recipes start vLLM inside the Job and send
  page requests concurrently. See the [server-mode OCR guide](SERVING.md) for
  supported models, setup and measured throughput. The
  [LightOnOCR-2](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/lighton-ocr2-saturate.py)
  and [OvisOCR2](https://huggingface.co/datasets/uv-scripts/ocr/blob/main/ovis-ocr2-saturate.py)
  `-saturate.py` variants add automatic concurrency and resumable output; their
  script headers explain how to run them and read their results.
- **Call OCR from an app or agent:** expose a model server with
  [Jobs serving](https://huggingface.co/docs/hub/jobs-serving). The endpoint stays
  available until you cancel the Job or its timeout is reached. The
  [Unlimited-OCR walkthrough](serving-unlimited-ocr.md) covers server setup,
  requests and parsing several pages together in one request.

The [PDF example above](#try-the-same-pages-as-a-pdf) processes pages independently
in a batch Job.

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
