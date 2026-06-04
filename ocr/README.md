---
viewer: false
tags: [uv-script, ocr, vision-language-model, document-processing, hf-jobs]
---

# OCR UV Scripts

> Part of [uv-scripts](https://huggingface.co/uv-scripts) - ready-to-run ML tools powered by UV and HuggingFace Jobs.

21 OCR scripts (text extraction) + 1 layout-detection script. Pick a model, point at your dataset, get markdown — no setup required. Layout-detection runs separately when you need bboxes for regions (text/title/table/figure/...) rather than the text itself.

## 🚀 Quick Start

Run OCR on any dataset without needing your own GPU:

```bash
# Quick test with 10 samples
hf jobs uv run --flavor l4x1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    your-input-dataset your-output-dataset \
    --max-samples 10
```

That's it! The script will:

- Process first 10 images from your dataset
- Add OCR results as a new `markdown` column
- Push the results to a new dataset
- View results at: `https://huggingface.co/datasets/[your-output-dataset]`

<details><summary>All scripts at a glance (sorted by model size)</summary>

| Script | Model | Size | Backend | Notes |
|--------|-------|------|---------|-------|
| `falcon-ocr.py` | [Falcon-OCR](https://huggingface.co/tiiuae/Falcon-OCR) | 0.3B | falcon-perception | Smallest in collection. #1 on multi-column docs and tables (olmOCR), Apache 2.0 |
| `smoldocling-ocr.py` | [SmolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview) | 256M | Transformers | DocTags structured output |
| `glm-ocr.py` | [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | vLLM | 94.62% OmniDocBench V1.5 |
| `paddleocr-vl.py` | [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) | 0.9B | Transformers | 4 task modes (ocr/table/formula/chart) |
| `paddleocr-vl-1.5.py` | [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | Transformers | 94.5% OmniDocBench, 6 task modes |
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

</details>

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

All scripts accept the same core flags. Model-specific defaults (batch size, context length, temperature) are tuned per model based on model card recommendations and can be overridden.

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

## Example: GLM-OCR

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B) scores 94.62% on OmniDocBench V1.5 and supports OCR, formula, and table extraction:

```bash
# Basic OCR
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    my-documents my-ocr-output

# Table extraction
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    my-documents my-tables --task table

# Test on 10 samples first
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    my-documents my-test --max-samples 10
```

## Example: NuExtract3 (markdown OCR **+ structured extraction**)

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

**Templates** (`--template`) and **JSON Schemas** (`--schema`) each accept **inline JSON, a URL, or a file path**. So a schema can be hosted once and reused:

```bash
# From a URL (e.g. an HF dataset's raw file)
... nuextract3.py docs out --template https://huggingface.co/datasets/ORG/REPO/raw/main/card.json

# From a JSON Schema / Pydantic model — Model.model_json_schema() dumped to JSON,
# auto-converted via numind's convert_json_schema_to_nuextract_template
... nuextract3.py docs out --schema invoice-schema.json

# From a mounted bucket (host configs in a bucket, mount read-only)
hf jobs uv run --flavor a100-large \
    --image vllm/vllm-openai:latest --python /usr/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages -s HF_TOKEN \
    -v hf://buckets/USER/configs:/configs:ro \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nuextract3.py \
    docs out --template /configs/card.json
```

Add `--enable-thinking` for harder layouts (slower; reasoning trace stored in a `<output-column>_reasoning` column). Template field names act as the model's extraction instructions, so name them descriptively — but note that overly leading names can prompt over-generation, so verify against a few examples.

<details><summary>Detailed per-model documentation</summary>

### PaddleOCR-VL-1.5 (`paddleocr-vl-1.5.py`) — 6 task modes

OCR using [PaddlePaddle/PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) with 94.5% accuracy:

- **94.5% on OmniDocBench v1.5** (0.9B parameters)
- 🧩 **Ultra-compact** - Only 0.9B parameters
- 📝 **OCR mode** - General text extraction to markdown
- 📊 **Table mode** - HTML table recognition
- 📐 **Formula mode** - LaTeX mathematical notation
- 📈 **Chart mode** - Chart and diagram analysis
- 🔍 **Spotting mode** - Text spotting with localization (higher resolution)
- 🔖 **Seal mode** - Seal and stamp recognition
- 🌍 **Multilingual** - Support for multiple languages

**Task Modes:**

- `ocr`: General text extraction (default)
- `table`: Table extraction to HTML
- `formula`: Mathematical formula to LaTeX
- `chart`: Chart and diagram analysis
- `spotting`: Text spotting with localization
- `seal`: Seal and stamp recognition

**Quick start:**

```bash
# Basic OCR mode
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl-1.5.py \
    your-input-dataset your-output-dataset \
    --max-samples 100

# Table extraction
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl-1.5.py \
    documents tables-extracted \
    --task-mode table

# Seal recognition
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl-1.5.py \
    documents seals-extracted \
    --task-mode seal
```

### PaddleOCR-VL (`paddleocr-vl.py`) 🎯 Smallest model with task-specific modes!

Ultra-compact OCR using [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) with only 0.9B parameters:

- 🎯 **Smallest model** - Only 0.9B parameters (even smaller than LightOnOCR!)
- 📝 **OCR mode** - General text extraction to markdown
- 📊 **Table mode** - HTML table recognition and extraction
- 📐 **Formula mode** - LaTeX mathematical notation
- 📈 **Chart mode** - Structured chart and diagram analysis
- 🌍 **Multilingual** - Support for multiple languages
- ⚡ **Fast initialization** - Tiny model size for quick startup
- 🔧 **ERNIE-4.5 based** - Different architecture from Qwen models

**Task Modes:**

- `ocr`: General text extraction (default)
- `table`: Table extraction to HTML
- `formula`: Mathematical formula to LaTeX
- `chart`: Chart and diagram analysis

**Quick start:**

```bash
# Basic OCR mode
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl.py \
    your-input-dataset your-output-dataset \
    --max-samples 100

# Table extraction
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl.py \
    documents tables-extracted \
    --task-mode table \
    --batch-size 32
```

### GLM-OCR (`glm-ocr.py`) 🏆 SOTA on OmniDocBench V1.5!

Compact high-performance OCR using [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) with 0.9B parameters:

- 🏆 **94.62% on OmniDocBench V1.5** - #1 overall ranking
- 🧠 **Multi-Token Prediction** - MTP loss + stable full-task RL for quality
- 📝 **Text recognition** - Clean markdown output
- 📐 **Formula recognition** - LaTeX mathematical notation
- 📊 **Table recognition** - Structured table extraction
- 🌍 **Multilingual** - zh, en, fr, es, ru, de, ja, ko
- ⚡ **Compact** - Only 0.9B parameters, MIT licensed
- 🔧 **CogViT + GLM** - Visual encoder with efficient token downsampling

**Task Modes:**

- `ocr`: Text recognition (default)
- `formula`: LaTeX formula recognition
- `table`: Table extraction

**Quick start:**

```bash
# Basic OCR
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    your-input-dataset your-output-dataset \
    --max-samples 100

# Formula recognition
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    scientific-papers formulas-extracted \
    --task formula

# Table extraction
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    documents tables-extracted \
    --task table
```

### LightOnOCR (`lighton-ocr.py`) ⚡ Good one to test first since it's small and fast!

Fast and compact OCR using [lightonai/LightOnOCR-1B-1025](https://huggingface.co/lightonai/LightOnOCR-1B-1025):

- ⚡ **Fastest**: 5.71 pages/sec on H100, ~6.25 images/sec on A100 with batch_size=4096
- 🎯 **Compact**: Only 1B parameters - quick to download and initialize
- 🌍 **Multilingual**: 3 vocabulary sizes for different use cases
- 📐 **LaTeX formulas**: Mathematical notation in LaTeX format
- 📊 **Table extraction**: Markdown table format
- 📝 **Document structure**: Preserves hierarchy and layout
- 🚀 **Production-ready**: 76.1% benchmark score, used in production

**Vocabulary sizes:**

- `151k`: Full vocabulary, all languages (default)
- `32k`: European languages, ~12% faster decoding
- `16k`: European languages, ~12% faster decoding

**Quick start:**

```bash
# Test on 100 samples with English text (32k vocab is fastest for European languages)
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr.py \
    your-input-dataset your-output-dataset \
    --vocab-size 32k \
    --batch-size 32 \
    --max-samples 100

# Full production run on A100 (can handle huge batches!)
hf jobs uv run --flavor a100-large \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr.py \
    your-input-dataset your-output-dataset \
    --vocab-size 32k \
    --batch-size 4096 \
    --temperature 0.0
```

### LightOnOCR-2 (`lighton-ocr2.py`) ⚡ Fastest OCR model!

Next-generation fast OCR using [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) with RLVR training:

- ⚡ **7× faster than v1**: 42.8 pages/sec on H100 (vs 5.71 for v1)
- 🎯 **Higher accuracy**: 83.2% on OlmOCR-Bench (+7.1% vs v1)
- 🧠 **RLVR trained**: Eliminates repetition loops and formatting errors
- 📚 **Better dataset**: 2.5× larger training data with cleaner annotations
- 🌍 **Multilingual**: Optimized for European languages
- 📐 **LaTeX formulas**: Mathematical notation support
- 📊 **Table extraction**: Markdown table format
- 💪 **Production-ready**: Outperforms models 9× larger

**Quick start:**

```bash
# Test on 100 samples
hf jobs uv run --flavor a100-large \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2.py \
    your-input-dataset your-output-dataset \
    --batch-size 32 \
    --max-samples 100

# Full production run
hf jobs uv run --flavor a100-large \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2.py \
    your-input-dataset your-output-dataset \
    --batch-size 32
```

### DeepSeek-OCR (`deepseek-ocr-vllm.py`)

Advanced document OCR using [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) with visual-text compression:

- 📐 **LaTeX equations** - Mathematical formulas in LaTeX format
- 📊 **Tables** - Extracted as HTML/markdown
- 📝 **Document structure** - Headers, lists, formatting preserved
- 🖼️ **Image grounding** - Spatial layout with bounding boxes
- 🔍 **Complex layouts** - Multi-column and hierarchical structures
- 🌍 **Multilingual** - Multiple language support
- 🎚️ **Resolution modes** - 5 presets for speed/quality trade-offs
- 💬 **Prompt modes** - 5 presets for different OCR tasks
- ⚡ **Fast batch processing** - vLLM acceleration

**Resolution Modes:**

- `tiny` (512×512): Fast, 64 vision tokens
- `small` (640×640): Balanced, 100 vision tokens
- `base` (1024×1024): High quality, 256 vision tokens
- `large` (1280×1280): Maximum quality, 400 vision tokens
- `gundam` (dynamic): Adaptive multi-tile (default)

**Prompt Modes:**

- `document`: Convert to markdown with grounding (default)
- `image`: OCR any image with grounding
- `free`: Fast OCR without layout
- `figure`: Parse figures from documents
- `describe`: Detailed image descriptions

### RolmOCR (`rolm-ocr.py`)

Fast general-purpose OCR using [reducto/RolmOCR](https://huggingface.co/reducto/RolmOCR) based on Qwen2.5-VL-7B:

- 🚀 **Fast extraction** - Optimized for speed and efficiency
- 📄 **Plain text output** - Clean, natural text representation
- 💪 **General-purpose** - Works well on various document types
- 🔥 **Large context** - Handles up to 16K tokens
- ⚡ **Batch optimized** - Efficient processing with vLLM

### Nanonets OCR (`nanonets-ocr.py`)

State-of-the-art document OCR using [nanonets/Nanonets-OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s) that handles:

- 📐 **LaTeX equations** - Mathematical formulas preserved
- 📊 **Tables** - Extracted as HTML format
- 📝 **Document structure** - Headers, lists, formatting maintained
- 🖼️ **Images** - Captions and descriptions included
- ☑️ **Forms** - Checkboxes rendered as ☐/☑

### Nanonets OCR2 (`nanonets-ocr2.py`)

Next-generation Nanonets OCR using [nanonets/Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) with improved accuracy:

- 🎯 **Enhanced quality** - 3.75B parameters for superior OCR accuracy
- 📐 **LaTeX equations** - Mathematical formulas preserved in LaTeX format
- 📊 **Advanced tables** - Improved HTML table extraction
- 📝 **Document structure** - Headers, lists, formatting maintained
- 🖼️ **Smart image captions** - Intelligent descriptions and captions
- ☑️ **Forms** - Checkboxes rendered as ☐/☑
- 🌍 **Multilingual** - Enhanced language support
- 🔧 **Based on Qwen2.5-VL** - Built on state-of-the-art vision-language model

### SmolDocling (`smoldocling-ocr.py`)

Ultra-compact document understanding using [ds4sd/SmolDocling-256M-preview](https://huggingface.co/ds4sd/SmolDocling-256M-preview) with only 256M parameters:

- 🏷️ **DocTags format** - Efficient XML-like representation
- 💻 **Code blocks** - Preserves indentation and syntax
- 🔢 **Formulas** - Mathematical expressions with layout
- 📊 **Tables & charts** - Structured data extraction
- 📐 **Layout preservation** - Bounding boxes and spatial info
- ⚡ **Ultra-fast** - Tiny model size for quick inference

### NuMarkdown (`numarkdown-ocr.py`)

Advanced reasoning-based OCR using [numind/NuMarkdown-8B-Thinking](https://huggingface.co/numind/NuMarkdown-8B-Thinking) that analyzes documents before converting to markdown:

- 🧠 **Reasoning Process** - Thinks through document layout before generation
- 📊 **Complex Tables** - Superior table extraction and formatting
- 📐 **Mathematical Formulas** - Accurate LaTeX/math notation preservation
- 🔍 **Multi-column Layouts** - Handles complex document structures
- ✨ **Thinking Traces** - Optional inclusion of reasoning process with `--include-thinking`

### dots.mocr (`dots-mocr.py`) — SVG generation + SOTA OCR

Advanced multilingual OCR and SVG generation using [rednote-hilab/dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr) with 3B parameters:

- 🌍 **100+ Languages** - Extensive multilingual support
- 📝 **Document OCR** - Clean text extraction (default mode)
- 📊 **Layout Analysis** - Structured output with bboxes and categories
- 📐 **Formula recognition** - LaTeX format support
- 🖼️ **SVG generation** - Convert charts, UI layouts, figures to editable SVG code
- 🔀 **8 prompt modes** - OCR, layout-all, layout-only, web-parsing, scene-spotting, grounding-ocr, svg, general
- 📄 **[Paper](https://arxiv.org/abs/2603.13032)** - 83.9% on olmOCR-Bench

**SVG variant:** Use `--model rednote-hilab/dots.mocr-svg` with `--prompt-mode svg` for best SVG results.

**Quick start:**

```bash
# Basic OCR
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-mocr.py \
    your-input-dataset your-output-dataset \
    --max-samples 100

# SVG generation from charts/figures
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-mocr.py \
    your-charts svg-output \
    --prompt-mode svg --model rednote-hilab/dots.mocr-svg

# Layout analysis with bounding boxes
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-mocr.py \
    your-documents layout-output \
    --prompt-mode layout-all
```

### DoTS.ocr v1 (`dots-ocr.py`)

Compact multilingual OCR using [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) with only 1.7B parameters:

- 🌍 **100+ Languages** - Extensive multilingual support
- 📝 **Simple OCR** - Clean text extraction (default mode)
- 📊 **Layout Analysis** - Optional structured output with bboxes and categories
- 📐 **Formula recognition** - LaTeX format support
- 🎯 **Compact** - Only 1.7B parameters, efficient on smaller GPUs
- 🔀 **Flexible prompts** - Switch between OCR, layout-all, and layout-only modes

### FireRed-OCR (`firered-ocr.py`)

Document OCR using [FireRedTeam/FireRed-OCR](https://huggingface.co/FireRedTeam/FireRed-OCR), a 2.1B model fine-tuned from Qwen3-VL-2B-Instruct:

- 📝 **Structured Markdown** - Preserves headings, paragraphs, lists
- 📐 **LaTeX formulas** - Inline and block math support
- 📊 **HTML tables** - Table extraction with `<table>` tags
- 🪶 **Lightweight** - 2.1B parameters, runs on L4 GPU
- 📜 **Apache 2.0** - Permissive license

**Quick start:**

```bash
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/firered-ocr.py \
    your-input-dataset your-output-dataset \
    --max-samples 100
```

### Qianfan-OCR (`qianfan-ocr.py`) — #1 on OmniDocBench v1.5

End-to-end document intelligence using [baidu/Qianfan-OCR](https://huggingface.co/baidu/Qianfan-OCR) with 4.7B parameters:

- **93.12 on OmniDocBench v1.5** — #1 end-to-end model
- **79.8 on OlmOCR Bench** — #1 end-to-end model
- 🧠 **Layout-as-Thought** — Optional reasoning phase for complex layouts (`--think`)
- 🌍 **192 languages** — Latin, CJK, Arabic, Cyrillic, and more
- 📝 **OCR mode** — Document parsing to markdown (default)
- 📊 **Table mode** — HTML table extraction
- 📐 **Formula mode** — LaTeX recognition
- 📈 **Chart mode** — Chart understanding and analysis
- 🔍 **Scene mode** — Scene text extraction
- 🔑 **KIE mode** — Key information extraction with custom prompts

**Prompt Modes:**

- `ocr`: Document parsing to markdown (default)
- `table`: Table extraction to HTML
- `formula`: Formula recognition to LaTeX
- `chart`: Chart understanding
- `scene`: Scene text extraction
- `kie`: Key information extraction (requires `--custom-prompt`)

**Quick start:**

```bash
# Basic OCR
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/qianfan-ocr.py \
    your-input-dataset your-output-dataset \
    --max-samples 100

# Layout-as-Thought for complex documents
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/qianfan-ocr.py \
    your-input-dataset your-output-dataset \
    --think --max-samples 50

# Key information extraction
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/qianfan-ocr.py \
    invoices extracted-fields \
    --prompt-mode kie --custom-prompt "Extract: name, date, total. Output as JSON."
```

### olmOCR2 (`olmocr2-vllm.py`)

High-quality document OCR using [allenai/olmOCR-2-7B-1025-FP8](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8) optimized with GRPO reinforcement learning:

- 🎯 **High accuracy** - 82.4 ± 1.1 on olmOCR-Bench (84.9% on math)
- 📐 **LaTeX equations** - Mathematical formulas in LaTeX format
- 📊 **Table extraction** - Structured table recognition
- 📑 **Multi-column layouts** - Complex document structures
- 🗜️ **FP8 quantized** - Efficient 8B model for faster inference
- 📜 **Degraded scans** - Works well on old/historical documents
- 📝 **Long text extraction** - Headers, footers, and full document content
- 🧩 **YAML metadata** - Structured front matter (language, rotation, content type)
- 🚀 **Based on Qwen2.5-VL-7B** - Fine-tuned with reinforcement learning

## 🆕 New Features

### Multi-Model Comparison Support

All scripts now include `inference_info` tracking for comparing multiple OCR models:

```bash
# First model
uv run rolm-ocr.py my-dataset my-dataset --max-samples 100

# Second model (appends to same dataset)
uv run nanonets-ocr.py my-dataset my-dataset --max-samples 100

# View all models used
python -c "import json; from datasets import load_dataset; ds = load_dataset('my-dataset'); print(json.loads(ds[0]['inference_info']))"
```

### Random Sampling

Get representative samples with the new `--shuffle` flag:

```bash
# Random 50 samples instead of first 50
uv run rolm-ocr.py ordered-dataset output --max-samples 50 --shuffle

# Reproducible random sampling
uv run nanonets-ocr.py dataset output --max-samples 100 --shuffle --seed 42
```

### Automatic Dataset Cards

Every OCR run now generates comprehensive dataset documentation including:

- Model configuration and parameters
- Processing statistics
- Column descriptions
- Reproduction instructions

## 💻 Usage Examples

### Run on HuggingFace Jobs (Recommended)

No GPU? No problem! Run on HF infrastructure:

```bash
# PaddleOCR-VL - Smallest model (0.9B) with task modes
hf jobs uv run --flavor l4x1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl.py \
    your-input-dataset your-output-dataset \
    --task-mode ocr \
    --max-samples 100

# PaddleOCR-VL - Extract tables from documents
hf jobs uv run --flavor l4x1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl.py \
    documents tables-dataset \
    --task-mode table

# PaddleOCR-VL - Formula recognition
hf jobs uv run --flavor l4x1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl.py \
    scientific-papers formulas-extracted \
    --task-mode formula \
    --batch-size 32

# GLM-OCR - SOTA 0.9B model (94.62% OmniDocBench)
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
    your-input-dataset your-output-dataset \
    --batch-size 16 \
    --max-samples 100

# DeepSeek-OCR - Real-world example (National Library of Scotland handbooks)
hf jobs uv run --flavor a100-large \
    -s HF_TOKEN \
    -e UV_TORCH_BACKEND=auto \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \
    NationalLibraryOfScotland/Britain-and-UK-Handbooks-Dataset \
    davanstrien/handbooks-deep-ocr \
    --max-samples 100 \
    --shuffle \
    --resolution-mode large

# DeepSeek-OCR - Fast testing with tiny mode
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    -e UV_TORCH_BACKEND=auto \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \
    your-input-dataset your-output-dataset \
    --max-samples 10 \
    --resolution-mode tiny

# DeepSeek-OCR - Parse figures from scientific papers
hf jobs uv run --flavor a100-large \
    -s HF_TOKEN \
    -e UV_TORCH_BACKEND=auto \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \
    scientific-papers figures-extracted \
    --prompt-mode figure

# Basic OCR job with Nanonets
hf jobs uv run --flavor l4x1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
    your-input-dataset your-output-dataset

# DoTS.ocr - Multilingual OCR with compact 1.7B model
hf jobs uv run --flavor a100-large \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-ocr.py \
    davanstrien/ufo-ColPali \
    your-username/ufo-ocr \
    --batch-size 256 \
    --max-samples 1000 \
    --shuffle

# Real example with UFO dataset 🛸
hf jobs uv run \
    --flavor a10g-large \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
    davanstrien/ufo-ColPali \
    your-username/ufo-ocr \
    --image-column image \
    --max-model-len 16384 \
    --batch-size 128

# Nanonets OCR2 - Next-gen quality with 3B model
hf jobs uv run \
    --flavor l4x1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr2.py \
    your-input-dataset \
    your-output-dataset \
    --batch-size 16

# NuMarkdown with reasoning traces for complex documents
hf jobs uv run \
    --flavor l4x4 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/numarkdown-ocr.py \
    your-input-dataset your-output-dataset \
    --max-samples 50 \
    --include-thinking \
    --shuffle

# olmOCR2 - High-quality OCR with YAML metadata
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/olmocr2-vllm.py \
    your-input-dataset your-output-dataset \
    --batch-size 16 \
    --max-samples 100

# Private dataset with custom settings
hf jobs uv run --flavor l40sx1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
    private-input private-output \
    --private \
    --batch-size 32
```

### Python API

```python
from huggingface_hub import run_uv_job

job = run_uv_job(
    "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py",
    args=["input-dataset", "output-dataset", "--batch-size", "16"],
    flavor="l4x1"
)
```

### Run Locally (Requires GPU)

```bash
# Clone and run
git clone https://huggingface.co/datasets/uv-scripts/ocr
cd ocr
uv run nanonets-ocr.py input-dataset output-dataset

# Or run directly from URL
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
    input-dataset output-dataset

# PaddleOCR-VL for task-specific OCR (smallest model!)
uv run paddleocr-vl.py documents extracted --task-mode ocr
uv run paddleocr-vl.py papers tables --task-mode table  # Extract tables
uv run paddleocr-vl.py textbooks formulas --task-mode formula  # LaTeX formulas

# RolmOCR for fast text extraction
uv run rolm-ocr.py documents extracted-text
uv run rolm-ocr.py images texts --shuffle --max-samples 100  # Random sample

# Nanonets OCR2 for highest quality
uv run nanonets-ocr2.py documents ocr-results

```

</details>

Works with any HuggingFace dataset containing images — documents, forms, receipts, books, handwriting.
