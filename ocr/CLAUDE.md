# OCR Scripts - Development Notes

## Active Scripts

### DeepSeek-OCR v1 (`deepseek-ocr-vllm.py`)
✅ **Production Ready** (Fixed 2026-02-12)
- Uses official vLLM offline pattern: `llm.generate()` with PIL images
- `NGramPerReqLogitsProcessor` prevents repetition on complex documents
- Resolution modes removed (handled by vLLM's multimodal processor)
- See: https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html

**Known issue (vLLM nightly, 2026-02-12):** Some images trigger a crop dimension validation error:
```
ValueError: images_crop dim[2] expected 1024, got 640. Expected shape: ('bnp', 3, 1024, 1024), but got torch.Size([0, 3, 640, 640])
```
This is a vLLM bug: the preprocessor defaults to gundam mode (image_size=640), but the tensor validator expects 1024x1024 even when the crop batch is empty (dim 0). Hit 2/10 on `davanstrien/ufo-ColPali`, 0/10 on NLS Medical History. Likely depends on image aspect ratios. No upstream issue filed yet. Related feature request: [vllm#28160](https://github.com/vllm-project/vllm/issues/28160) (no way to control resolution mode via mm-processor-kwargs).

### LightOnOCR-2-1B (`lighton-ocr2.py`)
✅ **Production Ready** (Fixed 2026-01-29)

**Status:** Working with vLLM nightly

**What was fixed:**
- Root cause was NOT vLLM - it was the deprecated `HF_HUB_ENABLE_HF_TRANSFER=1` env var
- The script was setting this env var but `hf_transfer` package no longer exists
- This caused download failures that manifested as "Can't load image processor" errors
- Fix: Removed the `HF_HUB_ENABLE_HF_TRANSFER=1` setting from the script

**Test results (2026-01-29):**
- 10/10 samples processed successfully
- Clean markdown output with proper headers and paragraphs
- Output dataset: `davanstrien/lighton-ocr2-test-v4`

**Example usage:**
```bash
hf jobs uv run --flavor a100-large \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr2.py \
    davanstrien/ufo-ColPali output-dataset \
    --max-samples 10 --shuffle --seed 42
```

**Model Info:**
- Model: `lightonai/LightOnOCR-2-1B`
- Architecture: Pixtral ViT encoder + Qwen3 LLM
- Training: RLVR (Reinforcement Learning with Verifiable Rewards)
- Performance: 83.2% on OlmOCR-Bench, 42.8 pages/sec on H100

### PaddleOCR-VL-1.5 (`paddleocr-vl-1.5.py`)
✅ **Production Ready** (Added 2026-01-30)

**Status:** Working with transformers

**Note:** Uses transformers backend (not vLLM) because PaddleOCR-VL only supports vLLM in server mode, which doesn't fit the single-command UV script pattern. Images are processed one at a time for stability.

**Test results (2026-01-30):**
- 10/10 samples processed successfully
- Processing time: ~50s per image on L4 GPU
- Output dataset: `davanstrien/paddleocr-vl15-final-test`

**Example usage:**
```bash
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl-1.5.py \
    davanstrien/ufo-ColPali output-dataset \
    --max-samples 10 --shuffle --seed 42
```

**Task modes:**
- `ocr` (default): General text extraction to markdown
- `table`: Table extraction to HTML format
- `formula`: Mathematical formula recognition to LaTeX
- `chart`: Chart and diagram analysis
- `spotting`: Text spotting with localization (uses higher resolution)
- `seal`: Seal and stamp recognition

**Model Info:**
- Model: `PaddlePaddle/PaddleOCR-VL-1.5`
- Size: 0.9B parameters (ultra-compact)
- Performance: 94.5% SOTA on OmniDocBench v1.5
- Backend: Transformers (single image processing)
- Requires: `transformers>=5.0.0`

### DoTS.ocr-1.5 (`dots-ocr-1.5.py`)
✅ **Production Ready** (Fixed 2026-03-14)

**Status:** Working with vLLM 0.17.1 stable

**Model availability:** The v1.5 model is NOT on HF from the original authors. We mirrored it from ModelScope to `davanstrien/dots.ocr-1.5`. Original: https://modelscope.cn/models/rednote-hilab/dots.ocr-1.5. License: MIT-based (with supplementary terms for responsible use).

**Key fix (2026-03-14):** Must pass `chat_template_content_format="string"` to `llm.chat()`. The model's `tokenizer_config.json` chat template expects string content (not openai-format lists). Without this, the model generates empty output (~1 token then EOS). The separate `chat_template.json` file handles multimodal content but vLLM uses the tokenizer_config template by default.

**Bbox coordinate system (layout modes):**
Bounding boxes from `layout-all` and `layout-only` modes are in the **resized image coordinate space**, not original image coordinates. The model uses `Qwen2VLImageProcessor` which resizes images via `smart_resize()`:
- `max_pixels=11,289,600`, `factor=28` (patch_size=14 × merge_size=2)
- Images are scaled down so `w×h ≤ max_pixels`, dims rounded to multiples of 28
- To map bboxes back to original image coordinates:
```python
import math

def smart_resize(height, width, factor=28, min_pixels=3136, max_pixels=11289600):
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

resized_h, resized_w = smart_resize(orig_h, orig_w)
scale_x = orig_w / resized_w
scale_y = orig_h / resized_h
# Then: orig_x = bbox_x * scale_x, orig_y = bbox_y * scale_y
```

**Test results (2026-03-14):**
- 3/3 samples on L4: OCR mode working, ~147 toks/s output
- 3/3 samples on L4: layout-all mode working, structured JSON with bboxes
- 10/10 samples on A100: layout-only mode on NLS Highland News, ~670 toks/s output
- Output datasets: `davanstrien/dots-ocr-1.5-smoke-test-v3`, `davanstrien/dots-ocr-1.5-layout-test`, `davanstrien/dots-ocr-1.5-nls-layout-test`

**Prompt modes:**
- `ocr` — text extraction (default)
- `layout-all` — layout + bboxes + categories + text (JSON)
- `layout-only` — layout + bboxes + categories only (JSON)
- `web-parsing` — webpage layout analysis (JSON) [new in v1.5]
- `scene-spotting` — scene text detection [new in v1.5]
- `grounding-ocr` — text from bounding box region [new in v1.5]
- `general` — free-form (use with `--custom-prompt`) [new in v1.5]

**Example usage:**
```bash
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    /path/to/dots-ocr-1.5.py \
    davanstrien/ufo-ColPali output-dataset \
    --model davanstrien/dots.ocr-1.5 \
    --max-samples 10 --shuffle --seed 42
```

**Model Info:**
- Original: `rednote-hilab/dots.ocr-1.5` (ModelScope only)
- Mirror: `davanstrien/dots.ocr-1.5` (HF)
- Parameters: 3B (1.2B vision encoder + 1.7B language model)
- Architecture: DotsOCRForCausalLM (custom code, trust_remote_code required)
- Precision: BF16
- GitHub: https://github.com/rednote-hilab/dots.ocr

---

## Pending Development

### DeepSeek-OCR-2 (`deepseek-ocr2-vllm.py`)
✅ **Production Ready** (2026-02-12)

**Status:** Working with vLLM nightly (requires nightly for `DeepseekOCR2ForCausalLM` support, not yet in stable 0.15.1)

**What was done:**
- Rewrote the broken draft script (which used base64/llm.chat/resolution modes)
- Uses the same proven pattern as v1: `llm.generate()` with PIL images + `NGramPerReqLogitsProcessor`
- Key v2 addition: `limit_mm_per_prompt={"image": 1}` in LLM init
- Added `addict` and `matplotlib` as dependencies (required by model's HF custom code)

**Test results (2026-02-12):**
- 10/10 samples processed successfully on L4 GPU
- Processing time: 6.4 min (includes model download + warmup)
- Model: 6.33 GiB, ~475 toks/s input, ~246 toks/s output
- Output dataset: `davanstrien/deepseek-ocr2-nls-test`

**Example usage:**
```bash
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr2-vllm.py \
    NationalLibraryOfScotland/medical-history-of-british-india output-dataset \
    --max-samples 10 --shuffle --seed 42
```

**Important notes:**
- Requires vLLM **nightly** (stable 0.15.1 does NOT include DeepSeek-OCR-2 support)
- The nightly index (`https://wheels.vllm.ai/nightly`) occasionally has transient build issues (e.g., only ARM wheels). If this happens, wait and retry.
- Uses same API pattern as v1: `NGramPerReqLogitsProcessor`, `SamplingParams(temperature=0, skip_special_tokens=False)`, `extra_args` for ngram settings

**Model Information:**
- Model ID: `deepseek-ai/DeepSeek-OCR-2`
- Model Card: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
- GitHub: https://github.com/deepseek-ai/DeepSeek-OCR-2
- Parameters: 3B
- Architecture: Visual Causal Flow
- Resolution: (0-6)x768x768 + 1x1024x1024 patches

## Other OCR Scripts

### Nanonets OCR (`nanonets-ocr.py`, `nanonets-ocr2.py`)
✅ Both versions working

### PaddleOCR-VL (`paddleocr-vl.py`)
✅ Working

---

## Future: OCR Smoke Test Dataset

**Status:** Idea (noted 2026-02-12)

Build a small curated dataset (`uv-scripts/ocr-smoke-test`?) with ~2-5 samples from diverse sources. Purpose: fast CI-style verification that scripts still work after dep updates, without downloading full datasets.

**Design goals:**
- Tiny (~20-30 images total) so download is seconds not minutes
- Covers the axes that break things: document type, image quality, language, layout complexity
- Has ground truth text where possible for quality regression checks
- All permissively licensed (CC0/CC-BY preferred)

**Candidate sources:**

| Source | What it covers | Why |
|--------|---------------|-----|
| `NationalLibraryOfScotland/medical-history-of-british-india` | Historical English, degraded scans | Has hand-corrected `text` column for comparison. CC0. Already tested with GLM-OCR. |
| `davanstrien/ufo-ColPali` | Mixed modern documents | Already used as our go-to test set. Varied layouts. |
| Something with **tables** | Structured data extraction | Tests `--task table` modes. Maybe a financial report or census page. |
| Something with **formulas/LaTeX** | Math notation | Tests `--task formula`. arXiv pages or textbook scans. |
| Something **multilingual** (CJK, Arabic, etc.) | Non-Latin scripts | GLM-OCR claims zh/ja/ko support. Good to verify. |
| Something **handwritten** | Handwriting recognition | Edge case that reveals model limits. |

**How it would work:**
```bash
# Quick smoke test for any script
uv run glm-ocr.py uv-scripts/ocr-smoke-test smoke-out --max-samples 5
# Or a dedicated test runner that checks all scripts against it
```

**Open questions:**
- Build as a proper HF dataset, or just a folder of images in the repo?
- Should we include expected output for regression testing (fragile if models change)?
- Could we add a `--smoke-test` flag to each script that auto-uses this dataset?
- Worth adding to HF Jobs scheduled runs for ongoing monitoring?

---

## OCR Benchmark Coordinator (`ocr-bench-run.py`)

**Status:** Working end-to-end (2026-02-14)

Launches N OCR models on the same dataset via `run_uv_job()`, each pushing to a shared repo as a separate config via `--config/--create-pr`. Eval done separately with `ocr-elo-bench.py`.

### Model Registry (4 models)

| Slug | Model ID | Size | Default GPU | Notes |
|------|----------|------|-------------|-------|
| `glm-ocr` | `zai-org/GLM-OCR` | 0.9B | l4x1 | |
| `deepseek-ocr` | `deepseek-ai/DeepSeek-OCR` | 4B | l4x1 | Auto-passes `--prompt-mode free` (no grounding tags) |
| `lighton-ocr-2` | `lightonai/LightOnOCR-2-1B` | 1B | a100-large | |
| `dots-ocr` | `rednote-hilab/dots.ocr` | 1.7B | l4x1 | Stable vLLM (>=0.9.1) |

Each model entry has a `default_args` list for model-specific flags (e.g., DeepSeek uses `["--prompt-mode", "free"]`).

### Workflow
```bash
# Launch all 4 models on same data
uv run ocr-bench-run.py source-dataset --output my-bench --max-samples 50

# Evaluate directly from PRs (no merge needed)
uv run ocr-elo-bench.py my-bench --from-prs --mode both

# Or merge + evaluate
uv run ocr-elo-bench.py my-bench --from-prs --merge-prs --mode both

# Other useful flags
uv run ocr-bench-run.py --list-models          # Show registry table
uv run ocr-bench-run.py ... --dry-run           # Preview without launching
uv run ocr-bench-run.py ... --wait              # Poll until complete
uv run ocr-bench-run.py ... --models glm-ocr dots-ocr  # Subset of models
```

### Eval script features (`ocr-elo-bench.py`)
- `--from-prs`: Auto-discovers open PRs on the dataset repo, extracts config names from PR title `[config-name]` suffix, loads data from `refs/pr/N` without merging
- `--merge-prs`: Auto-merges discovered PRs via `api.merge_pull_request()` before loading
- `--configs`: Manually specify which configs to load (for merged repos)
- `--mode both`: Runs pairwise ELO + pointwise scoring
- Flat mode (original behavior) still works when `--configs`/`--from-prs` not used

### Scripts pushed to Hub
All 4 scripts have been pushed to `uv-scripts/ocr` on the Hub with `--config`/`--create-pr` support:
- `glm-ocr.py` ✅
- `deepseek-ocr-vllm.py` ✅
- `lighton-ocr2.py` ✅
- `dots-ocr.py` ✅

### Benchmark Results

#### Run 1: NLS Medical History (2026-02-14) — Pilot

**Dataset:** `NationalLibraryOfScotland/medical-history-of-british-india` (10 samples, shuffled, seed 42)
**Output repo:** `davanstrien/ocr-bench-test` (4 open PRs)
**Judge:** `Qwen/Qwen2.5-VL-72B-Instruct` via HF Inference Providers
**Content:** Historical English, degraded scans of medical texts

**ELO (pairwise, 5 samples evaluated):**
1. DoTS.ocr — 1540 (67% win rate)
2. DeepSeek-OCR — 1539 (57%)
3. LightOnOCR-2 — 1486 (50%)
4. GLM-OCR — 1436 (29%)

**Pointwise (5 samples):**
1. DeepSeek-OCR — 5.0/5.0
2. GLM-OCR — 4.6
3. LightOnOCR-2 — 4.4
4. DoTS.ocr — 4.2

**Key finding:** DeepSeek-OCR's `--prompt-mode document` produces grounding tags (`<|ref|>`, `<|det|>`) that the judge penalizes heavily. Switching to `--prompt-mode free` (now the default in the registry) made it jump from last place to top 2.

**Caveat:** 5 samples is far too few for stable rankings. The judge VLM is called once per comparison (pairwise) or once per model-sample (pointwise) via HF Inference Providers API.

#### Run 2: Rubenstein Manuscript Catalog (2026-02-15) — First Full Benchmark

**Dataset:** `biglam/rubenstein-manuscript-catalog` (50 samples, shuffled, seed 42)
**Output repo:** `davanstrien/ocr-bench-rubenstein` (4 PRs)
**Judge:** Jury of 2 via `ocr-vllm-judge.py` — `Qwen/Qwen2.5-VL-7B-Instruct` + `Qwen/Qwen3-VL-8B-Instruct` on A100
**Content:** ~48K typewritten + handwritten manuscript catalog cards from Duke University (CC0)

**ELO (pairwise, 50 samples, 300 comparisons, 0 parse failures):**

| Rank | Model | ELO | W | L | T | Win% |
|------|-------|-----|---|---|---|------|
| 1 | LightOnOCR-2-1B | 1595 | 100 | 50 | 0 | 67% |
| 2 | DeepSeek-OCR | 1497 | 73 | 77 | 0 | 49% |
| 3 | GLM-OCR | 1471 | 57 | 93 | 0 | 38% |
| 4 | dots.ocr | 1437 | 70 | 80 | 0 | 47% |

**OCR job times** (all 50 samples each):
- dots-ocr: 5.3 min (L4)
- deepseek-ocr: 5.6 min (L4)
- glm-ocr: 5.7 min (L4)
- lighton-ocr-2: 6.4 min (A100)

**Key findings:**
- **LightOnOCR-2-1B dominates** on manuscript catalog cards (67% win rate, 100-point ELO gap over 2nd place) — a very different result from the NLS pilot where it placed 3rd
- **Rankings are dataset-dependent**: NLS historical medical texts favored DoTS.ocr and DeepSeek-OCR; Rubenstein typewritten/handwritten cards favor LightOnOCR-2
- **Jury of small models works well**: 0 parse failures on 300 comparisons thanks to vLLM structured output (xgrammar). Majority voting between 2 judges provides robustness
- **50 samples gives meaningful separation**: Clear ELO gaps (1595 → 1497 → 1471 → 1437) unlike the noisy 5-sample pilot
- This validates the multi-dataset benchmark approach — no single dataset tells the whole story

#### Run 3: UFO-ColPali (2026-02-15) — Cross-Dataset Validation

**Dataset:** `davanstrien/ufo-ColPali` (50 samples, shuffled, seed 42)
**Output repo:** `davanstrien/ocr-bench-ufo` (4 PRs)
**Judge:** `Qwen/Qwen3-VL-30B-A3B-Instruct` via `ocr-vllm-judge.py` on A100 (updated prompt)
**Content:** Mixed modern documents (invoices, reports, forms, etc.)

**ELO (pairwise, 50 samples, 294 comparisons):**

| Rank | Model | ELO | W | L | T | Win% |
|------|-------|-----|---|---|---|------|
| 1 | DeepSeek-OCR | 1827 | 130 | 17 | 0 | 88% |
| 2 | dots.ocr | 1510 | 64 | 83 | 0 | 44% |
| 3 | LightOnOCR-2-1B | 1368 | 77 | 70 | 0 | 52% |
| 4 | GLM-OCR | 1294 | 23 | 124 | 0 | 16% |

**Human validation (30 comparisons):** DeepSeek-OCR #1 (same as judge), LightOnOCR-2 #3 (same). Middle pack (GLM-OCR #2 human / #4 judge, dots.ocr #4 human / #2 judge) shuffled.

#### Cross-Dataset Comparison (Human-Validated)

| Model | Rubenstein Human | Rubenstein Kimi | UFO Human | UFO 30B |
|-------|:---------------:|:---------------:|:---------:|:-------:|
| DeepSeek-OCR | **#1** | **#1** | **#1** | **#1** |
| GLM-OCR | #2 | #3 | #2 | #4 |
| LightOnOCR-2 | #4 | #2 | #3 | #3 |
| dots.ocr | #3 | #4 | #4 | #2 |

**Conclusion:** DeepSeek-OCR is consistently #1 across datasets and evaluation methods. Middle-pack rankings are dataset-dependent. Updated prompt fixed the LightOnOCR-2 overrating seen with old prompt/small judges.

*Note: NLS pilot results (5 samples, 72B API judge) omitted — not comparable with newer methodology.*

### Known Issues / Next Steps

1. ✅ **More samples needed** — Done. Rubenstein run (2026-02-15) used 50 samples and produced clear ELO separation across all 4 models.
2. ✅ **Smaller judge model** — Tested with Qwen VL 7B + Qwen3 VL 8B via `ocr-vllm-judge.py`. Works well with structured output (0 parse failures). Jury of small models compensates for individual model weakness. See "Offline vLLM Judge" section below.
3. **Auto-merge in coordinator** — `--wait` could auto-merge PRs after successful jobs. Not yet implemented.
4. **Adding more models** — `rolm-ocr.py` exists but needs `--config`/`--create-pr` added. `deepseek-ocr2-vllm.py`, `paddleocr-vl-1.5.py`, etc. could also be added to the registry.
5. **Leaderboard Space** — See future section below.
6. ✅ **Result persistence** — `ocr-vllm-judge.py` now has `--save-results REPO_ID` flag. First dataset: `davanstrien/ocr-bench-rubenstein-judge`.
7. **More diverse datasets** — Rankings are dataset-dependent (LightOnOCR-2 wins on Rubenstein, DoTS.ocr won pilot on NLS). Need benchmarks on tables, formulas, multilingual, and modern documents for a complete picture.
8. ✅ **Human validation** — `ocr-human-eval.py` completed on Rubenstein (30/30). Tested 3 judge configs. **Kimi K2.5 (170B) via Novita + updated prompt = best human agreement** (only judge to match human's #1). Now default in `ocr-jury-bench.py`. See `OCR-BENCHMARK.md` for full comparison.

---

## Offline vLLM Judge (`ocr-vllm-judge.py`)

**Status:** Working end-to-end (2026-02-15)

Runs pairwise OCR quality comparisons using a local VLM judge via vLLM's offline `LLM()` pattern. Supports jury mode (multiple models vote sequentially on the same GPU) with majority voting.

### Why use this over the API judge (`ocr-jury-bench.py`)?

| | API judge (`ocr-jury-bench.py`) | Offline judge (`ocr-vllm-judge.py`) |
|---|---|---|
| Parse failures | Needs retries for malformed JSON | 0 failures — vLLM structured output guarantees valid JSON |
| Network | Rate limits, timeouts, transient errors | Zero network calls |
| Cost | Per-token API pricing | Just GPU time |
| Judge models | Limited to Inference Providers catalog | Any vLLM-supported VLM |
| Jury mode | Sequential API calls per judge | Sequential model loading, batch inference per judge |
| Best for | Quick spot-checks, access to 72B models | Batch evaluation (50+ samples), reproducibility |

**Pushed to Hub:** `uv-scripts/ocr` as `ocr-vllm-judge.py` (2026-02-15)

### Test Results (2026-02-15)

**Test 1 — Single judge, 1 sample, L4:**
- Qwen2.5-VL-7B-Instruct, 6/6 comparisons, 0 parse failures
- Total time: ~3 min (including model download + warmup)

**Test 2 — Jury of 2, 3 samples, A100:**
- Qwen2.5-VL-7B + Qwen3-VL-8B, 15/15 comparisons, 0 parse failures
- GPU cleanup between models: successful (nanobind warnings are cosmetic)
- Majority vote aggregation working (`[2/2]` unanimous, `[1/2]` split)
- Total time: ~4 min (including both model downloads)

**Test 3 — Full benchmark, 50 samples, A100 (Rubenstein Manuscript Catalog):**
- Qwen2.5-VL-7B + Qwen3-VL-8B jury, 300/300 comparisons, 0 parse failures
- Input: `davanstrien/ocr-bench-rubenstein` (4 PRs from `ocr-bench-run.py`)
- Produced clear ELO rankings with meaningful separation
- See "Benchmark Results → Run 2" in the OCR Benchmark Coordinator section above

### Usage

```bash
# Single judge on L4
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    ocr-vllm-judge.py davanstrien/ocr-bench-nls-50 --from-prs \
    --judge-model Qwen/Qwen2.5-VL-7B-Instruct --max-samples 10

# Jury of 2 on A100 (recommended for jury mode)
hf jobs uv run --flavor a100-large -s HF_TOKEN \
    ocr-vllm-judge.py davanstrien/ocr-bench-nls-50 --from-prs \
    --judge-model Qwen/Qwen2.5-VL-7B-Instruct \
    --judge-model Qwen/Qwen3-VL-8B-Instruct \
    --max-samples 50
```

### Implementation Notes
- Comparisons built upfront on CPU as `NamedTuple`s, then batched to vLLM in single `llm.chat()` call
- Structured output via compatibility shim: `StructuredOutputsParams` (vLLM >= 0.12) → `GuidedDecodingParams` (older) → prompt-based fallback
- GPU cleanup between jury models: `destroy_model_parallel()` + `gc.collect()` + `torch.cuda.empty_cache()`
- Position bias mitigation: A/B order randomized per comparison
- A100 recommended for jury mode; L4 works for single 7B judge

### Next Steps
1. ✅ **Scale test** — Completed on Rubenstein Manuscript Catalog (50 samples, 300 comparisons, 0 parse failures). Rankings differ from API-based pilot (different dataset + judge), validating multi-dataset approach.
2. ✅ **Result persistence** — Added `--save-results REPO_ID` flag. Pushes 3 configs to HF Hub: `comparisons` (one row per pairwise comparison), `leaderboard` (ELO + win/loss/tie per model), `metadata` (source dataset, judge models, seed, timestamp). First dataset: `davanstrien/ocr-bench-rubenstein-judge`.
3. **Integrate into `ocr-bench-run.py`** — Add `--eval` flag that auto-runs vLLM judge after OCR jobs complete

---

## Blind Human Eval (`ocr-human-eval.py`)

**Status:** Working (2026-02-15)

Gradio app for blind A/B comparison of OCR outputs. Shows document image + two anonymized OCR outputs, human picks winner or tie. Computes ELO rankings from human annotations and optionally compares against automated judge results.

### Usage

```bash
# Basic — blind human eval only
uv run ocr-human-eval.py davanstrien/ocr-bench-rubenstein --from-prs --max-samples 5

# With judge comparison — loads automated judge results for agreement analysis
uv run ocr-human-eval.py davanstrien/ocr-bench-rubenstein --from-prs \
    --judge-results davanstrien/ocr-bench-rubenstein-judge --max-samples 5
```

### Features
- **Blind evaluation**: Two-tab design — Evaluate tab never shows model names, Results tab reveals rankings
- **Position bias mitigation**: A/B order randomly swapped per comparison
- **Resume support**: JSON annotations saved atomically after each vote; restart app to resume where you left off
- **Live agreement tracking**: Per-vote feedback shows running agreement with automated judge (when `--judge-results` provided)
- **Split-jury prioritization**: Comparisons where automated judges disagreed ("1/2" agreement) shown first — highest annotation value per vote
- **Image variety**: Round-robin interleaving by sample so you don't see the same document image repeatedly
- **Soft/hard disagreement analysis**: Distinguishes between harmless ties-vs-winner disagreements and genuine opposite-winner errors

### First Validation Results (Rubenstein, 30 annotations)

Tested 3 judge configs against 30 human annotations. **Kimi K2.5 (170B) via Novita** is the only judge to match human's #1 pick (DeepSeek-OCR). Small models (7B/8B/30B) all overrate LightOnOCR-2 due to bias toward its commentary style. Updated prompt (prioritized faithfulness > completeness > accuracy) helps but model size is the bigger factor.

Full results and analysis in `OCR-BENCHMARK.md` → "Human Validation" section.

### Next Steps
1. **Second dataset** — Run on NLS Medical History for cross-dataset human validation
2. **Multiple annotators** — Currently single-user; could support annotator ID for inter-annotator agreement
3. **Remaining LightOnOCR-2 gap** — Still #2 (Kimi) vs #4 (human). May need to investigate on more samples or strip commentary in preprocessing

---

## Future: Leaderboard HF Space

**Status:** Idea (noted 2026-02-14)

Build a Hugging Face Space with a persistent leaderboard that gets updated after each benchmark run. This would give a public-facing view of OCR model quality.

**Design ideas:**
- Gradio or static Space displaying ELO ratings + pointwise scores
- `ocr-elo-bench.py` could push results to a dataset that the Space reads
- Or the Space itself could run evaluation on demand
- Show per-document comparisons (image + side-by-side OCR outputs)
- Historical tracking — how scores change across model versions
- Filter by document type (historical, modern, tables, formulas, multilingual)

**Open questions:**
- Should the eval script push structured results to a dataset (e.g., `uv-scripts/ocr-leaderboard-data`)?
- Static leaderboard (updated by CI/scheduled job) vs interactive (evaluate on demand)?
- Include sample outputs for qualitative comparison?
- How to handle different eval datasets (NLS medical history vs UFO vs others)?

---

## Incremental Uploads / Checkpoint Strategy — ON HOLD

**Status:** Waiting on HF Hub Buckets (noted 2026-02-20)

**Current state:**
- `glm-ocr.py` (v1): Simple batch-then-push. Works fine for most jobs.
- `glm-ocr-v2.py`: Adds CommitScheduler-based incremental uploads + checkpoint/resume. ~400 extra lines. Works but has tradeoffs (commit noise, `--create-pr` incompatible, complex resume metadata).

**Decision: Do NOT port v2 pattern to other scripts.** Wait for HF Hub Buckets instead.

**Why:** Two open PRs will likely make the v2 CommitScheduler approach obsolete:
- [huggingface_hub#3673](https://github.com/huggingface/huggingface_hub/pull/3673) — Buckets API: S3-like mutable object storage on HF, no git versioning overhead
- [huggingface_hub#3807](https://github.com/huggingface/huggingface_hub/pull/3807) — HfFileSystem support for buckets: fsspec-compatible, so pyarrow/pandas/datasets can read/write `hf://buckets/` paths directly

**What Buckets would replace:** Once landed, incremental saves become one line per batch:
```python
batch_ds.to_parquet(f"hf://buckets/{user}/ocr-scratch/shard-{batch_num:05d}.parquet")
```
No CommitScheduler, no CleanupScheduler, no resume metadata, no completed batch scanning. Just write to the bucket path via fsspec. Final step: read back from bucket, `push_to_hub` to a clean dataset repo (compatible with `--create-pr`).

**Action items when Buckets ships:**
1. Test `hf://buckets/` fsspec writes on one script (glm-ocr is the guinea pig)
2. Verify: write performance, atomicity (partial writes visible?), auth propagation in HF Jobs
3. If it works, adopt as the standard pattern for all scripts — simple enough to inline (~20 lines)
4. Retire `glm-ocr-v2.py` CommitScheduler approach

**Until then:** v1 scripts stay as-is. `glm-ocr-v2.py` exists if someone needs resume on a very large job today.

---

**Last Updated:** 2026-02-20
**Watch PRs:**
- **HF Hub Buckets API** ([#3673](https://github.com/huggingface/huggingface_hub/pull/3673)): Core buckets support. Will enable simpler incremental upload pattern for all scripts.
- **HfFileSystem Buckets** ([#3807](https://github.com/huggingface/huggingface_hub/pull/3807)): fsspec support for `hf://buckets/` paths. Key for zero-boilerplate writes from scripts.
- DeepSeek-OCR-2 stable vLLM release: Currently only in nightly. Watch for vLLM 0.16.0 stable release on PyPI to remove nightly dependency.
- nanobind leak warnings in vLLM structured output (xgrammar): Cosmetic only, does not affect results. May be fixed in future xgrammar release.
