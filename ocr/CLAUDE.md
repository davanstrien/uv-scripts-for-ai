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

### Unlimited-OCR (`unlimited-ocr-vllm.py`)
✅ **Production Ready — single-image** (added + validated 2026-06-28)

Baidu's `baidu/Unlimited-OCR` (3.3B, MIT, DeepSeek-OCR / DeepSeek-OCR-2 descendant). Offline vLLM
batch recipe adapted from `deepseek-ocr-vllm.py` — `llm.generate()` with PIL images +
`NGramPerReqLogitsProcessor` (imported from `vllm.model_executor.models.unlimited_ocr`), prompt
`<image>document parsing.`, `SamplingParams(temperature=0, skip_special_tokens=False,
extra_args=dict(ngram_size=35, window_size=128))`, `limit_mm_per_prompt={"image": 1}`. One image per
row → one markdown. `--strip-grounding` drops `<|det|>`/`<|ref|>` tags (verified locally on real
output: removes boxes, keeps inner text + LaTeX).

**⚠️ Dedicated image, not the standard one.** The arch is NOT in any stable vLLM pip wheel — must run
on Baidu's `vllm/vllm-openai:unlimited-ocr` (CUDA 13.0; `:unlimited-ocr-cu129` on Hopper). So `vllm`
and `torch` are **omitted from the PEP 723 deps** and come from the image via `PYTHONPATH`. The image
uses the **standard** layout: `--python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages`
(vLLM `0.23.1rc1.dev541` lives there; probed 2026-06-28). The `unlimited_ocr` module re-exports
`deepseek_ocr.NGramPerReqLogitsProcessor`. Recipe: https://recipes.vllm.ai/baidu/Unlimited-OCR

**Smoke tests (2026-06-28):**
- **ufo-ColPali** (5, l4x1): 5/5 OK, 2.3 min, ~200 tok/s. Clean layout-grounded markdown — accurate
  text, `<|det|>` bboxes (0–1000), multilingual (Spanish), LaTeX. Output `davanstrien/unlimited-ocr-smoke`.
- **encyclopaedia-britannica-1771** (8, l4x1, `--strip-grounding`): 6/6 content pages produced clean
  text matching the dataset's own `ocr_text` length almost exactly (e.g. row 1: md 5811 vs ocr_text
  5752), period-accurate 1771 OCR (long-ſ, archaic spelling). The 2 "empty" rows are genuinely blank
  pages (ground-truth `ocr_text` 3–24 chars). Output `davanstrien/unlimited-ocr-britannica-smoke`.

**Multi-page: BOTH engines work on clean docs; robustness differs on hard scans. (Corrected
2026-06-29 — earlier "vLLM multi-page is broken" was an input-difficulty artifact.)**
- **Control test that overturned the first read:** ran the SAME clean synthetic 2-page doc through the
  **vLLM server** that SGLang had aced. vLLM returned **`<PAGE>=2`, both pages, real text** (`Chapter
  One The Harbor` + lines / `Chapter Two The Market` + lines), with minor body-OCR slips ("early oakh",
  "Guile covered") — i.e. the model *misreading*, not the engine hallucinating. Worked with both 1×
  and 2× `<image>` prompt forms + `vllm_xargs.window_size=1024`. So **vLLM multi-page works**.
- **What the earlier garbling actually was:** my first vLLM multi-page tests used **hard** inputs —
  `unlimited-ocr-pdf-test` (blank + dense 1771 Britannica) and ufo newspaper clippings. On those, vLLM
  multi-page degraded to hallucination (counting garbage "SIGILLUM. 17. 96…", `2017年1月1日` loops,
  content in neither input). SGLang read the *same* hard ufo input as real content → **SGLang is more
  robust on hard/degraded scans**, but neither engine is "broken."
- **Offline `LLM().generate()`** still needs one `<image>` per image (single placeholder → assertion);
  offline multi-page was only tested on the hard Britannica PDF (garbled) — not re-tested on clean, so
  the recipe stays single-image (multi-page belongs to serving).
- `images_config`/`image_mode` are **SGLang-only** params (vLLM ignores them); on vLLM use one
  `<image>` per page + `window_size=1024` in `vllm_xargs`.
- **Upstream check (vllm-project/vllm#46564, "Support Unlimited OCR", merged 2026-06-28):** confirms
  this. Multi-image IS implemented (crop/gundam auto-disabled → base mode; one `<image>` placeholder
  per image). R-SWA needs the **FlexAttention** backend (auto on non-FA4 GPUs like L4) or FA4 on
  H20/H100 — our run correctly used FlexAttention. BUT: the PR's only benchmark is **single-page
  OmniDocBench** (FA4 92.12 / Flex 92.38); there is **no multi-page test, no `examples/`, no canonical
  multi-page prompt** in the merged code. PR-author comment: multi-page needs **V1 + NGramPerReq-
  LogitsProcessor** (V2 lacks custom logits processors), and their "14-page PDF merge" smoke test only
  confirmed "**R-SWA itself works**" (mechanism runs on long seqs) — *not* OCR quality. So nobody
  upstream has shown multi-page OCR quality; the tweet's "40+ pages, low edit distance" is ahead of the
  merged evidence. (Our own clean-doc control test later showed vLLM multi-page DOES read correctly —
  see the corrected block above; the earlier garbling was hard-input degradation, not an engine break.)
- **Conclusion:** the **batch recipe stays single-image** (offline multi-page is finicky and untested
  on clean; `--pdf-column` removed). For multi-page, **serve** the model — both engines read clean
  multi-page docs; route hard/degraded scans to **SGLang** (more robust; authors' `images_config` path;
  serving-unlimited-ocr.md Option B + §3). Image probed: `vllm 0.23.1rc1.dev541` (docs say "0.25.0+").
- **SGLang multi-page — ✅ FIXED + validated working (2026-06-28).** Multi-page is the model's headline
  feature and **SGLang delivers it robustly** (vLLM multi-page also works on clean docs but hallucinated
  on hard scans — see corrected block above). Two pins were needed:
  1. **Image `lmsysorg/sglang:v0.5.10.post1`** (not `:latest`). `:latest` drifted to sglang 0.5.14 /
     torch 2.11 / cu130; the wheel (`dev11416`) needs torch 2.9.1 / cuda-python 12.9 / flashinfer 0.6.7 /
     xgrammar 0.1.32 / transformers 5.3.0. Found v0.5.10.post1 by bisecting sglang release pyproject
     pins — the **last** release before the torch-2.11 bump; matches the wheel exactly.
  2. **`a100-large` + `--attention-backend flashinfer`** (not `h200`/`fa3`). `fa3` needs Hopper, but
     HF's `h200` nodes **fail GPU init with `CUDA error 802: system not yet initialized`, 3/3** (infra /
     Fabric-Manager — *all* working jobs this session were l4x1/a100, never h200). The version pin alone
     did NOT fix 802; the 802 is purely the h200 node. a100+flashinfer dodges it.
  - **Result:** server up; clean 2-page synthetic doc → **both pages read verbatim, `<PAGE>`-separated**
    (`Chapter One: The Harbor…` / `Chapter Two: The Market…`); ufo pages → **real content**
    (`OUT OF THIS WORLD / UFO FlyBys…`), *not* vLLM's hallucinated garbage. Client: OpenAI API,
    `images_config:{image_mode:base}` + `Multi page parsing.`; no per-request NGram processor (so harder
    scans show minor page-merge/OCR slips — fa3 + the custom logit processor would tighten quality; the
    mechanism works). Working command lives in `serving-unlimited-ocr.md` Option B; switch back to
    `fa3`/`h200` for exact R-SWA once the h200 802 infra issue clears.

**Example usage:**
```bash
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    --image vllm/vllm-openai:unlimited-ocr --python /usr/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
    ./ocr/unlimited-ocr-vllm.py davanstrien/ufo-ColPali output-dataset --max-samples 10 --shuffle
```

**Deferred follow-up (captured, not built):** a *multi-page batch* recipe that drives the **SGLang
server** in-job (server lifecycle + `ThreadPoolExecutor` over multi-page docs, like Baidu's `infer.py`,
→ Hub) — the only way to get robust multi-page at corpus scale, since SGLang offline-Engine is
non-viable (server-only, custom-logit-processor/R-SWA are server-side, `fa3` Hopper-only) and vLLM
offline needs one `<image>` per page and degrades on hard scans. Gate: a real corpus-scale multi-page
need **+** the h200/`fa3` infra fix (for exact R-SWA quality). Single-image vLLM (this recipe) stays
the batch default.

### Nanonets OCR (`nanonets-ocr.py`, `nanonets-ocr2.py`)
✅ Both versions working

### PaddleOCR-VL (`paddleocr-vl.py`)
✅ Working

### lift (`lift-extract.py`)
✅ **Both backends validated on Jobs** (added 2026-06-22)

Datalab's `lift` (9B, Qwen3.5-based) for **schema-constrained** structured extraction:
image *or* multi-page PDF + JSON Schema → JSON. Sits alongside `nuextract3.py` /
`lfm2-vl-extract.py` in the structured-extraction group, but it's the only one that
ingests PDFs directly (one row = one document, multi-page collapsed into one extraction).

**Shared rendering** comes from lift: we reuse `lift.input.load_file` (auto-detects PDF vs
image by content; `pypdfium2`, DPI/min-dim, `--page-range`) via a temp file per row. Each row
→ a list of page images → one extraction. Both backends share this.

**Backends (`--method`)** — both **in-process, single command** (no server):
- `hf` (default): drives the `lift-pdf` package directly — `InferenceManager(method="hf")` →
  `AutoModelForImageTextToText`, bf16, batches a list of `BatchInputItem` conversations with
  left padding. **No** constrained decoding (plain `model.generate`); trusts lift's training.
  Runs on the **default** uv image. Simplest path; best for small jobs.
- `vllm`: vLLM's **offline `LLM()` engine** + `llm.chat()` with structured outputs — the
  repo's standard fast-batch pattern. We reproduce lift's *own* vLLM recipe (their `generate_vllm`)
  rather than calling the package: `PROMPT_MAPPING["direct"]`, `scale_to_fit`,
  `mm_processor_kwargs={min_pixels:3136,max_pixels:861696}`, and the guided JSON schema
  (`json_schema_to_pydantic.create_model` → `make_properties_nullable` → `StructuredOutputsParams`,
  with the version shim from `ocr-vllm-judge.py`). Sampling matches lift exactly: `temperature=0.0,
  top_p=0.1, max_tokens=12384`. Needs the `vllm/vllm-openai` image (vLLM not in our deps; reused
  from the image via `PYTHONPATH`, which also wins the torch version → no clash). **Not mirrored:**
  lift's repeat-token retry loop (re-runs looped items at higher temp) — less critical here since
  the grammar constraint already prevents runaway repetition.

> **History:** the first `--method vllm` used the package's path, which is an OpenAI *client* →
> server (lift's `lift_vllm` shells out to `sudo docker run`, unusable in a Job). We built+validated
> an auto-launched `vllm serve` subprocess for it, then replaced the whole thing with the offline
> `LLM()` engine — cleaner single command, no HTTP, and the repo's established pattern.

**Model id:** card repo is `datalab-to/lift` (9.65B, license `openrail`, not gated). The
installed package's internal default was `datalab-to/lift-extract`; we pin `--model
datalab-to/lift` via the `MODEL_CHECKPOINT` env (set *before* importing lift, since settings
read env at import). Confirmed in the smoke test: `datalab-to/lift` (commit `3129597…`) loads.

**Naming gotcha:** the script must NOT be named `lift.py` — that shadows the installed `lift`
package (`import lift` resolves to the script itself → `ImportError: cannot import name
'resolve_schema'`). Hence `lift-extract.py`. Hit this on the first Jobs run.

**License:** code Apache-2.0, **weights modified OpenRAIL-M** (research/personal/<$5M, no
competitive use vs Datalab API). Surfaced in the docstring, the README entry, and the output
dataset card.

**Benchmark both backends:** `--config hf --create-pr` vs `--config vllm --create-pr` into one
repo (same multi-config pattern as the other OCR scripts).

**Smoke-test results (2026-06-22, `davanstrien/ufo-ColPali`, 3 samples, a100-large):**
- **HF backend** (default image): 3/3 valid JSON, batched (1 chunk of 3 at `--batch-size 8`, no
  padding/image-count issues), 1.8 min. Output `davanstrien/lift-smoke-hf`. Resolved
  `lift-pdf==0.1.1, transformers==5.12.1, torch==2.12.1, datasets==5.0.0`.
- **vLLM offline backend** (`vllm/vllm-openai` image): `LLM()` engine loaded (weights 18 GiB /
  59s via Xet high-perf), `llm.chat` batched all 3 prompts in one call (538 tok/s in), 3/3 valid
  JSON via `StructuredOutputsParams`, clean engine shutdown, 5.2 min (engine init + torch.compile
  warmup dominates at 3 samples; wins at scale). `vllm==0.23.0`, image's `torch==2.11.0+cu130` (no
  clash). Output `davanstrien/lift-smoke-vllm-offline`.
  - (The earlier server-subprocess vLLM also passed — `davanstrien/lift-smoke-vllm`, 5.3 min — but
    was replaced by the offline engine; see History above.)
- **All paths produce valid schema-shaped JSON**, e.g.
  `{"title": "OUT OF THIS WORLD UFO FlyBys in Middle Tennessee", "date": "Oct. 26, 1995"}`;
  absent fields → `null` (nullable-leaf transform). `parse_error_rate: 0.0`. Outputs agree across
  backends except minor low-temp content drift (offline-vLLM recovered a Spanish title hf left null).

**Still untested (lower risk — reuses lift's `load_file`, exercised on the image path):**
- PDF column path (`--pdf-column`, `--page-range`) on a real PDF-bytes dataset.
- `l4x1` for the hf backend (9B bf16 ≈ 19GB; default `a100-large` confirmed comfortable).

Requires Python ≥3.12 (lift-pdf constraint) — fine on the standard images.

### Surya OCR 2 (`surya-ocr.py`)
✅ **OCR + layout + table validated on Jobs** (added 2026-06-22)

Datalab's **Surya OCR 2** (`datalab-to/surya-ocr-2`, 650M, Qwen3.5-style) for **structured** OCR.
Unlike the flat-markdown scripts, it returns per-block HTML + bounding boxes + reading order. The
recipe writes **two columns**: `--output-column` (default `markdown`, flattened reading-order text)
**and** `surya_blocks` (the full structured result as JSON, one entry per page). `--task` switches
between `ocr` (RecognitionPredictor, full-page), `layout` (LayoutPredictor), and `table`
(TableRecPredictor; `--table-mode full` → HTML, `simple` → rows/cols/cells).

**Engine — offline vLLM batch, NO server (the whole trick).** Surya normally runs its VLM through a
**spawned server**: on GPU it `docker run`s `vllm/vllm-openai`, on CPU a `llama-server` subprocess
(`surya/inference/backends/{vllm,llamacpp}.py`). Docker-in-Docker isn't available inside a Job, so
the default path can't work. Instead we subclass Surya's `Backend` ABC
(`surya/inference/backends/base.py`: `start`/`stop`/`generate(batch)->List[BatchOutputItem]`) with an
in-process `OfflineVLLMBackend` that runs vLLM's offline `LLM().chat()` and inject it via
`manager.backend = ...` (bypassing `SuryaInferenceManager.__init__`'s autodetect). **Surya still owns
everything else** — prompts (`PROMPT_MAPPING`), image scaling (`scale_to_fit`), HTML/bbox parsing, the
repeat-loop fallback, the 0–1000→pixel bbox rescale, and the layout/table predictors — so we only swap
the transport. We reuse Surya's own `_build_messages`/`scale_to_fit`/`PROMPT_MAPPING` so the offline
path matches the server byte-for-byte. `mm_processor_kwargs={min_pixels:3136,max_pixels:6291456}`,
`dtype=bfloat16`, `max_model_len=18000`, sampling `temperature=0.0/top_p=0.1`, `logprobs=1` →
`mean_token_prob` → Surya's per-block `confidence`. Guided JSON (layout's `LAYOUT_JSON_SCHEMA`) maps to
`StructuredOutputsParams`/`GuidedDecodingParams` (same shim as `ocr-vllm-judge.py`). **Not mirrored:**
Surya's per-item repeat-token retry — its recognition layer already detects loops and falls back to
layout+block OCR, so the backend stays simple (like lift).

**⚠️ Image gotcha — pin `vllm/vllm-openai:v0.20.1` AND use the `site-packages` path.** Surya-2 is the
recent, **version-sensitive, hybrid (linear-attention) `qwen3_5`** architecture; v0.20.1 is Surya's
known-good vLLM. Unlike the other vLLM recipes (which use the unversioned image at
`/usr/bin/python3` + `dist-packages`), the **`:v0.20.1`** image puts python at `/usr/local/bin/python3`
and vLLM/torch at **`/usr/local/lib/python3.12/site-packages`**. The first smoke run used the old
`dist-packages` path → `No module named 'vllm'` → 0/5. Correct flags:
```bash
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    --image vllm/vllm-openai:v0.20.1 --python /usr/local/bin/python3 \
    -e PYTHONPATH=/usr/local/lib/python3.12/site-packages \
    ./ocr/surya-ocr.py davanstrien/ufo-ColPali OUTPUT --max-samples 5
```
`PYTHONPATH` is prepended ahead of the uv venv, so the **image's** torch 2.11.0+cu130 / transformers /
vLLM 0.20.1 win at import even though `surya-ocr` pulls its own torch into the venv (harmless, just a
wasted download). Confirmed via a probe job: vLLM at `…/site-packages/vllm`, python 3.12.13.

**Naming gotcha:** must be `surya-ocr.py`, never `surya.py` (would shadow the `surya` package on
import). Checked: no other `surya*` file in the repo.

**Smoke-test results (2026-06-22, `davanstrien/ufo-ColPali`, l4x1, `vllm/vllm-openai:v0.20.1`):**
- **ocr** (5 samples): 5/5 OK, 3.7 min (vLLM engine init ~113s incl. 34s compile + CUDA-graph capture,
  then inference). `markdown` clean reading-order text; `surya_blocks` valid JSON with **pixel-space**
  bboxes (e.g. `[21.6,65.5,30.9,343.4]` within `image_bbox=[0,0,618,1007]`), sequential `reading_order`,
  canonical labels (PageHeader/SectionHeader/Text/…), `confidence` ~0.94 (logprobs path works), per-block
  HTML (`<h1>`, `<sup>`, `<br/>`). Output `davanstrien/surya-smoke-ocr`. Resolved `vllm==0.20.1,
  torch==2.11.0+cu130, transformers==5.7.0, surya-ocr==0.20.0`.
- **layout** (3 samples): 3/3 OK; `surya_blocks` = `LayoutResult` per page (bboxes with `label`/
  `position`/`count`/`confidence`, guided-JSON enforced). Output `davanstrien/surya-smoke-layout`.
- **table** `--table-mode full` (3 samples): 3/3 OK; `TableResult` with `html` populated (rows/cols/cells
  empty in full mode, by design). ufo-ColPali has no real tables, so use a table dataset for meaningful
  output — the code path is what's validated. Output `davanstrien/surya-smoke-table`.

- **pdf** (`--pdf-column`/`--page-range`, real 14.8MB arXiv PDF, pages 0–2): 1/1 OK. Text
  concatenates the 3 pages (title/authors/abstract of arXiv:2606.17162 extracted in reading order);
  `surya_blocks` has **3 page entries** (`image_bbox=[0,0,1632,2112]` at 192 DPI) with sensible labels
  (PageHeader/SectionHeader/Text/Picture/Diagram/Caption/ListGroup/…). Source built by wrapping the PDF
  bytes into a `Value("binary")` column. Output `davanstrien/surya-smoke-pdf`.

**Still untested (low risk):** `--table-mode simple` (rows/cols/cells). Larger GPUs (l4x1 confirmed
comfortable for 650M).

### Bucket variant (`surya-ocr-bucket.py`) — issue #55 ✅
✅ **OCR a bucket of files directly, no dataset round-trip** (added 2026-06-22). Reuses the parent's
`OfflineVLLMBackend` / predictor dispatch / `serialize_pages` **verbatim**; grafts on the bucket I/O
from `pp-doclayout.py`. Two input strategies via `--io-mode {auto,mount,copy}`: **mount** reads off a
FUSE-mounted `/in` (`-v hf://buckets/<id>:/in:ro`); **copy** uses `huggingface_hub`
`list_bucket_tree` + `download_bucket_files` to batch-fetch each `--batch-size` chunk to temp, OCR, then
`shutil.rmtree` (peak disk = one batch — sidesteps the FUSE bulk-read stall). Two sinks (≥1, both
allowed): `--output-bucket` writes per-page `<rel>.md` + `<rel>.json` (`surya_blocks`) to a mounted dir
or `hf://buckets/...` URL (`batch_bucket_files`), **resume-by-skip keyed on the `.json`** (the parent
bucket recipes have no resume); `--output-dataset` buffers one row per file and `push_to_hub`. `.jp2` is
first-class (LoC/Chronicling America) with an `imagecodecs` fallback when the image's Pillow lacks
OpenJPEG.

**⚠️ Dependency gotcha (cost one job):** must pin **`surya-ocr==0.20.0`** in the PEP 723 header. Adding
`huggingface-hub>=1.6.0` (for the buckets API) loosened the resolve and uv backtracked to an ancient
surya without the `surya.inference` engine layout → `ModuleNotFoundError: No module named 'surya.inference'`.
Fix: pin surya, leave `huggingface-hub` unpinned — at runtime `PYTHONPATH` puts the pinned image's hub
(buckets API present) ahead of the venv, so there's no version tension.

**Smoke-tested on Jobs (2026-06-22, `davanstrien/chronicling-america-mirror-demo`, 1901 *The Commoner*
`.jp2`, l4x1):** copy→dataset, mount→mounted-bucket-files, copy→API-bucket-files, and resume re-run
(skip-all, no model load) all 8/8 OK with clean masthead/body OCR + valid pixel-space `surya_blocks`.
Mount-vs-copy benchmark (32-page seed-42 slice, l4x1, inference identical ~745s — confirms the I/O
split): **copy wins decisively** — listing **5.1s vs mount 134.2s** (FUSE `rglob` stats all 38k bucket
files; ~26×), batch-download I/O **57.6s vs FUSE-read 74.6s**. Mount *also* hit a transient
`Volume mount failed: init container exhausted retries` on the first attempt (needed a cold retry;
documented fresh-node CSI flake) — copy never mounts. → `auto` defaulting `hf://buckets/...` inputs to
**copy** is the right call (already the implemented default); mount stays for when the bucket is already
mounted or zero ephemeral disk is wanted.

**TODO(alto):** ALTO XML export from `surya_blocks` is its own follow-up issue (block-level
bbox→`HPOS/VPOS/WIDTH/HEIGHT`, label→`TextBlock`/`Illustration`, reading_order→order; line-level needs
Surya's `DetectionPredictor`; word-level out of scope). The test bucket ships CA's own ALTO `.xml` next
to each `.jp2` as a ready-made diff target.

**License:** code Apache-2.0, **weights modified OpenRAIL-M** (research/personal/<$5M, no competitive use
vs Datalab's API). Surfaced in the docstring, README entry, and output dataset card.

**Benchmark/compare:** `--config`/`--create-pr` push the same multi-config pattern as the other scripts.

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
