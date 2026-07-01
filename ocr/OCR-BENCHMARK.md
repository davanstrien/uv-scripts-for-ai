# OCR Benchmark — results & history

Result tables and validation history for the OCR benchmark tooling. **How to *run* the tools**
(`ocr-bench-run.py`, `ocr-vllm-judge.py`, `ocr-human-eval.py`) lives in `CLAUDE.md` → "Internal
tooling"; this file is the accumulated *evidence*.

## Model registry (as benchmarked)

| Slug | Model | Size | GPU | Notes |
|------|-------|------|-----|-------|
| `glm-ocr` | `zai-org/GLM-OCR` | 0.9B | l4x1 | |
| `deepseek-ocr` | `deepseek-ai/DeepSeek-OCR` | 4B | l4x1 | auto `--prompt-mode free` (no grounding tags) |
| `lighton-ocr-2` | `lightonai/LightOnOCR-2-1B` | 1B | a100-large | |
| `dots-ocr` | `rednote-hilab/dots.ocr` | 1.7B | l4x1 | stable vLLM (>=0.9.1) |

## Run 1 — NLS Medical History (2026-02-14, pilot)

`NationalLibraryOfScotland/medical-history-of-british-india`, 10 samples, seed 42. Judge:
`Qwen2.5-VL-72B` via Inference Providers. Historical English, degraded scans.

- **ELO (pairwise, 5 samples):** DoTS 1540 (67%) · DeepSeek 1539 (57%) · LightOnOCR-2 1486 (50%) · GLM 1436 (29%)
- **Pointwise (5):** DeepSeek 5.0 · GLM 4.6 · LightOnOCR-2 4.4 · DoTS 4.2
- **Key finding:** DeepSeek's `--prompt-mode document` emits grounding tags (`<|ref|>`/`<|det|>`) the
  judge penalises heavily; switching to `--prompt-mode free` moved it last→top-2 (now the registry default).
- **Caveat:** 5 samples is far too few for stable rankings.

## Run 2 — Rubenstein Manuscript Catalog (2026-02-15, first full run)

`biglam/rubenstein-manuscript-catalog`, 50 samples, seed 42. Judge: jury of `Qwen2.5-VL-7B` +
`Qwen3-VL-8B` on A100 (`ocr-vllm-judge.py`). ~48K typewritten + handwritten cards (Duke, CC0).

**ELO (50 samples, 300 comparisons, 0 parse failures):**

| Rank | Model | ELO | W | L | T | Win% |
|------|-------|-----|---|---|---|------|
| 1 | LightOnOCR-2-1B | 1595 | 100 | 50 | 0 | 67% |
| 2 | DeepSeek-OCR | 1497 | 73 | 77 | 0 | 49% |
| 3 | GLM-OCR | 1471 | 57 | 93 | 0 | 38% |
| 4 | dots.ocr | 1437 | 70 | 80 | 0 | 47% |

Job times (50 samples): dots 5.3 min (L4) · deepseek 5.6 (L4) · glm 5.7 (L4) · lighton 6.4 (A100).

**Findings:** LightOnOCR-2 dominates on manuscript cards (very different from the NLS pilot) — rankings
are **dataset-dependent**; a jury of small models works well (0 parse failures via vLLM structured output);
50 samples gives meaningful separation.

## Run 3 — UFO-ColPali (2026-02-15, cross-dataset validation)

`davanstrien/ufo-ColPali`, 50 samples, seed 42. Judge: `Qwen3-VL-30B-A3B` on A100 (updated prompt).
Mixed modern documents.

**ELO (50 samples, 294 comparisons):**

| Rank | Model | ELO | W | L | T | Win% |
|------|-------|-----|---|---|---|------|
| 1 | DeepSeek-OCR | 1827 | 130 | 17 | 0 | 88% |
| 2 | dots.ocr | 1510 | 64 | 83 | 0 | 44% |
| 3 | LightOnOCR-2-1B | 1368 | 77 | 70 | 0 | 52% |
| 4 | GLM-OCR | 1294 | 23 | 124 | 0 | 16% |

**Human validation (30 comparisons):** DeepSeek #1 (matches judge), LightOnOCR-2 #3 (matches). Middle
pack (GLM, dots) shuffled between human and judge.

## Cross-dataset comparison (human-validated)

| Model | Rubenstein Human | Rubenstein Kimi | UFO Human | UFO 30B |
|-------|:---:|:---:|:---:|:---:|
| DeepSeek-OCR | **#1** | **#1** | **#1** | **#1** |
| GLM-OCR | #2 | #3 | #2 | #4 |
| LightOnOCR-2 | #4 | #2 | #3 | #3 |
| dots.ocr | #3 | #4 | #4 | #2 |

**Conclusion:** DeepSeek-OCR is consistently #1 across datasets and eval methods; middle-pack rankings
are dataset-dependent. (NLS pilot omitted — 5 samples / 72B API judge, not comparable with the newer
methodology.)

## Judge validation — `ocr-vllm-judge.py` (2026-02-15)

- **Test 1** (single judge, 1 sample, L4): `Qwen2.5-VL-7B`, 6/6 comparisons, 0 parse failures, ~3 min.
- **Test 2** (jury of 2, 3 samples, A100): `Qwen2.5-VL-7B` + `Qwen3-VL-8B`, 15/15, 0 failures; GPU cleanup
  between models OK; majority-vote aggregation working (`[2/2]` unanimous, `[1/2]` split).
- **Test 3** (full, 50 samples, A100, Rubenstein): 300/300 comparisons, 0 parse failures; clear ELO
  separation. First saved dataset: `davanstrien/ocr-bench-rubenstein-judge`.

Structured output via a compatibility shim: `StructuredOutputsParams` (vLLM ≥0.12) → `GuidedDecodingParams`
(older) → prompt-based fallback. Position bias mitigated by A/B randomisation. A100 recommended for jury mode.

## Human eval — `ocr-human-eval.py` first validation (Rubenstein, 30 annotations)

Tested 3 judge configs against 30 human annotations. **Kimi K2.5 (170B) via Novita + the updated prompt**
is the only judge to match the human's #1 (DeepSeek-OCR); it's now the default in `ocr-jury-bench.py`.
Small models (7B/8B/30B) overrate LightOnOCR-2 (bias toward its commentary style); the updated prompt
(faithfulness > completeness > accuracy) helps, but model size is the bigger factor.
