---
name: bump-vllm-pins
description: "Internal, dev-only: smoke-test and bump the pinned vLLM recipes in this repo. Use periodically (~quarterly) or after a vLLM release to confirm each pinned vLLM script still runs on Jobs and to propose pin / image-tag bumps to a newer known-good. Tests on l4x1 via `hf jobs uv run`; reports a diff of proposed bumps; never auto-merges."
---

# bump-vllm-pins

Pins give stability; this prevents staleness. Internal/dev-only, not shipped.

## vLLM facts to apply (verified 2026-06)
- **Structured outputs:** use `StructuredOutputsParams` (`structured_outputs=`), floor `vllm>=0.11.0`. `GuidedDecodingParams` / `guided_decoding=` was **removed in 0.12.0** → ImportError on ≥0.12.0 (incl. 0.22.x). Param kinds carry over: `choice=`, `json=`, `regex=`, `grammar=`.
- **FlashInfer / nvcc at init:** recent FlashInfer JIT-compiles kernels needing the CUDA toolkit. Prefer running on the **`vllm/vllm-openai` image** (ships the toolkit) over the `VLLM_USE_FLASHINFER_SAMPLER=0` env opt-out. (The official [`vllm-project/vllm-skills`](https://github.com/vllm-project/vllm-skills) docker skill likewise recommends this image.)
- **Driver / flavor:** CUDA-13 builds need a newer driver. `a100-large`'s driver caps at CUDA 12.9 (`found version 12090`) → CUDA-13 wheels/images fail there; `l4x1` works. Prefer a CUDA-12.x image for broad flavor compat, or test on `l4x1`.
- **Pinning:** pin the **image tag** (`vllm/vllm-openai:<ver>`, not `:latest`) for the strongest reproducibility (freezes vLLM + toolkit together); also pin `vllm==` in the PEP 723 header for the local path. vLLM releases ~monthly.

## Procedure (per pinned script)
```
- [ ] 1. Smoke-test the current pin on l4x1 (image, small --max-samples, public dataset)
- [ ] 2. If green, try the latest vLLM image tag / pin; re-test
- [ ] 3. Record pass/fail (+ toks/s); propose the bump or hold
- [ ] 4. Report a diff of pin / image-tag changes for review — do NOT auto-merge
```
1. **Discover** pinned scripts: `grep -rnE "vllm" */*.py` and the documented `--image vllm/vllm-openai:<tag>` in docstrings.
2. **Smoke-test** (template):
   ```bash
   hf jobs uv run -d --flavor l4x1 --secrets HF_TOKEN \
     --image vllm/vllm-openai:<tag> --python /usr/bin/python3 \
     -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
     <script.py> <public-input-dataset> <out> --max-samples 5
   ```
   Poll `hf jobs inspect <id> --format json`. (`--python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages` make uv use the image's preinstalled Python 3.12 + vLLM instead of fetching its own — **required** for the `vllm/vllm-openai` image; verified to run clean on `l4x1`.)
3. **Green** = runs to completion + writes output. Flag failures by class: `ImportError` (API drift → re-harden to the new API), `nvcc`/FlashInfer (toolkit/image), or "driver too old" (CUDA-13-vs-flavor).
4. **Propose** bumps (image tag + `vllm==`) as a reviewable diff; maintainer merges.

## Capture-then-pin
Pin to a *tested* version, never a guessed one. Workflow: run loose (or on `:latest`) → **read the working versions from the green run** (vLLM logs its version at engine init; or `uv pip freeze` / `uv lock --script <file>` inside the job) → pin to those. Lead with the **image tag** (`vllm/vllm-openai:<that-version>`) — it freezes vLLM + torch + CUDA toolkit + FlashInfer together — and keep a `vllm==`/`>=` floor in the PEP 723 header for the local path.

## Output
A short report: per script — current pin, tested tag, pass/fail, proposed bump. No file changes without review.
