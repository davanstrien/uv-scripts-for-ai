---
name: review-recipe
description: "Internal, dev-only: review a new or changed OCR recipe against the conventions in ocr/CLAUDE.md before merging. Runs the static contract checker, then walks the judgment-only invariants the checker can't verify (model-card fidelity, context length, pin rationale, license, image bounding) and the new-recipe checklist. Use when adding or editing an ocr/*.py recipe."
disable-model-invocation: true
---

# review-recipe

Enforces the [Conventions & invariants](../../../ocr/CLAUDE.md) in `ocr/CLAUDE.md`. This
skill does **not** restate them — it points at them and drives the checks. The checker
covers the mechanical rules; you cover the judgment ones. Internal/dev-only, not shipped.

## 1. Run the static checker

```bash
uv run tools/check-contract.py ocr/<recipe>.py     # or no arg for all recipes
```

- **Fix every `error`.** They map to real bugs (missing `--create-pr`, `inference_info`
  key drift, no collision guard, no CUDA guard, env-guard ordering). Each finding prints
  the `ocr/CLAUDE.md` section to consult.
- **Justify or fix every `warning`** (bare push, non-`[OCR FAILED]` sentinel). A warning you
  keep needs a one-line reason; if the reason is "this file legitimately diverges", add it to
  `EXEMPT` in `tools/check-contract.py` with a why-comment — never to silence a real gap.

## 2. Judgment checks the checker can't make

Walk each against `ocr/CLAUDE.md` "Conventions & invariants". Completion criterion in **bold**.

- [ ] **Model-card fidelity** — every claim in the recipe docstring + generated dataset card
  (param count, benchmark score, prompt format, sampling defaults, languages) matches the
  model's *actual* HF card. **You opened the card and confirmed each number.**
- [ ] **Context-length invariant** — `--max-tokens` ≤ `--max-model-len` ≤ the model's real max
  context. **You ran the `config.json` check from the invariant and the defaults fit** (mind
  `text_config` / `rope_scaling`).
- [ ] **Pins carry rationale** — every image tag / `==` pin / nightly index has a **why it
  exists and what would loosen it** comment (per the "Pins are temporary" invariant).
- [ ] **License surfaced** — if weights are under a non-standard / restricted license, it is
  **stated in the docstring and the dataset card** (see the Surya / lift / Hunyuan examples).
- [ ] **Image bounding** — a full-page recipe **caps input pixels / resizes, or sizes
  `max_model_len` to fit**; it does not auto-size `max_model_len` from image dimensions.

## 3. New-recipe checklist

Only for a newly added recipe (per the "New-recipe checklist" invariant):

- [ ] Row in **both** README tables (models-at-a-glance + modes-and-flags).
- [ ] Script-status row in `ocr/CLAUDE.md` (⏳ until Jobs-smoke-tested; a gotcha section only
  if it has load-bearing quirks).
- [ ] Change-log line in `ocr/CLAUDE.md`.
- [ ] Smoke-tested on Jobs (`--max-samples 5`, l4x1) before flipping ⏳ → ✅.

## Output

A short verdict: checker errors fixed, each warning fixed-or-justified, the five judgment
boxes checked (with the numbers you verified), and the new-recipe rows added. No merge until
errors are zero.
