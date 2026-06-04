# LAUNCH.md — going public

Readiness checklist and launch plan for making `uv-scripts-for-ai` public. The repo is **private during the
build phase**; flip to public only when the gate below is green. See [AGENTS.md](AGENTS.md) for the
mirroring rules (and the one rule: never delete a file from the Hub).

## Strategy in one line

A GitHub-authoritative cookbook of self-contained UV scripts — for **humans and agents**, run **anywhere**
(`uv run` locally or `hf jobs uv run` on a GPU). GitHub earns the stars; the Hub earns the execution +
trackable usage. **OCR is the flagship.**

## Readiness gate (before flipping public)

**Blockers — must be done:**
- [ ] **`LICENSE`** at the repo root (recommend **Apache-2.0** — patent grant, HF-standard). Note in the
      README that *model* licenses vary per recipe.
- [ ] **Revoke the 2 HF tokens** pasted in chat during the build; confirm the `HF_TOKEN` GitHub secret is a
      fine-grained token scoped to **write on the `uv-scripts` org** (covers every mirrored repo, not just `ocr`).
- [ ] **GitHub repo metadata**: rewrite the description to match the reframe, add topics
      (`ocr`, `huggingface`, `uv`, `datasets`, `vlm`, `hf-jobs`, `agents`), set homepage → the `uv-scripts` org.
      *(Current description is the old "…run any in one command as a Job" framing.)*
- [ ] **Decide surface area** — see "Surface area" below.
- [ ] **Add GitHub backlinks** to each Hub README (held back while private; add at flip for the stars funnel).
- [ ] **Cold-test the quickstart** exactly as written: install uv → `uv run …--help` → `hf jobs uv run …`.
- [ ] **Final repo name** — keep `uv-scripts-for-ai` or rename before stars accrue (GitHub redirects on rename).

**Should-do — first-impression polish:**
- [ ] **Demo GIF / asciinema** at the top of the root README (assets exist: `ocr/demo.cast`, `ocr/*.gif`).
      "One command → a dataset" lands far better shown than told.
- [ ] Repo **social-preview image**; pin to profile; enable Issues + Discussions.
- [ ] **2–3 good-first-issues** so contributors have a door.
- [ ] Confirm caches (`.ruff_cache`, etc.) are gitignored and not shipping.
- [ ] **One real pipeline** end-to-end (OCR → `build-atlas`) so "composable" is demonstrable, not hypothetical.
- [ ] (Optional, strong) **`recipes.json` catalog** — a machine-readable index so the "built for agents"
      claim is a feature you can point at, not just words.

## Surface area (the pivotal call)

A one-folder repo undersells the "cookbook." **Decision: path B — flagship + a few more.**

- [x] `ocr/` — flagship, live + tidied
- [ ] `sam3/`, `transcription/`, `build-atlas/`, `vlm-object-detection/` — wave 2 (PR #5), opens
      vision + audio + the embeddings/atlas pipeline
- [ ] Later, traction-ordered: `openai-oss`, `vllm`, `gliner` (88 dl), `dataset-creation`, `iiif-tiles`, …
- [ ] **Out:** `embeddings` (private — never add to the tree); `training` / `transformers-training`
      (de-emphasized; the README keeps the door open via the "…and more" row + "What fits here")

## Levers for success

- **Lead with the form + a concrete win**, not "I made a monorepo": *"OCR a whole image dataset to markdown
  in one command — pick from 20+ models."* Agents angle = strong secondary hook (timely, differentiated).
- **Keep "runs locally" prominent** — Jobs is Pro/Team/Enterprise-gated; don't make the value contingent on it.
- **Pre-wire the discovery loop**: Hub ↔ GitHub backlinks, the [`jobs-examples`](https://huggingface.co/docs/hub/jobs-examples)
  docs page (already lists the org — enhance with a concrete OCR example), and [`ocr-bench`](https://github.com/davanstrien/ocr-bench).

## When

Sequence, not a date: **license + metadata + surface area (wave 2) + demo GIF + cold-tested quickstart → flip
public → announce.** Don't announce into a bare repo. Natural narrative anchor: the **OCR-affordances blog**
(launch alongside or just after). Dev-audience timing: Tue–Thu, US morning.

## How to share

- **Anchor**: a short post (or the OCR blog) telling the "one command, your data, the Hub" story; repo link inside.
- **Waves**: soft-share with a few HF colleagues / friendly users for feedback → X/Bluesky thread led by the
  **demo GIF** → LinkedIn → r/LocalLLaMA + r/MachineLearning → optionally *Show HN* once solid.
- **HF internal**: Slack amplification, HF Posts, docs/newsletter.
- **GitHub hygiene as marketing**: topics, social image, pinned repo, CONTRIBUTING + good-first-issues.

## Open decisions (owner: Daniel)

1. Surface area beyond wave 2 before launch? *(default: wave 2 is enough)*
2. License: **Apache-2.0** (recommended) or MIT?
3. Launch anchor: tie to the OCR-affordances blog, or standalone?
4. Final repo name?
