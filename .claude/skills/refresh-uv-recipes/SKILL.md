---
name: refresh-uv-recipes
description: "Audit the shippable `uv-recipes` skill against the live Hugging Face Hub and `hf` CLI to catch drift. Use periodically — monthly, after an `hf`/`huggingface_hub`/vLLM release, or when adding a recipe family — to confirm the run pattern (INPUT_DATASET OUTPUT_DATASET, Hub-in/Hub-out), the `hf jobs uv run` flags, the hardware flavors, the default image, the discovery commands, and the task-family list in `skills/uv-recipes/SKILL.md` still match reality. Reports what is stale and proposes minimal edits; does not change the shippable skill without review."
---

# refresh-uv-recipes

Verify that `skills/uv-recipes/SKILL.md` still reflects reality. The recipe *list* is
self-refreshing (discovered live), so this audits only the baked-in **assumptions**.

Copy this checklist and work through it; report results, propose edits, apply only after review.

```
- [ ] 1. Pattern still holds: recipes take INPUT_DATASET OUTPUT_DATASET and write to the Hub
- [ ] 2. Invocation flags unchanged (--flavor/--secrets/--image/--timeout) AND setup still valid
- [ ] 3. Hardware flavors + default image still as documented
- [ ] 4. Discovery endpoints still return the expected shape
- [ ] 5. Task families: any new org repo the description doesn't mention
- [ ] 6. Gotchas still accurate; any new common failure to surface
- [ ] 7. Best-practices conformance (description <1024 chars, 3rd person, refs 1 level deep)
```

**1. Core pattern.** Pick a few real recipes (via step 4) across families and read their headers,
e.g. `uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py --help`. Confirm
`INPUT OUTPUT` arg order and Hub output (`push_to_hub`). If a new dominant pattern appeared
(bucket-first I/O, multi-output, a non-dataset input), flag it — that changes the skill's premise.

**2. Invocation & setup.**
```bash
hf jobs uv run --help     # do --flavor/--secrets/--image/--timeout still exist, same names?
```
Also confirm the setup still works: the `hf` install URL (`https://hf.co/cli/install.sh`) resolves,
and `hf skills add` still installs the `hf-cli` skill (`hf skills list` to check the marketplace).

**3. Hardware & image.**
```bash
hf jobs hardware          # do the flavor names in SKILL.md still exist?
```
Confirm the default image is still `astral-sh/uv:python3.12-bookworm` and the 30-min default timeout.

**4. Discovery.**
```bash
curl -s "https://huggingface.co/api/datasets?author=uv-scripts" | jq -r '.[].id'
curl -s "https://huggingface.co/api/datasets/uv-scripts/ocr/tree/main" | jq -r '.[].path | select(endswith(".py"))'
```
Both must still return repos / `.py` files. If the API shape changed, update the commands.

**5. Coverage.** Diff the org repo list (step 4) against the task families named in the
`uv-recipes` description. A new family (e.g. an audio-generation repo) → propose adding it to the
description's trigger list so the skill activates for it.

**6. Gotchas.** Re-check documented gotchas (the `vllm/vllm-openai` image note) and skim recent
issues on the GitHub repo and the `uv-scripts/ocr` Hub repo for any new fleet-wide failure worth
surfacing (this is how the FlashInfer issue would have been caught).

**7. Conformance.** Description ≤1024 chars and third-person; SKILL.md body reasonable; any
reference files one level deep.

**Output.** A short verdict — *still sensible* / *needs update* — with a concrete diff per stale
item. Apply edits to `skills/uv-recipes/SKILL.md` only after the maintainer confirms.
