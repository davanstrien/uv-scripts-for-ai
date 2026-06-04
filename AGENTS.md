# AGENTS.md — operating this repo

This is a GitHub-authoritative cookbook of **UV scripts you run on Hugging Face Jobs in one command**
(`hf jobs uv run <url>`). Each top-level folder **mirrors to a Hugging Face dataset repo** under
[`uv-scripts`](https://huggingface.co/uv-scripts) via the [`huggingface/hub-sync`](https://github.com/huggingface/hub-sync)
Action on push to `main`. GitHub is the source of truth; the Hub repos are mirrors. (Claude Code: this
file is the runbook — `CLAUDE.md` only points here.)

## ⛔ The one rule: never delete a file from the Hub

The sync runs `hf upload <repo> <folder> --delete="*"` on every push. **Any file on the Hub repo that is
absent from its GitHub folder is permanently deleted.** Therefore:

- A folder gets a sync workflow **only after** it is a *superset* of that repo's current Hub contents.
- **Seed from the Hub, never from old local clones:**
  `hf download uv-scripts/<repo> --repo-type dataset --local-dir <repo>/ && rm -rf <repo>/.cache <repo>/.gitattributes`.
  Dropping `.gitattributes` is mandatory: it carries `*.gif`/`*.png`/`*.pdf filter=lfs` rules, and with
  git-lfs installed `git add` would turn binaries into pointers that the workflow (no `lfs: true` checkout)
  uploads as text — **corrupting the Hub asset**. The action excludes `.gitattributes` anyway, so removing
  it is lossless.
- **Never `hf download` (or otherwise add) the private `embeddings` repo into the tree** — it must never
  appear in the public monorepo.
- Edits to an existing repo's folder are **additive** — add files freely; do not remove anything that's on
  the Hub. (Retiring a Hub file is a separate, deliberate, confirmed action — never a seeding side effect.)
- The superset gate runs **both locally** (`tools/verify-superset.sh <folder> uv-scripts/<repo>` — respect a
  BLOCK) **and inside every sync workflow** as a hard pre-step, so it guards re-triggers too, not just first
  flips.

## Layout — one folder per Hub repo, named identically

`ocr/ → uv-scripts/ocr`, `sam3/ → uv-scripts/sam3`, … `subdirectory == folder == repo name`, so the mapping
can't be mis-typed. Domain grouping for readers lives in the root `README.md`, not in folder nesting. Don't
mirror the org's Spaces; don't add a folder for the private `embeddings` repo.

## Script conventions

- Self-contained **[PEP 723](https://peps.python.org/pep-0723/) / [UV script](https://docs.astral.sh/uv/guides/scripts/)** — deps inline, no `requirements.txt`, no shared importable lib:
  ```python
  # /// script
  # requires-python = ">=3.10"
  # dependencies = ["datasets", "transformers", "torch", "huggingface-hub[hf_transfer]"]
  # ///
  ```
- Each repo folder has a `README.md` with frontmatter `viewer: false` + `tags: [uv-script, …]` (these are
  dataset repos hosting code).
- Every script's docstring shows a runnable **`hf jobs uv run`** example. README examples use the **HF raw
  URL** `https://huggingface.co/datasets/uv-scripts/<repo>/raw/main/<script>.py` (the trackable invocation).
- GPU scripts check `torch.cuda.is_available()` and exit with a clear message. Write outputs to the Hub
  (`push_to_hub`) or a bucket (`-v hf://…`), never local paths (Jobs disk is ephemeral). Name by task, not tool.

## The mirror — how it works

- `.github/workflows/sync-<repo>.yml`, one **thin caller** per folder (triggered on `push` to `main` filtered
  to `['<repo>/**', '.github/workflows/sync-<repo>.yml']`), which calls the shared reusable workflow
  `.github/workflows/_sync-folder.yml` — the single place the gate + LFS guard + `hub-sync` logic lives, so an
  action-version bump or gate change is a one-file edit.
- It uploads file **bytes** over HTTP, creates the repo if missing (`--exist-ok`), makes **one squashed
  commit**, and **excludes `.git*` and `.github*`** (so `.gitattributes` is not synced). No git history carried.
- **`repo_type: dataset` is required** (the action defaults to `space`). Pin `huggingface/hub-sync@v0.1.0`.
- **Commit binaries as plain blobs — do NOT use Git LFS here.** Files are well under GitHub's 100 MB limit; HF
  stores large files as LFS/xet automatically on upload. Because every binary repo's `.gitattributes` has
  `*.gif`/`*.png`/`*.pdf filter=lfs`, **delete `.gitattributes` on seed** (above) so `git add` can't make a
  pointer; the workflow's LFS-pointer guard is the backstop. (If Git LFS is ever deliberately added, the
  checkout MUST use `lfs: true`, or the action uploads pointer text and corrupts the Hub file.)

## Add a repo to the mirror (checklist)

1. `uv-scripts/<repo>` exists? **Yes** → `hf download` it into `<repo>/` (+ `rm -rf <repo>/.cache
   <repo>/.gitattributes`), then `git add <repo>/` and run `tools/verify-superset.sh <repo> uv-scripts/<repo>`
   (must pass). **No** → create `<repo>/` with scripts + a `viewer:false` README; the action births it.
2. Add `.github/workflows/sync-<repo>.yml` — a 13-line caller from the template below (just set the workflow
   `name`, the `paths` filter, and `folder: <repo>`). All sync logic is inherited from `_sync-folder.yml`.
3. Make sure `HF_TOKEN` (repo secret, fine-grained Write on the `uv-scripts` org) covers `<repo>`.
4. Commit + push to `main`. Only this repo syncs. Then confirm on the Hub: file count ≥ before, named assets present.
5. Add the recipe to the root `README.md` under its domain group.

## Token

GitHub repo secret **`HF_TOKEN`** = a fine-grained HF token with **Write** on the `uv-scripts` org (Write covers
repo creation; it does NOT grant whole-repo deletion). Re-create/rotate rather than widening scope.

## Workflow template

Each folder gets a thin **caller**; the gate + LFS guard + `hub-sync` logic lives once in the reusable
`.github/workflows/_sync-folder.yml` (so an action-version bump or a gate change is a single edit, not one per
repo). To add a repo, copy this caller and change the two `<repo>` occurrences + `folder:`:

```yaml
# .github/workflows/sync-<repo>.yml
name: Sync <repo> → HF
on:
  push:
    branches: [main]
    paths: ['<repo>/**', '.github/workflows/sync-<repo>.yml']
jobs:
  sync:
    uses: ./.github/workflows/_sync-folder.yml
    with:
      folder: <repo>
    secrets:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
```
