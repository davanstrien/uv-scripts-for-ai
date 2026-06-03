# Contributing

Every recipe here is a single, self-contained **[UV script](https://docs.astral.sh/uv/guides/scripts/)**
([PEP 723](https://peps.python.org/pep-0723/) inline metadata) that runs on Hugging Face Jobs in one command. No shared library, no packaging — one file
you can `hf jobs uv run`.

## Add or change a recipe

1. Pick the folder that maps to the Hugging Face dataset repo it belongs to (folder name == repo name, e.g.
   `ocr/` → [`uv-scripts/ocr`](https://huggingface.co/datasets/uv-scripts/ocr)). To start a brand-new domain,
   create a new folder.
2. Write the script with an inline PEP 723 dependency block and a docstring that shows a runnable
   `hf jobs uv run` example.
3. Open a PR. On merge to `main`, the folder mirrors to its Hub repo automatically.

> ⛔ **Never delete a file that exists on the Hub.** The mirror removes Hub files that disappear from a folder.
> Maintainers: see [AGENTS.md](AGENTS.md) for the seed-then-flip rules and the superset gate.

## Script template (PEP 723)

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "huggingface-hub[hf_transfer]",
# ]
# ///
"""One-line description of what this recipe does.

Example (Hugging Face Jobs):
    hf jobs uv run --flavor l4x1 \
      https://huggingface.co/datasets/uv-scripts/<repo>/raw/main/<script>.py \
      input-dataset output-dataset
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # ...
    args = parser.parse_args()
    # ... core logic


if __name__ == "__main__":
    main()
```

## Conventions

- **Self-contained:** declare all deps inline; no `requirements.txt`, no shared importable modules.
- **Hub-native I/O:** read inputs from and write outputs to the Hub (`push_to_hub`) or a bucket (`-v hf://…`).
  Job disk is ephemeral — never rely on local output paths.
- **GPU scripts:** check `torch.cuda.is_available()` and exit with a clear message if a GPU is required.
- **Fast downloads:** prefer `huggingface-hub[hf_transfer]`.
- **Naming:** name by task, not tool (`classify-dataset.py`, not `bert-classify.py`).
- **Per-folder README:** each folder has a `README.md` with frontmatter `viewer: false` and `tags: [uv-script, …]`
  (these are dataset repos hosting code, so the dataset viewer is disabled).
- **Examples use the HF raw URL** `https://huggingface.co/datasets/uv-scripts/<repo>/raw/main/<script>.py` —
  that's the invocation the Hub can attribute.
