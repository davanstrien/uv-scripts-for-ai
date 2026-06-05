---
title: README
emoji: 📚
colorFrom: red
colorTo: indigo
sdk: static
pinned: false
---

# UV Scripts

**Run a data or ML task over a Hugging Face dataset in one command — for humans and agents.**

Each recipe is a single self-contained [UV script](https://docs.astral.sh/uv/guides/scripts/): dependencies are declared inline, so you run it straight from a URL with no clone, no virtualenv, no `pip install`. Run it locally with `uv run` where you have the hardware, or hand it to [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs) for a managed GPU. Most recipes read a Hub dataset and write a new one, so they chain into pipelines.

## Quickstart

```bash
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \
  davanstrien/ufo-ColPali your-username/ufo-ocr --max-samples 10
```

One command → a new dataset with a `markdown` column. No GPU of your own needed (Jobs is pay-per-second). Works locally too with just `uv run`.

## Browse recipes

- **[ocr](https://huggingface.co/datasets/uv-scripts/ocr)** — document → text & structured data (30+ models)
- **[sam3](https://huggingface.co/datasets/uv-scripts/sam3)** · **[object-detection](https://huggingface.co/datasets/uv-scripts/object-detection)** · **[vlm-object-detection](https://huggingface.co/datasets/uv-scripts/vlm-object-detection)** — detection & segmentation
- **[transcription](https://huggingface.co/datasets/uv-scripts/transcription)** — audio → text
- **[gliner](https://huggingface.co/datasets/uv-scripts/gliner)** — NER over text · **[classification](https://huggingface.co/datasets/uv-scripts/classification)** — text/image classification
- **[build-atlas](https://huggingface.co/datasets/uv-scripts/build-atlas)** — embed a dataset & build an interactive map
- **[vllm](https://huggingface.co/datasets/uv-scripts/vllm)** · **[openai-oss](https://huggingface.co/datasets/uv-scripts/openai-oss)** — batch LLM/VLM inference
- **[synthetic-data](https://huggingface.co/datasets/uv-scripts/synthetic-data)** · **[dataset-creation](https://huggingface.co/datasets/uv-scripts/dataset-creation)** · **[dataset-stats](https://huggingface.co/datasets/uv-scripts/dataset-stats)**

## Built for agents

Every recipe takes its arguments in the same `input output` order and runs from a URL, so an agent can pick one from its header and run it with no setup. The cookbook ships a ready-to-use **agent skill** for discovering, running, and adapting recipes — see the GitHub repo.

## Learn more

- **[GitHub: davanstrien/uv-scripts-for-ai](https://github.com/davanstrien/uv-scripts-for-ai)** — the full cookbook and how to contribute
- [Hugging Face Jobs docs](https://huggingface.co/docs/hub/jobs) — run `hf jobs hardware` for GPUs & pricing
- [UV scripts guide](https://docs.astral.sh/uv/guides/scripts/)
