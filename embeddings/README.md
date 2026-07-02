---
viewer: false
tags:
- uv-script
- embeddings
- sentence-transformers
- vector-search
---

# Embeddings

Generate embeddings for a Hugging Face dataset — text or images — with one command, on a
cloud GPU, no infra. The output lands back on the Hub as a new dataset (or, with the Lance
variant, as a **searchable vector index you can query over `hf://` without downloading**).

There is one simple default and two variants; they are separate single-file scripts because
their dependencies (sentence-transformers vs vLLM vs Lance) are too different to share one env.

| Script | Use it for | Engine |
|---|---|---|
| `generate-embeddings.py` | The default. Text or images. Simple, fast, runs anywhere. | sentence-transformers |
| `generate-embeddings-vllm.py` | Max throughput on large *decoder* embedding models (Qwen3-Embedding). | vLLM pooling |
| `embed-to-lance.py` | Get a **searchable vector index as a Hub dataset** (the "vector DB" path). | sentence-transformers + Lance |

## Quick start

```bash
# Text — pick a model from the MTEB leaderboard
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/generate-embeddings.py \
  stanfordnlp/imdb  your-name/imdb-embeddings --column text

# Images (CLIP)
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/embeddings/raw/main/generate-embeddings.py \
  your-name/photos your-name/photos-embeddings --modality image --column image --model clip-ViT-B-32
```

Always try `--max-samples 100 --private` first.

## Which model?

Any [sentence-transformers](https://huggingface.co/models?library=sentence-transformers) model; check the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

| Model | Params | Dim | Note |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | 384 | Fastest; great default |
| `BAAI/bge-base-en-v1.5` | 109M | 768 | Strong English quality/speed balance |
| `BAAI/bge-m3` | 568M | 1024 | Multilingual + long context (slower) |
| `Qwen/Qwen3-Embedding-0.6B` | 596M | 1024 | Top open MTEB; decoder → use the vLLM variant / A100 |

Images: `clip-ViT-B-32` (fast) or `clip-ViT-L-14` (higher quality).

## Which GPU? (measured, 20k rows, seq-cap 512)

Throughput (rows/s) and cost per 1M rows:

| Model | L4 ($0.80/hr) | A10G ($1.50/hr) | A100 ($2.50/hr) |
|---|---|---|---|
| all-MiniLM-L6-v2 | 912 · **$0.24/1M** | 1099 · $0.38/1M | 1372 · $0.51/1M |
| bge-base-en-v1.5 | 119 · **$1.87/1M** | 206 · $2.02/1M | 261 · $2.66/1M |
| Qwen3-Embedding-0.6B | 59 · $3.77/1M | 93 · $4.48/1M | 250 · **$2.78/1M** |

**Default to `l4x1`** — cheapest per 1M rows for encoder models. For **decoder** embedders
(Qwen3-Embedding) the A100 is both faster *and* cheaper per 1M (they use the extra compute),
and the vLLM variant roughly doubles throughput again.

## The vector-DB path (`embed-to-lance.py`)

Writes a [Lance](https://huggingface.co/docs/hub/datasets-lance) table with a vector index and
pushes it as a Hub dataset. You (or anyone you share it with) can then search it directly over
`hf://` **without downloading it**:

```python
import lance
ds = lance.dataset("hf://datasets/your-name/my-vecdb/vecdb.lance")   # opens in ~1s, no download
hits = ds.to_table(nearest={"column": "vector", "q": query_vec, "k": 5})
```

Best for share-and-search over a corpus; for high-QPS serving, pull the dataset local first.
