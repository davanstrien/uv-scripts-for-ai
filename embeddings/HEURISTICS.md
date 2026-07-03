# Embedding-on-Jobs heuristics — what we measured, and what to try

A field guide for the `embeddings/` recipes, for **humans and agents**. Everything here is
measured on HF Jobs (2026-07), not folklore. If you're an agent picking settings for a user's
data: read the TL;DR, match the data shape, use the defaults; the recipe's `--batch-size auto`
+ token sniffer handle the rest.

## First: pick a *current* model, not the frozen list below

Embedding quality moves fast. The specific models in this guide are what we **benchmarked in
2026-07** — a durable baseline, not a permanent answer. Before committing, find the current best
(an agent should do this automatically, not trust a hard-coded name):

- **MTEB leaderboard** — the canonical ranking: <https://huggingface.co/spaces/mteb/leaderboard>
- **The Hub, from the CLI** (scriptable, always current):
  ```bash
  hf models ls --filter sentence-transformers --sort trending_score --limit 20   # what's hot now
  hf models ls --filter sentence-transformers --sort downloads --limit 20        # proven workhorses
  hf models ls --search embedding --num-parameters min:0,max:1B --sort trending_score
  ```
  Sort by `trending_score` or `downloads`, **not `created_at`** — the newest list is mostly empty
  test repos. Then check the model card for its prompt convention (see Prompts) and license.

The **heuristics here are model-independent** — token-bound throughput, batch ~128, L4-for-encoders,
prompts-matter, seq-len cost hold whatever tops the leaderboard next week. Use them; swap in the
current model. (E.g. as of this writing the CLI surfaced newer options like jina-embeddings-v5,
voyage-4-nano, and LiquidAI's ColBERT that postdate parts of this guide.)

## TL;DR decision table
*(benchmarked examples, 2026-07 — run the queries above for the current best)*

| Your situation | Model | Flavor | Notes |
|---|---|---|---|
| Default / fast / cheap (English) | `all-MiniLM-L6-v2` | `l4x1` | ~900 rows/s, ~$0.24/1M rows, dim 384 |
| Better English quality | `BAAI/bge-base-en-v1.5` | `l4x1` | ~120 rows/s, ~$1.87/1M, dim 768 |
| Multilingual / long-context | `BAAI/bge-m3` | `l4x1`/`a10g-large` | slower (dim 1024); use only if you need it |
| Top open quality (decoder) | `Qwen/Qwen3-Embedding-0.6B` | `a100-large` + **vLLM variant** | A100 is 4× the L4 here AND cheaper/1M |
| Max quality, cost no object | `Qwen/Qwen3-Embedding-8B` (4B benched) | `a100-large` + vLLM | ~$7/1M, dim 2560 |
| Images, fast | `clip-ViT-B-32` | `l4x1` | ~395 img/s (bs=32), dim 512 |
| Images, higher quality | `clip-ViT-L-14` or SigLIP-2-large | `l4x1`/`a10g-large` | slower, larger dim |

**One-line scale proof:** 241,787 Wikipedia articles → a searchable Lance vector DB on the Hub
in **4.5 min for ~$0.07** on a single L4 (all-MiniLM). Cheap at scale is real.

## Text: the load-bearing heuristics

**1. Throughput is token-bound, not row-bound.** Same model (all-MiniLM, L4): short text
(AG News, median 53 tokens) = **2866 rows/s**; long text (IMDB, median ~300 tokens) = **912
rows/s**. A 3× swing from text length alone. So estimate cost in *tokens*, not rows, and know
that "rows/s" quoted anywhere is only meaningful with a text length attached.

**2. Batch size peaks around ~128 — bigger is usually *slower*.** Counter-intuitive but
consistent: at batch 128/256/512/1024, all-MiniLM on short text ran 2443 / 1981 / 1355 / 796
rows/s — monotonically *down*. sentence-transformers length-sorts internally, and larger
batches pad to the longest member + add overhead. **Don't crank the batch.** `--batch-size auto`
probes and lands here for you; the token sniffer widens the probe range for short text (and it
still picks ~128).

**3. Sequence length is a real cost lever — and bigger models feel it more.** On long text
(IMDB), capping `--max-seq-len` 512 → 128 sped things up **1.8×** for all-MiniLM (912 → 1682
rows/s) and **2.8×** for bge-base (119 → 332 rows/s). The larger model benefits *more* because
attention is O(n²), so shorter sequences help disproportionately, not just linearly.

| Model | seq-cap 128 | seq-cap 512 | speedup from capping |
|---|---|---|---|
| all-MiniLM-L6-v2 | 1682 rows/s | 912 rows/s | 1.8× |
| bge-base-en-v1.5 | 332 rows/s | 119 rows/s | 2.8× |

Rule of thumb: RAG-sized chunks live under 512 tokens, so the `--max-seq-len 512` default is
right for most retrieval corpora. **Lower the cap for a big, cheap speedup when your text is
short anyway or you can tolerate truncation** (the token sniffer tells you what fraction you'd
lose). Raise it only if the sniffer warns you're truncating a lot AND you need the tail —
budget for the slowdown, especially on bigger models.

**4. GPU: L4 for encoders, A100 only for decoders.** $/1M rows (encode-only):

| Model (type) | L4 $0.80 | A10G $1.50 | A100 $2.50 |
|---|---|---|---|
| all-MiniLM (encoder) | **$0.24** | $0.38 | $0.51 |
| bge-base (encoder) | **$1.87** | $2.02 | $2.66 |
| bge-m3 (encoder) | **$6.17** | $6.22 | $8.27 |
| Qwen3-Embedding-0.6B (decoder) | $3.77 | $4.48 | **$2.78** |

Encoders (MiniLM, bge) are cheapest on the L4 — the bigger GPU doesn't earn its price. **Decoder**
embedders (Qwen3-Embedding) are the exception: the A100 runs them ~4× faster AND cheaper per 1M,
because decoders actually use the extra compute.

**5. Engine: sentence-transformers by default; vLLM ~2× for decoder embedders.** On the same
Qwen3-Embedding-0.6B/L4, vLLM pooling hit 121 rows/s vs sentence-transformers' 59 (~2×). For
small encoders, sentence-transformers is already fast and far simpler — use the default. Switch
to `generate-embeddings-vllm.py` only for large decoder embedders at scale.

**6. Prompts are a correctness issue, not a nicety.** Many retrieval models need a *different*
prefix for documents vs queries, and mismatching them silently hurts retrieval. Gotcha we
verified: sentence-transformers reports an empty placeholder `{"query":"","document":""}` for
models that ship NO prompts — so e5 ("passage: "/"query: ") and nomic ("search_document:
"/"search_query: ") *look* prompt-less but aren't; their prefixes live only in the model card.
The recipe's built-in family table handles e5 / nomic / bge / Qwen3 automatically for a document
corpus; pass `--query-mode` for a query set, or `--prompt '<prefix>'` to override.

## Images: heuristics

- **Model choice:** `clip-ViT-B-32` (~395 img/s on L4 at bs=32, dim 512) is the fast default; `clip-ViT-L-14`
  (40 img/s, dim 768) or SigLIP-2-large (46 img/s, dim 1024) for quality. SigLIP-2 wins at the
  large tier but needs the transformers path (~4× the code); CLIP-via-sentence-transformers is the
  clean one-liner.
- **Flavor + cost: L4 is the cost pick for every image model.** clip-ViT-B-32 ≈ **$0.56/1M images**
  (pre-sized) or ~$1.03/1M (full-res); siglip2-large ≈ $4.84/1M. A10G is ~1.3× faster but 1.875× the
  rate (~$0.92/1M for B-32) → use it only when wall-clock, not cost, matters.
- **`--batch-size auto` uses 32/64/128 for images; 64 is a safe manual default** (only ~6% off the
  bs=32 peak on full-res images, and more robust).
- **Batch: images favor *small* batches — the opposite of the "fixed-size → batch big" hunch.**
  clip-ViT-B-32 on L4: bs=32 = **395 img/s** (fastest), then flat/slower above (343 / 330 / 333 /
  331 / 329 at bs 64 / 128 / 256 / 512 / 1024). Peak GPU mem stays tiny (0.7–4 GB even at bs=1024),
  so it's not a memory ceiling — it's per-batch pipeline overhead. So "don't crank the batch" holds
  for images too, at an even smaller optimum than text. `--batch-size auto` probes from 32 and lands
  on it — no image-specific tuning needed.
- **GPU memory = f(batch × model) only** — models resize to a fixed 224px, so source resolution
  never touches GPU memory (verified: identical peak across a 32px and a full-res dataset). BUT
  **full-res images run ~1.8× slower** than tiny ones (395 → 215 img/s for B-32) — decode/resize is a
  real CPU tax, negligible *only* for pre-sized/small images.

## Storage / dimensions

Bigger embedding dim = more storage + slower vector search. If your model supports **Matryoshka**
truncation (nomic-embed, Qwen3-Embedding), you can keep the first N dims (e.g. 256 of 768) for
much cheaper storage/search at a small quality cost — worth it for large indexes. Always
normalize (the recipe does) so cosine = dot product.

## Other modalities
Text and images are what these recipes cover. **Audio** (CLAP / speech-encoder embeddings) and
**code** (code-specialized text embedders) use different models — a separate recipe, not this one.
Note it and route users there rather than forcing them through the image/text path.

## The evidence (measured on HF Jobs, 2026-07)
Measured on HF Jobs (2026-07) with the scripts in this folder. Text: 20k rows, batch 64, seq-cap 512 unless
noted. All datasets used were public; all test outputs were private.
