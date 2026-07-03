# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "sentence-transformers>=3.0.0",
#     "torch",
#     "numpy",
#     "pillow",
#     "einops",
#     "huggingface-hub",
# ]
# ///
"""
Generate embeddings for a Hugging Face dataset (text OR images) with sentence-transformers,
and push the result back to the Hub as a new dataset with an `embeddings` column.

This is the simple, ergonomic default. It runs as one command on the bare uv image, on CPU
or any GPU flavor. For maximum throughput on large *decoder* embedding models (e.g.
Qwen3-Embedding), see the vLLM variant; to get a searchable vector index as a Hub dataset,
see the Lance variant.

PROMPTS (retrieval correctness — read this):
    Many embedding models need a DIFFERENT prefix/instruction for documents vs queries, and
    getting it wrong silently degrades retrieval. This script embeds a *document corpus* by
    default and picks the right document convention for you:
      1. the model's REGISTERED prompt if it ships one (e.g. Qwen3-Embedding), else
      2. a small built-in table of well-known families (e5, nomic, bge), else
      3. no prefix.
    Heads-up: current sentence-transformers injects a placeholder prompts dict
    {"query": "", "document": ""} even for models that register NOTHING — so e5 ("passage: "),
    nomic ("search_document: ") etc. look prompt-less via `.prompts`; their real prefixes live
    only in the model card. The built-in table handles that. Override with --prompt '<prefix>'
    or --prompt-name <registered-name>; embed a query set with --query-mode; force no prefix
    with --prompt ''. The chosen prompt is logged and recorded in the dataset card.

Benchmarks (20k rows, seq-cap 512): all-MiniLM-L6-v2 ~900 rows/s on an L4 (~$0.24/1M rows);
bge-base-en-v1.5 ~120 rows/s. L4 is the cheapest flavor for these encoder models.

Examples:
    # Text (default). Document convention auto-picked.
    hf jobs uv run --flavor l4x1 -s HF_TOKEN generate-embeddings.py \\
        stanfordnlp/imdb  your-name/imdb-embeddings \\
        --column text --model sentence-transformers/all-MiniLM-L6-v2

    # e5: docs auto-get "passage: ". (--prompt 'passage: ' would be the explicit form.)
    hf jobs uv run --flavor l4x1 -s HF_TOKEN generate-embeddings.py \\
        stanfordnlp/imdb  your-name/imdb-e5 --model intfloat/multilingual-e5-large

    # Images (CLIP) — prompts don't apply.
    hf jobs uv run --flavor l4x1 -s HF_TOKEN generate-embeddings.py \\
        your-name/photos  your-name/photos-embeddings \\
        --modality image --column image --model clip-ViT-B-32

    # Test on a small slice first, keep the output private
    hf jobs uv run --flavor l4x1 -s HF_TOKEN generate-embeddings.py \\
        stanfordnlp/imdb  your-name/imdb-emb --max-samples 100 --private
"""
import argparse
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("generate-embeddings")


def find_batch_size(model, sample, normalize, candidates=(32, 64, 128, 256)):
    """Probe for the fastest batch that fits (used by --batch-size auto). Throughput is NOT
    monotonic in batch size, so we time a few on a warmup sample and keep the fastest that doesn't
    OOM. Why bigger isn't better: for text, larger batches pad to the longest member + add overhead;
    for images, the ViT forward already saturates the GPU by ~batch 32. Works for text and images."""
    import time
    import torch
    warm = sample[: min(1024, len(sample))]
    try:  # one untimed warmup so cudnn autotune doesn't penalise the first probe
        model.encode(warm[:32], batch_size=32, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=normalize)
    except Exception:
        pass
    best_bs, best_rps = candidates[0], 0.0
    for bs in candidates:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            t = time.perf_counter()
            model.encode(warm, batch_size=bs, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=normalize)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rps = len(warm) / (time.perf_counter() - t)
            logger.info(f"  auto-batch probe bs={bs}: {rps:.0f} rows/s")
            if rps > best_rps:
                best_rps, best_bs = rps, bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"  auto-batch bs={bs} OOM → stopping probe")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            raise
    logger.info(f"auto-batch chose bs={best_bs} ({best_rps:.0f} rows/s on warmup)")
    return best_bs


def known_convention(model_id):
    """Best-effort (query_prefix, doc_prefix) for common families whose convention is
    documented in the model card but NOT registered in config_sentence_transformers.json.
    Returns None if unknown. Overridable with --prompt / --no-auto-prompt.

    Verified 2026-07-03 on HF Jobs: of e5 / nomic / bge-en / bge-m3 / Qwen3-Embedding, only
    Qwen3-Embedding registers real ST prompts; the rest ship none and rely on manual prefixes.
    """
    m = model_id.lower()
    # Instruction-style embedders (e5-*-instruct, gte-Qwen, ...): prefer the model's REGISTERED
    # prompt or an explicit --prompt; don't guess a literal prefix.
    if "instruct" in m:
        return None
    if "nomic-embed-text" in m:
        return ("search_query: ", "search_document: ")
    if "bge-m3" in m:  # bge-m3 uses no prompts
        return ("", "")
    if "e5" in m:  # e5-base/large/small, multilingual-e5-* (non-instruct)
        return ("query: ", "passage: ")
    if "bge" in m and "-en" in m:  # English bge retrieval: query instruction, docs raw
        return ("Represent this sentence for searching relevant passages: ", "")
    return None


def resolve_prompt(model, model_id, is_query, args):
    """Decide the prefix to prepend for this corpus, log the decision, warn only on real risk."""
    registered = dict(getattr(model, "prompts", {}) or {})
    # Current sentence-transformers injects a placeholder {"query":"","document":""} for models
    # with no config prompts; only non-empty values are real conventions.
    real = {k: v for k, v in registered.items() if v}
    logger.info(f"Registered prompts: {registered} · real (non-empty): {real or 'none'} · "
                f"default_prompt_name={getattr(model, 'default_prompt_name', None)}")
    side = "query" if is_query else "document"

    if args.prompt is not None:  # includes --prompt '' to force no prefix
        logger.info(f"Prompt: raw --prompt → {args.prompt!r}")
        return args.prompt
    if args.prompt_name:
        if args.prompt_name not in registered:
            logger.error(f"--prompt-name {args.prompt_name!r} not registered ({list(registered)}); "
                         f"use --prompt '<raw prefix>' instead.")
            sys.exit(1)
        logger.info(f"Prompt: registered prompt_name={args.prompt_name!r} → {registered[args.prompt_name]!r}")
        return registered[args.prompt_name]
    if registered.get(side):  # model ships a real prompt for this side (e.g. Qwen3 query)
        logger.info(f"Prompt: model-registered {side!r} → {registered[side]!r}")
        return registered[side]

    kc = known_convention(model_id)
    if kc is not None:
        chosen = kc[0] if is_query else kc[1]
        if args.no_auto_prompt:
            if chosen:
                logger.warning(f"--no-auto-prompt set: NOT applying the known {side} prefix {chosen!r} for "
                               f"{model_id}. Retrieval may degrade unless you pass --prompt.")
            return ""
        logger.info(f"Prompt: known-family {side} prefix → {chosen!r} (override with --prompt)"
                    if chosen else f"Prompt: known-family → no {side} prefix needed")
        return chosen

    logger.info(f"Prompt: none (no registered prompt or known convention for {model_id}). "
                f"If it's a retrieval model needing a query/document prefix, pass --prompt.")
    return ""


def sniff_token_lengths(model, texts, max_seq_len, sample=512):
    """Tokenize a sample to report the token-length distribution + how much --max-seq-len truncates,
    and return the median length (used to pick the auto-batch candidate range: short texts under-use
    the GPU at small batch, long texts waste compute on padding). Text only; returns None on failure."""
    try:
        tok = model.tokenizer
    except Exception:
        return None
    s = texts[: min(sample, len(texts))]
    lens = sorted(len(tok.encode(t, add_special_tokens=True)) for t in s)
    n = len(lens)
    if not n:
        return None
    median, p90, mx = lens[n // 2], lens[min(n - 1, int(n * 0.9))], lens[-1]
    pct_over = 100 * sum(1 for L in lens if L > max_seq_len) / n
    note = (f" → {pct_over:.0f}% exceed --max-seq-len {max_seq_len} and are truncated "
            f"(raise it to keep more, at higher cost/slower)" if pct_over >= 5
            else f" (all within --max-seq-len {max_seq_len})")
    logger.info(f"Token lengths (sample {n}): median {median}, p90 {p90}, max {mx}{note}")
    return median


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dataset", help="Input dataset ID on the Hugging Face Hub")
    p.add_argument("output_dataset", help="Output dataset ID to create on the Hub")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="sentence-transformers model (text or CLIP image model)")
    p.add_argument("--modality", choices=["text", "image"], default="text")
    p.add_argument("--column", default="text", help="Input column (text string, or image)")
    p.add_argument("--output-column", default="embeddings")
    p.add_argument("--config", default=None, help="Dataset config name (e.g. wikipedia needs one)")
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=None, help="Limit rows (for testing)")
    p.add_argument("--batch-size", default="auto",
                   help="'auto' probes for the fastest batch that fits, or pass an int")
    p.add_argument("--prompt", default=None,
                   help="Raw prefix to prepend to every text (e.g. 'passage: '). Highest precedence. "
                        "Use --prompt '' to force NO prefix.")
    p.add_argument("--prompt-name", default=None,
                   help="Name of a prompt REGISTERED by the model (e.g. 'query'); errors if not registered.")
    p.add_argument("--query-mode", action="store_true",
                   help="Embed inputs as QUERIES, not documents (flips the auto-picked convention).")
    p.add_argument("--no-auto-prompt", action="store_true",
                   help="Disable the built-in known-family prefix table (still honours registered prompts).")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="Truncate text to this many tokens (predictable cost; RAG-typical)")
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--private", action="store_true", help="Make the output dataset private")
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from huggingface_hub import DatasetCard, login
    from sentence_transformers import SentenceTransformer

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    if not torch.cuda.is_available():
        logger.warning("No CUDA — running on CPU (much slower). Prefer a GPU flavor, e.g. --flavor l4x1.")

    logger.info(f"Loading {args.input_dataset} [{args.split}]")
    ds = (load_dataset(args.input_dataset, args.config, split=args.split) if args.config
          else load_dataset(args.input_dataset, split=args.split))
    if args.column not in ds.column_names:
        logger.error(f"Column {args.column!r} not found. Available: {ds.column_names}")
        sys.exit(1)
    if args.output_column in ds.column_names:
        logger.error(f"Output column {args.output_column!r} already exists — choose another --output-column.")
        sys.exit(1)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    logger.info(f"{len(ds)} rows; modality={args.modality}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device, trust_remote_code=True)
    if args.modality == "text" and getattr(model, "max_seq_length", None):
        model.max_seq_length = min(model.max_seq_length, args.max_seq_len)
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model {args.model} on {device}; dim={dim}")

    # Prompt handling — many retrieval models need a query vs document/passage prefix (text only).
    prompt_str = ""
    if args.modality == "text":
        prompt_str = resolve_prompt(model, args.model, is_query=args.query_mode, args=args)
        items = [t if isinstance(t, str) and t.strip() else " " for t in ds[args.column]]
    else:
        if args.prompt or args.prompt_name:
            logger.warning("--prompt/--prompt-name ignored for image modality.")
        items = [im.convert("RGB") if hasattr(im, "convert") else im for im in ds[args.column]]

    median_tok = sniff_token_lengths(model, items, args.max_seq_len) if args.modality == "text" else None

    if str(args.batch_size).lower() == "auto":
        # Pick the probe range from the data shape. Images: the ViT forward saturates the GPU by
        # ~batch 32, so bigger only adds memory — probe low. Text: short texts under-use the GPU at
        # small batch (probe bigger); long texts pad-waste at big batch (stay modest). Probe verifies.
        if args.modality == "image":
            candidates = (32, 64, 128)
        elif median_tok is None or median_tok >= 256:
            candidates = (32, 64, 128, 256)
        elif median_tok >= 64:
            candidates = (64, 128, 256, 512)
        else:
            candidates = (128, 256, 512, 1024)
        logger.info(f"Finding batch size (--batch-size auto; candidates {candidates})...")
        batch_size = find_batch_size(model, items, args.normalize, candidates=candidates)
    else:
        batch_size = int(args.batch_size)

    t0 = time.perf_counter()
    emb = model.encode(items, batch_size=batch_size, show_progress_bar=True,
                       prompt=(prompt_str or None), convert_to_numpy=True,
                       normalize_embeddings=args.normalize)
    secs = time.perf_counter() - t0
    logger.info(f"Embedded {len(items)} in {secs:.1f}s ({len(items)/secs:.0f} rows/s), dim={dim}")

    ds = ds.add_column(args.output_column, [e.tolist() for e in emb])

    prompt_line = f"`{prompt_str}`" if prompt_str else "(none)"
    card = DatasetCard(
        f"# {args.output_dataset}\n\n"
        f"Embeddings of [`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}) "
        f"column `{args.column}`.\n\n"
        f"- Model: [`{args.model}`](https://huggingface.co/{args.model}) (dim {dim})\n"
        f"- Column: `{args.output_column}`  ·  normalized: {args.normalize}\n"
        f"- Prompt prepended ({'query' if args.query_mode else 'document'} side): {prompt_line}\n\n"
        f"Produced on Hugging Face Jobs with `uv-scripts/embeddings/generate-embeddings.py`.\n"
    )
    logger.info(f"Pushing to {args.output_dataset} (private={args.private})")
    ds.push_to_hub(args.output_dataset, private=args.private)
    try:
        card.push_to_hub(args.output_dataset, repo_type="dataset")
    except Exception as e:
        logger.warning(f"card push skipped: {e}")
    logger.info(f"✅ https://huggingface.co/datasets/{args.output_dataset}")


if __name__ == "__main__":
    main()
