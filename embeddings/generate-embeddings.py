# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "sentence-transformers>=3.0.0",
#     "torch",
#     "numpy",
#     "pillow",
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

Benchmarks (20k rows, seq-cap 512): all-MiniLM-L6-v2 ~900 rows/s on an L4 (~$0.24/1M rows);
bge-base-en-v1.5 ~120 rows/s. L4 is the cheapest flavor for these encoder models.

Examples:
    # Text (default). Pick a model off the MTEB leaderboard.
    hf jobs uv run --flavor l4x1 -s HF_TOKEN generate-embeddings.py \\
        stanfordnlp/imdb  your-name/imdb-embeddings \\
        --column text --model sentence-transformers/all-MiniLM-L6-v2

    # Images (CLIP)
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dataset", help="Input dataset ID on the Hugging Face Hub")
    p.add_argument("output_dataset", help="Output dataset ID to create on the Hub")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="sentence-transformers model (text or CLIP image model)")
    p.add_argument("--modality", choices=["text", "image"], default="text")
    p.add_argument("--column", default="text", help="Input column (text string, or image)")
    p.add_argument("--output-column", default="embeddings")
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, default=None, help="Limit rows (for testing)")
    p.add_argument("--batch-size", type=int, default=64)
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
    ds = load_dataset(args.input_dataset, split=args.split)
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
    model = SentenceTransformer(args.model, device=device)
    if args.modality == "text" and getattr(model, "max_seq_length", None):
        model.max_seq_length = min(model.max_seq_length, args.max_seq_len)
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model {args.model} on {device}; dim={dim}")

    if args.modality == "text":
        items = [t if isinstance(t, str) and t.strip() else " " for t in ds[args.column]]
    else:
        items = [im.convert("RGB") if hasattr(im, "convert") else im for im in ds[args.column]]

    t0 = time.perf_counter()
    emb = model.encode(items, batch_size=args.batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=args.normalize)
    secs = time.perf_counter() - t0
    logger.info(f"Embedded {len(items)} in {secs:.1f}s ({len(items)/secs:.0f} rows/s), dim={dim}")

    ds = ds.add_column(args.output_column, [e.tolist() for e in emb])

    card = DatasetCard(
        f"# {args.output_dataset}\n\n"
        f"Embeddings of [`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}) "
        f"column `{args.column}`.\n\n"
        f"- Model: [`{args.model}`](https://huggingface.co/{args.model}) (dim {dim})\n"
        f"- Column: `{args.output_column}`  ·  normalized: {args.normalize}\n\n"
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
