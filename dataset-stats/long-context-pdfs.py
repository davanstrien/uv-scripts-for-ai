# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars>=1.0",
#     "datasets",
# ]
# ///
"""
Extract long-context, high-quality PDFs from finepdfs.

Creates a curated subset of long documents with high OCR quality -
useful for long-context model training.

Examples:
    # Quick test (Welsh)
    uv run long-context-pdfs.py --lang cym_Latn --limit 100

    # English long docs
    uv run long-context-pdfs.py --lang eng_Latn --output user/finepdfs-eng-long

    # All Latin scripts
    uv run long-context-pdfs.py --lang "*_Latn" --output user/finepdfs-long-context

    # HF Jobs (memory-efficient with small chunk size)
    hf jobs uv run \\
        -s HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/dataset-stats/raw/main/long-context-pdfs.py \\
        -- --lang eng_Latn --output user/finepdfs-eng-long
"""

import argparse
import tempfile
from pathlib import Path

import polars as pl
from datasets import Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Extract long-context high-quality PDFs"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="cym_Latn",
        help="Language code or glob pattern (default: cym_Latn, use '*_Latn' for all Latin)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=10000,
        help="Minimum token count (default: 10000)",
    )
    parser.add_argument(
        "--min-lid-score",
        type=float,
        default=0.8,
        help="Minimum language ID score (default: 0.8)",
    )
    parser.add_argument("--limit", type=int, help="Limit rows")
    parser.add_argument("--output", type=str, help="Output dataset repo")
    parser.add_argument("--private", action="store_true")

    args = parser.parse_args()

    source = f"hf://datasets/HuggingFaceFW/finepdfs/data/{args.lang}/train/*.parquet"

    print("=" * 60)
    print("Long-Context High-Quality PDF Extraction")
    print("=" * 60)
    print(f"Source: {source}")
    print("Filters:")
    print(f"  - token_count >= {args.min_tokens}")
    print(f"  - page_average_lid_score >= {args.min_lid_score}")
    print("  - extractor == 'docling'")
    if args.limit:
        print(f"  - limit: {args.limit}")
    print("=" * 60)

    # Build query - simpler filters first, OCR quality filter can be tricky
    lf = (
        pl.scan_parquet(source)
        .filter(
            (pl.col("token_count") >= args.min_tokens)
            & (pl.col("page_average_lid_score") >= args.min_lid_score)
            & (pl.col("extractor") == "docling")
        )
        .select(
            [
                "id",
                "url",
                "text",
                "language",
                "token_count",
                "dump",
                "page_average_lid_score",
            ]
        )
    )

    if args.limit:
        lf = lf.limit(args.limit)

    # Preview
    print("\nPreviewing...")
    preview = lf.limit(5).collect()
    print(f"Sample rows: {len(preview)}")
    if len(preview) > 0:
        print(preview.select(["language", "token_count", "page_average_lid_score"]))
    else:
        print("No rows matched! Try lowering thresholds.")
        return

    if not args.output:
        print("\nNo --output specified. Use --output to push to Hub.")
        return

    # Rebuild query for streaming to parquet
    lf = (
        pl.scan_parquet(source)
        .filter(
            (pl.col("token_count") >= args.min_tokens)
            & (pl.col("page_average_lid_score") >= args.min_lid_score)
            & (pl.col("extractor") == "docling")
        )
        .select(
            [
                "id",
                "url",
                "text",
                "language",
                "token_count",
                "dump",
                "page_average_lid_score",
            ]
        )
    )
    if args.limit:
        lf = lf.limit(args.limit)

    # Use sink_parquet for true streaming (minimal memory)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "data.parquet"
        print("\nStreaming to parquet (sink_parquet)...")
        lf.sink_parquet(output_path)

        print(f"\nLoading parquet and pushing to {args.output}...")
        ds = Dataset.from_parquet(str(output_path))
        print(f"Dataset: {len(ds)} rows")
        ds.push_to_hub(args.output, private=args.private)

    print(f"\nDone! https://huggingface.co/datasets/{args.output}")


if __name__ == "__main__":
    main()
