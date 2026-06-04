# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "duckdb",
#     "huggingface-hub",
# ]
# ///

"""Prep Hacker News stories for atlas visualization.

Filters to stories with titles, adds year column for coloring.
Uses DuckDB to query HF parquet files directly (no full download).
Writes prepped parquet to output path (bucket mount or local).

Usage (as HF Job):
    hf jobs uv run --flavor cpu-upgrade \
        -v hf://buckets/davanstrien/atlas-data:/output \
        -s HF_TOKEN --timeout 1h \
        hn-prep.py --output /output/hn-stories/stories.parquet
"""

import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/output/hn-stories/stories.parquet")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap total rows after filtering")
    args = parser.parse_args()

    import duckdb

    start = time.time()

    con = duckdb.connect()
    con.execute("SET enable_http_metadata_cache=true")

    # DuckDB hf:// protocol picks up HF_TOKEN from env automatically

    source = "hf://datasets/open-index/hacker-news/data/**/*.parquet"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print("Querying HN stories from HF parquet files via DuckDB...")

    limit_clause = f"LIMIT {args.max_rows}" if args.max_rows else ""

    query = f"""
    COPY (
        SELECT
            id,
            title,
            score,
            CAST(year(time) AS VARCHAR) AS year,
            CASE
                WHEN score <= 5 THEN '0-5'
                WHEN score <= 25 THEN '6-25'
                WHEN score <= 100 THEN '26-100'
                WHEN score <= 500 THEN '101-500'
                ELSE '500+'
            END AS score_bucket,
            "by",
            url,
            descendants
        FROM '{source}'
        WHERE type = 1
          AND title IS NOT NULL
          AND trim(title) != ''
        ORDER BY random()
        {limit_clause}
    ) TO '{args.output}' (FORMAT PARQUET)
    """

    con.execute(query)
    elapsed = time.time() - start

    # Check output
    result = con.execute(f"SELECT count(*) FROM '{args.output}'").fetchone()
    size_mb = os.path.getsize(args.output) / (1024**2)
    print(f"\nWrote {result[0]:,} stories to {args.output} ({size_mb:.0f} MB)")
    print(f"Total time: {elapsed:.0f}s")

    # Quick stats
    stats = con.execute(f"""
        SELECT min(year) as min_year, max(year) as max_year, count(distinct year) as n_years
        FROM '{args.output}'
    """).fetchone()
    print(f"Year range: {stats[0]} - {stats[1]} ({stats[2]} years)")


if __name__ == "__main__":
    main()
