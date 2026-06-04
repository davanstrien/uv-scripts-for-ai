# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "duckdb",
#     "huggingface-hub",
# ]
# ///

"""Prep Open Library works for atlas visualization.

Filters to works with titles and subjects, adds broad category for coloring.
Uses DuckDB to query HF parquet files directly.

Usage (as HF Job):
    hf jobs uv run --flavor cpu-upgrade \
        -v hf://buckets/davanstrien/atlas-data:/output \
        -s HF_TOKEN --timeout 1h \
        open-library-prep.py --output /output/open-library/books.parquet
"""

import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/output/open-library/books.parquet")
    parser.add_argument("--max-rows", type=int, default=2000000)
    args = parser.parse_args()

    import duckdb

    start = time.time()
    con = duckdb.connect()
    con.execute("SET enable_http_metadata_cache=true")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    source = "hf://datasets/open-index/open-library/data/works/*.parquet"

    print(f"Querying Open Library works (max {args.max_rows:,} rows)...")

    query = f"""
    COPY (
        SELECT
            title,
            CASE
                WHEN subjects LIKE '%Fiction%' OR subjects LIKE '%Novel%' OR subjects LIKE '%Stories%' THEN 'Fiction'
                WHEN subjects LIKE '%History%' OR subjects LIKE '%Antiquities%' OR subjects LIKE '%Civilization%' THEN 'History'
                WHEN subjects LIKE '%Science%' OR subjects LIKE '%Physics%' OR subjects LIKE '%Chemistry%' OR subjects LIKE '%Biology%' OR subjects LIKE '%Geology%' OR subjects LIKE '%Astronomy%' THEN 'Science'
                WHEN subjects LIKE '%Religion%' OR subjects LIKE '%Theology%' OR subjects LIKE '%Bible%' OR subjects LIKE '%Church%' THEN 'Religion'
                WHEN subjects LIKE '%Biography%' OR subjects LIKE '%Correspondence%' THEN 'Biography'
                WHEN subjects LIKE '%Poetry%' OR subjects LIKE '%Drama%' OR subjects LIKE '%Literature%' THEN 'Literature'
                WHEN subjects LIKE '%Mathematics%' OR subjects LIKE '%Computer%' OR subjects LIKE '%Engineering%' OR subjects LIKE '%Technol%' THEN 'Tech & Engineering'
                WHEN subjects LIKE '%Music%' THEN 'Music'
                WHEN subjects LIKE '%Art%' OR subjects LIKE '%Photography%' OR subjects LIKE '%Architecture%' OR subjects LIKE '%Design%' THEN 'Art & Design'
                WHEN subjects LIKE '%Law%' OR subjects LIKE '%Politics%' OR subjects LIKE '%Government%' OR subjects LIKE '%Foreign relations%' THEN 'Law & Politics'
                WHEN subjects LIKE '%Education%' OR subjects LIKE '%Teaching%' THEN 'Education'
                WHEN subjects LIKE '%Philosophy%' OR subjects LIKE '%Psychology%' THEN 'Philosophy'
                WHEN subjects LIKE '%Medicine%' OR subjects LIKE '%Health%' OR subjects LIKE '%Disease%' THEN 'Medicine'
                WHEN subjects LIKE '%Econom%' OR subjects LIKE '%Business%' OR subjects LIKE '%Commerce%' OR subjects LIKE '%Finance%' THEN 'Business & Economics'
                WHEN subjects LIKE '%Children%' OR subjects LIKE '%Juvenile%' THEN 'Children'
                WHEN subjects LIKE '%Travel%' OR subjects LIKE '%Guidebook%' OR subjects LIKE '%Description and travel%' THEN 'Travel'
                WHEN subjects LIKE '%Agriculture%' OR subjects LIKE '%Gardening%' OR subjects LIKE '%Cook%' OR subjects LIKE '%Food%' THEN 'Food & Agriculture'
                WHEN subjects LIKE '%Social%' OR subjects LIKE '%Sociology%' OR subjects LIKE '%Women%' OR subjects LIKE '%Feminism%' THEN 'Society'
                WHEN subjects LIKE '%Military%' OR subjects LIKE '%War%' THEN 'Military'
                WHEN subjects LIKE '%Sport%' OR subjects LIKE '%Games%' OR subjects LIKE '%Baseball%' OR subjects LIKE '%Football%' THEN 'Sports'
                ELSE 'Other'
            END as category,
            first_publish_date,
            json_extract_string(subjects, '$[0]') as primary_subject
        FROM '{source}'
        WHERE subjects IS NOT NULL
          AND subjects != '[]'
          AND title IS NOT NULL
          AND trim(title) != ''
          AND length(title) > 3
        ORDER BY random()
        LIMIT {args.max_rows}
    ) TO '{args.output}' (FORMAT PARQUET)
    """

    con.execute(query)
    elapsed = time.time() - start

    # Stats
    result = con.execute(f"SELECT count(*) FROM '{args.output}'").fetchone()
    size_mb = os.path.getsize(args.output) / (1024**2)
    print(f"\nWrote {result[0]:,} books to {args.output} ({size_mb:.0f} MB)")
    print(f"Total time: {elapsed:.0f}s")

    cats = con.execute(f"""
        SELECT category, count(*) as cnt
        FROM '{args.output}'
        GROUP BY 1 ORDER BY 2 DESC
    """).df()
    print("\nCategory distribution:")
    for _, row in cats.iterrows():
        print(f"  {row['cnt']:6,}  {row['category']}")


if __name__ == "__main__":
    main()
