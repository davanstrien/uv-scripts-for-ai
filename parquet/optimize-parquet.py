# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=5"]
# ///
"""Convert CSV, JSON and Parquet files added to a bucket into optimized Parquet
in a second bucket (OUTPUT_BUCKET).

Meant to run as a Job triggered by a bucket webhook: the Job receives the list of
changed files in WEBHOOK_PAYLOAD. `datasets` writes optimized Parquet by default
(content-defined chunking, page index, row groups of at most 100MB).

Setup (once):

    # 1. A base Job for the webhook to re-run. With no payload, this first run exits.
    #    Use `hf jobs run ... uv run <url>`, not `hf jobs uv run <url>`: the latter uploads
    #    the script as a volume, and webhook runs don't keep volumes.
    hf jobs run --flavor cpu-upgrade --timeout 2h -e OUTPUT_BUCKET=<user>/<output-bucket> \\
        ghcr.io/astral-sh/uv:python3.12-bookworm \\
        uv run https://huggingface.co/datasets/uv-scripts/parquet/raw/main/optimize-parquet.py

    # 2. A webhook on the input bucket that re-runs that Job on every change.
    #    Webhook runs don't keep the Job's secrets: pass a token as the webhook secret.
    from huggingface_hub import create_webhook
    create_webhook(
        job_id="<job id from step 1>",
        watched=[{"type": "bucket", "name": "<user>/<input-bucket>"}],
        domains=["repo"],
        secret="<fine-grained token>",
    )

Then upload files to the input bucket, e.g.
`hf buckets cp data.csv hf://buckets/<user>/<input-bucket>/data.csv`,
and the output appears at `<output-bucket>/data.csv/data/train-00000-of-00001.parquet`.
"""

import json
import os
import shutil
import tempfile
from pathlib import PurePosixPath

from datasets import load_dataset

# Webhook runs don't keep the Job's secrets: the token arrives as the webhook secret.
if "HF_TOKEN" not in os.environ and "WEBHOOK_SECRET" in os.environ:
    os.environ["HF_TOKEN"] = os.environ["WEBHOOK_SECRET"]

BUILDERS = {".csv": "csv", ".json": "json", ".jsonl": "json", ".parquet": "parquet"}

event = json.loads(os.environ.get("WEBHOOK_PAYLOAD", "{}"))
input_bucket = os.environ.get("WEBHOOK_REPO_ID")
output_bucket = os.environ["OUTPUT_BUCKET"]
# Writing to the watched bucket would trigger this Job again for its own output.
if output_bucket == input_bucket:
    raise SystemExit("OUTPUT_BUCKET must be different from the watched bucket")

# A full load needs disk for the download, the Arrow cache and the output.
# Larger files are streamed instead.
free_disk = shutil.disk_usage(tempfile.gettempdir()).free
stream_above = int(os.environ.get("STREAM_ABOVE_BYTES", free_disk // 3))

for changed_file in event.get("updatedFiles", []):
    path = PurePosixPath(changed_file["path"])
    if changed_file["action"] != "add":
        continue
    if path.suffix not in BUILDERS:
        print(f"Skipping {path}: unsupported file type")
        continue

    streaming = changed_file["size"] > stream_above
    mode = "streaming" if streaming else "full load"
    print(f"{path} ({changed_file['size']:,} bytes): {mode}")
    dataset = load_dataset(
        BUILDERS[path.suffix],
        data_files=f"hf://buckets/{input_bucket}/{path}",
        split="train",
        streaming=streaming,
    )
    # a/b.csv -> <output bucket>/a/b.csv/data/train-*.parquet (keeps b.csv and b.jsonl apart)
    # Tabular files have no image/audio files to embed. Setting this also avoids a
    # crash when pushing a streamed CSV/JSON dataset (its features are not known yet).
    dataset.push_to_hub(f"buckets/{output_bucket}/{path}", embed_external_files=False)
    print(f"Wrote buckets/{output_bucket}/{path}")
