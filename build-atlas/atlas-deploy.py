# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub>=1.9.0",
# ]
# ///

"""Deploy an Embedding Atlas Space from bucket data.

Reads atlas-config.json from the bucket to generate the right Dockerfile,
creates a Docker Space, and prints instructions to mount the bucket.

Examples:

    # Deploy from existing atlas build in a bucket
    uv run atlas-deploy.py \\
        --name my-atlas \\
        --bucket user/atlas-data \\
        --space-id user/my-atlas-space

    # With custom Space hardware
    uv run atlas-deploy.py \\
        --name my-atlas \\
        --bucket user/atlas-data \\
        --space-id user/my-atlas-space \\
        --hardware cpu-upgrade
"""

import argparse
import json
import os

from huggingface_hub import HfApi, Volume, create_repo, upload_file


DOCKERFILE_TEMPLATE = """FROM python:3.12-slim

RUN useradd -m -u 1000 user
RUN pip install --no-cache-dir "embedding-atlas>=0.19.1"

USER user
EXPOSE 7860

CMD ["embedding-atlas", \\
     "/data/{name}/data/dataset.parquet", \\
     "--text", "{text_column}", \\
     "--x", "projection_x", \\
     "--y", "projection_y", \\
     "--disable-projection", \\
     "--duckdb", "server", \\
     "--host", "0.0.0.0", \\
     "--port", "7860"]
"""

README_TEMPLATE = """---
title: {title}
emoji: 🗺️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🗺️ {title}

Interactive embedding visualization of {sample_desc}.

Built with [HF Jobs](https://huggingface.co/docs/hub/jobs) + [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets) + [Embedding Atlas](https://github.com/apple/embedding-atlas).

## How it works

- **Data**: Stored in a Storage Bucket (mounted read-only)
- **Server**: embedding-atlas in server mode with DuckDB
- **Build**: GPU UMAP via cuml.accel ({build_info})

## Features

- Interactive scatter plot with WebGPU acceleration
- Real-time search and filtering
- SQL queries via DuckDB server mode
- Click points to see details
"""


def main():
    parser = argparse.ArgumentParser(description="Deploy an Atlas Space from bucket data")
    parser.add_argument("--name", required=True, help="Atlas name (subdirectory in bucket)")
    parser.add_argument("--bucket", required=True, help="Data bucket ID (e.g. user/atlas-data)")
    parser.add_argument("--space-id", default=None, help="Space ID (default: {user}/{name})")
    parser.add_argument("--hardware", default="cpu-basic", help="Space hardware (default: cpu-basic)")
    parser.add_argument("--text-column", default=None, help="Override text column (reads from config if not set)")
    parser.add_argument("--private", action="store_true", help="Make Space private")
    args = parser.parse_args()

    api = HfApi()

    # Resolve space ID
    if args.space_id is None:
        user = api.whoami()["name"]
        args.space_id = f"{user}/{args.name}"

    # Try to read config from bucket
    text_column = args.text_column or "text"
    sample_desc = "dataset"
    build_info = ""

    try:
        from huggingface_hub import download_bucket_files
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config_remote = f"{args.name}/atlas-config.json"
            config_local = os.path.join(tmp, "atlas-config.json")
            download_bucket_files(args.bucket, files=[(config_remote, config_local)])

            with open(config_local) as f:
                config = json.load(f)

            text_column = config.get("text_column", text_column)
            sample = config.get("sample")
            build_time = config.get("build_time_seconds")
            gpu = config.get("gpu_info", {}).get("gpu", "")

            if sample:
                sample_desc = f"{sample:,} samples"
            if build_time and gpu:
                build_info = f"{build_time:.0f}s on {gpu}"
            elif build_time:
                build_info = f"{build_time:.0f}s"

            print(f"Read config from bucket: text_column={text_column}, sample={sample}")
    except Exception as e:
        print(f"Could not read atlas-config.json from bucket: {e}")
        print(f"Using defaults: text_column={text_column}")

    if args.text_column:
        text_column = args.text_column

    # Create Space
    print(f"\nCreating Space: {args.space_id}")
    create_repo(
        args.space_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    # Generate and upload Dockerfile
    dockerfile = DOCKERFILE_TEMPLATE.format(name=args.name, text_column=text_column)
    upload_file(
        path_or_fileobj=dockerfile.encode(),
        path_in_repo="Dockerfile",
        repo_id=args.space_id,
        repo_type="space",
    )

    # Generate and upload README
    title = args.name.replace("-", " ").replace("_", " ").title()
    readme = README_TEMPLATE.format(
        title=title,
        sample_desc=sample_desc,
        build_info=build_info,
    )
    upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=args.space_id,
        repo_type="space",
    )

    # Set hardware
    if args.hardware != "cpu-basic":
        api.request_space_hardware(args.space_id, args.hardware)
        print(f"Hardware: {args.hardware}")

    # Mount bucket volume
    api.set_space_volumes(
        args.space_id,
        volumes=[
            Volume(type="bucket", source=args.bucket, mount_path="/data", read_only=True),
        ],
    )
    print(f"Bucket mounted: {args.bucket} -> /data (read-only)")

    space_url = f"https://huggingface.co/spaces/{args.space_id}"
    print(f"\nSpace deployed: {space_url}")


if __name__ == "__main__":
    main()
