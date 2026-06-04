# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "embedding-atlas>=0.19.1",
#     "datasets",
#     "cuml-cu12",
# ]
#
# [[tool.uv.index]]
# url = "https://pypi.nvidia.com/simple"
# ///

"""Build an Embedding Atlas visualization with GPU-accelerated UMAP.

Runs embedding-atlas with cuml.accel for ~50x faster UMAP on GPU.
Designed to run as an HF Job with a bucket volume mount for output.

Examples:

    # From a prepped parquet in a bucket
    hf jobs uv run --flavor a100-large \\
        -v hf://buckets/user/atlas-data:/data \\
        -s HF_TOKEN --timeout 2h \\
        atlas-build-gpu.py /data/books.parquet \\
        --text title --sample 2000000 --name my-atlas

    # From an HF dataset
    hf jobs uv run --flavor a100-large \\
        -v hf://buckets/user/atlas-data:/data \\
        -s HF_TOKEN --timeout 2h \\
        atlas-build-gpu.py stanfordnlp/imdb \\
        --text text --split train --name imdb-atlas
"""

import argparse
import json
import os
import subprocess
import sys
import time

os.environ["CUML_ACCEL_ENABLED"] = "1"


def main():
    parser = argparse.ArgumentParser(description="Build an Embedding Atlas with GPU UMAP")
    parser.add_argument("input", help="Parquet path or HF dataset ID")
    parser.add_argument("--name", required=True, help="Atlas name (output subdirectory)")
    parser.add_argument("--text", default="text", help="Text column name")
    parser.add_argument("--image", default=None, help="Image column name")
    parser.add_argument("--split", default=None, help="Dataset split")
    parser.add_argument("--sample", type=int, default=None, help="Number of rows to sample")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--model", default=None, help="Embedding model name")
    parser.add_argument("--output-dir", default="/data", help="Base output directory")
    args = parser.parse_args()

    atlas_output = os.path.join(args.output_dir, args.name)
    config_path = os.path.join(atlas_output, "atlas-config.json")

    print(f"Input: {args.input}")
    print(f"Name: {args.name}")
    print(f"Output: {atlas_output}")
    print(f"Sample: {args.sample}")
    print(f"Batch size: {args.batch_size}")

    # Report GPU/cuml status
    gpu_info = {}
    try:
        import cuml
        print(f"cuML: {cuml.__version__}")
        gpu_info["cuml_version"] = cuml.__version__
    except ImportError:
        print("WARNING: cuml not available, falling back to CPU UMAP")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["gpu"] = torch.cuda.get_device_name()
            print(f"GPU: {gpu_info['gpu']}")
    except ImportError:
        pass

    start = time.time()

    cmd = ["embedding-atlas", args.input, "--text", args.text,
           "--batch-size", str(args.batch_size),
           "--export-application", atlas_output]

    if args.image:
        cmd.extend(["--image", args.image])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.split:
        cmd.extend(["--split", args.split])
    if args.sample:
        cmd.extend(["--sample", str(args.sample)])

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=os.environ)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nFailed with exit code {result.returncode} after {elapsed:.1f}s")
        sys.exit(result.returncode)

    # Write config sidecar for atlas-deploy.py
    parquet_path = os.path.join(atlas_output, "data", "dataset.parquet")
    parquet_mb = os.path.getsize(parquet_path) / (1024**2) if os.path.exists(parquet_path) else 0

    config = {
        "name": args.name,
        "text_column": args.text,
        "image_column": args.image,
        "model": args.model,
        "sample": args.sample,
        "input": args.input,
        "parquet_size_mb": round(parquet_mb, 1),
        "build_time_seconds": round(elapsed, 1),
        "gpu_info": gpu_info,
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Parquet: {parquet_mb:.1f} MB")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
