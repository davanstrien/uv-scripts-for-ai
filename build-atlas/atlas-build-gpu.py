# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "embedding-atlas>=0.19.1",
#     "datasets",
#     "cuml-cu12",
# ]
#
# [[tool.uv.index]]
# url = "https://pypi.nvidia.com"
# ///

"""Build an Embedding Atlas visualization with GPU-accelerated UMAP.

Runs embedding-atlas with cuml.accel for ~50x faster UMAP on GPU.
Designed to run as an HF Job with a bucket volume mount for output.

Examples:

    # From a prepped parquet in a bucket
    hf jobs uv run --flavor a100-large \\
        -v hf://buckets/user/atlas-data:/output \\
        -s HF_TOKEN --timeout 2h \\
        atlas-build-gpu.py /output/books.parquet \\
        --text title --sample 2000000 --name my-atlas

    # From an HF dataset
    hf jobs uv run --flavor a100-large \\
        -v hf://buckets/user/atlas-data:/output \\
        -s HF_TOKEN --timeout 2h \\
        atlas-build-gpu.py stanfordnlp/imdb \\
        --text text --split train --name imdb-atlas

The bucket is mounted at /output, not /data: Jobs reserves /data for the uploaded
script artifact when you pass a local script path.
"""

import argparse
import json
import os
import sys
import time


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
    parser.add_argument("--output-dir", default="/output", help="Base output directory")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Run without a GPU (slow: CPU embedding + CPU UMAP)")
    args = parser.parse_args()

    atlas_output = os.path.join(args.output_dir, args.name)
    config_path = os.path.join(atlas_output, "atlas-config.json")

    print(f"Input: {args.input}")
    print(f"Name: {args.name}")
    print(f"Output: {atlas_output}")
    print(f"Sample: {args.sample}")
    print(f"Batch size: {args.batch_size}")

    gpu_info = {}
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    if cuda_available:
        gpu_info["gpu"] = torch.cuda.get_device_name()
        print(f"GPU: {gpu_info['gpu']}")
    elif not args.allow_cpu:
        print("ERROR: no CUDA GPU available. Run on a GPU flavor, or pass --allow-cpu "
              "to accept a much slower CPU build.")
        sys.exit(1)
    else:
        print("WARNING: no GPU — running embedding and UMAP on CPU (--allow-cpu)")

    # cuml.accel patches umap-learn so embedding-atlas's UMAP runs on GPU. It must be
    # installed in-process *before* embedding_atlas imports umap — an env var or a plain
    # subprocess does not engage it.
    if cuda_available:
        try:
            import cuml.accel

            cuml.accel.install()
            import cuml

            gpu_info["cuml_version"] = cuml.__version__
            print(f"cuml.accel installed (cuML {cuml.__version__}) — UMAP will run on GPU")
        except Exception as e:
            print(f"WARNING: cuml.accel unavailable ({e}) — UMAP falls back to CPU")

    from embedding_atlas.cli import main as atlas_cli

    cli_args = [args.input, "--text", args.text,
                "--batch-size", str(args.batch_size),
                "--export-application", atlas_output]

    if args.image:
        cli_args.extend(["--image", args.image])
    if args.model:
        cli_args.extend(["--model", args.model])
    if args.split:
        cli_args.extend(["--split", args.split])
    if args.sample:
        cli_args.extend(["--sample", str(args.sample)])

    print(f"\nRunning: embedding-atlas {' '.join(cli_args)}\n")
    start = time.time()

    returncode = 0
    try:
        atlas_cli(args=cli_args)
    except SystemExit as e:
        returncode = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    elapsed = time.time() - start

    if returncode != 0:
        print(f"\nFailed with exit code {returncode} after {elapsed:.1f}s")
        sys.exit(returncode)

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
