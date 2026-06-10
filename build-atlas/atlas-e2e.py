# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub>=1.12.0",
# ]
# ///

"""Build and deploy an Embedding Atlas end-to-end.

Orchestrates the full pipeline:
1. Creates a storage bucket (if needed)
2. Submits a GPU Job to build the atlas (embedding + UMAP)
3. Waits for the Job to complete
4. Deploys a Docker Space that serves the atlas from the bucket

⚠️ EXPERIMENTAL — this workflow is new and may change.

Examples:

    # Minimal — from HF dataset to deployed Space
    uv run atlas-e2e.py stanfordnlp/imdb \\
        --text text --split train \\
        --name imdb-atlas --sample 50000

    # From prepped parquet (already in a bucket)
    uv run atlas-e2e.py hf://buckets/user/atlas-data/books.parquet \\
        --text title --name open-library-atlas --sample 2000000

    # Full control
    uv run atlas-e2e.py my-org/my-dataset \\
        --text text --split train \\
        --name my-atlas \\
        --sample 1000000 \\
        --bucket user/atlas-data \\
        --space-id user/my-atlas-viz \\
        --flavor a100-large \\
        --timeout 2h \\
        --batch-size 512
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, Volume, create_bucket, get_token, run_uv_job

# Where the sibling scripts live on the Hub, so this orchestrator also works when run
# straight from a URL (uv downloads only this one file).
HUB_RAW = "https://huggingface.co/datasets/uv-scripts/build-atlas/raw/main"

TERMINAL_STAGES = ("COMPLETED", "ERROR", "CANCELED")


def resolve_script(name: str) -> str:
    """Use the sibling script if it exists locally, else its Hub raw URL."""
    local = Path(__file__).parent / name
    if local.exists():
        return str(local)
    url = f"{HUB_RAW}/{name}"
    print(f"{name} not found locally — using {url}")
    return url


def wait_for_job(api: HfApi, job_id: str, poll_interval: int = 30) -> str:
    """Poll a Job until it reaches a terminal stage. Returns the final stage."""
    print(f"\nWaiting for Job {job_id}...")
    while True:
        job = api.inspect_job(job_id=job_id)
        stage = job.status.stage
        if stage in TERMINAL_STAGES:
            msg = job.status.message or ""
            print(f"Job {stage}" + (f": {msg}" if msg else ""))
            return stage
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Build and deploy an Embedding Atlas end-to-end (experimental)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("input", help="HF dataset ID or parquet path")
    parser.add_argument("--name", required=True, help="Atlas name")
    parser.add_argument("--text", default="text", help="Text column name")

    # Dataset options
    parser.add_argument("--split", default=None, help="Dataset split")
    parser.add_argument("--sample", type=int, default=None, help="Number of rows")
    parser.add_argument("--image", default=None, help="Image column name")
    parser.add_argument("--model", default=None, help="Embedding model")

    # Infrastructure
    parser.add_argument("--bucket", default=None, help="Bucket ID (default: {user}/atlas-data)")
    parser.add_argument("--space-id", default=None, help="Space ID (default: {user}/{name})")
    parser.add_argument("--flavor", default="a100-large", help="Job GPU flavor (default: a100-large)")
    parser.add_argument("--timeout", default="2h", help="Job timeout (default: 2h)")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--space-hardware", default="cpu-basic", help="Space hardware (default: cpu-basic)")
    parser.add_argument("--private", action="store_true", help="Make Space private")

    # Workflow control
    parser.add_argument("--build-only", action="store_true", help="Only build, don't deploy Space")
    parser.add_argument("--deploy-only", action="store_true", help="Only deploy from existing bucket data")

    args = parser.parse_args()

    if args.build_only and args.deploy_only:
        parser.error("--build-only and --deploy-only are mutually exclusive")

    api = HfApi()
    user = api.whoami()["name"]

    # Resolve defaults
    if args.bucket is None:
        args.bucket = f"{user}/atlas-data"
    if args.space_id is None:
        args.space_id = f"{user}/{args.name}"

    print("=" * 60)
    print("Embedding Atlas — End-to-End Pipeline")
    print("=" * 60)
    print(f"Input:    {args.input}")
    print(f"Name:     {args.name}")
    print(f"Bucket:   {args.bucket}")
    print(f"Space:    {args.space_id}")
    print(f"Flavor:   {args.flavor}")
    print(f"Sample:   {args.sample}")
    print("=" * 60)

    # ── Step 1: Create bucket ──
    if not args.deploy_only:
        print(f"\n[1/3] Creating bucket {args.bucket}...")
        create_bucket(args.bucket, exist_ok=True)

    # ── Step 2: Submit build Job ──
    if not args.deploy_only:
        print(f"\n[2/3] Submitting build Job ({args.flavor})...")

        build_script = resolve_script("atlas-build-gpu.py")

        script_args = [
            args.input,
            "--name", args.name,
            "--text", args.text,
            "--batch-size", str(args.batch_size),
        ]
        if args.split:
            script_args.extend(["--split", args.split])
        if args.sample:
            script_args.extend(["--sample", str(args.sample)])
        if args.image:
            script_args.extend(["--image", args.image])
        if args.model:
            script_args.extend(["--model", args.model])

        job = run_uv_job(
            build_script,
            script_args=script_args,
            flavor=args.flavor,
            timeout=args.timeout,
            secrets={"HF_TOKEN": get_token()},
            volumes=[Volume(type="bucket", source=args.bucket, mount_path="/data")],
        )

        print(f"Job submitted: {job.id}")
        print(f"View: https://huggingface.co/jobs/{user}/{job.id}")

        # Wait for completion
        stage = wait_for_job(api, job.id)
        if stage != "COMPLETED":
            print(f"\nJob did not complete ({stage}). Check logs: https://huggingface.co/jobs/{user}/{job.id}")
            sys.exit(1)

        print("Build complete!")

    if args.build_only:
        print(f"\nBuild finished. Data in bucket: {args.bucket}/{args.name}/")
        print(f"Deploy later with: uv run atlas-deploy.py --name {args.name} --bucket {args.bucket}")
        return

    # ── Step 3: Deploy Space ──
    print(f"\n[3/3] Deploying Space {args.space_id}...")

    deploy_script = resolve_script("atlas-deploy.py")

    deploy_cmd = [
        "uv", "run",
        deploy_script,
        "--name", args.name,
        "--bucket", args.bucket,
        "--space-id", args.space_id,
        "--hardware", args.space_hardware,
        "--text-column", args.text,
    ]
    if args.private:
        deploy_cmd.append("--private")

    subprocess.run(deploy_cmd, check=True)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Space: https://huggingface.co/spaces/{args.space_id}")
    print(f"Bucket: https://huggingface.co/buckets/{args.bucket}")
    print("=" * 60)


if __name__ == "__main__":
    main()
