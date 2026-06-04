# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface-hub",
#     "torch",  # For GPU detection
#     "fsspec",  # For HfFileSystem glob
# ]
# ///

"""
Generate and deploy Embedding Atlas visualizations with remote data loading from HF datasets.

This script creates interactive embedding visualizations where the large parquet data file
is stored in a HF dataset repository while the viewer runs in a lightweight HF Space.
This solves storage limitations for large datasets (>10GB).

Requires embedding-atlas >= 0.18.0 (uses --export-metadata for clean remote data loading).

Example usage:
    # Basic usage - creates Space with remote data loading
    uv run atlas-export-remote.py \
        stanfordnlp/imdb \
        --space-name my-imdb-viz \
        --data-repo my-atlas-data

    # With custom model and configuration
    uv run atlas-export-remote.py \
        beans \
        --space-name bean-disease-atlas \
        --data-repo bean-atlas-data \
        --image-column image \
        --model openai/clip-vit-base-patch32 \
        --sample 10000

    # Run on HF Jobs with GPU
    hf jobs uv run --flavor t4-small -s HF_TOKEN \
        https://huggingface.co/datasets/uv-scripts/build-atlas/raw/main/atlas-export-remote.py \
        my-dataset \
        --space-name my-atlas \
        --data-repo my-atlas-data \
        --model nomic-ai/nomic-embed-text-v1.5

    # Use existing atlas export
    uv run atlas-export-remote.py \
        --from-export atlas_export.zip \
        --space-name my-viz \
        --data-repo my-data
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import (
    HfApi,
    HfFileSystem,
    create_repo,
    get_token,
    login,
    metadata_update,
    upload_file,
    upload_folder,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu_available() -> bool:
    """Check if GPU is available for computation."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def resolve_repo_id(
    repo_name: str,
    organization: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """Resolve a repo name to a full repo_id (org/name or user/name)."""
    if organization:
        return f"{organization}/{repo_name}"
    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info["name"]
    return f"{username}/{repo_name}"


def build_parquet_url(data_repo_id: str) -> str:
    """Build the deterministic parquet URL for a data repo."""
    return f"https://huggingface.co/datasets/{data_repo_id}/resolve/main/dataset.parquet"


def resolve_glob_pattern(
    pattern: str, max_shards: Optional[int] = None, seed: int = 42
) -> list[str]:
    """Resolve an HF glob pattern to a list of resolve URLs.

    Pattern format: hf://datasets/<repo_id>/<path>/*.parquet
    or just: datasets/<repo_id>/<path>/*.parquet
    """
    # Normalize: strip hf:// prefix if present
    fs_pattern = pattern.removeprefix("hf://")

    fs = HfFileSystem()
    matches = fs.glob(fs_pattern)

    if not matches:
        raise ValueError(f"No files matched glob pattern: {pattern}")

    logger.info(f"Glob matched {len(matches)} files")

    if max_shards and max_shards < len(matches):
        random.seed(seed)
        matches = random.sample(matches, max_shards)
        logger.info(f"Sampled {max_shards} shards (seed={seed})")

    # Convert HfFileSystem paths to resolve URLs
    # HfFileSystem paths look like: datasets/org/repo/path/file.parquet
    urls = [f"https://huggingface.co/{m}/resolve/main" for m in matches]
    # Actually, HfFileSystem paths include the revision already in their structure
    # The correct URL format is: https://huggingface.co/datasets/<repo>/resolve/main/<path>
    urls = []
    for m in matches:
        # m = "datasets/org/repo/path/to/file.parquet"
        # Split into repo_type/org/repo and the rest
        parts = m.split("/")
        repo_type = parts[0]  # "datasets"
        repo_id = f"{parts[1]}/{parts[2]}"
        file_path = "/".join(parts[3:])
        urls.append(
            f"https://huggingface.co/{repo_type}/{repo_id}/resolve/main/{file_path}"
        )

    return urls


def build_atlas_command(args, data_repo_id: str) -> Tuple[list, str]:
    """Build the embedding-atlas command with all parameters."""
    cmd = [
        "uvx",
        "--with",
        "datasets",
        "--with",
        "pylance",
        "embedding-atlas>=0.18.0",
    ]
    cmd.extend(args.inputs)

    # Add all optional parameters
    if args.model:
        cmd.extend(["--model", args.model])

    # Always specify text column to avoid interactive prompt
    text_col = args.text_column or "text"
    cmd.extend(["--text", text_col])

    if args.image_column:
        cmd.extend(["--image", args.image_column])

    if args.split:
        cmd.extend(["--split", args.split])

    if args.sample:
        cmd.extend(["--sample", str(args.sample)])

    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    if not args.compute_embeddings:
        cmd.append("--no-compute-embeddings")

        if args.x_column:
            cmd.extend(["--x", args.x_column])

        if args.y_column:
            cmd.extend(["--y", args.y_column])

        if args.neighbors_column:
            cmd.extend(["--neighbors", args.neighbors_column])

    # Export to a folder (not zip) with metadata for remote data loading
    export_dir = "atlas_export_dir"
    cmd.extend(["--export-application", export_dir])

    parquet_url = build_parquet_url(data_repo_id)
    metadata = json.dumps({"database": {"datasetUrl": parquet_url}})
    cmd.extend(["--export-metadata", metadata])

    return cmd, export_dir


def extract_atlas_export(zip_path: str, output_dir: Path) -> Tuple[Path, Path]:
    """
    Extract atlas export ZIP (used for --from-export path).
    Returns (assets_dir, parquet_path)
    """
    logger.info(f"Extracting {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # Find the parquet file
    parquet_path = output_dir / "data" / "dataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError("dataset.parquet not found in export")

    # Check parquet size
    size_gb = parquet_path.stat().st_size / (1024**3)
    logger.info(f"Parquet file size: {size_gb:.2f} GB")

    return output_dir, parquet_path


def update_metadata_with_dataset_url(metadata_path: Path, parquet_url: str) -> None:
    """Deep-merge datasetUrl into existing metadata.json (for --from-export path)."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if "database" not in metadata:
        metadata["database"] = {}

    metadata["database"]["datasetUrl"] = parquet_url

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Updated metadata.json with datasetUrl")


def upload_parquet_to_dataset(
    parquet_path: Path,
    data_repo_id: str,
    private: bool = False,
    hf_token: Optional[str] = None,
) -> str:
    """Upload parquet file to HF dataset repository and return the URL."""
    logger.info(f"Creating/updating dataset repository: {data_repo_id}")

    # Create dataset repository
    create_repo(
        data_repo_id,
        repo_type="dataset",
        private=private,
        token=hf_token,
        exist_ok=True,
    )
    logger.info(f"Dataset repository ready: {data_repo_id}")

    # Upload parquet file
    logger.info(
        f"Uploading {parquet_path.name} ({parquet_path.stat().st_size / (1024**3):.2f} GB)..."
    )
    upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="dataset.parquet",
        repo_id=data_repo_id,
        repo_type="dataset",
        token=hf_token,
    )

    parquet_url = build_parquet_url(data_repo_id)
    logger.info(f"Parquet uploaded to: {parquet_url}")

    return parquet_url


def create_space_readme(args, data_repo_id: str, parquet_size_gb: float) -> str:
    """Generate README.md content for the Space."""
    title = args.space_name.replace("-", " ").title()

    readme = f"""---
title: {title}
emoji: 🗺️
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
license: mit
---

# 🗺️ {title}

Interactive embedding visualization with remote data loading.

## Architecture

This visualization uses a split architecture for efficient large dataset handling:

- **Viewer**: Runs in this Space (~100MB)
- **Data**: Stored in [`{data_repo_id}`](https://huggingface.co/datasets/{data_repo_id}) ({parquet_size_gb:.1f}GB)

The viewer loads data on-demand using HTTP range requests, providing fast initial load and efficient memory usage.

## Features

- Interactive embedding visualization
- Real-time search and filtering
- Automatic clustering with labels
- WebGPU-accelerated rendering
- Remote data loading from HF datasets
"""

    if hasattr(args, "inputs") and args.inputs:
        if len(args.inputs) == 1 and not args.inputs[0].startswith("http"):
            dataset_id = args.inputs[0]
            readme += f"\n## Source Dataset\n\nVisualization of [{dataset_id}](https://huggingface.co/datasets/{dataset_id})\n"
        else:
            readme += "\n## Source Data\n\n"
            for inp in args.inputs:
                if inp.startswith("http"):
                    readme += f"- [{inp.split('/')[-1]}]({inp})\n"
                else:
                    readme += f"- [{inp}](https://huggingface.co/datasets/{inp})\n"

    if hasattr(args, "model") and args.model:
        readme += f"\n## Model\n\nEmbeddings generated using: `{args.model}`\n"

    if hasattr(args, "sample") and args.sample:
        readme += f"\n## Data\n\nVisualization includes {args.sample:,} samples from the dataset.\n"

    readme += """
## How to Use

- **Click and drag** to navigate
- **Scroll** to zoom in/out
- **Click** on points to see details
- **Search** using the search box
- **Filter** using metadata panels

---

*Generated with [UV Scripts Atlas Export (Remote)](https://huggingface.co/datasets/uv-scripts/build-atlas)*
"""

    return readme


def deploy_space_with_remote_data(
    assets_dir: Path,
    space_name: str,
    data_repo_id: str,
    parquet_size_gb: float,
    organization: Optional[str] = None,
    private: bool = False,
    hf_token: Optional[str] = None,
    args: argparse.Namespace = None,
) -> str:
    """Deploy the Space with remote data loading configuration."""
    api = HfApi(token=hf_token)

    # Construct full repo ID
    if organization:
        repo_id = f"{organization}/{space_name}"
    else:
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{space_name}"

    logger.info(f"Creating Space: {repo_id}")

    # Create the Space repository
    create_repo(
        repo_id,
        repo_type="space",
        space_sdk="static",
        private=private,
        token=hf_token,
        exist_ok=True,
    )
    logger.info(f"Space repository ready: {repo_id}")

    # Create README
    readme_content = create_space_readme(args, data_repo_id, parquet_size_gb)
    (assets_dir / "README.md").write_text(readme_content)

    # Upload all files EXCEPT the parquet
    logger.info("Uploading viewer assets to Space...")

    # Get list of files to upload (excluding parquet)
    files_to_upload = []
    for path in assets_dir.rglob("*"):
        if path.is_file() and path.name != "dataset.parquet":
            files_to_upload.append(path)

    logger.info(f"Uploading {len(files_to_upload)} files (excluding dataset.parquet)")

    # Upload the folder
    upload_folder(
        folder_path=str(assets_dir),
        repo_id=repo_id,
        repo_type="space",
        token=hf_token,
        ignore_patterns=["**/dataset.parquet"],
    )

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    logger.info(f"Space deployed: {space_url}")

    return space_url


def main():
    # Enable HF Transfer for faster downloads if available
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    parser = argparse.ArgumentParser(
        description="Deploy Embedding Atlas with remote data loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    parser.add_argument(
        "inputs",
        type=str,
        nargs="*",
        help="HuggingFace dataset ID(s) or parquet URL(s) to visualize (supports multiple)",
    )
    parser.add_argument(
        "--from-export",
        type=str,
        help="Use existing atlas_export.zip file",
    )
    parser.add_argument(
        "--glob",
        type=str,
        help="HF glob pattern for parquet shards, e.g. 'hf://datasets/org/repo/faq/*.parquet'",
    )
    parser.add_argument(
        "--glob-max-shards",
        type=int,
        help="Max number of shards to randomly sample from glob matches",
    )

    # Required Space and data repo configuration
    parser.add_argument(
        "--space-name",
        type=str,
        required=True,
        help="Name for the HuggingFace Space",
    )
    parser.add_argument(
        "--data-repo",
        type=str,
        required=True,
        help="Name for the HuggingFace dataset repository to store parquet",
    )

    # Organization and privacy
    parser.add_argument(
        "--organization",
        type=str,
        help="HuggingFace organization (default: your username)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make both Space and dataset private",
    )
    parser.add_argument(
        "--private-space",
        action="store_true",
        help="Make only the Space private",
    )
    parser.add_argument(
        "--private-data",
        action="store_true",
        help="Make only the dataset private",
    )

    # Atlas configuration (only used when generating new export)
    parser.add_argument(
        "--model",
        type=str,
        help="Embedding model to use",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        help="Name of text column",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        help="Name of image column for image datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for embedding computation. Larger values are faster but use more GPU memory.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in dataset/model",
    )

    # Pre-computed embeddings
    parser.add_argument(
        "--no-compute-embeddings",
        action="store_false",
        dest="compute_embeddings",
        help="Use pre-computed embeddings from dataset",
    )
    parser.add_argument(
        "--x-column",
        type=str,
        help="Column with X coordinates",
    )
    parser.add_argument(
        "--y-column",
        type=str,
        help="Column with Y coordinates",
    )
    parser.add_argument(
        "--neighbors-column",
        type=str,
        help="Column with neighbor indices",
    )

    # Additional options
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only prepare locally, don't deploy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Local directory for output",
    )

    args = parser.parse_args()

    # Resolve glob pattern into inputs
    if args.glob:
        if args.inputs:
            parser.error("Cannot use --glob together with positional inputs")
        args.inputs = resolve_glob_pattern(args.glob, args.glob_max_shards)

    # Validate input
    if not args.from_export and not args.inputs:
        parser.error(
            "Either provide inputs, --glob, or --from-export"
        )
    if args.from_export and args.inputs:
        parser.error("Cannot use --from-export together with positional inputs")

    # Check GPU availability
    if not args.from_export and check_gpu_available():
        logger.info("GPU detected - may accelerate embedding generation")
    elif not args.from_export:
        logger.info("Running on CPU - embedding generation may be slower")

    # Login to HuggingFace if needed
    if not args.local_only:
        hf_token = (
            args.hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or get_token()
        )

        if hf_token:
            login(token=hf_token)
            logger.info("Authenticated with Hugging Face")
        else:
            logger.error("No HF token found. Required for deployment.")
            logger.error("Please set HF_TOKEN environment variable or use --hf-token")
            sys.exit(1)
    else:
        hf_token = None

    # Resolve repo IDs early (needed before atlas export for --export-metadata)
    data_repo_id = resolve_repo_id(args.data_repo, args.organization, hf_token)

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = None
    else:
        temp_dir = tempfile.mkdtemp(prefix="atlas_remote_")
        output_dir = Path(temp_dir)
        logger.info(f"Using temporary directory: {output_dir}")

    try:
        if args.from_export:
            # --from-export path: extract ZIP, patch metadata
            assets_dir, parquet_path = extract_atlas_export(
                args.from_export, output_dir
            )
            parquet_size_gb = parquet_path.stat().st_size / (1024**3)

            # Patch metadata.json with datasetUrl
            parquet_url = build_parquet_url(data_repo_id)
            update_metadata_with_dataset_url(
                assets_dir / "data" / "metadata.json", parquet_url
            )
        else:
            # Fresh export: atlas handles metadata via --export-metadata
            cmd, export_dir_name = build_atlas_command(args, data_repo_id)
            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(cmd)

            if result.returncode != 0:
                logger.error(f"Atlas export failed with code {result.returncode}")
                sys.exit(1)

            logger.info("Atlas export completed")

            # The export is a folder (not a zip)
            export_dir = Path(export_dir_name)
            if not export_dir.exists():
                logger.error(
                    f"Expected export folder {export_dir_name} not found. "
                    "Check embedding-atlas version (requires >= 0.18.0)."
                )
                sys.exit(1)

            # Move export folder contents into output_dir if different
            if export_dir.resolve() != output_dir.resolve():
                # Copy contents into output_dir
                for item in export_dir.iterdir():
                    dest = output_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)
                shutil.rmtree(export_dir)

            assets_dir = output_dir
            parquet_path = assets_dir / "data" / "dataset.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError("dataset.parquet not found in export folder")
            parquet_size_gb = parquet_path.stat().st_size / (1024**3)

        if args.local_only:
            logger.info(f"Files prepared in: {output_dir}")
            logger.info(f"Parquet size: {parquet_size_gb:.2f} GB")
            logger.info("To deploy manually:")
            logger.info(f"1. Upload {parquet_path} to a HF dataset")
            logger.info(f"2. Deploy {assets_dir} to a HF Space")
        else:
            space_private = args.private or args.private_space
            data_private = args.private or args.private_data

            # Upload parquet to dataset repository
            upload_parquet_to_dataset(
                parquet_path, data_repo_id, data_private, hf_token
            )

            # Remove local parquet from assets before Space upload
            if parquet_path.exists():
                parquet_path.unlink()
                logger.info("Removed local parquet from Space assets")

            # Deploy Space
            space_url = deploy_space_with_remote_data(
                assets_dir,
                args.space_name,
                data_repo_id,
                parquet_size_gb,
                args.organization,
                space_private,
                hf_token,
                args,
            )

            # Link source datasets in Space metadata
            space_repo_id = space_url.split("/spaces/")[-1]
            dataset_ids = [data_repo_id]
            if args.inputs:
                for inp in args.inputs:
                    if not inp.startswith("http") and "/" in inp:
                        dataset_ids.append(inp)
            metadata_update(
                space_repo_id,
                {"datasets": dataset_ids},
                repo_type="space",
                overwrite=True,
                token=hf_token,
            )
            logger.info(f"Linked datasets in Space metadata: {dataset_ids}")

            # Summary
            logger.info("\n" + "=" * 50)
            logger.info("Deployment Complete!")
            logger.info("=" * 50)
            logger.info(
                f"Data Repository: https://huggingface.co/datasets/{data_repo_id}"
            )
            logger.info(f"   Size: {parquet_size_gb:.2f} GB")
            logger.info(f"Space: {space_url}")
            logger.info("   Size: ~100 MB (viewer only)")
            logger.info("\nThe visualization will be available in a few moments.")

    finally:
        # Clean up temp directory if used
        if temp_dir and not args.local_only:
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary files")


if __name__ == "__main__":
    # Show examples if no args
    if len(sys.argv) == 1:
        print("Example commands:\n")
        print("# Create new atlas with remote data:")
        print("uv run atlas-export-remote.py stanfordnlp/imdb \\")
        print("    --space-name imdb-atlas --data-repo imdb-atlas-data\n")
        print("# Use existing export:")
        print("uv run atlas-export-remote.py --from-export atlas_export.zip \\")
        print("    --space-name my-viz --data-repo my-data\n")
        print("# With custom model and sampling:")
        print("uv run atlas-export-remote.py my-dataset \\")
        print("    --space-name my-viz --data-repo my-data \\")
        print("    --model nomic-ai/nomic-embed-text-v1.5 --sample 100000\n")
        print("# Multiple parquet URLs as input:")
        print("uv run atlas-export-remote.py \\")
        print("    https://huggingface.co/.../shard-0.parquet \\")
        print("    https://huggingface.co/.../shard-1.parquet \\")
        print("    --space-name my-viz --data-repo my-data --sample 50000\n")
        print("# For HF Jobs with GPU:")
        print("hf jobs uv run --flavor t4-small -s HF_TOKEN \\")
        print(
            "    https://huggingface.co/datasets/uv-scripts/build-atlas/raw/main/atlas-export-remote.py \\"
        )
        print("    dataset --space-name viz --data-repo viz-data")
        sys.exit(0)

    main()
