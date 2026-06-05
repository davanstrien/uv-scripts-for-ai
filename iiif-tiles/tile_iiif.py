# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyvips[binary]",
#     "huggingface-hub",
# ]
# ///
"""Generate IIIF Level 0 static tiles from images in a HF Bucket.

Downloads source images from a bucket, generates IIIF Image API 3.0 tiles
using libvips, creates a IIIF Presentation v3 manifest, and syncs everything
to an output bucket for static serving via HF CDN.

Usage:
    uv run tile_iiif.py --source-bucket org/source --output-bucket org/tiles

HF Jobs:
    hf jobs uv run tile_iiif.py --source-bucket org/source --output-bucket org/tiles
"""

import argparse
import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyvips
from huggingface_hub import HfApi, sync_bucket

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp"}


def generate_tiles(image_path: Path, output_dir: Path, tile_size: int) -> dict:
    """Generate IIIF Level 0 tiles for a single image. Returns image metadata."""
    image = pyvips.Image.new_from_file(str(image_path))
    name = image_path.stem

    # dzsave with iiif3 layout generates the full tile tree + info.json
    image.dzsave(
        str(output_dir / name),
        layout="iiif3",
        tile_size=tile_size,
        suffix=".jpg[Q=85]",
        overlap=0,
    )

    # libvips doesn't generate full/max — create it so the manifest body resolves
    max_dir = output_dir / name / "full" / "max" / "0"
    max_dir.mkdir(parents=True, exist_ok=True)
    image.jpegsave(str(max_dir / "default.jpg"), Q=85)

    return {
        "name": name,
        "width": image.width,
        "height": image.height,
    }


def patch_info_json(tile_dir: Path, image_name: str, base_url: str):
    """Patch the info.json id field to point to the bucket URL."""
    info_path = tile_dir / image_name / "info.json"
    info = json.loads(info_path.read_text())
    info["id"] = f"{base_url}/{image_name}"
    info_path.write_text(json.dumps(info, indent=2))


def generate_manifest(
    images: list[dict], base_url: str, collection_name: str
) -> dict:
    """Generate a minimal IIIF Presentation v3 manifest."""
    items = []
    for img in images:
        canvas_id = f"{base_url}/{img['name']}/canvas"
        image_service_id = f"{base_url}/{img['name']}"
        full_image_id = (
            f"{base_url}/{img['name']}/full/max/0/default.jpg"
        )

        items.append(
            {
                "id": canvas_id,
                "type": "Canvas",
                "width": img["width"],
                "height": img["height"],
                "label": {"en": [img["name"]]},
                "items": [
                    {
                        "id": f"{canvas_id}/page",
                        "type": "AnnotationPage",
                        "items": [
                            {
                                "id": f"{canvas_id}/page/annotation",
                                "type": "Annotation",
                                "motivation": "painting",
                                "body": {
                                    "id": full_image_id,
                                    "type": "Image",
                                    "format": "image/jpeg",
                                    "width": img["width"],
                                    "height": img["height"],
                                    "service": [
                                        {
                                            "id": image_service_id,
                                            "type": "ImageService3",
                                            "profile": "level0",
                                        }
                                    ],
                                },
                                "target": canvas_id,
                            }
                        ],
                    }
                ],
            }
        )

    return {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": f"{base_url}/manifest.json",
        "type": "Manifest",
        "label": {"en": [collection_name]},
        "items": items,
    }


def collect_tile_files(tile_dir: Path, image_name: str) -> list[tuple[str, str]]:
    """Collect (local_path, remote_path) pairs for all tiles of an image."""
    image_tile_dir = tile_dir / image_name
    pairs = []
    for f in image_tile_dir.rglob("*"):
        if f.is_file():
            remote = str(f.relative_to(tile_dir))
            pairs.append((str(f), remote))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate IIIF tiles from images in a HF Bucket"
    )
    parser.add_argument(
        "--source-bucket",
        default=None,
        help="Source bucket with images (e.g., org/iiif-source)",
    )
    parser.add_argument(
        "--output-bucket",
        default=None,
        help="Output bucket for tiles (e.g., org/iiif-tiles)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512, help="Tile size in pixels (default: 512)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for IIIF ids (default: https://huggingface.co/buckets/{output-bucket}/resolve)",
    )
    parser.add_argument(
        "--collection-name",
        default="IIIF Collection",
        help="Name for the IIIF manifest",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Use a local directory as source instead of a bucket",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write tiles to a local directory instead of a bucket",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Concurrent workers for bucket-to-bucket mode (default: 3)",
    )
    args = parser.parse_args()

    if not args.source_dir and not args.source_bucket:
        parser.error("Either --source-bucket or --source-dir is required")

    base_url = args.base_url
    if not base_url:
        if args.output_bucket:
            base_url = f"https://huggingface.co/buckets/{args.output_bucket}/resolve"
        else:
            base_url = "http://localhost:8000"  # for local testing

    api = HfApi()
    use_streaming = args.source_bucket and args.output_bucket and not args.source_dir

    if use_streaming:
        # Streaming mode: process one image at a time to minimize local storage
        _run_streaming(api, args, base_url)
    else:
        # Batch mode: local source and/or local output
        _run_batch(api, args, base_url)


def _process_one_image(
    api: HfApi,
    source_bucket: str,
    output_bucket: str,
    bucket_file,
    work_dir: Path,
    tile_size: int,
    base_url: str,
    index: int,
    total: int,
) -> dict:
    """Download, tile, upload, and cleanup a single image. Returns metadata."""
    img_name = Path(bucket_file.path).name
    # Each worker gets its own subdirectory to avoid conflicts
    worker_dir = work_dir / f"worker_{index}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    # Download
    t1 = time.time()
    local_img = worker_dir / img_name
    api.download_bucket_files(
        source_bucket,
        files=[(bucket_file, str(local_img))],
    )
    t_dl = time.time() - t1

    # Tile
    t1 = time.time()
    tile_dir = worker_dir / "tiles"
    tile_dir.mkdir(exist_ok=True)
    meta = generate_tiles(local_img, tile_dir, tile_size)
    patch_info_json(tile_dir, meta["name"], base_url)
    for xml in tile_dir.rglob("vips-properties.xml"):
        xml.unlink()
    t_tile = time.time() - t1

    # Upload
    t1 = time.time()
    tile_pairs = collect_tile_files(tile_dir, meta["name"])
    api.batch_bucket_files(output_bucket, add=tile_pairs)
    t_ul = time.time() - t1

    print(
        f"  [{index + 1}/{total}] {img_name}: "
        f"download {t_dl:.1f}s, tile {t_tile:.1f}s, "
        f"upload {len(tile_pairs)} files {t_ul:.1f}s"
    )

    # Cleanup
    shutil.rmtree(worker_dir)

    return meta


def _run_streaming(api: HfApi, args, base_url: str):
    """Process images with concurrent workers to overlap I/O and tiling."""
    t0 = time.time()
    workers = args.workers

    # List source images
    print(f"Listing images in bucket: {args.source_bucket}")
    source_files = [
        f
        for f in api.list_bucket_tree(args.source_bucket)
        if hasattr(f, "path")
        and Path(f.path).suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not source_files:
        print("No images found in source bucket")
        return

    source_files.sort(key=lambda f: f.path)
    total = len(source_files)
    print(f"Found {total} images, processing with {workers} worker(s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        if workers == 1:
            # Sequential — simpler output
            image_metadata = []
            for i, bf in enumerate(source_files):
                meta = _process_one_image(
                    api, args.source_bucket, args.output_bucket,
                    bf, work_dir, args.tile_size, base_url, i, total,
                )
                image_metadata.append(meta)
        else:
            # Concurrent
            image_metadata = [None] * total
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _process_one_image,
                        api, args.source_bucket, args.output_bucket,
                        bf, work_dir, args.tile_size, base_url, i, total,
                    ): i
                    for i, bf in enumerate(source_files)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    image_metadata[idx] = future.result()

        # Generate and upload manifest
        manifest = generate_manifest(image_metadata, base_url, args.collection_name)
        manifest_json = json.dumps(manifest, indent=2)
        api.batch_bucket_files(
            args.output_bucket,
            add=[(manifest_json.encode(), "manifest.json")],
        )

    t_total = time.time() - t0
    print(f"\nDone in {t_total:.1f}s! View your manifest at:")
    print(f"  {base_url}/manifest.json")
    print("\nOpen in a IIIF viewer:")
    manifest_url = f"{base_url}/manifest.json"
    print(f"  https://projectmirador.org/embed/?iiif-content={manifest_url}")


def _run_batch(api: HfApi, args, base_url: str):
    """Batch mode: download all, tile all, upload all. For local sources/outputs."""
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        source_dir = Path(tmpdir) / "source"
        tile_dir = Path(args.output_dir) if args.output_dir else Path(tmpdir) / "tiles"
        source_dir.mkdir(exist_ok=True)
        tile_dir.mkdir(parents=True, exist_ok=True)

        # 1. Get source images
        if args.source_dir:
            source_dir = Path(args.source_dir)
            print(f"Using local source: {source_dir}")
        else:
            print(f"Syncing from bucket: {args.source_bucket}")
            sync_bucket(
                f"hf://buckets/{args.source_bucket}",
                str(source_dir),
            )

        # 2. Find images
        images = [
            f
            for f in source_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            print(f"No images found in {source_dir}")
            return

        print(f"Found {len(images)} images")

        # 3. Generate tiles
        image_metadata = []
        for img_path in sorted(images):
            print(f"  Tiling: {img_path.name}")
            meta = generate_tiles(img_path, tile_dir, args.tile_size)
            patch_info_json(tile_dir, meta["name"], base_url)
            image_metadata.append(meta)

        # 4. Clean up vips metadata files
        for xml in tile_dir.rglob("vips-properties.xml"):
            xml.unlink()

        # 5. Generate manifest
        manifest = generate_manifest(image_metadata, base_url, args.collection_name)
        manifest_path = tile_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Manifest written: {manifest_path}")

        # 6. Sync tiles to output bucket (or report local output)
        if args.output_bucket:
            print(f"Syncing tiles to bucket: {args.output_bucket}")
            sync_bucket(
                str(tile_dir),
                f"hf://buckets/{args.output_bucket}",
            )
            print("\nDone! View your manifest at:")
            print(f"  {base_url}/manifest.json")
            print("\nOpen in a IIIF viewer:")
            manifest_url = f"{base_url}/manifest.json"
            print(f"  https://projectmirador.org/embed/?iiif-content={manifest_url}")
        elif args.output_dir:
            print(f"\nTiles written to: {tile_dir}")
            print(f"To test locally: cd {tile_dir} && python -m http.server")
            print("Then open: http://localhost:8000/manifest.json")
        else:
            print("\nNo --output-bucket or --output-dir specified, tiles in temp dir (will be deleted)")

    t_total = time.time() - t0
    print(f"\nTotal time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
