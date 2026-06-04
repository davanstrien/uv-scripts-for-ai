# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "internetarchive",
#     "huggingface-hub",
# ]
# ///

"""
Download audio files from an Internet Archive item to an HF Bucket.

Designed to run as a CPU HF Job with a mounted output bucket.

Examples:

  # Local — download to a local directory
  uv run download-ia.py SUSPENSE ./suspense-audio

  # HF Jobs — download straight into a bucket
  hf jobs uv run \
      -v hf://buckets/user/audio-bucket:/output \
      download-ia.py SUSPENSE /output

  # With filters
  uv run download-ia.py SUSPENSE ./output --glob '43-*'  --max-files 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def main():
    parser = argparse.ArgumentParser(
        description="Download audio files from Internet Archive to a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run download-ia.py SUSPENSE ./output
  uv run download-ia.py SUSPENSE ./output --glob '43-*' --max-files 10
  uv run download-ia.py Dragnet_OTR ./output --max-files 20

HF Jobs with bucket volume:
  hf jobs uv run \\
      -v hf://buckets/user/audio-bucket:/output \\
      download-ia.py SUSPENSE /output
        """,
    )
    parser.add_argument("item_id", help="Internet Archive item identifier")
    parser.add_argument("output_dir", help="Directory to download files to")
    parser.add_argument(
        "--glob",
        default="*.mp3",
        help="Glob pattern for files to download (default: *.mp3)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to download",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import internetarchive as ia

    logger.info(f"Fetching file list for '{args.item_id}'...")
    item = ia.get_item(args.item_id)

    # Filter to audio files matching glob
    import fnmatch

    files = []
    for f in item.files:
        name = f["name"]
        ext = Path(name).suffix.lower()
        if ext in AUDIO_EXTENSIONS and fnmatch.fnmatch(name, args.glob):
            files.append(f)

    files.sort(key=lambda f: f["name"])

    if not files:
        logger.error(f"No audio files matching '{args.glob}' in item '{args.item_id}'")
        sys.exit(1)

    if args.max_files:
        files = files[: args.max_files]

    total_size = sum(int(f.get("size", 0)) for f in files)
    logger.info(
        f"Downloading {len(files)} file(s) "
        f"({total_size / 1024 / 1024:.1f} MB) from '{args.item_id}'"
    )

    start_time = time.time()
    downloaded = 0

    for i, f in enumerate(files, 1):
        name = f["name"]
        dest = output_dir / name
        if dest.exists():
            logger.info(f"  [{i}/{len(files)}] {name} (already exists, skipping)")
            downloaded += 1
            continue

        logger.info(f"  [{i}/{len(files)}] {name} ({int(f.get('size', 0)) / 1024 / 1024:.1f} MB)")
        item.download(files=[name], destdir=str(output_dir), no_directory=True)
        downloaded += 1

    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.1f}s"

    logger.info("=" * 50)
    logger.info(f"Done! Downloaded {downloaded} file(s) in {elapsed_str}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Total size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Download Internet Archive Audio to Directory/Bucket")
        print("=" * 60)
        print()
        print("Usage:")
        print("  uv run download-ia.py ITEM_ID OUTPUT_DIR")
        print()
        print("Examples:")
        print("  uv run download-ia.py SUSPENSE ./suspense-audio")
        print("  uv run download-ia.py Dragnet_OTR ./dragnet --max-files 20")
        print()
        print("HF Jobs with bucket:")
        print("  hf jobs uv run \\")
        print("      -v hf://buckets/user/audio-bucket:/output \\")
        print("      download-ia.py SUSPENSE /output")
        print()
        print("For full help: uv run download-ia.py --help")
        sys.exit(0)

    main()
