#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""
Simple network speed test for HF Jobs.

Tests download speed by fetching a file from Hugging Face Hub.
Useful for verifying network performance in different HF Jobs flavors.

Usage:
    # Run locally
    uv run network-speed-test.py

    # Run on HF Jobs
    hfjobs run --flavor l4x1 \
        -s HF_TOKEN \
        uv run https://huggingface.co/datasets/uv-scripts/jobs-utils/raw/main/network-speed-test.py

Example output:
    Testing download speed...
    Downloaded 100.0 MB in 2.34 seconds
    Download speed: 42.74 MB/s (341.89 Mbps)
"""

import sys
import time
import urllib.request
import argparse
from typing import Optional


def format_bytes(bytes_value: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def test_download_speed(url: str, size_hint: Optional[int] = None) -> tuple[float, int]:
    """
    Download a file and measure speed.

    Returns:
        Tuple of (duration_seconds, bytes_downloaded)
    """
    print(f"Testing download speed from: {url}")
    print("Downloading...", flush=True)

    start_time = time.time()

    with urllib.request.urlopen(url) as response:
        data = response.read()

    duration = time.time() - start_time
    bytes_downloaded = len(data)

    return duration, bytes_downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Test network speed for HF Jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://huggingface.co/datasets/OpenWebText/resolve/main/README.md",
        help="URL to download for testing (default: sample HF file)",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Use a larger file for testing (~100MB model file)",
    )

    args = parser.parse_args()

    # If --large flag is set, use a bigger file
    test_url = args.url
    if args.large:
        # Use a model file that's about 100MB
        test_url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin"
        print("Using large file test (~100MB)")

    print("=" * 60)
    print("HF Jobs Network Speed Test")
    print("=" * 60)
    print()

    try:
        duration, bytes_downloaded = test_download_speed(test_url)

        # Calculate speeds
        mb_downloaded = bytes_downloaded / (1024 * 1024)
        speed_mbps = (bytes_downloaded * 8) / (duration * 1_000_000)  # Megabits per second
        speed_mb_s = mb_downloaded / duration  # Megabytes per second

        print()
        print("=" * 60)
        print("Results:")
        print("=" * 60)
        print(f"Downloaded: {format_bytes(bytes_downloaded)}")
        print(f"Time: {duration:.2f} seconds")
        print(f"Speed: {speed_mb_s:.2f} MB/s ({speed_mbps:.2f} Mbps)")
        print("=" * 60)

        # Exit successfully
        return 0

    except urllib.error.URLError as e:
        print(f"\n❌ Error downloading file: {e}", file=sys.stderr)
        print("\nThis could indicate network connectivity issues.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
