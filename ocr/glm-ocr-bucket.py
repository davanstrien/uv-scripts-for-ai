# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pillow",
#     "pymupdf",
#     "vllm",
#     "torch",
# ]
#
# [[tool.uv.index]]
# url = "https://wheels.vllm.ai/nightly/cu129"
#
# [tool.uv]
# prerelease = "allow"
# override-dependencies = ["transformers>=5.1.0"]
# ///

"""
OCR images and PDFs from a directory using GLM-OCR, writing markdown files.

Designed to work with HF Buckets mounted as volumes via `hf jobs uv run -v ...`
(requires huggingface_hub with PR #3936 volume mounting support).

The script reads images/PDFs from INPUT_DIR, runs GLM-OCR via vLLM, and writes
one .md file per image (or per PDF page) to OUTPUT_DIR, preserving directory structure.

Input:                          Output:
  /input/page1.png        →      /output/page1.md
  /input/report.pdf       →      /output/report/page_001.md
  (3 pages)                       /output/report/page_002.md
                                  /output/report/page_003.md
  /input/sub/photo.jpg    →      /output/sub/photo.md

Examples:

  # Local test
  uv run glm-ocr-bucket.py ./test-images ./test-output

  # HF Jobs with bucket volumes (PR #3936)
  hf jobs uv run --flavor l4x1 \\
      -s HF_TOKEN \\
      -v bucket/user/ocr-input:/input:ro \\
      -v bucket/user/ocr-output:/output \\
      glm-ocr-bucket.py /input /output

Model: zai-org/GLM-OCR (0.9B, 94.62% OmniDocBench V1.5, MIT licensed)
"""

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "zai-org/GLM-OCR"

TASK_PROMPTS = {
    "ocr": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")


def make_ocr_message(image: Image.Image, task: str = "ocr") -> list[dict]:
    """Create chat message for GLM-OCR from a PIL Image."""
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": TASK_PROMPTS.get(task, TASK_PROMPTS["ocr"])},
            ],
        }
    ]


def discover_files(input_dir: Path) -> list[Path]:
    """Walk input_dir recursively, returning sorted list of image and PDF files."""
    files = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in IMAGE_EXTENSIONS or ext == ".pdf":
            files.append(path)
    return files


def prepare_images(
    files: list[Path], input_dir: Path, output_dir: Path, pdf_dpi: int
) -> list[tuple[Image.Image, Path]]:
    """
    Convert discovered files into (PIL.Image, output_md_path) pairs.

    Images map 1:1. PDFs expand to one image per page in a subdirectory.
    """
    import fitz  # pymupdf

    items: list[tuple[Image.Image, Path]] = []

    for file_path in files:
        rel = file_path.relative_to(input_dir)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            # PDF → one .md per page in a subdirectory named after the PDF
            pdf_output_dir = output_dir / rel.with_suffix("")
            try:
                doc = fitz.open(file_path)
                num_pages = len(doc)
                logger.info(f"PDF: {rel} ({num_pages} pages)")
                for page_num in range(num_pages):
                    page = doc[page_num]
                    # Render at specified DPI
                    zoom = pdf_dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    md_path = pdf_output_dir / f"page_{page_num + 1:03d}.md"
                    items.append((img, md_path))
                doc.close()
            except Exception as e:
                logger.error(f"Failed to open PDF {rel}: {e}")
        else:
            # Image → single .md
            try:
                img = Image.open(file_path).convert("RGB")
                md_path = output_dir / rel.with_suffix(".md")
                items.append((img, md_path))
            except Exception as e:
                logger.error(f"Failed to open image {rel}: {e}")

    return items


def main():
    parser = argparse.ArgumentParser(
        description="OCR images/PDFs from a directory using GLM-OCR, output markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task modes:
  ocr      Text recognition to markdown (default)
  formula  LaTeX formula recognition
  table    Table extraction (HTML)

Examples:
  uv run glm-ocr-bucket.py ./images ./output
  uv run glm-ocr-bucket.py /input /output --task table --pdf-dpi 200

HF Jobs with bucket volumes (requires huggingface_hub PR #3936):
  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\
      -v bucket/user/input-bucket:/input:ro \\
      -v bucket/user/output-bucket:/output \\
      glm-ocr-bucket.py /input /output
        """,
    )
    parser.add_argument("input_dir", help="Directory containing images and/or PDFs")
    parser.add_argument("output_dir", help="Directory to write markdown output files")
    parser.add_argument(
        "--task",
        choices=["ocr", "formula", "table"],
        default="ocr",
        help="OCR task mode (default: ocr)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for vLLM (default: 16)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Max model context length (default: 8192)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max output tokens (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=300,
        help="DPI for PDF page rendering (default: 300)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Sampling temperature (default: 0.01)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.00001, help="Top-p sampling (default: 0.00001)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print resolved package versions",
    )

    args = parser.parse_args()

    check_cuda_availability()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover and prepare
    start_time = time.time()

    logger.info(f"Scanning {input_dir} for images and PDFs...")
    files = discover_files(input_dir)
    if not files:
        logger.error(f"No image or PDF files found in {input_dir}")
        sys.exit(1)

    pdf_count = sum(1 for f in files if f.suffix.lower() == ".pdf")
    img_count = len(files) - pdf_count
    logger.info(f"Found {img_count} image(s) and {pdf_count} PDF(s)")

    logger.info("Preparing images (rendering PDFs)...")
    items = prepare_images(files, input_dir, output_dir, args.pdf_dpi)
    if not items:
        logger.error("No processable images after preparation")
        sys.exit(1)

    logger.info(f"Total images to OCR: {len(items)}")

    # Init vLLM
    logger.info(f"Initializing vLLM with {MODEL}...")
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    # Process in batches
    errors = 0
    processed = 0
    total = len(items)

    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch = items[batch_start:batch_end]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (total + args.batch_size - 1) // args.batch_size

        logger.info(f"Batch {batch_num}/{total_batches} ({processed}/{total} done)")

        try:
            messages = [make_ocr_message(img, task=args.task) for img, _ in batch]
            outputs = llm.chat(messages, sampling_params)

            for (_, md_path), output in zip(batch, outputs):
                text = output.outputs[0].text.strip()
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(text, encoding="utf-8")
                processed += 1

        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            # Write error markers for failed batch
            for _, md_path in batch:
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(f"[OCR ERROR: {e}]", encoding="utf-8")
            errors += len(batch)
            processed += len(batch)

    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.1f}s"

    logger.info("=" * 50)
    logger.info(f"Done! Processed {total} images in {elapsed_str}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Errors: {errors}")
    if total > 0:
        logger.info(f"  Speed: {total / elapsed:.2f} images/sec")

    if args.verbose:
        import importlib.metadata

        logger.info("--- Package versions ---")
        for pkg in ["vllm", "transformers", "torch", "pillow", "pymupdf"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("GLM-OCR Bucket Script")
        print("=" * 60)
        print("\nOCR images/PDFs from a directory → markdown files.")
        print("Designed for HF Buckets mounted as volumes (PR #3936).")
        print()
        print("Usage:")
        print("  uv run glm-ocr-bucket.py INPUT_DIR OUTPUT_DIR")
        print()
        print("Examples:")
        print("  uv run glm-ocr-bucket.py ./images ./output")
        print("  uv run glm-ocr-bucket.py /input /output --task table")
        print()
        print("HF Jobs with bucket volumes:")
        print("  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\")
        print("      -v bucket/user/ocr-input:/input:ro \\")
        print("      -v bucket/user/ocr-output:/output \\")
        print("      glm-ocr-bucket.py /input /output")
        print()
        print("For full help: uv run glm-ocr-bucket.py --help")
        sys.exit(0)

    main()
