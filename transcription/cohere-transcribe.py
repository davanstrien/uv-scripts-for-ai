# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "transformers>=4.56,<5.3,!=5.0.*,!=5.1.*",
#     "torch==2.6.0",
#     "huggingface-hub",
#     "soundfile",
#     "librosa",
#     "sentencepiece",
#     "protobuf",
# ]
# ///

"""
Transcribe audio files from a directory using Cohere Transcribe via transformers.

Uses model.transcribe() which handles long-form audio automatically (chunking,
overlap, reassembly). No nightly dependencies required.

Designed to work with HF Buckets mounted as volumes via `hf jobs uv run -v ...`.

Input:                              Output:
  /input/episode1.mp3         ->      /output/episode1.txt
  /input/sub/clip.wav         ->      /output/sub/clip.txt

Examples:

  # Local test (requires CUDA GPU)
  uv run transcribe-transformers.py ./test-audio ./test-output --language en

  # HF Jobs with bucket volumes
  hf jobs uv run --flavor l4x1 \\
      -s HF_TOKEN \\
      -v hf://buckets/user/audio-input:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      transcribe-transformers.py /input /output --language en --compile

Model: CohereLabs/cohere-transcribe-03-2026 (2B, Apache 2.0)
  - 14 languages: en, de, fr, it, es, pt, el, nl, pl, ar, vi, zh, ja, ko
  - Automatic long-form chunking (>35s handled transparently)
  - compile=True for torch.compile speedup (one-time warmup cost)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

MODEL = "CohereLabs/cohere-transcribe-03-2026"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"}

SUPPORTED_LANGUAGES = {
    "en", "de", "fr", "it", "es", "pt", "el", "nl", "pl", "ar", "vi", "zh", "ja", "ko",
}


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")


def discover_audio_files(input_dir: Path) -> list[Path]:
    """Walk input_dir recursively, returning sorted list of audio files."""
    files = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return files


def get_audio_duration(file_path: Path) -> float | None:
    """Get audio duration in seconds."""
    try:
        import librosa
        return librosa.get_duration(path=str(file_path))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Cohere Transcribe (transformers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Languages: en, de, fr, it, es, pt, el, nl, pl, ar, vi, zh, ja, ko

Examples:
  uv run transcribe-transformers.py ./audio ./output --language en
  uv run transcribe-transformers.py /input /output --language en --compile

HF Jobs with bucket volumes:
  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\
      -v hf://buckets/user/audio-bucket:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      transcribe-transformers.py /input /output --language en --compile
        """,
    )
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_dir", help="Directory to write transcript text files")
    parser.add_argument(
        "--language",
        required=True,
        choices=sorted(SUPPORTED_LANGUAGES),
        help="Language code (required, model does not auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster throughput (one-time warmup cost)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
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

    # Discover audio files
    logger.info(f"Scanning {input_dir} for audio files...")
    files = discover_audio_files(input_dir)
    if not files:
        logger.error(f"No audio files found in {input_dir}")
        logger.error(f"Supported extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        sys.exit(1)

    if args.max_files:
        files = files[: args.max_files]

    logger.info(f"Found {len(files)} audio file(s)")

    # Load model
    logger.info(f"Loading {MODEL}...")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    # Workaround: model.transcribe() -> _ensure_decode_pool accesses tokenizer
    # attributes that CohereAsrTokenizer doesn't expose via the standard
    # transformers interface. Patch them so transcribe() can build its pool.
    tokenizer = processor.tokenizer
    if not hasattr(tokenizer, "additional_special_tokens"):
        tokenizer.additional_special_tokens = []

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL, trust_remote_code=True
    ).to("cuda:0")
    model.eval()
    logger.info("Model loaded")

    # Transcribe all files — model.transcribe() handles chunking and batching
    file_paths = [str(f) for f in files]

    logger.info(
        f"Transcribing {len(files)} file(s) "
        f"(compile={args.compile}, batch_size={args.batch_size})..."
    )

    start_time = time.time()

    texts = model.transcribe(
        processor=processor,
        audio_files=file_paths,
        language=args.language,
        compile=args.compile,
        pipeline_detokenization=True,
        batch_size=args.batch_size,
    )

    elapsed = time.time() - start_time

    # Write outputs
    total_audio_duration = 0.0
    results = []

    for file_path, text in zip(files, texts):
        rel = file_path.relative_to(input_dir)
        txt_path = output_dir / rel.with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(text, encoding="utf-8")

        duration = get_audio_duration(file_path)
        if duration:
            total_audio_duration += duration

        results.append({
            "file": str(rel),
            "duration_s": round(duration, 1) if duration else None,
            "transcript_length": len(text),
            "word_count": len(text.split()),
        })
        logger.info(
            f"  {rel} -> {txt_path.name} "
            f"({len(text.split())} words"
            f"{f', {duration:.0f}s audio' if duration else ''})"
        )

    # Write summary
    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Report
    elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.1f}s"
    logger.info("=" * 50)
    logger.info(f"Done! Transcribed {len(files)} file(s) in {elapsed_str}")
    logger.info(f"  Output: {output_dir}")
    if total_audio_duration > 0:
        rtfx = total_audio_duration / elapsed
        logger.info(f"  Audio: {total_audio_duration / 60:.1f} min total")
        logger.info(f"  RTFx: {rtfx:.1f}x realtime")
    logger.info(f"  Summary: {summary_path}")

    if args.verbose:
        import importlib.metadata
        logger.info("--- Package versions ---")
        for pkg in ["transformers", "torch", "librosa", "soundfile", "huggingface-hub"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Audio Transcription with Cohere Transcribe (transformers)")
        print("=" * 60)
        print("\nTranscribe audio files from a directory -> text files.")
        print("Long audio handled automatically (chunking + overlap).")
        print("Designed for HF Buckets mounted as volumes.")
        print()
        print("Usage:")
        print("  uv run transcribe-transformers.py INPUT_DIR OUTPUT_DIR --language en")
        print()
        print("Examples:")
        print("  uv run transcribe-transformers.py ./audio ./output --language en")
        print("  uv run transcribe-transformers.py ./audio ./output --language en --compile")
        print()
        print("HF Jobs with bucket volumes:")
        print("  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\")
        print("      -v hf://buckets/user/audio-input:/input:ro \\")
        print("      -v hf://buckets/user/transcripts:/output \\")
        print("      transcribe-transformers.py /input /output --language en --compile")
        print()
        print("For full help: uv run transcribe-transformers.py --help")
        sys.exit(0)

    main()
