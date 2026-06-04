# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "vllm",
#     "torch",
#     "huggingface-hub",
#     "soundfile",
#     "librosa",
#     "numpy",
# ]
#
# [[tool.uv.index]]
# url = "https://wheels.vllm.ai/nightly/cu129"
#
# [tool.uv]
# prerelease = "allow"
# ///

"""
Transcribe audio files from a directory using Cohere Transcribe via vLLM.

Designed to work with HF Buckets mounted as volumes via `hf jobs uv run -v ...`.

Long audio files (>30s) are automatically chunked with overlap and reassembled.
All chunks from all files are batched through vLLM in a single call for maximum
throughput.

Input:                              Output:
  /input/episode1.mp3         ->      /output/episode1.txt
  /input/sub/clip.wav         ->      /output/sub/clip.txt

Examples:

  # Local test (requires CUDA GPU)
  uv run transcribe.py ./test-audio ./test-output --language en

  # HF Jobs with bucket volumes
  hf jobs uv run --flavor l4x1 \\
      -s HF_TOKEN \\
      -v hf://buckets/user/audio-input:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      transcribe.py /input /output --language en

Model: CohereLabs/cohere-transcribe-03-2026 (2B, Apache 2.0)
  - 14 languages: en, de, fr, it, es, pt, el, nl, pl, ar, vi, zh, ja, ko
  - Up to 3x faster than other dedicated ASR models in same size range
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
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

# Chunking config — matches model's internal chunking behavior
CHUNK_DURATION_S = 30  # seconds per chunk
OVERLAP_S = 2  # overlap between chunks to avoid cutting mid-word
SAMPLE_RATE = 16000


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


def load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    """Load audio file resampled to 16kHz mono."""
    import librosa
    audio, sr = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
    return audio, sr


def chunk_audio(audio: np.ndarray, sr: int) -> list[np.ndarray]:
    """Split audio into overlapping chunks. Short audio returned as single chunk."""
    chunk_samples = CHUNK_DURATION_S * sr
    overlap_samples = OVERLAP_S * sr
    step = chunk_samples - overlap_samples

    if len(audio) <= chunk_samples:
        return [audio]

    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunks.append(audio[start:end])
        if end >= len(audio):
            break
        start += step

    return chunks


def build_prompt(language: str) -> str:
    """Build the CohereASR prompt with language tokens."""
    return (
        f"<|startofcontext|><|startoftranscript|>"
        f"<|emo:undefined|><|{language}|><|{language}|><|pnc|><|noitn|>"
        f"<|notimestamp|><|nodiarize|>"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Cohere Transcribe via vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Languages: en, de, fr, it, es, pt, el, nl, pl, ar, vi, zh, ja, ko

Examples:
  uv run transcribe.py ./audio ./output --language en
  uv run transcribe.py /input /output --language en --max-files 3

HF Jobs with bucket volumes:
  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\
      -v hf://buckets/user/audio-bucket:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      transcribe.py /input /output --language en
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
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
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

    # Load and chunk audio files
    logger.info("Loading and chunking audio files...")
    file_audio = []  # (file_path, full_audio, sr)
    all_chunks = []  # (file_index, chunk_array) — flat list for batching
    for i, f in enumerate(files):
        audio, sr = load_audio(f)
        file_audio.append((f, audio, sr))
        chunks = chunk_audio(audio, sr)
        duration = len(audio) / sr
        logger.info(f"  {f.name}: {duration:.0f}s -> {len(chunks)} chunk(s)")
        for chunk in chunks:
            all_chunks.append((i, chunk))

    total_chunks = len(all_chunks)
    logger.info(f"Total chunks to transcribe: {total_chunks}")

    # Init vLLM
    logger.info(f"Initializing vLLM with {MODEL}...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        limit_mm_per_prompt={"audio": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )

    # Build inputs — one prompt per chunk
    prompt = build_prompt(args.language)
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"audio": [(chunk, SAMPLE_RATE)]},
        }
        for _, chunk in all_chunks
    ]

    # Transcribe all chunks in one batch
    logger.info(f"Transcribing {total_chunks} chunk(s) across {len(files)} file(s)...")
    start_time = time.time()

    outputs = llm.generate(inputs, sampling_params)

    elapsed = time.time() - start_time

    # Reassemble chunks per file
    file_texts: dict[int, list[str]] = {}
    for (file_idx, _), output in zip(all_chunks, outputs):
        text = output.outputs[0].text.strip()
        file_texts.setdefault(file_idx, []).append(text)

    # Write outputs
    total_audio_duration = 0.0
    results = []

    for i, (file_path, audio, sr) in enumerate(file_audio):
        chunks_text = file_texts.get(i, [])
        full_text = " ".join(chunks_text)

        rel = file_path.relative_to(input_dir)
        txt_path = output_dir / rel.with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(full_text, encoding="utf-8")

        duration = len(audio) / sr
        total_audio_duration += duration

        results.append({
            "file": str(rel),
            "duration_s": round(duration, 1),
            "chunks": len(chunks_text),
            "transcript_length": len(full_text),
            "word_count": len(full_text.split()),
        })
        logger.info(
            f"  {rel} -> {txt_path.name} "
            f"({len(full_text.split())} words, {len(chunks_text)} chunks, "
            f"{duration:.0f}s audio)"
        )

    # Write summary
    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Report
    elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.1f}s"
    rtfx = total_audio_duration / elapsed if elapsed > 0 else 0
    logger.info("=" * 50)
    logger.info(f"Done! Transcribed {len(files)} file(s) in {elapsed_str}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Audio: {total_audio_duration / 60:.1f} min total")
    logger.info(f"  RTFx: {rtfx:.1f}x realtime")
    logger.info(f"  Total chunks: {total_chunks}")
    logger.info(f"  Summary: {summary_path}")

    if args.verbose:
        import importlib.metadata
        logger.info("--- Package versions ---")
        for pkg in ["vllm", "transformers", "torch", "librosa", "soundfile"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Audio Transcription with Cohere Transcribe (vLLM)")
        print("=" * 60)
        print("\nTranscribe audio files from a directory -> text files.")
        print("Long audio automatically chunked with overlap.")
        print("Designed for HF Buckets mounted as volumes.")
        print()
        print("Usage:")
        print("  uv run transcribe.py INPUT_DIR OUTPUT_DIR --language en")
        print()
        print("Examples:")
        print("  uv run transcribe.py ./audio ./output --language en")
        print("  uv run transcribe.py ./audio ./output --language en --max-files 3")
        print()
        print("HF Jobs with bucket volumes:")
        print("  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\")
        print("      -v hf://buckets/user/audio-input:/input:ro \\")
        print("      -v hf://buckets/user/transcripts:/output \\")
        print("      transcribe.py /input /output --language en")
        print()
        print("For full help: uv run transcribe.py --help")
        sys.exit(0)

    main()
