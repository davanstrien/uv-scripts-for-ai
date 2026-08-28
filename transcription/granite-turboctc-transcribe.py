# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git@da7234ac435f",
#     "torch==2.8.0",
#     "torchaudio==2.8.0",
#     "huggingface-hub",
#     "soundfile",
#     "librosa",
#     "numpy",
# ]
# ///

"""
Transcribe English audio files with IBM Granite Speech 5.0 TurboCTC (470M).

Encoder-only CTC model: one forward pass + argmax per batch, no autoregressive
decoding, so it is very fast and cannot hallucinate or loop. English only.

Files are fed whole by default: block attention makes cost linear in length,
so a 30-minute file is fine (measured 10.3 GB peak for four 30-minute files
per batch, i.e. ~2.5 GB per 30 min of audio at bf16). For hour-plus files or
a 24 GB card, lower --batch-size or use --chunk-seconds N, which cuts every
file into N-second windows, pooled, length-sorted and batched (30 s windows
at batch 32 peak at 2.2 GB).
Audio decoding runs in a thread pool (--decode-workers) since it dominates
wall time for a model this fast.

Designed to work with HF Buckets mounted as volumes via `hf jobs uv run -v ...`.

Input:                              Output:
  /input/episode1.mp3         ->      /output/episode1.txt
  /input/sub/clip.wav         ->      /output/sub/clip.txt

Examples:

  # Local test (requires CUDA GPU)
  uv run granite-turboctc-transcribe.py ./test-audio ./test-output

  # HF Jobs with bucket volumes
  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\
      -e UV_TORCH_BACKEND=cu128 \\
      -v hf://buckets/user/audio-input:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      granite-turboctc-transcribe.py /input /output

Model: ibm-granite/granite-speech-5.0-470m-turboctc (Apache 2.0)
  - English only, 16 kHz mono (resampled on load)
  - transformers support landed on main 2026-08-25 (PR #48288); pinned to that
    commit until it ships in a release
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "ibm-granite/granite-speech-5.0-470m-turboctc"
SAMPLE_RATE = 16000

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"}


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")


def discover_audio_files(input_dir: Path) -> list[Path]:
    """Walk input_dir recursively, returning sorted list of audio files."""
    return [
        p
        for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]


def load_audio(path: Path) -> np.ndarray:
    """Decode to float32 mono at 16 kHz."""
    import librosa

    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def chunk_audio(audio: np.ndarray, chunk_seconds: float) -> list[np.ndarray]:
    """Fixed non-overlapping windows; the last one keeps the remainder."""
    if chunk_seconds <= 0 or len(audio) <= chunk_seconds * SAMPLE_RATE:
        return [audio]
    step = int(chunk_seconds * SAMPLE_RATE)
    return [audio[i : i + step] for i in range(0, len(audio), step)]


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe English audio with Granite Speech 5.0 TurboCTC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run granite-turboctc-transcribe.py ./audio ./output
  uv run granite-turboctc-transcribe.py ./audio ./output --chunk-seconds 30 --batch-size 32

HF Jobs with bucket volumes:
  hf jobs uv run --flavor l4x1 -s HF_TOKEN -e UV_TORCH_BACKEND=cu128 \\
      -v hf://buckets/user/audio-bucket:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      granite-turboctc-transcribe.py /input /output
        """,
    )
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_dir", help="Directory to write transcript text files")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.0,
        help="Cut files into N-second windows; 0 = feed each file whole (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Files (or windows) per forward pass (default: 4; 32+ with --chunk-seconds 30)",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Threads for audio decoding (default: CPU count)",
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

    logger.info(f"Scanning {input_dir} for audio files...")
    files = discover_audio_files(input_dir)
    if not files:
        logger.error(f"No audio files found in {input_dir}")
        logger.error(f"Supported extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        sys.exit(1)

    if args.max_files:
        files = files[: args.max_files]

    logger.info(f"Found {len(files)} audio file(s)")

    wall_start = time.time()

    # Decode in a thread pool while the model loads, then feed the GPU as files
    # land: decoding is CPU-bound and, for a model this fast, the bottleneck.
    # segments: (file_idx, chunk_idx, audio)
    logger.info(f"Decoding audio with {args.decode_workers} worker(s)...")
    decode_start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    futures = {pool.submit(load_audio, path): fi for fi, path in enumerate(files)}

    logger.info(f"Loading {MODEL}...")
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForCTC.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    logger.info("Model loaded")

    durations = [0.0] * len(files)
    texts: dict[tuple[int, int], str] = {}
    n_segments = 0
    longest_s = 0.0
    gpu_time = 0.0
    buffer: list[tuple[int, int, np.ndarray]] = []

    def run_batch(batch):
        nonlocal gpu_time, longest_s
        t = time.time()
        inputs = processor(
            [seg[2] for seg in batch], sampling_rate=SAMPLE_RATE, device=model.device
        )
        inputs = inputs.to(model.device, dtype=model.dtype)
        ids = model.generate(**inputs)
        decoded = processor.batch_decode(ids, skip_special_tokens=True)
        for seg, text in zip(batch, decoded):
            texts[seg[:2]] = text.strip()
        torch.cuda.synchronize()
        gpu_time += time.time() - t
        longest_s = max(longest_s, max(len(seg[2]) for seg in batch) / SAMPLE_RATE)

    logger.info(
        f"Transcribing (chunk_seconds={args.chunk_seconds}, batch_size={args.batch_size})..."
    )
    torch.cuda.reset_peak_memory_stats()
    decode_time = None
    with torch.inference_mode():
        for fut in as_completed(futures):
            fi = futures[fut]
            audio = fut.result()
            durations[fi] = len(audio) / SAMPLE_RATE
            for ci, chunk in enumerate(chunk_audio(audio, args.chunk_seconds)):
                buffer.append((fi, ci, chunk))
                n_segments += 1
            if len(futures) and all(f.done() for f in futures) and decode_time is None:
                decode_time = time.time() - decode_start
            while len(buffer) >= args.batch_size:
                run_batch(buffer[: args.batch_size])
                del buffer[: args.batch_size]
            logger.info(
                f"  {files[fi].name}: decoded, {len(texts)}/{n_segments} segments done"
            )
        if buffer:
            run_batch(buffer)
            buffer.clear()
    pool.shutdown()
    decode_time = decode_time or (time.time() - decode_start)
    total_audio = sum(durations)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    model_time = gpu_time

    # Reassemble per file and write outputs
    results = []
    for fi, path in enumerate(files):
        parts = [texts[k] for k in sorted(k for k in texts if k[0] == fi)]
        text = " ".join(p for p in parts if p)
        rel = path.relative_to(input_dir)
        txt_path = output_dir / rel.with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(text, encoding="utf-8")
        results.append(
            {
                "file": str(rel),
                "duration_s": round(durations[fi], 1),
                "segments": len(parts),
                "transcript_length": len(text),
                "word_count": len(text.split()),
            }
        )
        logger.info(
            f"  {rel} -> {txt_path.name} ({len(text.split())} words, {durations[fi]:.0f}s audio)"
        )

    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)

    wall = time.time() - wall_start
    logger.info("=" * 50)
    logger.info(
        f"Done! Transcribed {len(files)} file(s), {total_audio / 60:.1f} min of audio"
    )
    logger.info(f"  Output: {output_dir}")
    logger.info(
        f"  Audio decode (overlapped with model load + GPU): {decode_time:.1f}s"
    )
    logger.info(
        f"  GPU busy (features+forward+decode): {model_time:.1f}s -> RTFx {total_audio / model_time:.0f}x"
    )
    logger.info(
        f"  Wall (incl. decode + model load): {wall:.1f}s -> RTFx {total_audio / wall:.0f}x"
    )
    logger.info(
        f"  Peak GPU memory: {peak_gb:.2f} GB "
        f"(batch_size={args.batch_size}, longest segment {longest_s:.0f}s)"
    )
    logger.info(f"  Summary: {summary_path}")

    if args.verbose:
        import importlib.metadata

        logger.info("--- Package versions ---")
        for pkg in [
            "transformers",
            "torch",
            "torchaudio",
            "librosa",
            "soundfile",
            "huggingface-hub",
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Audio Transcription with Granite Speech 5.0 TurboCTC (470M, English)")
        print("=" * 60)
        print("\nTranscribe audio files from a directory -> text files.")
        print("Encoder-only CTC: fast, batched, no hallucination loops.")
        print("Designed for HF Buckets mounted as volumes.")
        print()
        print("Usage:")
        print("  uv run granite-turboctc-transcribe.py INPUT_DIR OUTPUT_DIR")
        print()
        print("HF Jobs with bucket volumes:")
        print("  hf jobs uv run --flavor l4x1 -s HF_TOKEN -e UV_TORCH_BACKEND=cu128 \\")
        print("      -v hf://buckets/user/audio-input:/input:ro \\")
        print("      -v hf://buckets/user/transcripts:/output \\")
        print("      granite-turboctc-transcribe.py /input /output")
        print()
        print("For full help: uv run granite-turboctc-transcribe.py --help")
        sys.exit(0)

    main()
