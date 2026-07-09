# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "moss-transcribe-diarize @ git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize@b5ad0f8386b155ddb89f9332ba3ca71891900357",
#     "transformers>=5.0,<6",
#     "torch>=2.8",
#     "huggingface-hub",
#     "librosa",
#     "soundfile",
# ]
# ///

"""
Transcribe + diarize audio files using MOSS-Transcribe-Diarize (0.9B).

Joint transcription, speaker attribution, and timestamps in a single
generation pass — no separate ASR/diarization/alignment stages. The model
handles long-form audio internally (128k context, up to ~90 min per file),
so files are never pre-chunked: speaker labels ([S01], [S02], ...) stay
consistent across the whole recording.

Designed to work with HF Buckets mounted as volumes via `hf jobs uv run -v ...`.

Input:                              Output:
  /input/meeting.mp3          ->      /output/meeting.json   (segments)
  /input/sub/interview.mp4    ->      /output/sub/interview.json
                                      (+ .txt / .srt with --emit-txt / --emit-srt)

Examples:

  # Local test (requires CUDA GPU)
  uv run moss-transcribe-diarize.py ./audio ./output --emit-txt

  # HF Jobs with bucket volumes
  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\
      -e UV_TORCH_BACKEND=cu128 \\
      -v hf://buckets/user/audio-files:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      https://huggingface.co/datasets/uv-scripts/transcription/raw/main/moss-transcribe-diarize.py \\
      /input /output --emit-txt --emit-srt

Model: OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B, Apache 2.0, not gated)
  - Languages: en, zh (no --language flag needed)
  - Also accepts video containers (mp4, mov, mkv, ...) — audio track is decoded
  - Hotword biasing: --hotwords "Acme Corp,Kubernetes,Dr. Chen"
  - Inference helpers installed from the model's GitHub repo (not on PyPI),
    pinned to a commit for reproducibility
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
# The git-pinned helper package above and the model's remote code are
# co-released, so pin the model revision they were verified against.
# Loosen once upstream stabilizes (model is days old and actively updated).
REVISION = "d7231bbae2587a4af278735eb765b318c4f64edd"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".wmv"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Auto max_new_tokens: model default 5120, ceiling 65536 (docs' long-form value).
# ~100 tokens covers well under 10s of dense multi-speaker speech, so 20 tok/s
# of audio is a safe over-provision; generation stops at EOS anyway.
MAX_NEW_TOKENS_FLOOR = 5120
MAX_NEW_TOKENS_CEILING = 65536
TOKENS_PER_AUDIO_SECOND = 20

PROGRESS_LOG_EVERY_TOKENS = 4096


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")


def discover_media_files(input_dir: Path) -> list[Path]:
    """Walk input_dir recursively, returning sorted list of audio/video files."""
    files = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
            files.append(path)
    return files


def get_media_duration(file_path: Path) -> float | None:
    """Get duration in seconds; PyAV fallback covers video containers."""
    try:
        import librosa

        return librosa.get_duration(path=str(file_path))
    except Exception:
        pass
    try:
        import av

        with av.open(str(file_path)) as container:
            if container.duration is not None:
                return container.duration / av.time_base
    except Exception:
        pass
    return None


def auto_max_new_tokens(duration_s: float | None) -> int:
    """Scale the token budget with audio length; unknown duration gets the ceiling."""
    if duration_s is None:
        return MAX_NEW_TOKENS_CEILING
    return min(
        MAX_NEW_TOKENS_CEILING,
        max(MAX_NEW_TOKENS_FLOOR, int(duration_s * TOKENS_PER_AUDIO_SECOND)),
    )


def write_txt(segments, path: Path):
    """Readable transcript: one `[start - end] SPEAKER: text` line per segment."""
    lines = [
        f"[{seg.start:.2f} - {seg.end:.2f}] {seg.speaker}: {seg.text}"
        for seg in segments
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe + diarize audio using MOSS-Transcribe-Diarize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Languages: en, zh (auto — no language flag)

Examples:
  uv run moss-transcribe-diarize.py ./audio ./output --emit-txt
  uv run moss-transcribe-diarize.py ./audio ./output --hotwords "Acme,Dr. Chen"
  uv run moss-transcribe-diarize.py /input /output --max-files 1 --max-new-tokens 2048

HF Jobs with bucket volumes:
  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\
      -e UV_TORCH_BACKEND=cu128 \\
      -v hf://buckets/user/audio-bucket:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      moss-transcribe-diarize.py /input /output --emit-txt --emit-srt
        """,
    )
    parser.add_argument("input_dir", help="Directory containing audio/video files")
    parser.add_argument("output_dir", help="Directory to write transcript JSON files")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Max generated tokens per file. 0 = auto-scale with audio duration "
        f"(min {MAX_NEW_TOKENS_FLOOR}, max {MAX_NEW_TOKENS_CEILING})",
    )
    parser.add_argument(
        "--hotwords",
        default=None,
        help="Comma-separated terms (names, products, jargon) appended to the "
        "prompt to bias recognition",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Full prompt override (replaces the built-in transcribe+diarize "
        "prompt; overrides --hotwords)",
    )
    parser.add_argument(
        "--emit-txt",
        action="store_true",
        help="Also write .txt transcripts ([start - end] SPEAKER: text per line)",
    )
    parser.add_argument(
        "--emit-srt",
        action="store_true",
        help="Also write .srt subtitles with speaker prefixes",
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

    # Discover media files
    logger.info(f"Scanning {input_dir} for audio/video files...")
    files = discover_media_files(input_dir)
    if not files:
        logger.error(f"No media files found in {input_dir}")
        logger.error(f"Supported extensions: {', '.join(sorted(MEDIA_EXTENSIONS))}")
        sys.exit(1)

    if args.max_files:
        files = files[: args.max_files]

    logger.info(f"Found {len(files)} file(s)")

    # Load model
    logger.info(f"Loading {MODEL}...")
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        DEFAULT_PROMPT,
        build_transcription_messages,
        generate_transcription,
    )
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=True, dtype="auto"
        )
        .to(dtype=dtype)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=True
    )
    logger.info("Model loaded")

    if args.prompt:
        prompt = args.prompt
    elif args.hotwords:
        # Hotword convention from the model card: append a 热词提示 (hotword
        # hint) line to the default prompt.
        terms = ", ".join(t.strip() for t in args.hotwords.split(",") if t.strip())
        prompt = f"{DEFAULT_PROMPT}热词提示：{terms}"
        logger.info(f"Hotwords: {terms}")
    else:
        prompt = DEFAULT_PROMPT

    # Transcribe files one at a time (no pre-chunking: the model handles
    # long-form internally, which is what keeps speaker labels consistent).
    # Outputs are written per file so partial progress survives job timeouts.
    start_time = time.time()
    total_audio_duration = 0.0
    results = []

    for i, file_path in enumerate(files, 1):
        rel = file_path.relative_to(input_dir)
        duration = get_media_duration(file_path)
        max_new_tokens = (
            args.max_new_tokens
            if args.max_new_tokens > 0
            else auto_max_new_tokens(duration)
        )

        duration_str = f"{duration:.0f}s" if duration else "unknown length"
        logger.info(
            f"[{i}/{len(files)}] {rel} ({duration_str}, "
            f"max_new_tokens={max_new_tokens})..."
        )

        def log_progress(n, _budget=max_new_tokens):
            if n % PROGRESS_LOG_EVERY_TOKENS == 0:
                logger.info(f"    ... {n}/{_budget} tokens")

        file_start = time.time()
        messages = build_transcription_messages(file_path, prompt=prompt)
        result = generate_transcription(
            model,
            processor,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device=device,
            dtype=dtype,
            token_callback=log_progress,
        )
        file_elapsed = time.time() - file_start

        raw_text = result["text"]
        generated_tokens = result["generated_tokens"]
        truncated = generated_tokens >= max_new_tokens
        if truncated:
            logger.warning(
                f"    Hit max_new_tokens={max_new_tokens} — transcript is likely "
                f"incomplete. Re-run with a higher --max-new-tokens."
            )

        segments = parse_transcript(raw_text)
        speakers = sorted({seg.speaker for seg in segments})

        record = {
            "file": str(rel),
            "model": MODEL,
            "duration_s": round(duration, 1) if duration else None,
            "raw_transcript": raw_text,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": seg.speaker,
                    "text": seg.text,
                }
                for seg in segments
            ],
            "num_segments": len(segments),
            "num_speakers": len(speakers),
            "generated_tokens": generated_tokens,
            "truncated": truncated,
        }

        json_path = output_dir / rel.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if args.emit_txt:
            write_txt(segments, json_path.with_suffix(".txt"))

        if args.emit_srt:
            from moss_transcribe_diarize.subtitle import (
                export_srt,
                subtitle_segments_from_transcript_segments,
            )

            srt_text = export_srt(
                subtitle_segments_from_transcript_segments(segments),
                show_speaker=True,
            )
            json_path.with_suffix(".srt").write_text(srt_text, encoding="utf-8")

        if duration:
            total_audio_duration += duration

        results.append(
            {
                "file": str(rel),
                "duration_s": round(duration, 1) if duration else None,
                "num_segments": len(segments),
                "num_speakers": len(speakers),
                "generated_tokens": generated_tokens,
                "truncated": truncated,
                "elapsed_s": round(file_elapsed, 1),
            }
        )
        logger.info(
            f"    -> {json_path.name}: {len(segments)} segments, "
            f"{len(speakers)} speaker(s), {generated_tokens} tokens, "
            f"{file_elapsed:.0f}s"
        )

    elapsed = time.time() - start_time

    # Write summary
    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.1f}s"
    truncated_count = sum(1 for r in results if r["truncated"])
    logger.info("=" * 50)
    logger.info(f"Done! Processed {len(files)} file(s) in {elapsed_str}")
    logger.info(f"  Output: {output_dir}")
    if total_audio_duration > 0:
        rtfx = total_audio_duration / elapsed
        logger.info(f"  Audio: {total_audio_duration / 60:.1f} min total")
        logger.info(f"  RTFx: {rtfx:.1f}x realtime")
    if truncated_count:
        logger.warning(f"  Truncated: {truncated_count} file(s) hit max_new_tokens")
    logger.info(f"  Summary: {summary_path}")

    if args.verbose:
        import importlib.metadata

        logger.info("--- Package versions ---")
        for pkg in [
            "moss-transcribe-diarize",
            "transformers",
            "torch",
            "av",
            "librosa",
            "soundfile",
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Transcription + Diarization with MOSS-Transcribe-Diarize")
        print("=" * 60)
        print("\nTranscribe audio/video from a directory -> JSON segments")
        print("with timestamps and speaker labels ([S01], [S02], ...).")
        print("One pass per file — no chunking, labels stay consistent.")
        print("Designed for HF Buckets mounted as volumes.")
        print()
        print("Usage:")
        print("  uv run moss-transcribe-diarize.py INPUT_DIR OUTPUT_DIR")
        print()
        print("Examples:")
        print("  uv run moss-transcribe-diarize.py ./audio ./output --emit-txt")
        print(
            "  uv run moss-transcribe-diarize.py ./audio ./output --hotwords 'Acme,Dr. Chen'"
        )
        print()
        print("HF Jobs with bucket volumes:")
        print("  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\")
        print("      -e UV_TORCH_BACKEND=cu128 \\")
        print("      -v hf://buckets/user/audio-files:/input:ro \\")
        print("      -v hf://buckets/user/transcripts:/output \\")
        print("      moss-transcribe-diarize.py /input /output --emit-txt --emit-srt")
        print()
        print("For full help: uv run moss-transcribe-diarize.py --help")
        sys.exit(0)

    main()
