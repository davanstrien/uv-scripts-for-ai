# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
#     "librosa",
#     "soundfile",
# ]
# ///

"""
High-throughput transcription + diarization via an in-job sgl-omni server.

Same model and outputs as moss-transcribe-diarize.py, but serves
MOSS-Transcribe-Diarize behind sglang-omni inside the job and posts files
concurrently — continuous batching decodes many tapes at once (measured
43x realtime aggregate at just 2 streams on a100-large, vs 3.2x for the
sequential transformers recipe).

This script is the *driver* half: it expects the server on localhost
(started by the job command below), splits long audio to fit the model's
context window, posts files concurrently, and writes JSON (+ .txt).

Requires a GPU with enough KV-cache room for long audio — use a100-large
(80 GB). 24 GB cards (l4x1, a10g) OOM on tapes over ~30 min.

Run on HF Jobs (single command — installs sglang-omni per upstream docs,
starts the server, then runs this driver against it):

  hf jobs run --detach --flavor a100-large -s HF_TOKEN --timeout 8h \\
      -v hf://buckets/user/audio-files:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      lmsysorg/sglang:nightly-dev-cu13-20260709-074bb928 -- \\
      bash -c "pip install -q uv; git clone --depth 1 https://github.com/sgl-project/sglang-omni.git && cd sglang-omni && uv venv .venv -p 3.12 && . .venv/bin/activate && uv pip install . && (sgl-omni serve --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize --host 0.0.0.0 --port 8000 --max-running-requests 16 --mem-fraction-static 0.80 &) && uv run https://huggingface.co/datasets/uv-scripts/transcription/raw/main/moss-transcribe-diarize-server.py /input /output --emit-txt"

Input:                              Output:
  /input/tape1.mp3            ->      /output/tape1.json  (segments; parts merged)
  /input/sub/tape2.mp3        ->      /output/sub/tape2.json

Model: OpenMOSS-Team/MOSS-Transcribe-Diarize (0.9B, Apache 2.0)
  - Long tapes are split into clips that fit the model's context window;
    speaker labels are consistent WITHIN a clip but reset BETWEEN clips
    (clip index is recorded on every segment as "part").
  - The model occasionally stops generating early (EOS mid-tape); the driver
    detects short coverage and automatically continues from the last
    timestamp. Compare coverage_s vs duration_s in the output JSON.
"""

import argparse
import concurrent.futures
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"}

SAMPLE_RATE = 16000

# Context budget. The model's window is 131072 tokens and the server
# reserves max_new_tokens from it, so input audio gets
# (131072 - max_new_tokens - margin) tokens. Audio tokenizes at ~17.5
# tokens/sec (30s Whisper chunks -> temporal merge), so requesting 65536
# output tokens silently truncates input past ~62 min. We split audio into
# parts that fit alongside a generation budget sized to the part.
CONTEXT_TOKENS = 131072
AUDIO_TOKENS_PER_SECOND = 17.5
CONTEXT_MARGIN_TOKENS = 2048
GENERATION_TOKENS_PER_AUDIO_SECOND = 20  # same over-provision as sibling script
MIN_GENERATION_TOKENS = 5120

# Longest part where input + generation + margin fits the window:
# d*17.5 + max(5120, d*20) + 2048 <= 131072  ->  d ~ 3440s. Stay under it.
DEFAULT_PART_SECONDS = 3300  # 55 min


def generation_budget(duration_s: float) -> int:
    """Output-token budget for a clip, capped so input still fits the window."""
    want = max(
        MIN_GENERATION_TOKENS, int(duration_s * GENERATION_TOKENS_PER_AUDIO_SECOND)
    )
    room = (
        CONTEXT_TOKENS
        - CONTEXT_MARGIN_TOKENS
        - int(duration_s * AUDIO_TOKENS_PER_SECOND)
    )
    return max(MIN_GENERATION_TOKENS, min(want, room))


def discover_audio_files(input_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]


def transcribe_clip(server: str, clip: Path, duration_s: float, timeout_s: int) -> dict:
    import requests

    with open(clip, "rb") as f:
        resp = requests.post(
            f"{server}/v1/audio/transcriptions",
            data={
                "model": MODEL,
                "response_format": "verbose_json",
                "max_new_tokens": generation_budget(duration_s),
            },
            files={"file": (clip.name, f)},
            timeout=timeout_s,
        )
    resp.raise_for_status()
    return resp.json()


SPEAKER_RE = re.compile(r"^\[(S\d+)\]\s*")


def parse_verbose_segments(
    payload: dict, offset_s: float, part_index: int
) -> list[dict]:
    """Normalize sgl-omni verbose_json segments; shift by part offset."""
    out = []
    for seg in payload.get("segments", []):
        m = SPEAKER_RE.match(seg["text"])
        out.append(
            {
                "start": round(seg["start"] + offset_s, 2),
                "end": round(seg["end"] + offset_s, 2),
                "speaker": m.group(1) if m else None,
                "text": SPEAKER_RE.sub("", seg["text"]).strip(),
                "part": part_index,
            }
        )
    return out


def write_txt(segments: list[dict], path: Path):
    lines = [
        f"[{s['start']:.2f} - {s['end']:.2f}] {s['speaker'] or '?'}: {s['text']}"
        for s in segments
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wait_for_server(server: str, timeout_s: int = 1800):
    import requests

    logger.info(f"Waiting for server at {server}...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{server}/health", timeout=5).status_code == 200:
                logger.info("Server is ready")
                return
        except requests.RequestException:
            pass
        time.sleep(10)
    logger.error(f"Server did not become ready within {timeout_s}s")
    sys.exit(1)


# A clip's transcript should reach near its end. If generation stops more
# than this many seconds short (the model sometimes emits EOS early — and
# always does past ~62 min of input), transcribe the remainder as a new clip.
COVERAGE_SLACK_S = 240
MIN_CONTINUATION_PROGRESS_S = 60


def process_file(
    file_path: Path,
    input_dir: Path,
    output_dir: Path,
    server: str,
    part_seconds: int,
    request_timeout: int,
    emit_txt: bool,
    workdir: Path,
) -> dict:
    import librosa
    import soundfile

    rel = file_path.relative_to(input_dir)
    start_time = time.time()

    audio, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / SAMPLE_RATE

    # Work queue of (offset_s) positions still needing transcription. Each
    # round transcribes up to part_seconds from the offset; if the model
    # stops early (early EOS or the serve stack's ~62-min input cap), the
    # uncovered remainder is re-queued as a continuation. Speaker labels are
    # consistent within a clip and reset between clips ("part" index).
    segments: list[dict] = []
    offset = 0.0
    part_index = 0
    while duration - offset > COVERAGE_SLACK_S / 2:
        clip_dur = min(part_seconds, duration - offset)
        if offset == 0.0 and clip_dur >= duration:
            clip_path = file_path  # single-clip file: send as-is
        else:
            clip_path = workdir / f"{file_path.stem}.part{part_index:02d}.wav"
            lo = int(offset * SAMPLE_RATE)
            hi = int((offset + clip_dur) * SAMPLE_RATE)
            soundfile.write(str(clip_path), audio[lo:hi], SAMPLE_RATE)

        payload = transcribe_clip(server, clip_path, clip_dur, request_timeout)
        if clip_path != file_path:
            clip_path.unlink(missing_ok=True)
        clip_segments = parse_verbose_segments(payload, offset, part_index)
        segments.extend(clip_segments)

        covered = (
            (max(s["end"] for s in clip_segments) - offset) if clip_segments else 0.0
        )
        part_index += 1
        if clip_dur - covered <= COVERAGE_SLACK_S:
            offset += clip_dur  # clip fully covered — next part
        elif covered >= MIN_CONTINUATION_PROGRESS_S:
            logger.info(
                f"  {rel}: early stop at {offset + covered:.0f}s of "
                f"{offset + clip_dur:.0f}s — continuing from there"
            )
            offset += covered
        else:
            # No meaningful progress (e.g. trailing static) — skip this clip
            # to avoid looping; the gap is visible in the coverage stats.
            logger.warning(
                f"  {rel}: no progress on clip at {offset:.0f}s "
                f"({len(clip_segments)} segments) — skipping {clip_dur:.0f}s"
            )
            offset += clip_dur

    coverage_s = round(max((s["end"] for s in segments), default=0.0), 1)
    speakers_per_part = {
        p: sorted({s["speaker"] for s in segments if s["part"] == p and s["speaker"]})
        for p in sorted({s["part"] for s in segments})
    }
    record = {
        "file": str(rel),
        "model": MODEL,
        "duration_s": round(duration, 1),
        "coverage_s": coverage_s,
        "num_parts": part_index,
        "segments": segments,
        "num_segments": len(segments),
        "speakers_per_part": speakers_per_part,
    }

    json_path = output_dir / rel.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if emit_txt:
        write_txt(segments, json_path.with_suffix(".txt"))

    elapsed = time.time() - start_time
    logger.info(
        f"  {rel}: {duration / 60:.0f} min, {part_index} clip(s), "
        f"{len(segments)} segments, coverage {coverage_s / max(duration, 1) * 100:.0f}% "
        f"in {elapsed:.0f}s"
    )
    return {
        "file": str(rel),
        "duration_s": round(duration, 1),
        "coverage_s": coverage_s,
        "num_parts": part_index,
        "num_segments": len(segments),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Concurrent transcription + diarization via in-job sgl-omni server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for the full `hf jobs run` command.",
    )
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("output_dir", help="Directory to write transcript JSON files")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8000",
        help="sgl-omni server base URL (default: in-job localhost:8000)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent transcription requests (default: 4; KV-cache bound)",
    )
    parser.add_argument(
        "--part-minutes",
        type=float,
        default=DEFAULT_PART_SECONDS / 60,
        help="Split audio into parts of this length (default: 55 min — the "
        "longest that fits the model's context window with generation room). "
        "Speaker labels reset between parts.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=3600,
        help="Per-request timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--emit-txt", action="store_true", help="Also write .txt transcripts"
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Limit files (for testing)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_audio_files(input_dir)
    if not files:
        logger.error(f"No audio files found in {input_dir}")
        sys.exit(1)
    if args.max_files:
        files = files[: args.max_files]
    logger.info(f"Found {len(files)} audio file(s)")

    wait_for_server(args.server)

    part_seconds = int(args.part_minutes * 60)
    start_time = time.time()
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = {
                pool.submit(
                    process_file,
                    f,
                    input_dir,
                    output_dir,
                    args.server,
                    part_seconds,
                    args.request_timeout,
                    args.emit_txt,
                    workdir,
                ): f
                for f in files
            }
            for fut in concurrent.futures.as_completed(futures):
                f = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    logger.error(f"  {f.name} FAILED: {exc}")
                    results.append({"file": f.name, "error": str(exc)})

    elapsed = time.time() - start_time

    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in results if "error" not in r]
    total_audio = sum(r["duration_s"] for r in ok)
    elapsed_str = f"{elapsed / 60:.1f} min" if elapsed > 60 else f"{elapsed:.1f}s"
    logger.info("=" * 50)
    logger.info(f"Done! {len(ok)}/{len(files)} file(s) in {elapsed_str}")
    if total_audio:
        logger.info(f"  Audio: {total_audio / 3600:.1f} h total")
        logger.info(f"  RTFx: {total_audio / elapsed:.1f}x realtime (aggregate)")
    if len(ok) < len(files):
        logger.warning(f"  Failed: {len(files) - len(ok)} file(s) — see {summary_path}")
    logger.info(f"  Summary: {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Concurrent Transcription + Diarization (sgl-omni server)")
        print("=" * 60)
        print("\nDriver for an in-job sgl-omni server: splits long audio to")
        print("fit the model's context window, posts files concurrently,")
        print("writes JSON segments with timestamps + speaker labels.")
        print("\nSee module docstring for the full `hf jobs run` command")
        print("(server + driver in one job, a100-large recommended).")
        print("\nFor full help: uv run moss-transcribe-diarize-server.py --help")
        sys.exit(0)

    main()
