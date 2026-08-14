# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]",
#     "vllm",
# ]
# ///

"""
Caption whole videos — no chunking — with Qwen3.8-27B, writing a resumable dataset.

Qwen3.8-27B (Apache 2.0, not gated) ingests videos natively at hour scale: an
11-minute film is one request, and in --timestamps mode the model returns a
timeline of <start - end> events whose times are read off per-frame time tags
(verified accurate to the ~2s sampling interval, with on-screen text quoted
verbatim). This is the structural difference from marlin-caption.py, which must
split anything over ~60s into chunks.

Serves the model on vLLM inside the job and pumps every video in INPUT_DIR
through it with saturate: crash-safe parquet out, exact resume (re-running
skips completed videos), congestion-aware concurrency.

Input:                              Output (one parquet dataset):
  /input/film.mp4   (11 min)  ->      1 row per video — columns: video,
  /input/clip.mp4   (45 s)    ->      duration_s, caption, events (JSON,
                                      --timestamps mode), token counts

Examples:

  # Caption a bucket of videos on HF Jobs
  hf jobs uv run --image vllm/vllm-openai:latest --flavor a100-large \\
      -s HF_TOKEN \\
      -v hf://buckets/user/my-videos:/input:ro \\
      qwen38-caption.py /input hf://buckets/user/my-videos/captions

  # Timestamped event timeline per film instead of a prose caption
  ... qwen38-caption.py /input hf://buckets/user/out --timestamps

  # Local machine with a CUDA GPU (needs ~60 GB VRAM)
  uv run qwen38-caption.py ./videos ./captions-out

Thinking mode is OFF by default (captioning gains little from it and it
multiplies latency); --thinking turns it back on with a generous budget and
the model card's thinking-mode sampling params.

Model: Qwen/Qwen3.8-27B (27B dense, hybrid GDN linear attention, vision tower).
bf16 weights are ~52 GB — a100-large is the smallest sensible Jobs flavor;
h200 generates ~2.3x faster if latency matters.
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen3.8-27B"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}

CAPTION_PROMPT = (
    "Describe this video: the setting, the people, what happens over its course, "
    "and any text that appears on screen."
)
TIMESTAMPS_PROMPT = (
    "Watch this video and produce a timeline of events covering the WHOLE video "
    "from start to finish. For each event give the time range as <start - end> "
    "in seconds (e.g. <95.0 - 112.5>) and a one-sentence description. "
    "Include title cards, on-screen text (quoted), and scene changes."
)

THINK = re.compile(r"<think>.*?</think>\s*|^.*?</think>\s*", re.DOTALL)
EVENT_LINE = re.compile(r"<(\d+\.?\d*)\s*-\s*(\d+\.?\d*)>\s*[:–\-]?\s*(.+)")


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120)
        return round(float(out.stdout.strip()), 2)
    except (ValueError, subprocess.SubprocessError):
        return None


def parse_events(text: str) -> list[dict]:
    return [{"start": float(a), "end": float(b), "text": d.strip()}
            for a, b, d in EVENT_LINE.findall(text)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="Directory of videos (e.g. a mounted bucket)")
    parser.add_argument("output", help="Dataset output: hf://datasets/..., hf://buckets/..., or local path")
    parser.add_argument("--timestamps", action="store_true",
                        help="Return a <start - end> event timeline per video instead of a prose caption")
    parser.add_argument("--prompt", help="Override the built-in prompt")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode (slower; uses the model card's thinking sampling params)")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frame sampling rate passed to the processor (default 2)")
    parser.add_argument("--max-videos", type=int, help="Only process the first N videos (testing)")
    parser.add_argument("--window-max", type=int, default=8,
                        help="Max in-flight requests (default 8 — whole-video requests are large)")
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-attempt rows that errored in a previous run")
    args = parser.parse_args()

    if shutil.which("ffprobe") is None:
        sys.exit("ffprobe not found — use the vllm/vllm-openai image (has it) or install ffmpeg")

    input_root = Path(args.input_dir).resolve()
    videos = sorted(p for p in input_root.rglob("*")
                    if p.suffix.lower() in VIDEO_EXTENSIONS)
    if args.max_videos:
        videos = videos[: args.max_videos]
    if not videos:
        sys.exit(f"no videos found under {input_root}")
    logger.info("found %d videos", len(videos))

    rows = [(str(p.relative_to(input_root)),
             {"video": str(p.relative_to(input_root)), "path": str(p),
              "duration_s": probe_duration(p)})
            for p in videos]

    prompt = args.prompt or (TIMESTAMPS_PROMPT if args.timestamps else CAPTION_PROMPT)
    # Timeline output scales with video length; measured 9.6K tokens for an
    # 11-min film — leave generous headroom. Thinking needs its own budget on top.
    max_tokens = 16384 if args.timestamps else 4096
    if args.thinking:
        max_tokens += 32768

    def to_request(row: dict) -> dict:
        req = {
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"file://{row['path']}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": max_tokens,
        }
        if args.thinking:  # model-card sampling params per mode
            req.update({"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        else:
            req.update({"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False}})
        return req

    def parse(row: dict, resp: dict) -> dict:
        text = THINK.sub("", resp["choices"][0]["message"]["content"] or "").strip()
        usage = resp.get("usage") or {}
        out = {"video": row["video"], "duration_s": row["duration_s"],
               "caption": text,
               "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens")}
        if args.timestamps:
            out["events"] = json.dumps(parse_events(text))
        return out

    from saturate import Auto, Engine, pump

    with Engine(
        MODEL,
        engine="vllm",
        extra_args=[
            "--allowed-local-media-path", str(input_root),
            "--max-model-len", str(args.max_model_len),
            "--mm-processor-kwargs",
            json.dumps({"fps": args.fps, "do_sample_frames": True}),
            # Unbounded growth on distinct videos — OOM-kills the node if left on.
            "--mm-processor-cache-gb", "0",
        ],
    ) as endpoint:
        stats = pump(
            rows,
            to_request=to_request,
            parse=parse,
            endpoint=endpoint,
            output=args.output,
            window=Auto(initial=4, max_limit=args.window_max),
            retry_errors=args.retry_errors,
        )

    logger.info("done: %d ok, %d failed, %.1f tok/s (window settled at %d)",
                stats.rows_processed, stats.rows_failed,
                stats.tokens_per_sec, stats.final_limit)
    if stats.rows_failed:
        logger.warning("failed rows are recorded in the output; re-run with "
                       "--retry-errors to attempt them again")


if __name__ == "__main__":
    main()
