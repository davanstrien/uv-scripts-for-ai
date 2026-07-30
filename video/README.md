---
viewer: false
tags:
  - uv-script
  - video
  - video-text-to-text
  - video-captioning
  - temporal-grounding
---

# Video

Scripts for captioning and temporally grounding video files using HF Buckets and Jobs.

What the output looks like — a frame from *Joan Avoids a Cold* (1947, Prelinger Archives) with the event Marlin-2B produced for that moment:

![Example: film frame with its timestamped event caption](example-card.jpg)

## Quick Start

Scripts run directly from their Hub URL — no clone or local checkout needed:

```bash
# Caption every video in a bucket: dense scene captions + timestamped events
hf jobs uv run --image vllm/vllm-openai:latest --flavor a10g-small \
    -s HF_TOKEN \
    -v hf://buckets/user/my-videos:/input:ro \
    https://huggingface.co/datasets/uv-scripts/video/raw/main/marlin-caption.py \
    /input hf://buckets/user/my-videos/captions

# Temporal grounding: when does an event happen?
hf jobs uv run --image vllm/vllm-openai:latest --flavor a10g-small \
    -s HF_TOKEN \
    -v hf://buckets/user/my-videos:/input:ro \
    https://huggingface.co/datasets/uv-scripts/video/raw/main/marlin-caption.py \
    /input hf://buckets/user/out --find "a person enters the room"
```

## Scripts

### marlin-caption.py

Runs [NemoStation/Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B) (2B video
VLM, gated — accept the license on the model page first) over a directory of videos
via vLLM. Output is a resumable parquet dataset: one row per ~60s chunk with `scene`,
`caption`, and an `events` column of `<start - end>` descriptions in seconds.
Re-running skips completed rows; failed rows are recorded, not dropped
(`--retry-errors` re-attempts them).

Videos longer than ~60s are split into chunks and event timestamps offset back to
global film time. This is required for correct timestamps, not an optimisation:
Marlin was trained on short clips and compresses any input onto a ~60s timeline.

`--find "event"` switches to grounding mode: each chunk returns a candidate
`(span_start, span_end)`. Spans are candidates, not detections — the model cannot
say "not present", so filter or verify downstream. When the event is real, spans
are precise to fractions of a second.

**Cost**: ~3s of GPU per minute of film on `a10g-small` at batch scale — about
$0.05 per hour of footage.

**Memory**: defaults encode a measured config (`--mm-processor-cache-gb 0`, in-flight
window capped at 24). vLLM's multimodal cache grows without bound on distinct videos
and will OOM a 15 GB node if re-enabled. On `a10g-large` and up, `--window-max 64`
is safe.

Run `--help` on the script for all options.
