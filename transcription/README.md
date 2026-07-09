---
viewer: false
tags:
  - uv-script
  - audio
  - transcription
  - automatic-speech-recognition
  - speaker-diarization
---

# Transcription

Scripts for transcribing — and diarizing — audio files using HF Buckets and Jobs.

## Quick Start

Scripts run directly from their Hub URL — no clone or local checkout needed:

```bash
# 1. Download audio from Internet Archive straight into a bucket
hf jobs uv run \
    -v hf://buckets/user/audio-files:/output \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/download-ia.py \
    SUSPENSE /output

# 2. Transcribe — audio bucket in, transcript bucket out
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    -e UV_TORCH_BACKEND=cu128 \
    -v hf://buckets/user/audio-files:/input:ro \
    -v hf://buckets/user/transcripts:/output \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/cohere-transcribe.py \
    /input /output --language en --compile

# Or: transcribe + figure out WHO said what (speaker diarization + timestamps)
hf jobs uv run --flavor l4x1 -s HF_TOKEN \
    -e UV_TORCH_BACKEND=cu128 \
    -v hf://buckets/user/audio-files:/input:ro \
    -v hf://buckets/user/transcripts:/output \
    https://huggingface.co/datasets/uv-scripts/transcription/raw/main/moss-transcribe-diarize.py \
    /input /output --emit-txt
```

No download/upload step. Buckets are mounted directly as volumes via [hf-mount](https://github.com/huggingface/hf-mount).

> **Local dev**: if you've cloned this repo, swap the URL for the local filename (e.g. `cohere-transcribe.py /input /output ...`).

## Scripts

### Transcription

| Script | Model | Backend | Output | Speed |
|--------|-------|---------|--------|-------|
| `cohere-transcribe.py` | Cohere Transcribe (2B) | transformers | `.txt` | 161x RT (A100) |
| `cohere-transcribe-vllm.py` | Cohere Transcribe (2B) | vLLM nightly | `.txt` | 214x RT (A100) |
| `easytranscriber-transcribe.py` | Cohere Transcribe 2B (default) or Whisper variants | [easytranscriber](https://github.com/kb-labb/easytranscriber) | JSON word timestamps (+ optional `.txt` / `.srt`) | 42.9x RT (L4) |
| `moss-transcribe-diarize.py` | [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) (0.9B) | transformers (remote code) | JSON speaker segments `{start, end, speaker, text}` (+ optional `.txt` / `.srt`) | 3.2x RT (A10G, 74-min file) |
| `moss-transcribe-diarize-server.py` | [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) (0.9B) | in-job sgl-omni server | JSON speaker segments (+ optional `.txt`) | 37.8x RT aggregate (A100, 4 streams) |

**`cohere-transcribe.py`** (recommended for plain text) — uses `model.transcribe()` with automatic long-form chunking, overlap, and reassembly. Stable dependencies.

**`cohere-transcribe-vllm.py`** — experimental vLLM variant. Faster but requires nightly vLLM and has minor duplication at chunk boundaries.

**`easytranscriber-transcribe.py`** — when you need **word-level timestamps** (subtitles, search indexing, forced alignment). Runs VAD → ASR → wav2vec2 emissions → forced alignment. Defaults to the Cohere backend so you get the same model as the other scripts with alignment on top; swap to `--backend ct2` + a Whisper model for languages Cohere doesn't cover (e.g. Swedish via `KBLab/kb-whisper-large`).

**`moss-transcribe-diarize.py`** — when you need **who said what**. Joint transcription + speaker diarization + timestamps in one generation pass (no separate ASR/diarization/alignment stages). 128k context handles up to ~90 min of audio per file internally, so files are never pre-chunked and the anonymous speaker labels (`[S01]`, `[S02]`, ...) stay consistent across the whole recording. English + Chinese; also accepts video containers (mp4, mkv, ...); supports hotword biasing for names/jargon.

**`moss-transcribe-diarize-server.py`** — same model at **~12x the throughput** for many files: serves it behind sglang-omni inside the job and posts files concurrently (continuous batching). Splits long tapes into ≤55-min clips to fit the context window (speaker labels reset between clips, recorded as `part` on each segment) and automatically continues from the last timestamp when the model stops early, reporting `coverage_s` per file. Needs `a100-large` — 24 GB cards OOM on long-tape KV. The `hf jobs run` command lives in the script docstring (server + driver in one job).

#### Options — `cohere-transcribe.py` / `cohere-transcribe-vllm.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | required | en, de, fr, it, es, pt, el, nl, pl, ar, vi, zh, ja, ko |
| `--compile` | off | torch.compile encoder (one-time warmup, faster after) |
| `--batch-size` | 16 | Batch size for inference |
| `--max-files` | all | Limit files to process (for testing) |

#### Options — `easytranscriber-transcribe.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--language` | required | ISO 639-1 code. Cohere supports the same 14 languages as above; ct2/hf support any Whisper language |
| `--backend` | `cohere` | `cohere`, `ct2` (CTranslate2 Whisper, fastest for Whisper), or `hf` (transformers) |
| `--transcription-model` | Cohere 2B / distil-whisper-large-v3.5 | HF model ID; override to use KB-Whisper, Whisper-large-v3, etc. |
| `--emissions-model` | per-language default | wav2vec2 for forced alignment: en→`wav2vec2-base-960h`, sv→`voxrex-swedish`, else→`facebook/mms-1b-all` |
| `--vad` | `silero` | `silero` (no auth) or `pyannote` (requires accepting terms + HF_TOKEN) |
| `--tokenizer-lang` | derived from `--language` | NLTK Punkt language name for sentence tokenization |
| `--emit-txt` | off | Also write `.txt` transcripts alongside the JSON alignments |
| `--emit-srt` | off | Also write `.srt` subtitles derived from alignment segments |
| `--batch-size-features` | 8 | Feature-extraction batch size |
| `--batch-size-transcribe` | 16 | ASR batch size (where backend supports it) |
| `--max-files` | all | Limit files to process (for testing) |

#### Options — `moss-transcribe-diarize-server.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--concurrency` | 4 | Concurrent transcription requests (KV-cache bound; 6 works on A100) |
| `--part-minutes` | 55 | Clip length for splitting long audio (longest that fits the context window). Speaker labels reset between clips |
| `--server` | `http://127.0.0.1:8000` | sgl-omni server URL (in-job localhost by default) |
| `--request-timeout` | 3600 | Per-request timeout in seconds |
| `--emit-txt` | off | Also write `.txt` transcripts |
| `--max-files` | all | Limit files to process (for testing) |

#### Options — `moss-transcribe-diarize.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--max-new-tokens` | 0 (auto) | Token budget per file. Auto scales with audio duration (min 5120, max 65536). Output JSON sets `"truncated": true` if the budget was hit — re-run with a higher value |
| `--hotwords` | none | Comma-separated terms (names, companies, jargon) appended to the prompt to bias recognition |
| `--prompt` | built-in | Full prompt override (replaces the built-in transcribe+diarize prompt) |
| `--emit-txt` | off | Also write `.txt` transcripts (`[start - end] SPEAKER: text` per line) |
| `--emit-srt` | off | Also write `.srt` subtitles with speaker prefixes |
| `--max-files` | all | Limit files to process (for testing) |

#### Benchmarks

CBS Suspense (1940s radio drama), 66 episodes, 33 hours of audio.

**`cohere-transcribe.py`** (plain text):

| GPU | Time | RTFx |
|-----|------|------|
| A100-SXM4-80GB | 12.3 min | 161x realtime |
| L4 | ~64s / 30 min episode | 28x realtime |

**`easytranscriber-transcribe.py`** (JSON alignments + optional .txt/.srt; VAD → ASR → wav2vec2 → forced alignment):

| GPU | Time | RTFx | Output |
|-----|------|------|--------|
| L4 | 46.2 min | 42.9x realtime | 66 JSON + SRT + TXT (42,633 segments, 295k words) |

**`moss-transcribe-diarize.py`** (Apollo 11 mission audio, one 74-min multi-speaker tape):

| GPU | Time | RTFx | Output |
|-----|------|------|--------|
| A10G | 23.3 min | 3.2x realtime | 743 segments, 7 speakers, 23k tokens, no truncation |

Long files are generation-bound (the whole diarized transcript is decoded in one pass), so RTFx drops as recordings grow; short clips run far faster. `l4x1` also works — same class of GPU.

**`moss-transcribe-diarize-server.py`** (Apollo 11 mission audio, 3 tapes / 2.7 h, concurrency 4):

| GPU | Time | RTFx | Output |
|-----|------|------|--------|
| A100 | 4.3 min | 37.8x realtime aggregate | 1,697 segments, 96–99% coverage per tape |

### Data

| Script | Description |
|--------|-------------|
| `download-ia.py` | Download audio from Internet Archive into a mounted bucket |

## Serve as a live endpoint

MOSS-Transcribe-Diarize can be served as an OpenAI-compatible transcription API on Jobs (see [Serving on Jobs](https://huggingface.co/docs/hub/jobs-serving)). Two verified paths; both expose `/v1/audio/transcriptions` at `https://<job-id>--8000.hf.jobs` (requests need an HF token with read access to your namespace).

**vLLM** (simplest; returns the raw `[start][Sxx]text[end]` transcript). Model support was merged into vLLM on 2026-07-08 — after the v0.24.0 release — so use a commit-pinned nightly image until the next release (then plain `vllm/vllm-openai:latest` works). The image needs the audio extras installed first:

```bash
hf jobs run --detach --expose 8000 --flavor l4x1 -s HF_TOKEN --timeout 2h \
    vllm/vllm-openai:nightly-2c17d33f4291a55b447317640c81eb61077b1b00 -- \
    bash -c "pip install librosa soundfile && vllm serve OpenMOSS-Team/MOSS-Transcribe-Diarize --trust-remote-code"
```

Parse the response `text` into segments with `parse_transcript` from the model's [GitHub package](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize).

**sglang-omni** (upstream's recommended production server; adds `response_format=verbose_json` with parsed speaker segments). Its prebuilt image predates this model, so install it at job start over a current [sglang nightly image](https://hub.docker.com/r/lmsysorg/sglang/tags) — the tag below is the tested pin; newer nightlies should also work. The clone + fresh-venv sequence is [upstream's documented install](https://github.com/sgl-project/sglang-omni/blob/main/docs/get_started/installation.md) and adds under a minute:

```bash
hf jobs run --detach --expose 8000 --flavor l4x1 -s HF_TOKEN --timeout 2h \
    lmsysorg/sglang:nightly-dev-cu13-20260709-074bb928 -- \
    bash -c "pip install -q uv 2>/dev/null; git clone --depth 1 https://github.com/sgl-project/sglang-omni.git && cd sglang-omni && uv venv .venv -p 3.12 && . .venv/bin/activate && uv pip install . && sgl-omni serve --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize --host 0.0.0.0 --port 8000 --max-running-requests 16 --mem-fraction-static 0.80"
```

Query either server the same way (`response_format=verbose_json` on sglang-omni only; raise `max_new_tokens`, e.g. `65536`, for long recordings). **Caution — long audio on the serve path**: sglang-omni silently truncates input past ~62 minutes (observed deterministically at ~62.5 min on the same tape that transcribes fully via the transformers recipe, regardless of `max_new_tokens`). Split longer recordings before posting — `moss-transcribe-diarize-server.py` does this automatically:

```bash
curl -X POST https://<job-id>--8000.hf.jobs/v1/audio/transcriptions \
    -H "Authorization: Bearer $HF_TOKEN" \
    -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
    -F file=@meeting.wav \
    -F response_format=verbose_json \
    -F max_new_tokens=65536
```

The job — and its billing — stops at `--timeout` or `hf jobs cancel <job_id>`.

## Notes

- **Gated model**: Accept terms at the [model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) before use.
- **Tokenizer workaround**: `cohere-transcribe.py` applies a one-line patch for a tokenizer compat issue. Will be removed once upstream fixes land ([model discussion](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/11)).
- **easytranscriber**: the Cohere backend requires `transformers>=5.4.0` (pinned in the script). Pyannote VAD is gated — accept terms at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) if using `--vad pyannote`. Otherwise stick with the default Silero VAD.
- **moss-transcribe-diarize**: not gated (Apache 2.0). Uses `trust_remote_code=True` (model code lives in the [model repo](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)); inference helpers install from the model's [GitHub repo](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) pinned to a commit (the package isn't on PyPI). Generation time scales with audio length — long multi-speaker recordings emit tens of thousands of tokens, so watch the `truncated` flag in the output JSON.
- **sglang-omni install**: the `lmsysorg/sglang-omni:dev` image predates this model — don't use it. Install from git over a current `lmsysorg/sglang` nightly image as shown above, and follow the clone + fresh-venv sequence exactly: installing outside the repo skips its `[tool.uv]` dependency overrides and fails on a protobuf conflict. Once they publish a fresh image, `sgl-omni serve` alone will do.
