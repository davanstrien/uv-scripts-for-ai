---
viewer: false
tags:
  - uv-script
  - audio
  - transcription
  - automatic-speech-recognition
private: true
---

# Transcription

Scripts for transcribing audio files using HF Buckets and Jobs.

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

**`cohere-transcribe.py`** (recommended for plain text) — uses `model.transcribe()` with automatic long-form chunking, overlap, and reassembly. Stable dependencies.

**`cohere-transcribe-vllm.py`** — experimental vLLM variant. Faster but requires nightly vLLM and has minor duplication at chunk boundaries.

**`easytranscriber-transcribe.py`** — when you need **word-level timestamps** (subtitles, search indexing, forced alignment). Runs VAD → ASR → wav2vec2 emissions → forced alignment. Defaults to the Cohere backend so you get the same model as the other scripts with alignment on top; swap to `--backend ct2` + a Whisper model for languages Cohere doesn't cover (e.g. Swedish via `KBLab/kb-whisper-large`).

**`moss-transcribe-diarize.py`** — when you need **who said what**. Joint transcription + speaker diarization + timestamps in one generation pass (no separate ASR/diarization/alignment stages). 128k context handles up to ~90 min of audio per file internally, so files are never pre-chunked and the anonymous speaker labels (`[S01]`, `[S02]`, ...) stay consistent across the whole recording. English + Chinese; also accepts video containers (mp4, mkv, ...); supports hotword biasing for names/jargon.

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

### Data

| Script | Description |
|--------|-------------|
| `download-ia.py` | Download audio from Internet Archive into a mounted bucket |

## Notes

- **Gated model**: Accept terms at the [model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) before use.
- **Tokenizer workaround**: `cohere-transcribe.py` applies a one-line patch for a tokenizer compat issue. Will be removed once upstream fixes land ([model discussion](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/11)).
- **easytranscriber**: the Cohere backend requires `transformers>=5.4.0` (pinned in the script). Pyannote VAD is gated — accept terms at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) if using `--vad pyannote`. Otherwise stick with the default Silero VAD.
- **moss-transcribe-diarize**: not gated (Apache 2.0). Uses `trust_remote_code=True` (model code lives in the [model repo](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)); inference helpers install from the model's [GitHub repo](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) pinned to a commit (the package isn't on PyPI). Generation time scales with audio length — long multi-speaker recordings emit tens of thousands of tokens, so watch the `truncated` flag in the output JSON.
- **Serving MOSS-Transcribe-Diarize**: upstream's production path is an OpenAI-compatible `/v1/audio/transcriptions` server via [sglang-omni](https://github.com/sgl-project/sglang-omni) (no offline engine — high-concurrency batch goes through the server). Not runnable on Jobs yet: as of 2026-07-09 the only published image (`lmsysorg/sglang-omni:dev`, 2026-06-16) predates the model's support (added 2026-07-04) and the `sgl-omni` CLI, and the newer code needs a newer core `sglang` than the image ships. Revisit when a fresh image lands.
