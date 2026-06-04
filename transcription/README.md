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

**`cohere-transcribe.py`** (recommended for plain text) — uses `model.transcribe()` with automatic long-form chunking, overlap, and reassembly. Stable dependencies.

**`cohere-transcribe-vllm.py`** — experimental vLLM variant. Faster but requires nightly vLLM and has minor duplication at chunk boundaries.

**`easytranscriber-transcribe.py`** — when you need **word-level timestamps** (subtitles, search indexing, forced alignment). Runs VAD → ASR → wav2vec2 emissions → forced alignment. Defaults to the Cohere backend so you get the same model as the other scripts with alignment on top; swap to `--backend ct2` + a Whisper model for languages Cohere doesn't cover (e.g. Swedish via `KBLab/kb-whisper-large`).

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

### Data

| Script | Description |
|--------|-------------|
| `download-ia.py` | Download audio from Internet Archive into a mounted bucket |

## Notes

- **Gated model**: Accept terms at the [model page](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) before use.
- **Tokenizer workaround**: `cohere-transcribe.py` applies a one-line patch for a tokenizer compat issue. Will be removed once upstream fixes land ([model discussion](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/11)).
- **easytranscriber**: the Cohere backend requires `transformers>=5.4.0` (pinned in the script). Pyannote VAD is gated — accept terms at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) if using `--vad pyannote`. Otherwise stick with the default Silero VAD.
