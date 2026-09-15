# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "transformers>=4.57,<6",
#     "torch==2.8.0",
#     "huggingface-hub",
#     "soundfile",
#     "librosa",
#     "numpy",
# ]
# ///

"""
Transcribe audio files in one of 102 languages with a BuzzASR monolingual model.

BuzzASR (lemn-lab, Findings of EMNLP 2026) is one Whisper-large-v3 fine-tune
per FLEURS language, published as BuzzASR/<language> on the Hub. Pick the
language and the script loads that checkpoint; the language/task prompt is
baked into each model, so nothing is auto-detected and nothing can drift to
the wrong language.

Long audio: --long-form picks how files longer than Whisper's 30 s window are
decoded. `sequential` (default) is Whisper's own long-form algorithm: timestamp
tokens place each next window and degenerate windows are re-decoded at higher
temperature, so nothing is skipped. `chunked` is the transformers pipeline
(overlapping 30 s windows merged on the overlap): ~3.5x faster, but the merge
drops sentences when a window decodes badly (measured 14% fewer words on
Dutch audiobooks). Files are batched together; audio decoding runs in a thread
pool while the model loads.

Designed to work with HF Buckets mounted as volumes via `hf jobs uv run -v ...`.

Input:                              Output:
  /input/episode1.mp3         ->      /output/episode1.txt
  /input/sub/clip.wav         ->      /output/sub/clip.txt

Examples:

  # Local test (requires CUDA GPU)
  uv run buzzasr-transcribe.py ./test-audio ./test-output --language dutch

  # HF Jobs with bucket volumes
  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\
      -e UV_TORCH_BACKEND=cu128 \\
      -v hf://buckets/user/audio-input:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      buzzasr-transcribe.py /input /output --language dutch

  # List the languages
  uv run buzzasr-transcribe.py --list-languages

Models: https://huggingface.co/BuzzASR (MIT), 1.55B params each, fp16 safetensors
  - `--language` takes the Hub slug (dutch, sorani-kurdish, ...) or its ISO code (nl, ckb, ...)
  - `--model` runs any other Whisper checkpoint through the same loop (e.g. openai/whisper-large-v3
    as a zero-shot control)
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

SAMPLE_RATE = 16000
# Whisper's receptive field; the pipeline overlaps windows by CHUNK_SECONDS / 6.
CHUNK_SECONDS = 30
# Audio buffered per generate call (float32 at 16 kHz is ~3.8 MB per minute).
GROUP_SECONDS = 7200

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus"}

# Hub slug -> language code from each model card's `language:` field (2026-09-15).
LANGUAGES = {
    "afrikaans": "af",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "asturian": "ast",
    "azerbaijani": "az",
    "belarusian": "be",
    "bengali": "bn",
    "bosnian": "bs",
    "bulgarian": "bg",
    "burmese": "my",
    "cantonese": "yue",
    "catalan": "ca",
    "cebuano": "ceb",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "filipino": "fil",
    "finnish": "fi",
    "french": "fr",
    "fulah": "ff",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "hausa": "ha",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "igbo": "ig",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jv",
    "kabuverdianu": "kea",
    "kamba": "kam",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "kyrgyz": "ky",
    "lao": "lo",
    "latvian": "lv",
    "lingala": "ln",
    "lithuanian": "lt",
    "luganda": "lg",
    "luo": "luo",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "mandarin": "cmn",
    "maori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "nepali": "ne",
    "northern-sotho": "nso",
    "norwegian": "nb",
    "nyanja": "ny",
    "occitan": "oc",
    "oriya": "or",
    "oromo": "om",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "shona": "sn",
    "sindhi": "sd",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "sorani-kurdish": "ckb",
    "spanish": "es",
    "swahili": "sw",
    "swedish": "sv",
    "tajik": "tg",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "umbundu": "umb",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "wolof": "wo",
    "xhosa": "xh",
    "yoruba": "yo",
    "zulu": "zu",
}
CODE_TO_SLUG = {code: slug for slug, code in LANGUAGES.items()}

# From the model cards' usage snippet (greedy + anti-loop settings).
GENERATE_KWARGS = {"num_beams": 1, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2}
# Whisper's long-form fallbacks (from the transformers Whisper docs): retry a
# window at higher temperature when the output looks degenerate.
SEQUENTIAL_KWARGS = {
    "return_timestamps": True,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "logprob_threshold": -1.0,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
}


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")


def resolve_model(language: str | None, model: str | None) -> str:
    """Map --language (slug or ISO code) to BuzzASR/<slug>, unless --model overrides."""
    if model:
        return model
    if not language:
        logger.error("--language is required (or pass --model). See --list-languages.")
        sys.exit(1)
    key = language.strip().lower()
    slug = key if key in LANGUAGES else CODE_TO_SLUG.get(key)
    if slug is None:
        logger.error(f"Unknown language {language!r}. See --list-languages.")
        sys.exit(1)
    return f"BuzzASR/{slug}"


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


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_generation_config(model_id: str):
    """Reload generation_config.json without the `_from_model_config` flag.

    The full-fine-tune BuzzASR checkpoints (native tokenizer) ship a minimal
    generation config saved with `"_from_model_config": true`. transformers
    treats such a file as auto-derived and drops every extra field on load, so
    `lang_to_id`, `language`, `task` and `no_timestamps_token_id` vanish: the
    language prompt is no longer forced and long-form generation errors out.
    Rebuilding from the raw JSON keeps the authors' settings. Harmless for the
    simple-fine-tune checkpoints, which ship the full Whisper config.
    """
    from huggingface_hub import hf_hub_download
    from transformers import GenerationConfig

    local = Path(model_id) / "generation_config.json"
    path = (
        local
        if local.is_file()
        else hf_hub_download(model_id, "generation_config.json")
    )
    raw = json.loads(Path(path).read_text())
    was_flagged = raw.pop("_from_model_config", False)
    config = GenerationConfig(**raw)
    if not hasattr(config, "max_initial_timestamp_index"):
        config.max_initial_timestamp_index = 50  # Whisper default
    config.is_multilingual = True
    if was_flagged:
        logger.info(
            "generation_config.json was flagged _from_model_config; rebuilt it so the "
            f"language prompt ({config.language}) and timestamp ids are kept"
        )
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with a BuzzASR monolingual Whisper-large-v3 fine-tune.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run buzzasr-transcribe.py ./audio ./output --language dutch
  uv run buzzasr-transcribe.py ./audio ./output --language nl --batch-size 32
  uv run buzzasr-transcribe.py ./audio ./output --model openai/whisper-large-v3   # zero-shot control

HF Jobs with bucket volumes:
  hf jobs uv run --flavor l4x1 -s HF_TOKEN -e UV_TORCH_BACKEND=cu128 \\
      -v hf://buckets/user/audio-bucket:/input:ro \\
      -v hf://buckets/user/transcripts:/output \\
      buzzasr-transcribe.py /input /output --language dutch
        """,
    )
    parser.add_argument("input_dir", nargs="?", help="Directory containing audio files")
    parser.add_argument(
        "output_dir", nargs="?", help="Directory to write transcript text files"
    )
    parser.add_argument(
        "--language",
        help="BuzzASR language: Hub slug (dutch, sorani-kurdish) or ISO code (nl, ckb)",
    )
    parser.add_argument(
        "--model",
        help="Override the checkpoint (any Whisper model on the Hub); --language is then ignored",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Print the 102 supported languages and exit",
    )
    parser.add_argument(
        "--long-form",
        choices=["sequential", "chunked"],
        default="sequential",
        help="How files longer than 30 s are decoded: Whisper's sequential algorithm "
        "(complete) or the transformers chunked pipeline (faster, drops text) (default: sequential)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Files per generate call (sequential) or 30 s windows per forward pass (chunked) (default: 16)",
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

    if args.list_languages:
        for slug, code in sorted(LANGUAGES.items()):
            print(f"{slug:16s} {code:4s} BuzzASR/{slug}")
        sys.exit(0)

    if not args.input_dir or not args.output_dir:
        parser.error("input_dir and output_dir are required")

    model_id = resolve_model(args.language, args.model)

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

    # Decode in a thread pool while the model loads; files are grouped as they
    # land and each group is transcribed as one batched call.
    logger.info(f"Decoding audio with {args.decode_workers} worker(s)...")
    decode_start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    futures = {pool.submit(load_audio, path): fi for fi, path in enumerate(files)}

    logger.info(f"Loading {model_id}...")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16
    )
    model.generation_config = load_generation_config(model_id)
    model.to("cuda:0").eval()
    asr = None
    if args.long_form == "chunked":
        from transformers import pipeline

        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=CHUNK_SECONDS,
            batch_size=args.batch_size,
            device=0,
        )
    logger.info("Model loaded")

    durations = [0.0] * len(files)
    texts: dict[int, str] = {}
    gpu_time = 0.0
    group: list[tuple[int, np.ndarray]] = []
    group_seconds = 0.0

    def generate_batch(audios: list[np.ndarray]) -> list[str]:
        """Feature-extract + generate + decode one batch of whole files."""
        inputs = processor(
            audios,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to("cuda:0", torch.float16)
        attention_mask = inputs.attention_mask.to("cuda:0")
        if input_features.shape[-1] <= 3000:  # every file fits one window: short-form
            ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                **GENERATE_KWARGS,
            )
            return [
                t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)
            ]
        # Decode segment by segment: the fine-tunes start a segment without a
        # leading-space token, so decoding the whole sequence at once glues the
        # last word of one segment to the first word of the next.
        out = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            return_segments=True,
            **GENERATE_KWARGS,
            **SEQUENTIAL_KWARGS,
        )
        texts = []
        for segments in out["segments"]:
            parts = processor.batch_decode(
                [seg["tokens"].tolist() for seg in segments], skip_special_tokens=True
            )
            texts.append(" ".join(t.strip() for t in parts if t.strip()))
        return texts

    def run_group():
        nonlocal gpu_time
        t = time.time()
        if args.long_form == "chunked":
            inputs = [
                {"raw": audio, "sampling_rate": SAMPLE_RATE} for _, audio in group
            ]
            outputs = asr(inputs, generate_kwargs=GENERATE_KWARGS)
            for (fi, _), out in zip(group, outputs):
                texts[fi] = out["text"].strip()
        else:
            for batch in batched(group, args.batch_size):
                decoded = generate_batch([audio for _, audio in batch])
                for (fi, _), text in zip(batch, decoded):
                    texts[fi] = text
        torch.cuda.synchronize()
        gpu_time += time.time() - t
        logger.info(
            f"  group of {len(group)} file(s), {group_seconds / 60:.1f} min audio done "
            f"({len(texts)}/{len(files)} files)"
        )

    logger.info(
        f"Transcribing (long_form={args.long_form}, batch_size={args.batch_size}, {model_id})..."
    )
    torch.cuda.reset_peak_memory_stats()
    decode_time = None
    with torch.inference_mode():
        for fut in as_completed(futures):
            fi = futures[fut]
            audio = fut.result()
            durations[fi] = len(audio) / SAMPLE_RATE
            group.append((fi, audio))
            group_seconds += durations[fi]
            if all(f.done() for f in futures) and decode_time is None:
                decode_time = time.time() - decode_start
            if group_seconds >= GROUP_SECONDS:
                run_group()
                group.clear()
                group_seconds = 0.0
        if group:
            run_group()
            group.clear()
    pool.shutdown()
    decode_time = decode_time or (time.time() - decode_start)
    total_audio = sum(durations)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # Write outputs
    results = []
    for fi, path in enumerate(files):
        text = texts[fi]
        rel = path.relative_to(input_dir)
        txt_path = output_dir / rel.with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(text, encoding="utf-8")
        results.append(
            {
                "file": str(rel),
                "model": model_id,
                "long_form": args.long_form,
                "duration_s": round(durations[fi], 1),
                "transcript_length": len(text),
                "word_count": len(text.split()),
            }
        )
        logger.info(
            f"  {rel} -> {txt_path.name} ({len(text.split())} words, {durations[fi]:.0f}s audio)"
        )

    summary_path = output_dir / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)

    wall = time.time() - wall_start
    logger.info("=" * 50)
    logger.info(
        f"Done! Transcribed {len(files)} file(s), {total_audio / 60:.1f} min of audio with {model_id}"
    )
    logger.info(f"  Output: {output_dir}")
    logger.info(
        f"  Audio decode (overlapped with model load + GPU): {decode_time:.1f}s"
    )
    logger.info(
        f"  GPU busy (features+generate+decode): {gpu_time:.1f}s -> RTFx {total_audio / gpu_time:.0f}x"
    )
    logger.info(
        f"  Wall (incl. decode + model load): {wall:.1f}s -> RTFx {total_audio / wall:.0f}x"
    )
    logger.info(
        f"  Peak GPU memory: {peak_gb:.2f} GB "
        f"(long_form={args.long_form}, batch_size={args.batch_size})"
    )
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
        print(
            "Audio Transcription with BuzzASR (102 monolingual Whisper-large-v3 fine-tunes)"
        )
        print("=" * 60)
        print("\nTranscribe audio files from a directory -> text files.")
        print("One model per language; long audio chunked and batched.")
        print("Designed for HF Buckets mounted as volumes.")
        print()
        print("Usage:")
        print("  uv run buzzasr-transcribe.py INPUT_DIR OUTPUT_DIR --language dutch")
        print("  uv run buzzasr-transcribe.py --list-languages")
        print()
        print("HF Jobs with bucket volumes:")
        print("  hf jobs uv run --flavor l4x1 -s HF_TOKEN -e UV_TORCH_BACKEND=cu128 \\")
        print("      -v hf://buckets/user/audio-input:/input:ro \\")
        print("      -v hf://buckets/user/transcripts:/output \\")
        print("      buzzasr-transcribe.py /input /output --language dutch")
        print()
        print("For full help: uv run buzzasr-transcribe.py --help")
        sys.exit(0)

    main()
