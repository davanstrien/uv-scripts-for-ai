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
baked into each model, so the model is never asked to detect the language.

Long audio: --long-form picks how files longer than Whisper's 30 s window are
decoded. `sequential` (default) is Whisper's own long-form algorithm: timestamp
tokens place each next window and degenerate windows are re-decoded at higher
temperature (no window was skipped in our tests). `chunked` is the transformers pipeline
(overlapping 30 s windows merged on the overlap): ~2.5x faster, but the merge
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

  # Which model is Sorani Kurdish? Each card tags both the name and the ISO code
  hf models list --author BuzzASR --filter ckb --format quiet     # -> BuzzASR/sorani-kurdish
  uv run buzzasr-transcribe.py --list-languages                    # all of them, live from the Hub

Models: https://huggingface.co/BuzzASR (MIT), 1.55B params each, fp16 safetensors
  - `--language` takes the Hub name (dutch, sorani-kurdish, ...) or ISO code (nl, ckb, ...);
    it is resolved against the org's model tags at run time, so new languages need no code change
  - `--model` runs any other Whisper checkpoint through the same loop (e.g. openai/whisper-large-v3
    as a zero-shot control)
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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

ORG = "BuzzASR"

# From the model cards' usage snippet (greedy + anti-loop settings).
GENERATE_KWARGS = {"num_beams": 1, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2}
# Whisper's long-form fallback (from the transformers Whisper docs): retry a
# window at higher temperature when its output looks degenerate. The docs also
# set logprob_threshold=-1.0, but that makes transformers keep every token's
# full-vocabulary scores on the host for the whole batch (measured 17.6 GB
# peak RSS for 178 min of audio vs 4.7 GB on the GPU), so only the
# compression-ratio check is used here.
SEQUENTIAL_KWARGS = {
    "return_timestamps": True,
    "condition_on_prev_tokens": False,
    "compression_ratio_threshold": 1.35,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
}


def check_cuda_availability():
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        sys.exit(1)
    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")


def list_org_models() -> list[tuple[str, str, str]]:
    """(name, language code, repo id) for every model in the org, from the card metadata."""
    from huggingface_hub import HfApi

    rows = []
    for m in HfApi().list_models(author=ORG, expand=["cardData"], limit=1000):
        code = (m.card_data or {}).get("language") or "?"
        if isinstance(code, list):
            code = code[0]
        rows.append((m.id.split("/", 1)[1], code, m.id))
    return sorted(rows)


def resolve_model(language: str | None, model: str | None) -> str:
    """Map --language to the org's repo for it, unless --model overrides.

    Every BuzzASR card tags both the language name and its ISO code, so a Hub
    tag filter on the org resolves either spelling (`hf models list --author
    BuzzASR --filter nl`). No table to keep in sync with the org.
    """
    if model:
        return model
    if not language:
        logger.error("--language is required (or pass --model). See --list-languages.")
        sys.exit(1)
    from huggingface_hub import HfApi

    api = HfApi()
    key = language.strip().lower()
    # Tag filter first (name and ISO code are both tags on every card), then a
    # name search for repo names the tag filter misses (hyphenated ones).
    try:
        matches = [m.id for m in api.list_models(author=ORG, filter=key, limit=10)]
        if not matches:
            found = [m.id for m in api.list_models(author=ORG, search=key, limit=10)]
            exact = [i for i in found if i.split("/", 1)[1] == key]
            matches = exact or found
    except Exception as err:  # network / Hub API failure, not "no such language"
        logger.error(
            f"Could not query the Hub to resolve --language {key!r} ({err}). "
            f"Pass --model {ORG}/<name> to skip the lookup."
        )
        sys.exit(1)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        logger.error(
            f"No {ORG} model is tagged {key!r}. Try the language name or ISO code, "
            f"`hf models list --author {ORG} --search {key}`, or --list-languages."
        )
    else:
        logger.error(
            f"{key!r} matches several {ORG} models: {', '.join(matches)}. Pass --model."
        )
    sys.exit(1)


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
        help="Language name (dutch, sorani-kurdish) or ISO code (nl, ckb); resolved via the org's model tags",
    )
    parser.add_argument(
        "--model",
        help="Override the checkpoint (any Whisper model on the Hub); --language is then ignored",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List the org's models with their language codes (live from the Hub) and exit",
    )
    parser.add_argument(
        "--long-form",
        choices=["sequential", "chunked"],
        default="sequential",
        help="How files longer than 30 s are decoded: Whisper's sequential algorithm "
        "(skipped no windows in our tests) or the transformers chunked pipeline "
        "(faster, drops text at bad merges) (default: sequential)",
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
        for name, code, repo_id in list_org_models():
            print(f"{name:16s} {code:4s} {repo_id}")
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

    # Output is <name>.txt, so episode.mp3 and episode.wav in one directory
    # would overwrite each other. Refuse up front rather than lose a transcript.
    out_paths = [
        output_dir / f.relative_to(input_dir).with_suffix(".txt") for f in files
    ]
    seen: dict[Path, Path] = {}
    for src, dst in zip(files, out_paths):
        if dst in seen:
            logger.error(f"{src} and {seen[dst]} would both write {dst}; rename one.")
            sys.exit(1)
        seen[dst] = src

    wall_start = time.time()

    # Decode in a thread pool while the model loads. Only a few files are in
    # flight at once (decoded audio is 3.8 MB/min), and each group is
    # transcribed and written out as soon as it is full, so memory is bounded
    # by GROUP_SECONDS plus the decode-ahead window and a crash keeps the
    # transcripts already written.
    logger.info(f"Decoding audio with {args.decode_workers} worker(s)...")
    decode_start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    to_decode = list(enumerate(files))
    pending: dict = {}  # future -> file index
    decode_ahead = 2 * args.decode_workers

    def timed_load(path: Path) -> tuple[np.ndarray, float]:
        return load_audio(path), time.time()

    def top_up_decoding():
        while to_decode and len(pending) < decode_ahead:
            fi, path = to_decode.pop(0)
            pending[pool.submit(timed_load, path)] = fi

    top_up_decoding()

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
        if inputs.input_features.shape[-1] <= 3000:
            # Every file fits one window: short-form. Re-extract padded to the
            # full 30 s window (3000 frames); the encoder and Whisper's language
            # detection (used by non-BuzzASR checkpoints) expect exactly that.
            feats = processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            ids = model.generate(
                input_features=feats.input_features.to("cuda:0", torch.float16),
                **GENERATE_KWARGS,
            )
            return [
                t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)
            ]
        input_features = inputs.input_features.to("cuda:0", torch.float16)
        attention_mask = inputs.attention_mask.to("cuda:0")
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
            f"  group of {len(group)} file(s), {group_seconds / 60:.1f} min audio done"
        )

    summary_path = output_dir / "summary.jsonl"
    summary_path.write_text("", encoding="utf-8")

    def write_group():
        """Write this group's transcripts and append their summary rows, in file order."""
        with open(summary_path, "a", encoding="utf-8") as f:
            for fi, _ in sorted(group):
                text = texts.pop(fi)
                rel = files[fi].relative_to(input_dir)
                txt_path = out_paths[fi]
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                txt_path.write_text(text, encoding="utf-8")
                row = {
                    "file": str(rel),
                    "model": model_id,
                    "long_form": args.long_form,
                    "duration_s": round(durations[fi], 1),
                    "transcript_length": len(text),
                    "word_count": len(text.split()),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                logger.info(
                    f"  {rel} -> {txt_path.name} ({len(text.split())} words, "
                    f"{durations[fi]:.0f}s audio)"
                )

    logger.info(
        f"Transcribing (long_form={args.long_form}, batch_size={args.batch_size}, {model_id})..."
    )
    torch.cuda.reset_peak_memory_stats()
    last_decoded_at = decode_start

    def flush_group():
        nonlocal group_seconds
        run_group()
        write_group()
        group.clear()
        group_seconds = 0.0

    with torch.inference_mode():
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                fi = pending.pop(fut)
                audio, decoded_at = fut.result()
                last_decoded_at = max(last_decoded_at, decoded_at)
                durations[fi] = len(audio) / SAMPLE_RATE
                # Flush before an addition would push the group past its budget;
                # a single file longer than the budget is a group on its own.
                if group and group_seconds + durations[fi] > GROUP_SECONDS:
                    flush_group()
                group.append((fi, audio))
                group_seconds += durations[fi]
            top_up_decoding()
            if not pending and group:
                flush_group()
    pool.shutdown()
    decode_time = last_decoded_at - decode_start
    total_audio = sum(durations)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    import resource

    peak_rss_gb = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    )  # kB on Linux

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
        f"  Peak GPU memory: {peak_gb:.2f} GB, peak host RSS: {peak_rss_gb:.1f} GB "
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
        print(
            "  hf models list --author BuzzASR --filter ckb --format quiet   # name from ISO code"
        )
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
