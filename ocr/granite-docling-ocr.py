# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "vllm",
#     "torch",
#     "toolz",
#     "tqdm",
#     "transformers",
#     "docling-core>=2.23.0",  # DocTags -> DoclingDocument -> markdown/html conversion
# ]
#
# ///

"""
Convert document images to markdown using Granite-Docling-258M with vLLM.

Granite-Docling (IBM Research) is the successor to SmolDocling: an ultra-compact
258M-parameter Idefics3-based VLM (SigLIP2 vision encoder + Granite 165M LLM) that
converts full pages to DocTags — Docling's structured document format with layout,
tables (OTSL), code blocks, formulas (LaTeX), and per-element bounding boxes.
This script converts the DocTags to markdown via docling-core by default; pass
`--output-format doctags` to keep the raw structured output.

Features:
- 258M parameters — cheap and fast, runs comfortably on the smallest GPU flavors
- Full-page conversion to DocTags (layout, reading order, bboxes)
- Strong table (0.97 TEDS FinTabNet), code, and formula recognition
- Element-level task modes: chart -> table, formula -> LaTeX, code -> text, table -> OTSL
- Apache 2.0 (model and weights)

Model: https://huggingface.co/ibm-granite/granite-docling-258M

NOTE on `--revision untied` (the default): vLLM cannot load the tied-weight
checkpoint on the `main` branch (`AttributeError: 'LlamaModel' object has no
attribute 'wte'` at engine init — see the model card's troubleshooting section),
so IBM publishes an untied copy on the `untied` branch and vLLM examples on the
card use it. Loosen to `main` once vLLM's tied-weight loading for granite-docling
is confirmed fixed in a stable release.

Usage:
    # Quick test with 10 samples
    uv run granite-docling-ocr.py your-input-dataset your-output-dataset --max-samples 10

    # Full dataset, keep raw DocTags instead of markdown
    uv run granite-docling-ocr.py your-input-dataset your-output-dataset \\
        --output-format doctags

    # Element-level tasks (crops of tables/formulas/code/charts, not full pages)
    uv run granite-docling-ocr.py table-crops otsl-out --task-mode table

    # On HF Jobs (no local GPU needed)
    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/granite-docling-ocr.py \\
        your-input-dataset your-output-dataset --max-samples 10

Hardware note: prefer a bfloat16-capable GPU (compute capability >= 8.0: L4, A10,
A100...). On older GPUs (e.g. T4) the model emits only `!` in bfloat16, so this
script auto-falls back to float32 there (slower, more memory — fine at 258M).
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Union

import torch
from datasets import load_dataset
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from huggingface_hub import DatasetCard, login
from PIL import Image, UnidentifiedImageError
from toolz import partition_all
from tqdm.auto import tqdm
from transformers import AutoProcessor

# Disable vLLM's FlashInfer sampler: it JIT-compiles a CUDA kernel needing nvcc, which the
# default uv-script image lacks (engine init then crashes). Greedy OCR doesn't use it; this
# lets the plain default-image command work. On the vllm/vllm-openai image it's a harmless no-op.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ibm-granite/granite-docling-258M"

# Instructions from the model card's "Supported Instructions" table. Only "full"
# produces a full-page DocTags document (convertible to markdown); the others are
# element-level tasks whose output (OTSL / LaTeX / plain text) is stored raw.
TASK_PROMPTS = {
    "full": "Convert this page to docling.",
    "chart": "Convert chart to table.",
    "formula": "Convert formula to LaTeX.",
    "code": "Convert code to text.",
    "table": "Convert table to OTSL.",
}


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def pick_dtype() -> str:
    """bfloat16 where supported, float32 otherwise.

    Model-card troubleshooting: on GPUs without bfloat16 (compute capability < 8.0,
    e.g. T4) the model outputs only exclamation marks; the documented workaround is
    dtype float32. At 258M params float32 is still tiny.
    """
    major, _minor = torch.cuda.get_device_capability()
    if major >= 8:
        return "bfloat16"
    logger.warning(
        "GPU lacks bfloat16 (compute capability < 8.0) — using float32 to avoid "
        "the all-'!' output failure documented on the model card."
    )
    return "float32"


def ensure_output_columns_free(dataset, columns, overwrite=False):
    """Fail fast if an output column would collide with an existing input column.

    Adding a column that already exists silently overwrites it (e.g. a ground-truth
    `text`/`markdown` column) or crashes on push with a duplicate-column error only
    *after* inference has run. Catch it up front. With overwrite=True, drop the clashing
    column(s) here instead (logged) so the later add_column is clean.
    """
    clash = [c for c in columns if c in dataset.column_names]
    if not clash:
        return dataset
    if overwrite:
        logger.warning(f"--overwrite: replacing existing column(s) {clash}")
        return dataset.remove_columns(clash)
    logger.error(
        f"Output column(s) {clash} already exist in the input dataset "
        f"(columns: {dataset.column_names})."
    )
    logger.error(
        "Choose a different --output-column, or pass --overwrite to replace them."
    )
    sys.exit(1)


def to_pil_image(image: Union[Image.Image, Dict[str, Any], str]) -> Image.Image:
    """Convert a dataset image cell to an RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image)}")


def doctags_to_markdown(doctags: str, image: Image.Image) -> str:
    """Convert a full-page DocTags string to markdown via docling-core.

    Falls back to the raw DocTags (with a warning) if parsing fails, so a
    conversion hiccup never loses the model output.
    """
    try:
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
        return doc.export_to_markdown()
    except Exception as e:
        logger.warning(
            f"DocTags -> markdown conversion failed ({e}); storing raw DocTags"
        )
        return doctags


def clean_model_output(text: str) -> str:
    """Strip the trailing end-of-utterance marker kept by skip_special_tokens=False."""
    return text.replace("<end_of_utterance>", "").strip()


def create_dataset_card(
    source_dataset: str,
    model: str,
    revision: str,
    num_samples: int,
    processing_time: str,
    output_column: str,
    output_format: str,
    task_mode: str,
    prompt_text: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- granite-docling
- docling
- doctags
- uv-script
- generated
---

# Document Conversion using {model_name}

This dataset contains document conversion results from images in
[{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using
[Granite-Docling](https://huggingface.co/{model}).

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model}) (revision `{revision}`)
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `{output_column}`
- **Output Format**: {output_format}
- **Task Mode**: `{task_mode}` (prompt: "{prompt_text}")
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

Granite-Docling-258M (IBM Research, Apache 2.0) is an ultra-compact document
conversion VLM — the successor to SmolDocling — built on the Idefics3 architecture
with a SigLIP2 vision encoder and a Granite 165M language model. It converts pages
to **DocTags**, Docling's structured format with layout, reading order, tables
(OTSL), code, formulas (LaTeX), and bounding boxes.

## Dataset Structure

The dataset contains all original columns plus:

- `{output_column}`: {"markdown converted from the model's DocTags output via docling-core" if output_format == "markdown" else "the model's raw output (DocTags / OTSL / LaTeX / text depending on task mode)"}
- `inference_info`: JSON list tracking the models applied to this dataset

## Reproduction

```bash
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \\
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/granite-docling-ocr.py \\
    {source_dataset} <output-dataset> \\
    --image-column {image_column} \\
    --output-format {output_format} \\
    --task-mode {task_mode} \\
    --batch-size {batch_size} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens}
```

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 32,
    model: str = DEFAULT_MODEL,
    revision: str = "untied",
    max_model_len: int = 8192,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    hf_token: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    private: bool = False,
    output_column: str = "markdown",
    overwrite: bool = False,
    output_format: str = "markdown",
    task_mode: str = "full",
    custom_prompt: Optional[str] = None,
    shuffle: bool = False,
    seed: int = 42,
    config: Optional[str] = None,
    create_pr: bool = False,
    verbose: bool = False,
):
    """Process images from an HF dataset through Granite-Docling and push results."""

    check_cuda_availability()

    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Resolve the prompt: --custom-prompt wins, else the task-mode instruction.
    prompt_text = custom_prompt or TASK_PROMPTS[task_mode]

    # Markdown conversion only makes sense for full-page DocTags output. Element-level
    # task modes (and custom prompts) return OTSL/LaTeX/text snippets — store those raw.
    convert_to_markdown = (
        output_format == "markdown" and task_mode == "full" and not custom_prompt
    )
    if output_format == "markdown" and not convert_to_markdown:
        logger.info(
            "Task mode is not 'full' (or a custom prompt is set) — storing the raw "
            "model output instead of converting DocTags to markdown."
        )

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Fail fast if the output column would collide with an existing input column
    dataset = ensure_output_columns_free(dataset, [output_column], overwrite=overwrite)

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Build the chat prompt once via the model's own chat template (model-card recipe).
    processor = AutoProcessor.from_pretrained(model, revision=revision)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Initialize vLLM. revision="untied" by default: vLLM can't load the tied-weight
    # main branch (see module docstring). dtype: bfloat16, or float32 on pre-Ampere
    # GPUs (T4 emits only '!' in bfloat16 — model-card troubleshooting).
    logger.info(f"Initializing vLLM with model: {model} (revision: {revision})")
    llm = LLM(
        model=model,
        revision=revision,
        dtype=pick_dtype(),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    # Sampling per the model card's vLLM example: greedy, keep special tokens
    # (DocTags markup is emitted as special tokens — skipping them empties the output).
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        skip_special_tokens=False,
    )

    all_output = []

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Task mode: {task_mode} | prompt: {prompt_text!r}")
    logger.info(f"Output format: {output_format}")

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="OCR processing",
    ):
        batch_indices = list(batch_indices)

        # Fetch and decode images first, with per-batch fallback for unreadable files.
        try:
            batch_images = [
                to_pil_image(dataset[i][image_column]) for i in batch_indices
            ]
        except (UnidentifiedImageError, OSError, ValueError) as e:
            logger.warning(
                f"Skipping batch of {len(batch_indices)} — unreadable image "
                f"in batch: {type(e).__name__}: {e}"
            )
            all_output.extend(["[OCR SKIPPED — UNREADABLE IMAGE]"] * len(batch_indices))
            continue

        try:
            batch_inputs = [
                {"prompt": prompt, "multi_modal_data": {"image": img}}
                for img in batch_images
            ]
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

            for img, output in zip(batch_images, outputs):
                doctags = clean_model_output(output.outputs[0].text)
                if convert_to_markdown:
                    all_output.append(doctags_to_markdown(doctags, img))
                else:
                    all_output.append(doctags)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            all_output.extend(["[OCR FAILED]"] * len(batch_images))

    # Add output column to dataset
    logger.info(f"Adding {output_column} column to dataset")
    dataset = dataset.add_column(output_column, all_output)

    # inference_info — standard schema: a JSON-string LIST of entries, each carrying
    # model_id (leaderboard label) and column_name (the output text column); script
    # extras ride along as extra keys.
    inference_entry = {
        "model_id": model,
        "column_name": output_column,
        "script": "granite-docling-ocr.py",
        "revision": revision,
        "task_mode": task_mode,
        "output_format": output_format,
        "max_tokens": max_tokens,
        "timestamp": datetime.now().isoformat(),
    }

    if "inference_info" in dataset.column_names:
        logger.info("Updating existing inference_info column")

        def update_inference_info(example):
            try:
                existing_info = (
                    json.loads(example["inference_info"])
                    if example["inference_info"]
                    else []
                )
            except (json.JSONDecodeError, TypeError):
                existing_info = []
            existing_info.append(inference_entry)
            return {"inference_info": json.dumps(existing_info)}

        dataset = dataset.map(update_inference_info)
    else:
        logger.info("Creating new inference_info column")
        inference_list = [json.dumps([inference_entry])] * len(dataset)
        dataset = dataset.add_column("inference_info", inference_list)

    processing_duration = datetime.now() - start_time
    processing_time = f"{processing_duration.total_seconds() / 60:.1f} minutes"

    # Push to hub with retry and XET fallback
    logger.info(f"Pushing to {output_dataset}")
    commit_info = None
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            commit_info = dataset.push_to_hub(
                output_dataset,
                private=private,
                token=HF_TOKEN,
                max_shard_size="500MB",
                **({"config_name": config} if config else {}),
                create_pr=create_pr,
                commit_message=f"Add {model} OCR results ({len(dataset)} samples)"
                + (f" [{config}]" if config else ""),
            )
            break
        except Exception as e:
            logger.error(f"Upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                delay = 30 * (2 ** (attempt - 1))
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("All upload attempts failed. OCR results are lost.")
                sys.exit(1)

    # Create and push dataset card
    logger.info("Creating dataset card...")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=model,
        revision=revision,
        num_samples=len(dataset),
        processing_time=processing_time,
        output_column=output_column,
        output_format=output_format,
        task_mode=task_mode,
        prompt_text=prompt_text,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        image_column=image_column,
        split=split,
    )
    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Granite-Docling processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    if create_pr and getattr(commit_info, "pr_url", None):
        logger.info(f"Pull request created: {commit_info.pr_url}")
    logger.info(f"Processing time: {processing_time}")

    if verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in [
            "vllm",
            "transformers",
            "torch",
            "datasets",
            "pyarrow",
            "pillow",
            "docling-core",
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 80)
        print("Granite-Docling Document Conversion (258M, Apache 2.0)")
        print("=" * 80)
        print("\nConvert document images to markdown (via DocTags) using")
        print("IBM's Granite-Docling-258M with vLLM.")
        print("\nExample usage:")
        print("\n1. Basic conversion to markdown:")
        print("   uv run granite-docling-ocr.py document-images converted-docs")
        print("\n2. Keep the raw DocTags structured output:")
        print(
            "   uv run granite-docling-ocr.py papers doc-analysis --output-format doctags"
        )
        print("\n3. Element-level tasks (image crops of tables/formulas/code/charts):")
        print("   uv run granite-docling-ocr.py table-crops otsl-out --task-mode table")
        print("\n4. Test on a random sample first:")
        print(
            "   uv run granite-docling-ocr.py big-dataset test-out --max-samples 10 --shuffle"
        )
        print("\n5. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/granite-docling-ocr.py \\"
        )
        print("       your-document-dataset your-output-dataset")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run granite-docling-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Convert document images to markdown using Granite-Docling-258M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage — adds a `markdown` column
  uv run granite-docling-ocr.py my-images-dataset converted-output

  # Keep the raw DocTags (structured format with layout + bboxes)
  uv run granite-docling-ocr.py documents doc-analysis --output-format doctags

  # Element-level task modes (for datasets of table/formula/code/chart crops)
  uv run granite-docling-ocr.py formula-crops latex-out --task-mode formula

  # Test with a random sample
  uv run granite-docling-ocr.py large-dataset test-output --max-samples 100 --shuffle
        """,
    )

    parser.add_argument("input_dataset", help="Input dataset ID from Hugging Face Hub")
    parser.add_argument("output_dataset", help="Output dataset ID for Hugging Face Hub")
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (default: image)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--revision",
        default="untied",
        help="Model revision (default: untied — vLLM can't load the tied-weight "
        "main branch; see the model card's troubleshooting section)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192, the model's max)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192, per the model card)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token")
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Column name for output text (default: markdown)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--output-format",
        default="markdown",
        choices=["markdown", "doctags"],
        help="Output format: 'markdown' (DocTags converted via docling-core) or "
        "'doctags' (the model's raw structured output). Default: markdown",
    )
    parser.add_argument(
        "--task-mode",
        default="full",
        choices=sorted(TASK_PROMPTS),
        help="Task instruction from the model card: 'full' page conversion (default), "
        "or element-level 'chart' (-> table), 'formula' (-> LaTeX), 'code' (-> text), "
        "'table' (-> OTSL). Non-'full' modes store the raw model output.",
    )
    parser.add_argument(
        "--custom-prompt",
        help="Custom instruction overriding --task-mode (e.g. the model card's "
        "location-based instructions). Output is stored raw.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset before processing (useful for random sampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--config",
        help="Config/subset name when pushing to Hub (for benchmarking multiple models in one repo)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of pushing directly (for parallel benchmarking)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after processing (useful for pinning deps)",
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        model=args.model,
        revision=args.revision,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        output_column=args.output_column,
        overwrite=args.overwrite,
        output_format=args.output_format,
        task_mode=args.task_mode,
        custom_prompt=args.custom_prompt,
        shuffle=args.shuffle,
        seed=args.seed,
        config=args.config,
        create_pr=args.create_pr,
        verbose=args.verbose,
    )
