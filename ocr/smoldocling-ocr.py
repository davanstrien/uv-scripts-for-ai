# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "pillow",
#     "vllm",
#     "tqdm",
#     "toolz",
#     "torch",  # Added for CUDA check
#     "docling-core",  # For DocTags conversion
# ]
#
# ///

"""
Extract structured documents using SmolDocling-256M with vLLM.

This script processes images through the SmolDocling model to extract
structured document content with DocTags format, ideal for documents
with code, formulas, tables, and complex layouts.

Features:
- Ultra-compact 256M parameter model
- DocTags format for efficient representation
- Code block recognition with indentation
- Mathematical formula detection
- Table and chart extraction
- Layout preservation with bounding boxes
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Union

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image, UnidentifiedImageError
from toolz import partition_all
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def prepare_llm_input(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt_text: str = "Convert page to Docling.",
) -> Dict:
    """Prepare input for vLLM processing."""
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    elif isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Create chat template - exact format from the example
    chat_template = (
        f"<|im_start|>User:<image>{prompt_text}<end_of_utterance>\nAssistant:"
    )

    # Return in the format expected by vLLM generate
    return {"prompt": chat_template, "multi_modal_data": {"image": pil_img}}


def convert_doctags_to_markdown(doctags_output: str) -> str:
    """Convert DocTags output to markdown format."""
    # For now, just return the raw output as-is
    # We'll focus on getting the basic vLLM inference working first
    return doctags_output.strip()


def create_dataset_card(
    source_dataset: str,
    model: str,
    num_samples: int,
    processing_time: str,
    output_column: str,
    output_format: str,
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
- smoldocling
- doctags
- structured-extraction
- uv-script
- generated
---

# Document Processing using {model_name}

This dataset contains structured document extraction from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using SmolDocling.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `{output_column}`
- **Output Format**: {output_format}
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

SmolDocling-256M is an ultra-compact multimodal model that excels at:
- 💻 **Code Recognition** - Detects and formats code blocks with proper indentation
- 🔢 **Formula Recognition** - Identifies and processes mathematical expressions
- 📊 **Tables & Charts** - Extracts structured data from tables and charts
- 📐 **Layout Preservation** - Maintains document structure with bounding boxes
- 🏷️ **DocTags Format** - Efficient minimal representation for documents
- ⚡ **Fast Inference** - Only 256M parameters for quick processing

## Dataset Structure

The dataset contains all original columns plus:
- `{output_column}`: The extracted {"DocTags JSON" if output_format == "doctags" else "markdown"} from each image
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset
import json
{"from docling_core.types.doc import DoclingDocument" if output_format == "doctags" else ""}
{"from docling_core.types.doc.document import DocTagsDocument" if output_format == "doctags" else ""}

# Load the dataset
dataset = load_dataset("{{output_dataset_id}}", split="{split}")

# Access the extracted content
for example in dataset:
    {"# Parse DocTags and convert to desired format" if output_format == "doctags" else ""}
    {f"doc_tags = DocTagsDocument.model_validate_json(example['{output_column}'])" if output_format == "doctags" else f"print(example['{output_column}'])"}
    {"doc = DoclingDocument.from_doctags(doc_tags)" if output_format == "doctags" else ""}
    {"print(doc.export(format='md').text)  # Or 'html', 'json'" if output_format == "doctags" else ""}
    break

# View all OCR models applied to this dataset
inference_info = json.loads(dataset[0]["inference_info"])
for info in inference_info:
    print(f"Column: {{info['column_name']}} - Model: {{info['model_id']}}")
```

## Reproduction

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) SmolDocling script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/smoldocling-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --output-format {output_format} \\
    --batch-size {batch_size} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens} \\
    --gpu-memory-utilization {gpu_memory_utilization}
```

## Performance

- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.1f} images/second
- **Model Size**: 256M parameters (ultra-compact)
- **GPU Configuration**: vLLM with {gpu_memory_utilization:.0%} GPU memory utilization

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 32,
    model: str = "ds4sd/SmolDocling-256M-preview",
    max_model_len: int = 8192,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    output_column: str = "markdown",
    output_format: str = "markdown",
    shuffle: bool = False,
    seed: int = 42,
    prompt: str = "Convert page to Docling.",
    config: str = None,
    create_pr: bool = False,
    verbose: bool = False,
):
    """Process images from HF dataset through SmolDocling model."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    # Validate image column
    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Validate output format
    if output_format not in ["markdown", "doctags"]:
        raise ValueError(
            f"Invalid output format '{output_format}'. Must be 'markdown' or 'doctags'"
        )

    # Shuffle if requested
    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Initialize vLLM
    logger.info(f"Initializing vLLM with model: {model}")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for OCR
        max_tokens=max_tokens,
    )

    # Process images in batches
    all_output = []

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output format: {output_format}")

    # Process in batches to avoid memory issues
    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="OCR processing",
    ):
        batch_indices = list(batch_indices)

        # Fetch images first, with per-batch fallback for unreadable files.
        # One corrupt image used to take down the entire run via the list
        # comprehension; now we mark the whole batch as skipped and continue.
        try:
            batch_images = [dataset[i][image_column] for i in batch_indices]
        except (UnidentifiedImageError, OSError) as e:
            logger.warning(
                f"Skipping batch of {len(batch_indices)} — unreadable image "
                f"in batch: {type(e).__name__}: {e}"
            )
            all_output.extend(
                ["[OCR SKIPPED — UNREADABLE IMAGE]"] * len(batch_indices)
            )
            continue

        try:
            # Prepare inputs for batch
            batch_inputs = [prepare_llm_input(img, prompt) for img in batch_images]

            # Process with vLLM using generate
            outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

            # Extract text from outputs
            for i, output in enumerate(outputs):
                raw_output = output.outputs[0].text.strip()

                # Convert to markdown if requested
                if output_format == "markdown":
                    processed_output = convert_doctags_to_markdown(raw_output)
                else:
                    processed_output = raw_output

                all_output.append(processed_output)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_output.extend(["[OCR FAILED]"] * len(batch_images))

    # Add output column to dataset
    logger.info(f"Adding {output_column} column to dataset")
    dataset = dataset.add_column(output_column, all_output)

    # Handle inference_info tracking
    inference_entry = {
        "model_id": model,
        "model_name": "SmolDocling-256M",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "output_format": output_format,
        "max_tokens": max_tokens,
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

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time = f"{processing_duration.total_seconds() / 60:.1f} minutes"

    # Push to hub with retry and XET fallback
    logger.info(f"Pushing to {output_dataset}")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                logger.warning("Disabling XET (fallback to HTTP upload)")
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            dataset.push_to_hub(
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
        num_samples=len(dataset),
        processing_time=processing_time,
        output_column=output_column,
        output_format=output_format,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)
    logger.info("Dataset card created and pushed!")

    logger.info("SmolDocling processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time}")

    if verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        for pkg in ["vllm", "transformers", "torch", "datasets", "pyarrow", "pillow"]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("SmolDocling Ultra-Compact Document Processing")
        print("=" * 80)
        print("\nThis script extracts structured document content using")
        print("the SmolDocling-256M model with vLLM acceleration.")
        print("\nFeatures:")
        print("- Ultra-compact 256M parameter model")
        print("- DocTags format for efficient representation")
        print("- Code block recognition with indentation")
        print("- Mathematical formula detection")
        print("- Table and chart extraction")
        print("- Layout preservation with bounding boxes")
        print("\nExample usage:")
        print("\n1. Basic document conversion to markdown:")
        print("   uv run smoldocling-ocr.py document-images extracted-docs")
        print("\n2. Extract with DocTags format:")
        print("   uv run smoldocling-ocr.py scientific-papers doc-analysis \\")
        print("       --output-format doctags")
        print("\n3. Custom settings:")
        print("   uv run smoldocling-ocr.py code-docs structured-output \\")
        print("       --image-column page \\")
        print("       --batch-size 64 \\")
        print("       --gpu-memory-utilization 0.9")
        print("\n4. Process a subset for testing:")
        print("   uv run smoldocling-ocr.py large-dataset test-output --max-samples 10")
        print("\n5. Random sample from ordered dataset:")
        print(
            "   uv run smoldocling-ocr.py ordered-dataset random-test --max-samples 50 --shuffle"
        )
        print("\n6. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print(
            '     -e HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())") \\'
        )
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/smoldocling-ocr.py \\"
        )
        print("       your-document-dataset \\")
        print("       your-structured-output")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run smoldocling-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Extract structured documents using SmolDocling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run smoldocling-ocr.py my-images-dataset structured-output

  # With DocTags format output
  uv run smoldocling-ocr.py documents doc-analysis --output-format doctags

  # Process subset for testing
  uv run smoldocling-ocr.py large-dataset test-output --max-samples 100

  # Random sample of 100 images
  uv run smoldocling-ocr.py ordered-dataset random-sample --max-samples 100 --shuffle

  # Custom output column name (default: smoldocling_text)
  uv run smoldocling-ocr.py images texts --output-column extracted_content
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
        default="ds4sd/SmolDocling-256M-preview",
        help="Model to use (default: ds4sd/SmolDocling-256M-preview)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192)",
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
        "--output-format",
        default="markdown",
        choices=["markdown", "doctags"],
        help="Output format: 'markdown' or 'doctags' (default: markdown)",
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
        "--prompt",
        default="Convert page to Docling.",
        help="Custom prompt for the model (default: 'Convert page to Docling.')",
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
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        output_column=args.output_column,
        output_format=args.output_format,
        shuffle=args.shuffle,
        seed=args.seed,
        prompt=args.prompt,
        config=args.config,
        create_pr=args.create_pr,
        verbose=args.verbose,
    )
