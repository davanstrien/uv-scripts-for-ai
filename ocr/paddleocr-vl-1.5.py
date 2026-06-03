# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "pillow",
#     "torch",
#     "torchvision",
#     "transformers>=5.0.0",
#     "accelerate",
#     "tqdm",
# ]
# ///

"""
Convert document images to text/tables/formulas using PaddleOCR-VL-1.5 with transformers.

PaddleOCR-VL-1.5 is a compact 0.9B OCR model that achieves 94.5% SOTA accuracy on
OmniDocBench v1.5. It supports multiple task modes including OCR, table recognition,
formula extraction, chart analysis, text spotting, and seal recognition.

NOTE: This script uses transformers batch inference (not vLLM) because PaddleOCR-VL
only supports vLLM in server mode, which doesn't fit the UV script pattern.
Transformers batch inference via apply_chat_template(padding=True) provides
efficient batching while keeping the single-command interface.

Features:
- 🎯 SOTA Performance: 94.5% on OmniDocBench v1.5
- 🧩 Ultra-compact: Only 0.9B parameters
- 📝 OCR mode: General text extraction to markdown
- 📊 Table mode: HTML table recognition
- 📐 Formula mode: LaTeX mathematical notation
- 📈 Chart mode: Structured chart analysis
- 🔍 Spotting mode: Text spotting with localization
- 🔖 Seal mode: Seal/stamp recognition
- 🌍 Multilingual support
- ⚡ Fast inference with batch processing

Model: PaddlePaddle/PaddleOCR-VL-1.5
Backend: Transformers (batch inference)
Performance: 94.5% SOTA on OmniDocBench v1.5
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Union

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model configuration
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"

# Task mode configurations from official PaddleOCR-VL-1.5 documentation
TASK_MODES = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

# Task descriptions for dataset card
TASK_DESCRIPTIONS = {
    "ocr": "General text extraction to markdown format",
    "table": "Table extraction to HTML format",
    "formula": "Mathematical formula recognition to LaTeX",
    "chart": "Chart and diagram analysis",
    "spotting": "Text spotting with localization",
    "seal": "Seal and stamp recognition",
}

# Max pixels configuration - spotting uses higher resolution
MAX_PIXELS = {
    "spotting": 2048 * 28 * 28,  # Higher resolution for spotting
    "default": 1280 * 28 * 28,
}


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def prepare_image(
    image: Union[Image.Image, Dict[str, Any], str],
) -> Image.Image:
    """
    Prepare image for PaddleOCR-VL-1.5 processing.

    Note: Image resizing is handled by the processor via images_kwargs.

    Args:
        image: PIL Image, dict with bytes, or file path

    Returns:
        Processed PIL Image in RGB format
    """
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to RGB
    pil_img = pil_img.convert("RGB")

    return pil_img


def create_message(image: Image.Image, task_mode: str) -> List[Dict]:
    """
    Create chat message for PaddleOCR-VL-1.5 processing.

    Args:
        image: Prepared PIL Image
        task_mode: Task mode (ocr, table, formula, chart, spotting, seal)

    Returns:
        Message in chat format
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TASK_MODES[task_mode]},
            ],
        }
    ]


def create_dataset_card(
    source_dataset: str,
    model: str,
    task_mode: str,
    num_samples: int,
    processing_time: str,
    max_tokens: int,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    task_description = TASK_DESCRIPTIONS[task_mode]

    return f"""---
tags:
- ocr
- document-processing
- paddleocr-vl-1.5
- {task_mode}
- uv-script
- generated
---

# Document Processing using PaddleOCR-VL-1.5 ({task_mode.upper()} mode)

This dataset contains {task_mode.upper()} results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using PaddleOCR-VL-1.5, an ultra-compact 0.9B SOTA OCR model.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Task Mode**: `{task_mode}` - {task_description}
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `paddleocr_1.5_{task_mode}`
- **Dataset Split**: `{split}`
- **Max Output Tokens**: {max_tokens:,}
- **Backend**: Transformers (single image processing)

## Model Information

PaddleOCR-VL-1.5 is a state-of-the-art, resource-efficient model for document parsing:
- 🎯 **SOTA Performance** - 94.5% on OmniDocBench v1.5
- 🧩 **Ultra-compact** - Only 0.9B parameters
- 📝 **OCR mode** - General text extraction
- 📊 **Table mode** - HTML table recognition
- 📐 **Formula mode** - LaTeX mathematical notation
- 📈 **Chart mode** - Structured chart analysis
- 🔍 **Spotting mode** - Text spotting with localization
- 🔖 **Seal mode** - Seal and stamp recognition
- 🌍 **Multilingual** - Support for multiple languages
- ⚡ **Fast** - Efficient batch inference

### Task Modes

- **OCR**: Extract text content to markdown format
- **Table Recognition**: Extract tables to HTML format
- **Formula Recognition**: Extract mathematical formulas to LaTeX
- **Chart Recognition**: Analyze and describe charts/diagrams
- **Spotting**: Text spotting with location information
- **Seal Recognition**: Extract text from seals and stamps

## Dataset Structure

The dataset contains all original columns plus:
- `paddleocr_1.5_{task_mode}`: The extracted content based on task mode
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("{{output_dataset_id}}", split="{split}")

# Access the extracted content
for example in dataset:
    print(example["paddleocr_1.5_{task_mode}"])
    break

# View all OCR models applied to this dataset
inference_info = json.loads(dataset[0]["inference_info"])
for info in inference_info:
    print(f"Task: {{info['task_mode']}} - Model: {{info['model_id']}}")
```

## Reproduction

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) PaddleOCR-VL-1.5 script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl-1.5.py \\
    {source_dataset} \\
    <output-dataset> \\
    --task-mode {task_mode} \\
    --image-column {image_column}
```

## Performance

- **Model Size**: 0.9B parameters
- **Benchmark Score**: 94.5% SOTA on OmniDocBench v1.5
- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.2f} images/second
- **Backend**: Transformers (single image processing)

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    task_mode: str = "ocr",
    max_tokens: int = 512,  # model card example uses 512 (element-level); increase for full pages
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
    config: str = None,
    create_pr: bool = False,
    verbose: bool = False,
):
    """Process images from HF dataset through PaddleOCR-VL-1.5 model."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Validate task mode
    if task_mode not in TASK_MODES:
        raise ValueError(
            f"Invalid task_mode '{task_mode}'. Choose from: {list(TASK_MODES.keys())}"
        )

    logger.info(f"Using task mode: {task_mode} - {TASK_DESCRIPTIONS[task_mode]}")
    logger.info(f"Output will be written to column: {output_column}")

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    # Validate image column
    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Shuffle if requested
    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Import transformers components
    logger.info(f"Loading model: {MODEL_ID}")
    logger.info("This may take a minute on first run...")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Load processor and model (native transformers 5.0+ support)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Set left padding for correct batch generation with decoder-only models
    processor.tokenizer.padding_side = "left"

    model = (
        AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        .to("cuda")
        .eval()
    )

    logger.info(f"Model loaded on {next(model.parameters()).device}")
    max_pixels = MAX_PIXELS.get(task_mode, MAX_PIXELS["default"])
    logger.info(f"Processing {len(dataset)} images (one at a time for stability)")
    logger.info(f"Image resizing: max_pixels={max_pixels:,} (handled by processor)")

    # Process images one at a time
    # Note: Batch processing with transformers VLMs can be unreliable,
    # so we process individually for stability
    all_outputs = []

    for i in tqdm(range(len(dataset)), desc=f"PaddleOCR-VL-1.5 {task_mode.upper()}"):
        try:
            # Prepare image and create message
            image = dataset[i][image_column]
            pil_image = prepare_image(image)
            messages = create_message(pil_image, task_mode)

            # Process with transformers
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                images_kwargs={
                    "size": {
                        "shortest_edge": processor.image_processor.min_pixels,
                        "longest_edge": max_pixels,
                    }
                },
            ).to(model.device)

            # Generate output
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )

            # Decode output - skip input tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0, input_len:]
            result = processor.decode(generated_ids, skip_special_tokens=True)
            all_outputs.append(result.strip())

        except Exception as e:
            logger.error(f"Error processing image {i}: {e}")
            all_outputs.append(f"[{task_mode.upper()} ERROR: {str(e)[:100]}]")

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column to dataset
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Handle inference_info tracking (for multi-model comparisons)
    inference_entry = {
        "model_id": MODEL_ID,
        "model_name": "PaddleOCR-VL-1.5",
        "model_size": "0.9B",
        "task_mode": task_mode,
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "max_tokens": max_tokens,
        "backend": "transformers",
    }

    if "inference_info" in dataset.column_names:
        # Append to existing inference info
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
        # Create new inference_info column
        logger.info("Creating new inference_info column")
        inference_list = [json.dumps([inference_entry])] * len(dataset)
        dataset = dataset.add_column("inference_info", inference_list)

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
                commit_message=f"Add {MODEL_ID} OCR results ({len(dataset)} samples)"
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
    logger.info("Creating dataset card")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=MODEL_ID,
        task_mode=task_mode,
        num_samples=len(dataset),
        processing_time=processing_time_str,
        max_tokens=max_tokens,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("PaddleOCR-VL-1.5 processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")
    logger.info(
        f"Processing speed: {len(dataset) / processing_duration.total_seconds():.2f} images/sec"
    )
    logger.info(f"Task mode: {task_mode} - {TASK_DESCRIPTIONS[task_mode]}")

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
        print("PaddleOCR-VL-1.5 Document Processing")
        print("=" * 80)
        print("\nSOTA 0.9B OCR model (94.5% on OmniDocBench v1.5)")
        print("\nFeatures:")
        print("- 🎯 SOTA Performance - 94.5% on OmniDocBench v1.5")
        print("- 🧩 Ultra-compact - Only 0.9B parameters")
        print("- 📝 OCR mode - General text extraction")
        print("- 📊 Table mode - HTML table recognition")
        print("- 📐 Formula mode - LaTeX mathematical notation")
        print("- 📈 Chart mode - Structured chart analysis")
        print("- 🔍 Spotting mode - Text spotting with localization")
        print("- 🔖 Seal mode - Seal and stamp recognition")
        print("- 🌍 Multilingual support")
        print("- ⚡ Fast batch inference with transformers")
        print("\nTask Modes:")
        for mode, description in TASK_DESCRIPTIONS.items():
            print(f"  {mode:10} - {description}")
        print("\nExample usage:")
        print("\n1. Basic OCR (default mode):")
        print("   uv run paddleocr-vl-1.5.py input-dataset output-dataset")
        print("\n2. Table extraction:")
        print("   uv run paddleocr-vl-1.5.py docs tables-extracted --task-mode table")
        print("\n3. Formula recognition:")
        print("   uv run paddleocr-vl-1.5.py papers formulas --task-mode formula")
        print("\n4. Text spotting (higher resolution):")
        print("   uv run paddleocr-vl-1.5.py images spotted --task-mode spotting")
        print("\n5. Seal recognition:")
        print("   uv run paddleocr-vl-1.5.py seals recognized --task-mode seal")
        print("\n6. Test with small sample:")
        print("   uv run paddleocr-vl-1.5.py dataset test --max-samples 10 --shuffle")
        print("\n7. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print("     -s HF_TOKEN \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl-1.5.py \\"
        )
        print("       input-dataset output-dataset --task-mode ocr")
        print("\n" + "=" * 80)
        print("\nBackend: Transformers (batch inference)")
        print(
            "Note: Uses transformers instead of vLLM for single-command UV script compatibility"
        )
        print("\nFor full help, run: uv run paddleocr-vl-1.5.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document processing using PaddleOCR-VL-1.5 (0.9B SOTA model, transformers backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task Modes:
  ocr       General text extraction to markdown (default)
  table     Table extraction to HTML format
  formula   Mathematical formula recognition to LaTeX
  chart     Chart and diagram analysis
  spotting  Text spotting with localization (higher resolution)
  seal      Seal and stamp recognition

Examples:
  # Basic text OCR
  uv run paddleocr-vl-1.5.py my-docs analyzed-docs

  # Extract tables from documents
  uv run paddleocr-vl-1.5.py papers tables --task-mode table

  # Recognize mathematical formulas
  uv run paddleocr-vl-1.5.py textbooks formulas --task-mode formula

  # Text spotting (uses higher resolution)
  uv run paddleocr-vl-1.5.py images spotted --task-mode spotting

  # Seal recognition
  uv run paddleocr-vl-1.5.py documents seals --task-mode seal

  # Test with random sampling
  uv run paddleocr-vl-1.5.py large-dataset test --max-samples 50 --shuffle --task-mode ocr

  # Disable smart resize for original resolution
  uv run paddleocr-vl-1.5.py images output --no-smart-resize

Backend: Transformers batch inference (not vLLM)
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
        "--task-mode",
        choices=list(TASK_MODES.keys()),
        default="ocr",
        help="Task type: ocr (default), table, formula, chart, spotting, or seal",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512, per model card element-level example; increase for full pages)",
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
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Column name for output text (default: markdown)",
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
        task_mode=args.task_mode,
        max_tokens=args.max_tokens,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        config=args.config,
        create_pr=args.create_pr,
        verbose=args.verbose,
    )
