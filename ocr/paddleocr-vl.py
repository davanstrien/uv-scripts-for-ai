# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "vllm>=0.15.1",
#     "tqdm",
#     "toolz",
#     "torch",
#     "pyarrow",
#     "transformers",
# ]
# ///

"""
Convert document images to text/tables/formulas using PaddleOCR-VL with vLLM.

PaddleOCR-VL is a compact 0.9B OCR model with task-specific capabilities for
document parsing. It combines a NaViT-style dynamic resolution visual encoder
with the ERNIE-4.5-0.3B language model for accurate element recognition.

Features:
- 🎯 Ultra-compact: Only 0.9B parameters (smallest OCR model)
- 📝 OCR mode: General text extraction to markdown
- 📊 Table mode: HTML table recognition and extraction
- 📐 Formula mode: LaTeX mathematical notation
- 📈 Chart mode: Structured chart analysis
- 🌍 Multilingual support
- ⚡ Fast initialization due to small size
- 🔧 Based on ERNIE-4.5 (different from Qwen-based models)

Model: PaddlePaddle/PaddleOCR-VL
vLLM: Requires nightly build for full support

IMPORTANT: As of Nov 2024, PaddleOCR-VL batch processing support in vLLM is still
being finalized. The model works with `vllm serve` but may have issues with the
LLM batch class. Use alternative models like LightOnOCR or DoTS for now, or run
PaddleOCR-VL in server mode as shown in the vLLM documentation.
"""

import argparse
import base64
import io
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Union
from datetime import datetime

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from toolz import partition_all
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Task mode configurations from official PaddleOCR-VL documentation
TASK_MODES = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

# Task descriptions for dataset card
TASK_DESCRIPTIONS = {
    "ocr": "General text extraction to markdown format",
    "table": "Table extraction to HTML format",
    "formula": "Mathematical formula recognition to LaTeX",
    "chart": "Chart and diagram analysis",
}


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 28 * 28 * 130,
    max_pixels: int = 28 * 28 * 1280,
) -> tuple[int, int]:
    """
    PaddleOCR-VL's intelligent resize logic.

    Rescales the image so that:
    1. Both dimensions are divisible by 'factor' (28)
    2. Total pixels are within [min_pixels, max_pixels]
    3. Aspect ratio is maintained as closely as possible

    Args:
        height: Original image height
        width: Original image width
        factor: Dimension divisibility factor (default: 28)
        min_pixels: Minimum total pixels (default: 100,880)
        max_pixels: Maximum total pixels (default: 1,003,520)

    Returns:
        Tuple of (new_height, new_width)
    """
    if height < factor:
        width = round((width * factor) / height)
        height = factor

    if width < factor:
        height = round((height * factor) / width)
        width = factor

    if max(height, width) / min(height, width) > 200:
        logger.warning(
            f"Extreme aspect ratio detected: {max(height, width) / min(height, width):.1f}"
        )
        # Continue anyway, but warn about potential issues

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    task_mode: str = "ocr",
    apply_smart_resize: bool = True,
) -> List[Dict]:
    """
    Create chat message for PaddleOCR-VL processing.

    PaddleOCR-VL expects a specific format with the task prefix after the image.
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

    # Apply smart resize if requested
    if apply_smart_resize:
        original_size = pil_img.size
        new_width, new_height = smart_resize(pil_img.height, pil_img.width)
        if (new_width, new_height) != (pil_img.width, pil_img.height):
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {original_size} to {pil_img.size}")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # PaddleOCR-VL message format: image first, then task prefix
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
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
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    temperature: float,
    apply_smart_resize: bool,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    task_description = TASK_DESCRIPTIONS[task_mode]

    return f"""---
tags:
- ocr
- document-processing
- paddleocr-vl
- {task_mode}
- uv-script
- generated
---

# Document Processing using PaddleOCR-VL ({task_mode.upper()} mode)

This dataset contains {task_mode.upper()} results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using PaddleOCR-VL, an ultra-compact 0.9B OCR model.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Task Mode**: `{task_mode}` - {task_description}
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `markdown`
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Smart Resize**: {"Enabled" if apply_smart_resize else "Disabled"}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **Temperature**: {temperature}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

PaddleOCR-VL is a state-of-the-art, resource-efficient model tailored for document parsing:
- 🎯 **Ultra-compact** - Only 0.9B parameters (smallest OCR model)
- 📝 **OCR mode** - General text extraction
- 📊 **Table mode** - HTML table recognition
- 📐 **Formula mode** - LaTeX mathematical notation
- 📈 **Chart mode** - Structured chart analysis
- 🌍 **Multilingual** - Support for multiple languages
- ⚡ **Fast** - Quick initialization and inference
- 🔧 **ERNIE-4.5 based** - Different architecture from Qwen models

### Task Modes

- **OCR**: Extract text content to markdown format
- **Table Recognition**: Extract tables to HTML format
- **Formula Recognition**: Extract mathematical formulas to LaTeX
- **Chart Recognition**: Analyze and describe charts/diagrams

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted content based on task mode
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("{{output_dataset_id}}", split="{split}")

# Access the extracted content
for example in dataset:
    print(example["markdown"])
    break

# View all OCR models applied to this dataset
inference_info = json.loads(dataset[0]["inference_info"])
for info in inference_info:
    print(f"Task: {{info['task_mode']}} - Model: {{info['model_id']}}")
```

## Reproduction

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) PaddleOCR-VL script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl.py \\
    {source_dataset} \\
    <output-dataset> \\
    --task-mode {task_mode} \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens} \\
    --gpu-memory-utilization {gpu_memory_utilization}
```

## Performance

- **Model Size**: 0.9B parameters (smallest among OCR models)
- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.2f} images/second
- **Architecture**: NaViT visual encoder + ERNIE-4.5-0.3B language model

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    task_mode: str = "ocr",
    max_model_len: int = 8192,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.8,
    apply_smart_resize: bool = True,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = None,
    verbose: bool = False,
):
    """Process images from HF dataset through PaddleOCR-VL model."""

    # Check CUDA availability first
    check_cuda_availability()

    # Compatibility warning
    logger.warning("⚠️  PaddleOCR-VL batch processing in vLLM is experimental.")
    logger.warning("If initialization fails, consider using:")
    logger.warning("  1. LightOnOCR (1B) - smallest stable model")
    logger.warning("  2. DoTS OCR (1.7B) - multilingual support")
    logger.warning("  3. vllm serve mode for PaddleOCR-VL")

    # Track processing start time
    start_time = datetime.now()

    # Enable high-performance Xet downloads
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Validate task mode
    if task_mode not in TASK_MODES:
        raise ValueError(
            f"Invalid task_mode '{task_mode}'. Choose from: {list(TASK_MODES.keys())}"
        )

    # Default output column is 'markdown' for consistency across scripts
    if output_column is None:
        output_column = "markdown"

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

    # Initialize vLLM model
    model_name = "PaddlePaddle/PaddleOCR-VL"
    logger.info(f"Initializing vLLM with {model_name}")
    logger.info("This may take a minute on first run (model is only 0.9B)...")

    # Note: PaddleOCR-VL requires specific vLLM configuration
    # The model needs to be loaded with specific settings
    try:
        # Try with standard configuration first
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            max_num_batched_tokens=16384,
            enable_prefix_caching=False,
            enforce_eager=True,
        )
    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR-VL with vLLM: {e}")
        logger.error("PaddleOCR-VL may require a newer vLLM version or server mode.")
        logger.error("Try running with vllm serve instead:")
        logger.error(f"  vllm serve {model_name} --trust-remote-code \\")
        logger.error("    --max-num-batched-tokens 16384 \\")
        logger.error("    --no-enable-prefix-caching \\")
        logger.error("    --mm-processor-cache-gb 0")
        sys.exit(1)

    # Sampling parameters - deterministic for OCR
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    if apply_smart_resize:
        logger.info("Smart resize enabled (PaddleOCR-VL's adaptive resolution)")

    # Process images in batches
    all_outputs = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc=f"PaddleOCR-VL {task_mode.upper()} processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch with task-specific prefix
            batch_messages = [
                make_ocr_message(
                    img, task_mode=task_mode, apply_smart_resize=apply_smart_resize
                )
                for img in batch_images
            ]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                all_outputs.append(text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_outputs.extend([f"[{task_mode.upper()} ERROR]"] * len(batch_images))

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column to dataset
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Handle inference_info tracking (for multi-model comparisons)
    inference_entry = {
        "model_id": model_name,
        "model_name": "PaddleOCR-VL",
        "model_size": "0.9B",
        "task_mode": task_mode,
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "smart_resize": apply_smart_resize,
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

    # Push to hub
    logger.info(f"Pushing to {output_dataset}")
    dataset.push_to_hub(output_dataset, private=private, token=HF_TOKEN)

    # Create and push dataset card
    logger.info("Creating dataset card")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=model_name,
        task_mode=task_mode,
        num_samples=len(dataset),
        processing_time=processing_time_str,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        temperature=temperature,
        apply_smart_resize=apply_smart_resize,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("✅ PaddleOCR-VL processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")
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
        print("PaddleOCR-VL Document Processing")
        print("=" * 80)
        print("\nUltra-compact 0.9B OCR model with task-specific capabilities")
        print("\nFeatures:")
        print("- 🎯 Smallest OCR model - Only 0.9B parameters")
        print("- 📝 OCR mode - General text extraction")
        print("- 📊 Table mode - HTML table recognition")
        print("- 📐 Formula mode - LaTeX mathematical notation")
        print("- 📈 Chart mode - Structured chart analysis")
        print("- 🌍 Multilingual support")
        print("- ⚡ Fast initialization and inference")
        print("- 🔧 Based on ERNIE-4.5 (unique architecture)")
        print("\nTask Modes:")
        for mode, description in TASK_DESCRIPTIONS.items():
            print(f"  {mode:8} - {description}")
        print("\nExample usage:")
        print("\n1. Basic OCR (default mode):")
        print("   uv run paddleocr-vl.py input-dataset output-dataset")
        print("\n2. Table extraction:")
        print("   uv run paddleocr-vl.py docs tables-extracted --task-mode table")
        print("\n3. Formula recognition:")
        print(
            "   uv run paddleocr-vl.py papers formulas --task-mode formula --batch-size 32"
        )
        print("\n4. Chart analysis:")
        print("   uv run paddleocr-vl.py diagrams charts-analyzed --task-mode chart")
        print("\n5. Test with small sample:")
        print("   uv run paddleocr-vl.py dataset test --max-samples 10 --shuffle")
        print("\n6. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print(
            '     -e HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())") \\'
        )
        print("     -e HF_XET_HIGH_PERFORMANCE=1 \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/paddleocr-vl.py \\"
        )
        print("       input-dataset output-dataset --task-mode ocr")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run paddleocr-vl.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document processing using PaddleOCR-VL (0.9B task-specific model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task Modes:
  ocr       General text extraction to markdown (default)
  table     Table extraction to HTML format
  formula   Mathematical formula recognition to LaTeX
  chart     Chart and diagram analysis

Examples:
  # Basic text OCR
  uv run paddleocr-vl.py my-docs analyzed-docs

  # Extract tables from documents
  uv run paddleocr-vl.py papers tables --task-mode table

  # Recognize mathematical formulas
  uv run paddleocr-vl.py textbooks formulas --task-mode formula

  # Analyze charts and diagrams
  uv run paddleocr-vl.py reports charts --task-mode chart

  # Test with random sampling
  uv run paddleocr-vl.py large-dataset test --max-samples 50 --shuffle --task-mode ocr

  # Disable smart resize for original resolution
  uv run paddleocr-vl.py images output --no-smart-resize
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
        default=16,
        help="Batch size for processing (default: 16)",
    )
    parser.add_argument(
        "--task-mode",
        choices=list(TASK_MODES.keys()),
        default="ocr",
        help="Task type: ocr (default), table, formula, or chart",
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
        default=4096,
        help="Maximum tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--no-smart-resize",
        action="store_true",
        help="Disable PaddleOCR-VL's smart resize, use original image size",
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
        help="Column name for output (default: markdown)",
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
        task_mode=args.task_mode,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        apply_smart_resize=not args.no_smart_resize,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        verbose=args.verbose,
    )
