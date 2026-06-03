# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "pillow",
#     "vllm",
#     "tqdm",
#     "toolz",
#     "torch",
#     "triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels",
# ]
#
# [[tool.uv.index]]
# url = "https://wheels.vllm.ai/nightly/cu129"
#
# [tool.uv]
# prerelease = "allow"
# ///

"""
Convert document images to markdown using LightOnOCR with vLLM.

LightOnOCR is a compact 1B multilingual OCR model optimized for production speed.
Combines Pixtral ViT encoder with Qwen3 language model for efficient document parsing.

NOTE: Requires vLLM nightly wheels for LightOnOCR support. First run may take
a few minutes to download and install dependencies.

Features:
- ⚡ Fastest: 5.71 pages/sec on H100 GPU
- 🎯 Compact: Only 1B parameters
- 🌍 Multilingual with European language optimization
- 📐 LaTeX formula recognition
- 📊 Table extraction (markdown format)
- 📝 Document structure preservation
- 🔤 3 vocabulary sizes (151k/32k/16k tokens)

Model: lightonai/LightOnOCR-1B-1025
vLLM: Requires nightly build from main branch
Performance: 76.1% overall benchmark score
"""

import argparse
import base64
import io
import json
import logging
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


# Model variants with different vocabulary sizes
MODEL_VARIANTS = {
    "151k": "lightonai/LightOnOCR-1B-1025",  # Full vocabulary (default)
    "32k": "lightonai/LightOnOCR-0.9B-32k-1025",  # European languages optimized
    "16k": "lightonai/LightOnOCR-0.9B-16k-1025",  # European languages optimized
}


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def resize_image_to_target(image: Image.Image, target_size: int = 1540) -> Image.Image:
    """
    Resize image so longest dimension is target_size while maintaining aspect ratio.

    LightOnOCR was trained with images at 1540px max resolution and 200 DPI.
    """
    width, height = image.size

    # If image is already smaller, don't upscale
    if max(width, height) <= target_size:
        return image

    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    resize: bool = True,
    target_size: int = 1540,
) -> List[Dict]:
    """
    Create chat message for OCR processing.

    LightOnOCR was trained with 1540px max resolution at 200 DPI for optimal results.
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

    # Resize to optimal dimensions for LightOnOCR
    if resize:
        pil_img = resize_image_to_target(pil_img, target_size)
        logger.debug(f"Resized image to {pil_img.size}")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # LightOnOCR uses message format with empty text prompt before image
    # (matching official demo: text first, then image)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]


def create_dataset_card(
    source_dataset: str,
    model: str,
    vocab_size: str,
    num_samples: int,
    processing_time: str,
    batch_size: int,
    max_model_len: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    temperature: float,
    top_p: float,
    target_size: int,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- lighton-ocr
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using LightOnOCR, a fast and compact 1B OCR model.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Vocabulary Size**: {vocab_size} tokens
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `markdown`
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Target Image Size**: {target_size}px (longest dimension)
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **Temperature**: {temperature}
- **Top P**: {top_p}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

LightOnOCR is a fast, compact OCR model that excels at:
- ⚡ **Production Speed** - 5.71 pages/second on H100 GPU
- 🎯 **Compact Size** - Only 1B parameters
- 📐 **LaTeX formulas** - Mathematical notation in LaTeX format
- 📊 **Tables** - Extracted and formatted as markdown
- 📝 **Document structure** - Hierarchy and layout preservation
- 🌍 **Multilingual** - Optimized for European languages
- 🔤 **Flexible vocabulary** - 151k/32k/16k token variants

### Vocabulary Variants

- **151k tokens**: Full vocabulary, supports all languages
- **32k tokens**: European languages optimized (~12% faster decoding)
- **16k tokens**: European languages optimized (~12% faster decoding)

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format with LaTeX formulas
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("{{output_dataset_id}}", split="{split}")

# Access the markdown text
for example in dataset:
    print(example["markdown"])
    break

# View all OCR models applied to this dataset
inference_info = json.loads(dataset[0]["inference_info"])
for info in inference_info:
    print(f"Column: {{info['column_name']}} - Model: {{info['model_id']}}")
```

## Reproduction

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) LightOnOCR script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --vocab-size {vocab_size} \\
    --image-column {image_column} \\
    --batch-size {batch_size}
```

## Performance

- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.2f} images/second
- **Benchmark Score**: 76.1% overall (across diverse document types)
- **Optimization**: Native resolution ViT + lightweight decoder

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    vocab_size: str = "151k",
    max_model_len: int = 8192,
    max_tokens: int = 6500,
    temperature: float = 0.2,
    top_p: float = 0.9,
    gpu_memory_utilization: float = 0.8,
    target_size: int = 1540,
    no_resize: bool = False,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
):
    """Process images from HF dataset through LightOnOCR model."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Get model ID from vocabulary size
    if vocab_size not in MODEL_VARIANTS:
        raise ValueError(
            f"Invalid vocab_size '{vocab_size}'. Choose from: {list(MODEL_VARIANTS.keys())}"
        )
    model = MODEL_VARIANTS[vocab_size]
    logger.info(f"Using model: {model} ({vocab_size} vocabulary)")

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
    logger.info("Initializing vLLM with LightOnOCR")
    logger.info("This may take a few minutes on first run...")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},  # One image per prompt
        enforce_eager=False,  # Use torch.compile for better performance
    )

    # LightOnOCR recommended sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")
    if not no_resize:
        logger.info(f"Images will be resized to {target_size}px (longest dimension)")

    # Process images in batches
    all_outputs = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="LightOnOCR processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [
                make_ocr_message(img, resize=not no_resize, target_size=target_size)
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
            all_outputs.extend(["[OCR ERROR]"] * len(batch_images))

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column to dataset
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Handle inference_info tracking (for multi-model comparisons)
    inference_entry = {
        "model_id": model,
        "model_name": "LightOnOCR",
        "vocab_size": vocab_size,
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "target_size": target_size if not no_resize else "original",
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
        model=model,
        vocab_size=vocab_size,
        num_samples=len(dataset),
        processing_time=processing_time_str,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        temperature=temperature,
        top_p=top_p,
        target_size=target_size,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("✅ LightOnOCR processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")
    logger.info(
        f"Processing speed: {len(dataset) / processing_duration.total_seconds():.2f} images/sec"
    )


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("LightOnOCR Document Processing")
        print("=" * 80)
        print("\nFast, compact 1B OCR model for production workloads")
        print("\nFeatures:")
        print("- ⚡ Fastest processing: 5.71 pages/sec on H100")
        print("- 🎯 Compact: Only 1B parameters")
        print("- 🌍 Multilingual with European language optimization")
        print("- 📐 LaTeX formula recognition")
        print("- 📊 Table extraction (markdown format)")
        print("- 🔤 3 vocabulary sizes for speed/quality tradeoffs")
        print("\nExample usage:")
        print("\n1. Basic OCR (full vocabulary):")
        print("   uv run lighton-ocr.py input-dataset output-dataset")
        print("\n2. European languages optimized (faster):")
        print("   uv run lighton-ocr.py docs results --vocab-size 32k")
        print("\n3. Custom batch size for performance:")
        print("   uv run lighton-ocr.py docs results --batch-size 32")
        print("\n4. Test with small sample:")
        print("   uv run lighton-ocr.py large-dataset test --max-samples 50 --shuffle")
        print("\n5. Original image size (no resize):")
        print("   uv run lighton-ocr.py docs output --no-resize")
        print("\n6. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print(
            '     -e HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())") \\'
        )
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr.py \\"
        )
        print("       input-dataset output-dataset --vocab-size 32k")
        print("\n" + "=" * 80)
        print("\nVocabulary Size Options:")
        print("  151k - Full vocabulary (all languages)")
        print("  32k  - European languages (~12% faster)")
        print("  16k  - European languages (~12% faster)")
        print("\nFor full help, run: uv run lighton-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using LightOnOCR (fast 1B model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vocabulary Size Options:
  151k    Full vocabulary supporting all languages (default)
  32k     European languages optimized (~12% faster decoding)
  16k     European languages optimized (~12% faster decoding)

Examples:
  # Basic text OCR with full vocabulary
  uv run lighton-ocr.py my-docs analyzed-docs

  # Fast processing for European languages
  uv run lighton-ocr.py papers results --vocab-size 32k

  # Test with random sampling
  uv run lighton-ocr.py large-dataset test --max-samples 50 --shuffle

  # Custom batch size for GPU optimization
  uv run lighton-ocr.py dataset output --batch-size 32 --gpu-memory-utilization 0.9
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
        "--vocab-size",
        default="151k",
        choices=list(MODEL_VARIANTS.keys()),
        help="Vocabulary size variant (default: 151k)",
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
        default=6500,
        help="Maximum tokens to generate (default: 6500)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1540,
        help="Target size for longest image dimension in pixels (default: 1540, matching training)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Don't resize images (use original size)",
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

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        target_size=args.target_size,
        no_resize=args.no_resize,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
    )
