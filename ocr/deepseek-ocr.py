# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "pillow",
#     "torch",
#     "torchvision",
#     "transformers==4.46.3",
#     "tokenizers==0.20.3",
#     "tqdm",
#     "addict",
#     "matplotlib",
#     "einops",
#     "easydict",
# ]
#
# ///

"""
Convert document images to markdown using DeepSeek-OCR with Transformers.

This script processes images through the DeepSeek-OCR model to extract
text and structure as markdown, using the official Transformers API.

Features:
- Multiple resolution modes (Tiny/Small/Base/Large/Gundam)
- LaTeX equation recognition
- Table extraction and formatting
- Document structure preservation
- Image grounding and descriptions
- Multilingual support

Note: This script processes images sequentially (no batching) using the
official transformers API. It's slower than vLLM-based scripts but uses
the well-supported official implementation.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolution mode presets
RESOLUTION_MODES = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},  # Dynamic resolution
}


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def create_dataset_card(
    source_dataset: str,
    model: str,
    num_samples: int,
    processing_time: str,
    resolution_mode: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- deepseek
- deepseek-ocr
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains markdown-formatted OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using DeepSeek-OCR.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `markdown`
- **Dataset Split**: `{split}`
- **Resolution Mode**: {resolution_mode}
- **Base Size**: {base_size}
- **Image Size**: {image_size}
- **Crop Mode**: {crop_mode}

## Model Information

DeepSeek-OCR is a state-of-the-art document OCR model that excels at:
- 📐 **LaTeX equations** - Mathematical formulas preserved in LaTeX format
- 📊 **Tables** - Extracted and formatted as HTML/markdown
- 📝 **Document structure** - Headers, lists, and formatting maintained
- 🖼️ **Image grounding** - Spatial layout and bounding box information
- 🔍 **Complex layouts** - Multi-column and hierarchical structures
- 🌍 **Multilingual** - Supports multiple languages

### Resolution Modes

- **Tiny** (512×512): Fast processing, 64 vision tokens
- **Small** (640×640): Balanced speed/quality, 100 vision tokens
- **Base** (1024×1024): High quality, 256 vision tokens
- **Large** (1280×1280): Maximum quality, 400 vision tokens
- **Gundam** (dynamic): Adaptive multi-tile processing for large documents

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format with preserved structure
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("{{{{output_dataset_id}}}}", split="{split}")

# Access the markdown text
for example in dataset:
    print(example["markdown"])
    break

# View all OCR models applied to this dataset
inference_info = json.loads(dataset[0]["inference_info"])
for info in inference_info:
    print(f"Column: {{{{info['column_name']}}}} - Model: {{{{info['model_id']}}}}")
```

## Reproduction

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) DeepSeek OCR script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --resolution-mode {resolution_mode} \\
    --image-column {image_column}
```

## Performance

- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.1f} images/second
- **Processing Method**: Sequential (Transformers API, no batching)

Note: This uses the official Transformers implementation. For faster batch processing,
consider using the vLLM version once DeepSeek-OCR is officially supported by vLLM.

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def process_single_image(
    model,
    tokenizer,
    image: Image.Image,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    temp_image_path: str,
    temp_output_dir: str,
) -> str:
    """Process a single image through DeepSeek-OCR."""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save to temp file (model.infer expects a file path)
    image.save(temp_image_path, format="PNG")

    # Run inference
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=temp_image_path,
        output_path=temp_output_dir,  # Need real directory path
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=False,
        test_compress=False,
    )

    return result if isinstance(result, str) else str(result)


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    model: str = "deepseek-ai/DeepSeek-OCR",
    resolution_mode: str = "gundam",
    base_size: Optional[int] = None,
    image_size: Optional[int] = None,
    crop_mode: Optional[bool] = None,
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
):
    """Process images from HF dataset through DeepSeek-OCR model."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()



    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Determine resolution settings
    if resolution_mode in RESOLUTION_MODES:
        mode_config = RESOLUTION_MODES[resolution_mode]
        final_base_size = base_size if base_size is not None else mode_config["base_size"]
        final_image_size = image_size if image_size is not None else mode_config["image_size"]
        final_crop_mode = crop_mode if crop_mode is not None else mode_config["crop_mode"]
        logger.info(f"Using resolution mode: {resolution_mode}")
    else:
        # Custom mode - require all parameters
        if base_size is None or image_size is None or crop_mode is None:
            raise ValueError(
                f"Invalid resolution mode '{resolution_mode}'. "
                f"Use one of {list(RESOLUTION_MODES.keys())} or specify "
                f"--base-size, --image-size, and --crop-mode manually."
            )
        final_base_size = base_size
        final_image_size = image_size
        final_crop_mode = crop_mode
        resolution_mode = "custom"

    logger.info(
        f"Resolution: base_size={final_base_size}, "
        f"image_size={final_image_size}, crop_mode={final_crop_mode}"
    )

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

    # Initialize model
    logger.info(f"Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    try:
        model_obj = AutoModel.from_pretrained(
            model,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
    except Exception as e:
        logger.warning(f"Failed to load with flash_attention_2: {e}")
        logger.info("Falling back to standard attention...")
        model_obj = AutoModel.from_pretrained(
            model,
            trust_remote_code=True,
            use_safetensors=True,
        )

    model_obj = model_obj.eval().cuda().to(torch.bfloat16)
    logger.info("Model loaded successfully")

    # Process images sequentially
    all_markdown = []

    logger.info(f"Processing {len(dataset)} images (sequential, no batching)")
    logger.info("Note: This may be slower than vLLM-based scripts")

    # Create temp directories for image files and output (simple local dirs)
    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)
    temp_image_path = str(temp_dir / "temp_image.png")

    temp_output_dir = Path("temp_output")
    temp_output_dir.mkdir(exist_ok=True)

    try:
        for i in tqdm(range(len(dataset)), desc="OCR processing"):
            try:
                image = dataset[i][image_column]

                # Handle different image formats
                if isinstance(image, dict) and "bytes" in image:
                    from io import BytesIO
                    image = Image.open(BytesIO(image["bytes"]))
                elif isinstance(image, str):
                    image = Image.open(image)
                elif not isinstance(image, Image.Image):
                    raise ValueError(f"Unsupported image type: {type(image)}")

                # Process image
                result = process_single_image(
                    model_obj,
                    tokenizer,
                    image,
                    prompt,
                    final_base_size,
                    final_image_size,
                    final_crop_mode,
                    temp_image_path,
                    str(temp_output_dir),
                )

                all_markdown.append(result)

            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                all_markdown.append("[OCR FAILED]")

    finally:
        # Clean up temp directories
        try:
            shutil.rmtree(temp_dir)
            shutil.rmtree(temp_output_dir)
        except:
            pass

    # Add markdown column to dataset
    logger.info("Adding markdown column to dataset")
    dataset = dataset.add_column("markdown", all_markdown)

    # Handle inference_info tracking
    logger.info("Updating inference_info...")

    # Check for existing inference_info
    if "inference_info" in dataset.column_names:
        try:
            existing_info = json.loads(dataset[0]["inference_info"])
            if not isinstance(existing_info, list):
                existing_info = [existing_info]
        except (json.JSONDecodeError, TypeError):
            existing_info = []
        dataset = dataset.remove_columns(["inference_info"])
    else:
        existing_info = []

    # Add new inference info
    new_info = {
        "column_name": "markdown",
        "model_id": model,
        "processing_date": datetime.now().isoformat(),
        "resolution_mode": resolution_mode,
        "base_size": final_base_size,
        "image_size": final_image_size,
        "crop_mode": final_crop_mode,
        "prompt": prompt,
        "script": "deepseek-ocr.py",
        "script_version": "1.0.0",
        "script_url": "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr.py",
        "implementation": "transformers (sequential)",
    }
    existing_info.append(new_info)

    # Add updated inference_info column
    info_json = json.dumps(existing_info, ensure_ascii=False)
    dataset = dataset.add_column("inference_info", [info_json] * len(dataset))

    # Push to hub
    logger.info(f"Pushing to {output_dataset}")
    dataset.push_to_hub(output_dataset, private=private, token=HF_TOKEN)

    # Calculate processing time
    end_time = datetime.now()
    processing_duration = end_time - start_time
    processing_time = f"{processing_duration.total_seconds() / 60:.1f} minutes"

    # Create and push dataset card
    logger.info("Creating dataset card...")
    card_content = create_dataset_card(
        source_dataset=input_dataset,
        model=model,
        num_samples=len(dataset),
        processing_time=processing_time,
        resolution_mode=resolution_mode,
        base_size=final_base_size,
        image_size=final_image_size,
        crop_mode=final_crop_mode,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)
    logger.info("✅ Dataset card created and pushed!")

    logger.info("✅ OCR conversion complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("DeepSeek-OCR to Markdown Converter (Transformers)")
        print("=" * 80)
        print("\nThis script converts document images to markdown using")
        print("DeepSeek-OCR with the official Transformers API.")
        print("\nFeatures:")
        print("- Multiple resolution modes (Tiny/Small/Base/Large/Gundam)")
        print("- LaTeX equation recognition")
        print("- Table extraction and formatting")
        print("- Document structure preservation")
        print("- Image grounding and spatial layout")
        print("- Multilingual support")
        print("\nNote: Sequential processing (no batching). Slower than vLLM scripts.")
        print("\nExample usage:")
        print("\n1. Basic OCR conversion (Gundam mode - dynamic resolution):")
        print("   uv run deepseek-ocr.py document-images markdown-docs")
        print("\n2. High quality mode (Large - 1280×1280):")
        print("   uv run deepseek-ocr.py scanned-pdfs extracted-text --resolution-mode large")
        print("\n3. Fast processing (Tiny - 512×512):")
        print("   uv run deepseek-ocr.py quick-test output --resolution-mode tiny")
        print("\n4. Process a subset for testing:")
        print("   uv run deepseek-ocr.py large-dataset test-output --max-samples 10")
        print("\n5. Custom resolution:")
        print("   uv run deepseek-ocr.py dataset output \\")
        print("       --base-size 1024 --image-size 640 --crop-mode")
        print("\n6. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print('     --secrets HF_TOKEN \\')
        print("     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr.py \\")
        print("       your-document-dataset \\")
        print("       your-markdown-output")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run deepseek-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="OCR images to markdown using DeepSeek-OCR (Transformers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Resolution Modes:
  tiny      512×512 pixels, fast processing (64 vision tokens)
  small     640×640 pixels, balanced (100 vision tokens)
  base      1024×1024 pixels, high quality (256 vision tokens)
  large     1280×1280 pixels, maximum quality (400 vision tokens)
  gundam    Dynamic multi-tile processing (adaptive)

Examples:
  # Basic usage with default Gundam mode
  uv run deepseek-ocr.py my-images-dataset ocr-results

  # High quality processing
  uv run deepseek-ocr.py documents extracted-text --resolution-mode large

  # Fast processing for testing
  uv run deepseek-ocr.py dataset output --resolution-mode tiny --max-samples 100

  # Custom resolution settings
  uv run deepseek-ocr.py dataset output --base-size 1024 --image-size 640 --crop-mode
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
        "--model",
        default="deepseek-ai/DeepSeek-OCR",
        help="Model to use (default: deepseek-ai/DeepSeek-OCR)",
    )
    parser.add_argument(
        "--resolution-mode",
        default="gundam",
        choices=list(RESOLUTION_MODES.keys()) + ["custom"],
        help="Resolution mode preset (default: gundam)",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        help="Base resolution size (overrides resolution-mode)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Image tile size (overrides resolution-mode)",
    )
    parser.add_argument(
        "--crop-mode",
        action="store_true",
        help="Enable dynamic multi-tile cropping (overrides resolution-mode)",
    )
    parser.add_argument(
        "--prompt",
        default="<image>\n<|grounding|>Convert the document to markdown.",
        help="Prompt for OCR (default: grounding markdown conversion)",
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

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        model=args.model,
        resolution_mode=args.resolution_mode,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode if args.crop_mode else None,
        prompt=args.prompt,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
    )
