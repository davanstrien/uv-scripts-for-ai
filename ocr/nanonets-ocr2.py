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
# ]
#
# ///

"""
Convert document images to markdown using Nanonets-OCR2-3B with vLLM.

This script processes images through the Nanonets-OCR2-3B model (3.75B params)
to extract text and structure as markdown, ideal for document understanding tasks.

Features:
- LaTeX equation recognition
- Table extraction and formatting (HTML)
- Document structure preservation
- Image descriptions and captions
- Signature and watermark detection
- Checkbox recognition
- Multilingual support
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


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.",
) -> List[Dict]:
    """Create chat message for OCR processing."""
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Return message in vLLM format
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def create_dataset_card(
    source_dataset: str,
    model: str,
    num_samples: int,
    processing_time: str,
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
- nanonets
- nanonets-ocr2
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains markdown-formatted OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using Nanonets-OCR2-3B.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Model Size**: 3.75B parameters
- **Number of Samples**: {num_samples:,}
- **Processing Time**: {processing_time}
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

### Configuration

- **Image Column**: `{image_column}`
- **Output Column**: `markdown`
- **Dataset Split**: `{split}`
- **Batch Size**: {batch_size}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

Nanonets-OCR2-3B is a state-of-the-art document OCR model that excels at:
- 📐 **LaTeX equations** - Mathematical formulas preserved in LaTeX format
- 📊 **Tables** - Extracted and formatted as HTML
- 📝 **Document structure** - Headers, lists, and formatting maintained
- 🖼️ **Images** - Captions and descriptions included in `<img>` tags
- ☑️ **Forms** - Checkboxes rendered as ☐/☑
- 🔖 **Watermarks** - Wrapped in `<watermark>` tags
- 📄 **Page numbers** - Wrapped in `<page_number>` tags
- 🌍 **Multilingual** - Supports multiple languages

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

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) Nanonets OCR2 script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr2.py \\
    {source_dataset} \\
    <output-dataset> \\
    --model {model} \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens} \\
    --gpu-memory-utilization {gpu_memory_utilization}
```

## Performance

- **Processing Speed**: ~{num_samples / (float(processing_time.split()[0]) * 60):.1f} images/second
- **GPU Configuration**: vLLM with {gpu_memory_utilization:.0%} GPU memory utilization

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    model: str = "nanonets/Nanonets-OCR2-3B",
    max_model_len: int = 8192,
    max_tokens: int = 15000,
    gpu_memory_utilization: float = 0.8,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    verbose: bool = False,
):
    """Process images from HF dataset through Nanonets-OCR2-3B model."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Enable high-performance Xet downloads
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

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
    all_markdown = []

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")

    # Process in batches to avoid memory issues
    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="OCR processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [make_ocr_message(img) for img in batch_images]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract markdown from outputs
            for output in outputs:
                markdown_text = output.outputs[0].text.strip()
                all_markdown.append(markdown_text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_markdown.extend(["[OCR FAILED]"] * len(batch_images))

    # Add markdown column to dataset
    logger.info("Adding markdown column to dataset")
    dataset = dataset.add_column("markdown", all_markdown)

    # Handle inference_info tracking
    logger.info("Updating inference_info...")

    inference_entry = {
        "model_id": model,
        "model_name": "Nanonets-OCR2-3B",
        "column_name": "markdown",
        "timestamp": datetime.now().isoformat(),
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "script": "nanonets-ocr2.py",
        "script_url": "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr2.py",
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
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
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
        print("Nanonets OCR2-3B to Markdown Converter")
        print("=" * 80)
        print("\nThis script converts document images to structured markdown using")
        print("the Nanonets-OCR2-3B model (3.75B params) with vLLM acceleration.")
        print("\nFeatures:")
        print("- LaTeX equation recognition")
        print("- Table extraction and formatting (HTML)")
        print("- Document structure preservation")
        print("- Image descriptions and captions")
        print("- Signature and watermark detection")
        print("- Checkbox recognition (☐/☑)")
        print("- Multilingual support")
        print("\nExample usage:")
        print("\n1. Basic OCR conversion:")
        print("   uv run nanonets-ocr2.py document-images markdown-docs")
        print("\n2. With custom settings:")
        print("   uv run nanonets-ocr2.py scanned-pdfs extracted-text \\")
        print("       --image-column page \\")
        print("       --batch-size 32 \\")
        print("       --gpu-memory-utilization 0.8")
        print("\n3. Process a subset for testing:")
        print("   uv run nanonets-ocr2.py large-dataset test-output --max-samples 10")
        print("\n4. Random sample from ordered dataset:")
        print("   uv run nanonets-ocr2.py ordered-dataset random-test \\")
        print("       --max-samples 50 --shuffle")
        print("\n5. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print(
            '     -e HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())") \\'
        )
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr2.py \\"
        )
        print("       your-document-dataset \\")
        print("       your-markdown-output")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run nanonets-ocr2.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="OCR images to markdown using Nanonets-OCR2-3B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run nanonets-ocr2.py my-images-dataset ocr-results

  # With specific image column
  uv run nanonets-ocr2.py documents extracted-text --image-column scan

  # Process subset for testing
  uv run nanonets-ocr2.py large-dataset test-output --max-samples 100

  # Random sample from ordered dataset
  uv run nanonets-ocr2.py ordered-dataset random-sample --max-samples 50 --shuffle
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
        "--model",
        default="nanonets/Nanonets-OCR2-3B",
        help="Model to use (default: nanonets/Nanonets-OCR2-3B)",
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
        default=15000,
        help="Maximum tokens to generate (default: 15000, per model card recommendation)",
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
        shuffle=args.shuffle,
        seed=args.seed,
        verbose=args.verbose,
    )
