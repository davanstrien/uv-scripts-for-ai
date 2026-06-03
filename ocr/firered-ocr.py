# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
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
Convert document images to markdown using FireRed-OCR with vLLM.

FireRed-OCR is a 2.1B document OCR model fine-tuned from Qwen3-VL-2B-Instruct.
It converts PDF/document images to structured Markdown with LaTeX formulas and
HTML tables. Apache 2.0 licensed.

Model: FireRedTeam/FireRed-OCR
vLLM: Uses stable Qwen3-VL support (>=0.15.1)
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Union

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from toolz import partition_all
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# FireRed-OCR Prompt (from official conv_for_infer.py)
# Source: https://github.com/FireRedTeam/FireRed-OCR/blob/main/conv_for_infer.py
# ────────────────────────────────────────────────────────────────

FIRERED_OCR_PROMPT = """You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \\( \\). For example: This is an inline formula \\( E = mc^2 \\)
- Enclose block formulas with \\[ \\]. For example: \\[ \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments."""


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
    prompt: str = FIRERED_OCR_PROMPT,
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

    # Convert to RGB
    pil_img = pil_img.convert("RGB")

    # Convert to base64 data URI (JPEG is faster than PNG for encoding)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    data_uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

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
- firered-ocr
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using FireRed-OCR, a 2.1B model fine-tuned from Qwen3-VL-2B-Instruct.

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
- **Batch Size**: {batch_size}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

FireRed-OCR is a document OCR model that converts images to structured Markdown:
- Fine-tuned from Qwen3-VL-2B-Instruct (2.1B parameters)
- LaTeX formula support (inline and block)
- HTML table extraction
- Layout-aware text extraction
- Apache 2.0 licensed

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format
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

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) FireRed-OCR script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/firered-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens} \\
    --gpu-memory-utilization {gpu_memory_utilization}
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    model: str = "FireRedTeam/FireRed-OCR",
    max_model_len: int = 32768,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
    config: str = None,
    create_pr: bool = False,
):
    """Process images from HF dataset through FireRed-OCR model."""

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

    # Shuffle if requested
    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Initialize vLLM model
    logger.info(f"Initializing vLLM with model: {model}")
    logger.info("This may take a few minutes on first run...")
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

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")

    # Process images in batches
    all_outputs = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="FireRed-OCR processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [
                make_ocr_message(img, FIRERED_OCR_PROMPT) for img in batch_images
            ]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                all_outputs.append(text)

        except Exception as e:
            logger.warning(
                f"Batch failed ({len(batch_images)} images), retrying individually: {e}"
            )
            for img in batch_images:
                try:
                    msg = make_ocr_message(img, FIRERED_OCR_PROMPT)
                    out = llm.chat([msg], sampling_params)
                    all_outputs.append(out[0].outputs[0].text.strip())
                except Exception as img_e:
                    logger.error(f"Image failed: {img_e}")
                    all_outputs.append("[OCR ERROR]")

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column to dataset (remove existing column if present)
    if output_column in dataset.column_names:
        logger.info(f"Removing existing '{output_column}' column before adding new results")
        dataset = dataset.remove_columns([output_column])
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Handle inference_info tracking (for multi-model comparisons)
    inference_entry = {
        "model_id": model,
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
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
    dataset.push_to_hub(
        output_dataset,
        private=private,
        token=HF_TOKEN,
        **({"config_name": config} if config else {}),
        create_pr=create_pr,
        commit_message=f"Add {model} OCR results ({len(dataset)} samples)"
        + (f" [{config}]" if config else ""),
    )

    # Create and push dataset card (skip for PR-based benchmark runs)
    if not create_pr:
        logger.info("Creating dataset card")
        card_content = create_dataset_card(
            source_dataset=input_dataset,
            model=model,
            num_samples=len(dataset),
            processing_time=processing_time_str,
            batch_size=batch_size,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            image_column=image_column,
            split=split,
        )

        card = DatasetCard(card_content)
        card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("FireRed-OCR processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("FireRed-OCR Document Processing")
        print("=" * 80)
        print("\n2.1B document OCR model (Qwen3-VL-2B fine-tune, Apache 2.0)")
        print("\nFeatures:")
        print("- Structured Markdown output")
        print("- LaTeX formula support (inline and block)")
        print("- HTML table extraction")
        print("- Layout-aware text extraction")
        print("\nExample usage:")
        print("\n1. Basic OCR:")
        print("   uv run firered-ocr.py input-dataset output-dataset")
        print("\n2. With custom settings:")
        print(
            "   uv run firered-ocr.py docs analyzed-docs --batch-size 20 --max-samples 100"
        )
        print("\n3. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print("     -s HF_TOKEN \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/firered-ocr.py \\"
        )
        print("       input-dataset output-dataset")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run firered-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using FireRed-OCR (2.1B, Qwen3-VL fine-tune)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic text OCR
  uv run firered-ocr.py my-docs analyzed-docs

  # Random sampling for testing
  uv run firered-ocr.py large-dataset test --max-samples 50 --shuffle

  # Benchmark mode (push as config with PR)
  uv run firered-ocr.py source-data bench-repo --config firered-ocr --create-pr

  # HF Jobs
  hf jobs uv run --flavor l4x1 -s HF_TOKEN \\
      https://huggingface.co/datasets/uv-scripts/ocr/raw/main/firered-ocr.py \\
      input-dataset output-dataset --max-samples 50
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
        default="FireRedTeam/FireRed-OCR",
        help="Model to use (default: FireRedTeam/FireRed-OCR)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model context length (default: 32768)",
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
        output_column=args.output_column,
        config=args.config,
        create_pr=args.create_pr,
    )
