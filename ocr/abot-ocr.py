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
Convert document images to Markdown using ABot-OCR with vLLM.

ABot-OCR (acvlab/ABot-OCR) is a compact Qwen3-VL-based document OCR model that
converts PDF/document page images into structured Markdown, including:
- Text with original document structure (headings, paragraphs, lists)
- Mathematical formulas in LaTeX (inline \\( \\) and block \\[ \\])
- Tables in HTML (<table>…</table>)

Model:  https://huggingface.co/acvlab/ABot-OCR
Paper:  https://arxiv.org/abs/2605.27978
Code:   https://github.com/amap-cvlab/ABot-OCR

HF Jobs note: ABot-OCR (Qwen3-VL) uses vLLM's flashinfer sampler, which needs CUDA
kernels the default uv-script image can't build ("Could not find nvcc"). Run on Jobs
with the pre-built vLLM image so the kernels are reused (same image-mode pattern as
nuextract3.py / paddleocr-vl-1.6.py):
    --image vllm/vllm-openai:latest
    --python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages
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

# ABot-OCR's recommended document→Markdown prompt (from the model's reference
# inference script; LaTeX delimiters de-mangled to proper \( \) and \[ \]).
DEFAULT_PROMPT = r"""You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
- Enclose block formulas with \[ \]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

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


def post_process_text(text: str, threshold: int = 8000) -> str:
    """Trim runaway repetition loops that VLM OCR can fall into on dense pages.

    Mirrors ABot-OCR's reference post-processing: if the tail of a long output
    repeats the same short substring many times, collapse it to a single copy.
    """
    n = len(text)
    if n < threshold:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[: n - length * (count - 1)]
    return text


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = DEFAULT_PROMPT,
) -> List[Dict]:
    """Create a vLLM chat message for OCR processing."""
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

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
- abot
- abot-ocr
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains Markdown-formatted OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using [ABot-OCR](https://huggingface.co/{model}).

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Paper**: [arxiv.org/abs/2605.27978](https://arxiv.org/abs/2605.27978)
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

ABot-OCR is a compact Qwen3-VL-based document OCR model that converts page images to Markdown:
- 📐 **LaTeX equations** — inline `\\( \\)` and block `\\[ \\]`
- 📊 **Tables** — extracted as HTML (`<table>…</table>`)
- 📝 **Document structure** — headings, paragraphs, and lists preserved

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in Markdown format with preserved structure
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{{{{output_dataset_id}}}}", split="{split}")
for example in dataset:
    print(example["markdown"])
    break
```

## Reproduction

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) ABot-OCR script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/abot-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --max-model-len {max_model_len} \\
    --max-tokens {max_tokens} \\
    --gpu-memory-utilization {gpu_memory_utilization}
```

Generated with 🤖 [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    model: str = "acvlab/ABot-OCR",
    max_model_len: int = 16384,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    verbose: bool = False,
):
    """Process images from a HF dataset through the ABot-OCR model."""

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
        enforce_eager=True,  # avoid warmup kernel compilation (no nvcc in the image)
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for OCR
        max_tokens=max_tokens,
    )

    # Process images in batches
    all_markdown = []

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="OCR processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            batch_messages = [make_ocr_message(img) for img in batch_images]
            outputs = llm.chat(batch_messages, sampling_params)
            for output in outputs:
                markdown_text = post_process_text(output.outputs[0].text.strip())
                all_markdown.append(markdown_text)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            all_markdown.extend(["[OCR FAILED]"] * len(batch_images))

    # Add markdown column to dataset
    logger.info("Adding markdown column to dataset")
    dataset = dataset.add_column("markdown", all_markdown)

    # Handle inference_info tracking
    logger.info("Updating inference_info...")

    inference_entry = {
        "model_id": model,
        "model_name": "ABot-OCR",
        "column_name": "markdown",
        "timestamp": datetime.now().isoformat(),
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "script": "abot-ocr.py",
        "script_url": "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/abot-ocr.py",
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
        print("ABot-OCR to Markdown Converter")
        print("=" * 80)
        print("\nConverts document images to structured Markdown using the")
        print("ABot-OCR model (Qwen3-VL based) with vLLM acceleration.")
        print("\nFeatures:")
        print("- Document structure preserved (headings, paragraphs, lists)")
        print("- LaTeX equation recognition (inline \\( \\) and block \\[ \\])")
        print("- Table extraction as HTML")
        print("\nExample usage:")
        print("\n1. Basic OCR conversion:")
        print("   uv run abot-ocr.py document-images markdown-docs")
        print("\n2. Process a subset for testing:")
        print("   uv run abot-ocr.py large-dataset test-output --max-samples 10")
        print("\n3. Running on HF Jobs (use the pre-built vLLM image for flashinfer kernels):")
        print("   hf jobs uv run --flavor l4x1 \\")
        print("     --image vllm/vllm-openai:latest \\")
        print(
            "     --python /usr/bin/python3 -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \\"
        )
        print("     -s HF_TOKEN \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/abot-ocr.py \\"
        )
        print("       your-document-dataset \\")
        print("       your-markdown-output \\")
        print("       --max-samples 10")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run abot-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="OCR images to Markdown using ABot-OCR (Qwen3-VL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run abot-ocr.py my-images-dataset ocr-results

  # With specific image column
  uv run abot-ocr.py documents extracted-text --image-column scan

  # Process subset for testing
  uv run abot-ocr.py large-dataset test-output --max-samples 100

  # Random sample from ordered dataset
  uv run abot-ocr.py ordered-dataset random-sample --max-samples 50 --shuffle
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
        default="acvlab/ABot-OCR",
        help="Model to use (default: acvlab/ABot-OCR)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum model context length (default: 16384)",
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
