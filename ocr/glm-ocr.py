# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=3.1.0",
#     "pyarrow>=17.0.0,<18.0.0",
#     "huggingface-hub",
#     "pillow",
#     "vllm",
#     "toolz",
#     "torch",
# ]
#
# [[tool.uv.index]]
# url = "https://wheels.vllm.ai/nightly/cu129"
#
# [tool.uv]
# prerelease = "allow"
# override-dependencies = ["transformers>=5.1.0"]
# ///

"""
Convert document images to markdown using GLM-OCR with vLLM.

GLM-OCR is a compact 0.9B parameter OCR model achieving 94.62% on OmniDocBench V1.5.
Uses CogViT visual encoder with GLM-0.5B language decoder and Multi-Token Prediction
(MTP) loss for fast, accurate document parsing.

NOTE: Requires vLLM nightly wheels from cu129 variant (GLM-OCR added in v0.16.0,
PR #33005) and transformers>=5.1.0 (GLM-OCR support landed in stable release).
Uses https://wheels.vllm.ai/nightly/cu129 which has x86_64 wheels.
First run may take a few minutes to download and install dependencies.

Features:
- 0.9B parameters (ultra-compact)
- 94.62% on OmniDocBench V1.5 (SOTA for sub-1B models)
- Text recognition with markdown output
- LaTeX formula recognition
- Table extraction (HTML format)
- Multilingual: zh, en, fr, es, ru, de, ja, ko
- MIT licensed

Model: zai-org/GLM-OCR
vLLM: Requires vLLM nightly build + transformers>=5.1.0
Performance: 94.62% on OmniDocBench V1.5
"""

import argparse
import base64
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
from toolz import partition_all
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "zai-org/GLM-OCR"

# Task prompts as specified by the model
TASK_PROMPTS = {
    "ocr": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}


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
    task: str = "ocr",
) -> List[Dict]:
    """
    Create chat message for OCR processing.

    GLM-OCR uses a chat format with an image and a task prompt prefix.
    Supported tasks: ocr, formula, table.
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

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    prompt_text = TASK_PROMPTS.get(task, TASK_PROMPTS["ocr"])

    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt_text},
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
    temperature: float,
    top_p: float,
    task: str,
    image_column: str = "image",
    split: str = "train",
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]
    task_desc = {
        "ocr": "text recognition",
        "formula": "formula recognition",
        "table": "table recognition",
    }

    return f"""---
tags:
- ocr
- document-processing
- glm-ocr
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using GLM-OCR, a compact 0.9B OCR model achieving SOTA performance.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Task**: {task_desc.get(task, task)}
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
- **Temperature**: {temperature}
- **Top P**: {top_p}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

GLM-OCR is a compact, high-performance OCR model:
- 0.9B parameters
- 94.62% on OmniDocBench V1.5
- CogViT visual encoder + GLM-0.5B language decoder
- Multi-Token Prediction (MTP) loss for efficiency
- Multilingual: zh, en, fr, es, ru, de, ja, ko
- MIT licensed

## Dataset Structure

The dataset contains all original columns plus:
- `markdown`: The extracted text in markdown format
- `inference_info`: JSON list tracking all OCR models applied to this dataset

## Reproduction

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --task {task}
```

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    max_model_len: int = 8192,
    max_tokens: int = 8192,
    temperature: float = 0.01,
    top_p: float = 0.00001,
    repetition_penalty: float = 1.1,
    gpu_memory_utilization: float = 0.8,
    task: str = "ocr",
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "markdown",
    verbose: bool = False,
    config: str = None,
    create_pr: bool = False,
):
    """Process images from HF dataset through GLM-OCR model."""

    check_cuda_availability()

    start_time = datetime.now()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Validate task
    if task not in TASK_PROMPTS:
        logger.error(f"Unknown task '{task}'. Supported: {list(TASK_PROMPTS.keys())}")
        sys.exit(1)

    logger.info(f"Using model: {MODEL}")
    logger.info(f"Task: {task} (prompt: '{TASK_PROMPTS[task]}')")

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)

    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Initialize vLLM
    logger.info("Initializing vLLM with GLM-OCR")
    logger.info("This may take a few minutes on first run...")
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    # Sampling defaults from GLM-OCR SDK (github.com/zai-org/GLM-OCR)
    # glmocr/config.py PageLoaderConfig: temperature=0.01, top_p=0.00001,
    # top_k=1, repetition_penalty=1.1, max_tokens=16384
    # generation_config.json on HF also sets do_sample=false (greedy)
    # Note: SDK uses max_tokens=16384 but vLLM caps at max_model_len (8192)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")

    all_outputs = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    processed = 0

    for batch_num, batch_indices in enumerate(
        partition_all(batch_size, range(len(dataset))), 1
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        logger.info(
            f"Batch {batch_num}/{total_batches} "
            f"({processed}/{len(dataset)} images done)"
        )

        try:
            batch_messages = [make_ocr_message(img, task=task) for img in batch_images]

            outputs = llm.chat(batch_messages, sampling_params)

            for output in outputs:
                text = output.outputs[0].text.strip()
                all_outputs.append(text)

            processed += len(batch_images)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            all_outputs.extend(["[OCR ERROR]"] * len(batch_images))
            processed += len(batch_images)

    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Inference info tracking
    inference_entry = {
        "model_id": MODEL,
        "model_name": "GLM-OCR",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
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
                commit_message=f"Add {MODEL} OCR results ({len(dataset)} samples)"
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
        model=MODEL,
        num_samples=len(dataset),
        processing_time=processing_time_str,
        batch_size=batch_size,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        temperature=temperature,
        top_p=top_p,
        task=task,
        image_column=image_column,
        split=split,
    )

    card = DatasetCard(card_content)
    card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("Done! GLM-OCR processing complete.")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")
    logger.info(
        f"Processing speed: {len(dataset) / processing_duration.total_seconds():.2f} images/sec"
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
    if len(sys.argv) == 1:
        print("=" * 70)
        print("GLM-OCR Document Processing")
        print("=" * 70)
        print("\n0.9B OCR model - 94.62% on OmniDocBench V1.5")
        print("\nTask modes:")
        print("  ocr      - Text recognition (default)")
        print("  formula  - LaTeX formula recognition")
        print("  table    - Table extraction")
        print("\nExamples:")
        print("\n1. Basic OCR:")
        print("   uv run glm-ocr.py input-dataset output-dataset")
        print("\n2. Formula recognition:")
        print("   uv run glm-ocr.py docs results --task formula")
        print("\n3. Table extraction:")
        print("   uv run glm-ocr.py docs results --task table")
        print("\n4. Test with small sample:")
        print("   uv run glm-ocr.py large-dataset test --max-samples 10 --shuffle")
        print("\n5. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print("     -s HF_TOKEN \\")
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr.py \\"
        )
        print("       input-dataset output-dataset --batch-size 16")
        print("\nFor full help: uv run glm-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using GLM-OCR (0.9B, 94.62% OmniDocBench V1.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task modes:
  ocr      Text recognition to markdown (default)
  formula  LaTeX formula recognition
  table    Table extraction

Examples:
  uv run glm-ocr.py my-docs analyzed-docs
  uv run glm-ocr.py docs results --task formula
  uv run glm-ocr.py large-dataset test --max-samples 50 --shuffle
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
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192, capped by max-model-len)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Sampling temperature (default: 0.01, near-greedy for OCR accuracy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.00001,
        help="Top-p sampling parameter (default: 0.00001, near-greedy)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty to prevent loops (default: 1.1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--task",
        choices=["ocr", "formula", "table"],
        default="ocr",
        help="OCR task mode (default: ocr)",
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
        "--config",
        help="Config/subset name when pushing to Hub (for benchmarking multiple models in one repo)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of pushing directly (for parallel benchmarking)",
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
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        gpu_memory_utilization=args.gpu_memory_utilization,
        task=args.task,
        hf_token=args.hf_token,
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        verbose=args.verbose,
        config=args.config,
        create_pr=args.create_pr,
    )
