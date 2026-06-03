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
# ///

"""
Convert document images to markdown using HunyuanOCR with vLLM.

HunyuanOCR is a lightweight 1B parameter VLM from Tencent designed for complex
multilingual document parsing. This script uses vLLM for processing.

Features:
- 📝 Full document parsing to markdown
- 📊 Table extraction (HTML format)
- 📐 Formula recognition (LaTeX format)
- 📍 Text spotting with coordinates
- 🔍 Information extraction (key-value, fields, subtitles)
- 🌐 Photo translation
- 🎯 Compact model (1B parameters)

Model: tencent/HunyuanOCR
vLLM: Requires nightly build (trust_remote_code=True)

Note: Due to vLLM V1 engine batching issues with HunyuanOCR, batch_size defaults to 1.
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
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# HunyuanOCR Prompt Templates (from official README)
# Source: https://huggingface.co/tencent/HunyuanOCR
# ────────────────────────────────────────────────────────────────

PROMPT_TEMPLATES = {
    # Parsing prompts
    "parse-document": {
        "en": "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.",
        "cn": "提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。",
    },
    "parse-formula": {
        "en": "Identify the formula in the image and represent it using LaTeX format.",
        "cn": "识别图片中的公式，用 LaTeX 格式表示。",
    },
    "parse-table": {
        "en": "Parse the table in the image into HTML.",
        "cn": "把图中的表格解析为 HTML。",
    },
    "parse-chart": {
        "en": "Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.",
        "cn": "解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示。",
    },
    # Spotting prompt
    "spot": {
        "en": "Detect and recognize text in the image, and output the text coordinates in a formatted manner.",
        "cn": "检测并识别图片中的文字，将文本坐标格式化输出。",
    },
    # Extraction prompts
    "extract-subtitles": {
        "en": "Extract the subtitles from the image.",
        "cn": "提取图片中的字幕。",
    },
    # Translation prompt (requires target_language substitution)
    "translate": {
        "en": "First extract the text, then translate the text content into {target_language}. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format.",
        "cn": "先提取文字，再将文字内容翻译为{target_language}。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。",
    },
}

# Templates that require dynamic substitution
EXTRACT_KEY_TEMPLATE = {
    "en": "Output the value of {key}.",
    "cn": "输出 {key} 的值。",
}

EXTRACT_FIELDS_TEMPLATE = {
    "en": "Extract the content of the fields: {fields} from the image and return it in JSON format.",
    "cn": "提取图片中的: {fields} 的字段内容，并按照 JSON 格式返回。",
}


def clean_repeated_substrings(text: str, threshold: int = 10) -> str:
    """
    Remove repeated substrings from long outputs.

    HunyuanOCR can sometimes produce repeated patterns in very long outputs.
    This utility detects and removes substrings that repeat more than threshold times.

    From: https://huggingface.co/tencent/HunyuanOCR
    """
    if len(text) <= 8000:
        return text

    # Check the last portion of the text for repetition
    check_portion = text[-4000:]

    # Find repeated patterns of various lengths
    for pattern_len in range(10, 200):
        pattern = check_portion[-pattern_len:]
        count = check_portion.count(pattern)

        if count >= threshold:
            # Find where the repetition starts and truncate
            first_occurrence = text.find(pattern)
            if first_occurrence != -1:
                return text[: first_occurrence + len(pattern)]

    return text


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def get_prompt(
    prompt_mode: str,
    use_chinese: bool = False,
    target_language: str = None,
    key: str = None,
    fields: List[str] = None,
) -> str:
    """Get the appropriate prompt for the given mode."""
    lang = "cn" if use_chinese else "en"

    if prompt_mode == "extract-key":
        if not key:
            raise ValueError("--key is required for extract-key mode")
        template = EXTRACT_KEY_TEMPLATE[lang]
        return template.format(key=key)

    if prompt_mode == "extract-fields":
        if not fields:
            raise ValueError("--fields is required for extract-fields mode")
        template = EXTRACT_FIELDS_TEMPLATE[lang]
        fields_str = str(fields)
        return template.format(fields=fields_str)

    if prompt_mode == "translate":
        if not target_language:
            raise ValueError("--target-language is required for translate mode")
        template = PROMPT_TEMPLATES["translate"][lang]
        return template.format(target_language=target_language)

    if prompt_mode not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown prompt mode: {prompt_mode}. "
            f"Available: {list(PROMPT_TEMPLATES.keys()) + ['extract-key', 'extract-fields']}"
        )

    return PROMPT_TEMPLATES[prompt_mode][lang]


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str,
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

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # HunyuanOCR format: image before text
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
    prompt_mode: str = "parse-document",
    use_chinese: bool = False,
) -> str:
    """Create a dataset card documenting the OCR process."""
    model_name = model.split("/")[-1]

    return f"""---
tags:
- ocr
- document-processing
- hunyuan-ocr
- multilingual
- markdown
- uv-script
- generated
---

# Document OCR using {model_name}

This dataset contains OCR results from images in [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using HunyuanOCR, a lightweight 1B VLM from Tencent.

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
- **Prompt Mode**: {prompt_mode}
- **Prompt Language**: {"Chinese" if use_chinese else "English"}
- **Max Model Length**: {max_model_len:,} tokens
- **Max Output Tokens**: {max_tokens:,}
- **GPU Memory Utilization**: {gpu_memory_utilization:.1%}

## Model Information

HunyuanOCR is a lightweight 1B VLM that excels at:
- 📝 **Document Parsing** - Full markdown extraction with reading order
- 📊 **Table Extraction** - HTML format tables
- 📐 **Formula Recognition** - LaTeX format formulas
- 📈 **Chart Parsing** - Mermaid/Markdown format
- 📍 **Text Spotting** - Detection with coordinates
- 🔍 **Information Extraction** - Key-value, fields, subtitles
- 🌐 **Translation** - Multilingual photo translation

## Prompt Modes Available

- `parse-document` - Full document parsing (default)
- `parse-formula` - LaTeX formula extraction
- `parse-table` - HTML table extraction
- `parse-chart` - Chart/flowchart parsing
- `spot` - Text detection with coordinates
- `extract-key` - Extract specific key value
- `extract-fields` - Extract multiple fields as JSON
- `extract-subtitles` - Subtitle extraction
- `translate` - Document translation

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

This dataset was generated using the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) HunyuanOCR script:

```bash
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hunyuan-ocr.py \\
    {source_dataset} \\
    <output-dataset> \\
    --image-column {image_column} \\
    --batch-size {batch_size} \\
    --prompt-mode {prompt_mode} \\
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
    batch_size: int = 1,  # Default to 1 due to vLLM V1 batching issues with HunyuanOCR
    model: str = "tencent/HunyuanOCR",
    max_model_len: int = 16384,
    max_tokens: int = 16384,
    gpu_memory_utilization: float = 0.8,
    hf_token: str = None,
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    prompt_mode: str = "parse-document",
    target_language: str = None,
    key: str = None,
    fields: List[str] = None,
    use_chinese: bool = False,
    custom_prompt: str = None,
    output_column: str = "markdown",
    clean_output: bool = True,
    config: str = None,
    create_pr: bool = False,
    verbose: bool = False,
):
    """Process images from HF dataset through HunyuanOCR model."""

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Login to HF if token provided
    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    # Determine prompt to use
    if custom_prompt:
        prompt = custom_prompt
        logger.info(f"Using custom prompt: {prompt[:50]}...")
    else:
        prompt = get_prompt(
            prompt_mode=prompt_mode,
            use_chinese=use_chinese,
            target_language=target_language,
            key=key,
            fields=fields,
        )
        lang_str = "Chinese" if use_chinese else "English"
        logger.info(f"Using prompt mode: {prompt_mode} ({lang_str})")

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

    # Note: HunyuanOCR has batching issues with vLLM V1 engine when batch_size > 1
    # Using limit_mm_per_prompt for stability
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,
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
        desc="HunyuanOCR processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [make_ocr_message(img, prompt) for img in batch_images]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                # Clean repeated substrings if enabled
                if clean_output:
                    text = clean_repeated_substrings(text)
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
        "model_name": "HunyuanOCR",
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "prompt_mode": prompt_mode if not custom_prompt else "custom",
        "prompt_language": "cn" if use_chinese else "en",
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
    commit_msg = f"Add HunyuanOCR OCR results ({len(dataset)} samples)" + (
        f" [{config}]" if config else ""
    )
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
                commit_message=commit_msg,
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

    # Create and push dataset card (skip when creating PR to avoid conflicts)
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
            prompt_mode=prompt_mode if not custom_prompt else "custom",
            use_chinese=use_chinese,
        )

        card = DatasetCard(card_content)
        card.push_to_hub(output_dataset, token=HF_TOKEN)

    logger.info("HunyuanOCR processing complete!")
    logger.info(
        f"Dataset available at: https://huggingface.co/datasets/{output_dataset}"
    )
    logger.info(f"Processing time: {processing_time_str}")

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
        ]:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")
        logger.info("--- End versions ---")


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("HunyuanOCR Document Processing")
        print("=" * 80)
        print("\nLightweight 1B VLM from Tencent for multilingual document parsing")
        print("\nFeatures:")
        print("- 📝 Full document parsing to markdown")
        print("- 📊 Table extraction (HTML format)")
        print("- 📐 Formula recognition (LaTeX format)")
        print("- 📍 Text spotting with coordinates")
        print("- 🔍 Information extraction (key-value, fields)")
        print("- 🌐 Photo translation")
        print("\nExample usage:")
        print("\n1. Basic document parsing:")
        print("   uv run hunyuan-ocr.py input-dataset output-dataset")
        print("\n2. Formula extraction:")
        print("   uv run hunyuan-ocr.py math-docs formulas --prompt-mode parse-formula")
        print("\n3. Table extraction:")
        print("   uv run hunyuan-ocr.py docs tables --prompt-mode parse-table")
        print("\n4. Text spotting with coordinates:")
        print("   uv run hunyuan-ocr.py images spotted --prompt-mode spot")
        print("\n5. Extract specific field:")
        print(
            '   uv run hunyuan-ocr.py invoices data --prompt-mode extract-key --key "Total Amount"'
        )
        print("\n6. Extract multiple fields as JSON:")
        print(
            '   uv run hunyuan-ocr.py forms data --prompt-mode extract-fields --fields "name,date,amount"'
        )
        print("\n7. Translate document to English:")
        print(
            "   uv run hunyuan-ocr.py cn-docs en-docs --prompt-mode translate --target-language English"
        )
        print("\n8. Use Chinese prompts:")
        print("   uv run hunyuan-ocr.py docs output --use-chinese-prompts")
        print("\n9. Running on HF Jobs:")
        print("   hf jobs uv run --flavor l4x1 \\")
        print(
            '     -e HF_TOKEN=$(python3 -c "from huggingface_hub import get_token; print(get_token())") \\'
        )
        print(
            "     https://huggingface.co/datasets/uv-scripts/ocr/raw/main/hunyuan-ocr.py \\"
        )
        print("       input-dataset output-dataset")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run hunyuan-ocr.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using HunyuanOCR (1B lightweight VLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prompt Modes (official HunyuanOCR prompts):
  parse-document   - Full document parsing to markdown (default)
  parse-formula    - LaTeX formula extraction
  parse-table      - HTML table extraction
  parse-chart      - Chart/flowchart parsing (Mermaid/Markdown)
  spot             - Text detection with coordinates
  extract-key      - Extract specific key value (requires --key)
  extract-fields   - Extract multiple fields as JSON (requires --fields)
  extract-subtitles - Subtitle extraction
  translate        - Document translation (requires --target-language)

Examples:
  # Basic document OCR (default)
  uv run hunyuan-ocr.py my-docs analyzed-docs

  # Extract formulas as LaTeX
  uv run hunyuan-ocr.py math-papers formulas --prompt-mode parse-formula

  # Extract tables as HTML
  uv run hunyuan-ocr.py reports tables --prompt-mode parse-table

  # Text spotting with coordinates
  uv run hunyuan-ocr.py images spotted --prompt-mode spot

  # Extract specific key from forms
  uv run hunyuan-ocr.py invoices amounts --prompt-mode extract-key --key "Total"

  # Extract multiple fields as JSON
  uv run hunyuan-ocr.py forms data --prompt-mode extract-fields --fields "name,date,amount"

  # Translate Chinese documents to English
  uv run hunyuan-ocr.py cn-docs translated --prompt-mode translate --target-language English

  # Use Chinese prompts
  uv run hunyuan-ocr.py docs output --use-chinese-prompts

  # Random sampling for testing
  uv run hunyuan-ocr.py large-dataset test --max-samples 50 --shuffle
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
        default=1,
        help="Batch size for processing (default: 1, higher values may cause vLLM errors)",
    )
    parser.add_argument(
        "--model",
        default="tencent/HunyuanOCR",
        help="Model to use (default: tencent/HunyuanOCR)",
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
        default=16384,
        help="Maximum tokens to generate (default: 16384)",
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
        "--prompt-mode",
        choices=[
            "parse-document",
            "parse-formula",
            "parse-table",
            "parse-chart",
            "spot",
            "extract-key",
            "extract-fields",
            "extract-subtitles",
            "translate",
        ],
        default="parse-document",
        help="Prompt template to use (default: parse-document)",
    )
    parser.add_argument(
        "--target-language",
        help="Target language for translation mode (e.g., 'English', 'Chinese')",
    )
    parser.add_argument(
        "--key",
        help="Key to extract for extract-key mode",
    )
    parser.add_argument(
        "--fields",
        help="Comma-separated list of fields for extract-fields mode",
    )
    parser.add_argument(
        "--use-chinese-prompts",
        action="store_true",
        help="Use Chinese versions of prompts",
    )
    parser.add_argument(
        "--custom-prompt",
        help="Custom prompt text (overrides --prompt-mode)",
    )
    parser.add_argument(
        "--output-column",
        default="markdown",
        help="Column name for output text (default: markdown)",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Disable cleaning of repeated substrings in output",
    )
    parser.add_argument(
        "--config",
        help="Dataset config name for multi-model benchmarks",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Push results as a pull request instead of direct commit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions at the end of the run",
    )

    args = parser.parse_args()

    # Parse fields if provided
    fields_list = None
    if args.fields:
        fields_list = [f.strip() for f in args.fields.split(",")]

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
        prompt_mode=args.prompt_mode,
        target_language=args.target_language,
        key=args.key,
        fields=fields_list,
        use_chinese=args.use_chinese_prompts,
        custom_prompt=args.custom_prompt,
        output_column=args.output_column,
        clean_output=not args.no_clean_output,
        config=args.config,
        create_pr=args.create_pr,
        verbose=args.verbose,
    )
