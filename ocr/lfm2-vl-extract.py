# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub",
#     "pillow",
#     "vllm",
#     "transformers",
#     "tqdm",
#     "toolz",
#     "torch",
# ]
# ///
"""
Extract structured JSON from images using LiquidAI's LFM2.5-VL-1.6B-Extract with vLLM.

LFM2.5-VL-1.6B-Extract (1.6B = LFM2 1.2B LM + SigLIP2 0.4B vision) is a compact
vision-language model purpose-built for *schema-guided* extraction: you give it a
list of fields, it returns a flat JSON object with those fields filled from the image.
It reports 99.6 JSON-validity / F1 on its benchmark, beating similarly-sized VLMs.

Unlike the markdown-OCR scripts here, this one needs a SCHEMA (a field list). Pass
`--schema` as inline JSON, a URL, or a file path, mapping field names to short
descriptions:

    --schema '{"invoice_number": "the invoice number", "total": "the total amount"}'

Model:  https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-Extract
Docs:   https://docs.liquid.ai/deployment/gpu-inference/vllm

HF Jobs note: run on the vLLM image so the CUDA toolkit + prebuilt FlashInfer kernels
are present and startup is fast (it reuses the image's CUDA-matched vLLM build):

    hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
        --image vllm/vllm-openai --python /usr/bin/python3 \
        -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lfm2-vl-extract.py \
        INPUT OUTPUT --schema '{"title": "the document title", "date": "any date shown"}'

It also runs on the default uv image, just with a slower first-time vLLM build. Deps are
left unpinned so uv resolves a vLLM that supports the LFM2-VL (transformers 5) architecture,
and FlashInfer sampling is disabled (VLLM_USE_FLASHINFER_SAMPLER=0, see below) so the engine
never JIT-compiles a kernel that needs nvcc — absent from the default image.
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.request import urlopen

# Disable vLLM's FlashInfer top-k/top-p sampler before the engine starts: it JIT-compiles
# at warmup and needs nvcc (absent from the default uv image). Harmless for greedy decoding.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from toolz import partition_all
from tqdm import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "LiquidAI/LFM2.5-VL-1.6B-Extract"


def check_cuda_availability() -> None:
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Run on Hugging Face Jobs with: hf jobs uv run --flavor l4x1 ...")
        sys.exit(1)
    logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name()}")


def load_schema_arg(value: str) -> Dict[str, str]:
    """Resolve --schema (inline JSON, URL, or file path) into a {field: description} dict."""
    text = value.strip()
    if text.startswith("http://") or text.startswith("https://"):
        logger.info(f"Loading schema from URL: {text}")
        text = urlopen(text).read().decode("utf-8")
    elif not text.startswith("{") and not text.startswith("["):
        if os.path.exists(text):
            logger.info(f"Loading schema from file: {text}")
            with open(text) as f:
                text = f.read()
    parsed = json.loads(text)
    # Accept {"field": "description"} or ["field1", "field2"]
    if isinstance(parsed, list):
        return {str(field): "" for field in parsed}
    if isinstance(parsed, dict):
        return {str(k): str(v) for k, v in parsed.items()}
    raise ValueError("--schema must be a JSON object {field: description} or a JSON list of field names.")


def build_system_prompt(schema: Dict[str, str]) -> str:
    """LFM2.5-VL-Extract prompt: a field list in the system message → flat JSON out."""
    lines = []
    for field, desc in schema.items():
        lines.append(f"{field}: {desc}" if desc else field)
    fields_block = "\n".join(lines)
    return (
        f"Extract the following from the image:\n\n{fields_block}\n\n"
        "Respond with only a JSON object."
    )


def image_to_data_uri(image: Union[Image.Image, Dict[str, Any], str]) -> str:
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        image = Image.open(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_message(image: Any, system_prompt: str) -> List[Dict]:
    data_uri = image_to_data_uri(image)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_uri}}]},
    ]


def parse_json_output(text: str) -> tuple[Optional[Any], bool]:
    """Return (parsed, ok). Strips ```json fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
    stripped = stripped.strip()
    try:
        return json.loads(stripped), True
    except (json.JSONDecodeError, ValueError):
        return None, False


def main(
    input_dataset: str,
    output_dataset: str,
    schema: str,
    image_column: str = "image",
    output_column: str = "extraction",
    split: str = "train",
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 16,
    model: str = DEFAULT_MODEL,
    max_model_len: int = 4096,
    max_tokens: int = 1024,
    private: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    check_cuda_availability()

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    schema_dict = load_schema_arg(schema)
    system_prompt = build_system_prompt(schema_dict)
    logger.info(f"Extraction fields: {list(schema_dict.keys())}")

    logger.info(f"Loading dataset: {input_dataset} (split={split})")
    dataset = load_dataset(input_dataset, split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"Processing {len(dataset)} examples")

    if image_column not in dataset.column_names:
        logger.error(f"Image column '{image_column}' not found. Columns: {dataset.column_names}")
        sys.exit(1)

    logger.info(f"Loading model: {model}")
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    all_outputs: List[str] = []
    n_valid = 0
    images = dataset[image_column]
    for batch in tqdm(list(partition_all(batch_size, images)), desc="Extracting"):
        batch_messages = [make_message(img, system_prompt) for img in batch]
        outputs = llm.chat(batch_messages, sampling_params)
        for out in outputs:
            text = out.outputs[0].text.strip()
            parsed, ok = parse_json_output(text)
            if ok:
                n_valid += 1
                all_outputs.append(json.dumps(parsed, ensure_ascii=False))
            else:
                all_outputs.append(text)  # keep raw on parse failure

    logger.info(f"Valid JSON: {n_valid}/{len(all_outputs)}")

    dataset = dataset.add_column(output_column, all_outputs)

    inference_entry = {
        "model": model,
        "column_name": output_column,
        "task": "schema-guided extraction",
        "fields": list(schema_dict.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": "lfm2-vl-extract.py",
    }
    if "inference_info" in dataset.column_names:
        def update_info(example):
            try:
                existing = json.loads(example["inference_info"]) if example["inference_info"] else []
            except (json.JSONDecodeError, TypeError):
                existing = []
            existing.append(inference_entry)
            return {"inference_info": json.dumps(existing)}
        dataset = dataset.map(update_info)
    else:
        dataset = dataset.add_column(
            "inference_info", [json.dumps([inference_entry])] * len(dataset)
        )

    logger.info(f"Pushing to {output_dataset}")
    dataset.push_to_hub(output_dataset, private=private, token=HF_TOKEN)

    card_text = f"""---
tags:
- uv-script
- extraction
- lfm2-vl
- json
---

# Structured extraction with LFM2.5-VL-1.6B-Extract

JSON fields extracted from images in [{input_dataset}](https://huggingface.co/datasets/{input_dataset})
using [{model}](https://huggingface.co/{model}).

- **Source**: `{input_dataset}` (split `{split}`)
- **Model**: `{model}`
- **Fields**: {", ".join(f"`{k}`" for k in schema_dict.keys())}
- **Output column**: `{output_column}` (JSON string per row)
- **Valid JSON**: {n_valid}/{len(all_outputs)}
- **Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

Generated with the [uv-scripts/ocr](https://huggingface.co/datasets/uv-scripts/ocr) `lfm2-vl-extract.py` script.
"""
    try:
        card = DatasetCard(card_text)
        card.push_to_hub(output_dataset, token=HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not push dataset card: {e}")

    logger.info("Done! Extraction complete.")
    logger.info(f"Dataset: https://huggingface.co/datasets/{output_dataset}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("LFM2.5-VL-1.6B-Extract — schema-guided JSON extraction from images")
        print("\nUsage:")
        print("  uv run lfm2-vl-extract.py INPUT OUTPUT --schema SCHEMA [options]")
        print("\nExample:")
        print('  uv run lfm2-vl-extract.py my-images my-extractions \\')
        print('    --schema \'{"title": "the document title", "date": "any date shown"}\'')
        print("\n  --schema accepts inline JSON, a URL, or a file path.")
        print("\nFor full help: uv run lfm2-vl-extract.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Schema-guided JSON extraction from images using LFM2.5-VL-1.6B-Extract",
    )
    parser.add_argument("input_dataset", help="Input dataset ID (with images)")
    parser.add_argument("output_dataset", help="Output dataset ID")
    parser.add_argument(
        "--schema", required=True,
        help="Fields to extract: inline JSON {field: description}, a URL, or a file path",
    )
    parser.add_argument("--image-column", default="image", help="Image column (default: image)")
    parser.add_argument("--output-column", default="extraction", help="Output column (default: extraction)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before sampling")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max context length (default: 4096)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens (default: 1024)")
    parser.add_argument("--private", action="store_true", help="Make output dataset private")
    parser.add_argument("--hf-token", help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        schema=args.schema,
        image_column=args.image_column,
        output_column=args.output_column,
        split=args.split,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        private=args.private,
        hf_token=args.hf_token,
    )
