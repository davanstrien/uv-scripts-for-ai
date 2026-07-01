# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "lift-pdf[hf]",
#     "datasets>=3.1.0",
#     "huggingface-hub",
#     "pillow",
#     "toolz",
#     "tqdm",
# ]
# ///
"""
Extract structured JSON from document images OR multi-page PDFs using Datalab's
`lift` model (`datalab-to/lift`, 9B, Qwen3.5-based).

Unlike the markdown-OCR scripts here, lift does *schema-constrained* extraction:
you give it a JSON Schema, it returns a JSON object matching that schema. It
natively handles multi-page documents — a whole PDF is collapsed into a single
extraction.

Two in-process backends, selected with `--method` (no server, single command):

  --method hf    (default)  Transformers via the `lift-pdf` package. Runs on the
                            default uv image. Simplest path; best for small jobs.
  --method vllm             vLLM's offline `LLM()` engine (`llm.chat`) with
                            structured-output decoding — the fast batched path the
                            other vLLM OCR scripts here use. Needs the
                            `vllm/vllm-openai` image (which ships vLLM). Reproduces
                            lift's own prompt + guided-JSON recipe against the
                            offline engine. Wins on large jobs via continuous batching.

Benchmark the two by pushing each to one repo with `--config hf` / `--config vllm`.

Input is one document per row:
  --image-column COL   (default `image`)  one image per row  -> one extraction
  --pdf-column COL                        PDF bytes per row -> one extraction
                                          (multi-page; respects --page-range)

Pass `--schema` as inline JSON, a URL, or a file path (standard JSON Schema):

    --schema '{"type":"object","properties":{"invoice_number":{"type":"string"},
               "total":{"type":"number"}},"required":["invoice_number"]}'

LICENSE NOTE: lift's *code* is Apache-2.0 but the *weights* are a modified
OpenRAIL-M license — free for research, personal use, and startups under $5M
funding/revenue, but restricted from competitive use against Datalab's API.
Confirm you are within those terms before using it. https://huggingface.co/datalab-to/lift

HF Jobs — HF backend (default image is fine; 9B needs a roomy GPU):

    hf jobs uv run --flavor a100-large -s HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lift-extract.py \\
        INPUT_DATASET OUTPUT_DATASET \\
        --schema '{"type":"object","properties":{"title":{"type":"string"}}}' \\
        --max-samples 5 --shuffle --seed 42

HF Jobs — vLLM offline backend (use the vllm image so vLLM is present):

    hf jobs uv run --flavor a100-large -s HF_TOKEN \\
        --image vllm/vllm-openai --python /usr/bin/python3 \\
        -e PYTHONPATH=/usr/local/lib/python3.12/dist-packages \\
        https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lift-extract.py \\
        INPUT_DATASET OUTPUT_DATASET --method vllm \\
        --schema '{"type":"object","properties":{"title":{"type":"string"}}}' \\
        --max-samples 5

Model: datalab-to/lift  (package: lift-pdf, https://github.com/datalab-to/lift)
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen

from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from PIL import Image
from toolz import partition_all
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The package default checkpoint drifts between releases (e.g. "datalab-to/lift-extract");
# pin to the canonical card repo so the script is stable across lift-pdf versions.
DEFAULT_MODEL = "datalab-to/lift"
DEFAULT_MAX_TOKENS = 12384  # lift-pdf's own MAX_OUTPUT_TOKENS default

# A processed document: (parsed JSON or None, error flag, raw model text).
DocResult = Tuple[Optional[Any], bool, str]


def check_cuda_availability() -> None:
    """Exit early with a clear message if there's no GPU."""
    import torch

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error(
            "Run on Hugging Face Jobs with: hf jobs uv run --flavor a100-large ..."
        )
        sys.exit(1)
    logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def ensure_output_columns_free(dataset, columns, overwrite=False):
    """Fail fast if an output column would collide with an existing input column.

    Adding a column that already exists silently overwrites it (e.g. a ground-truth
    `text`/`markdown` column) or crashes on push with a duplicate-column error only
    *after* inference has run. Catch it up front. With overwrite=True, drop the clashing
    column(s) here instead (logged) so the later add_column is clean.
    """
    clash = [c for c in columns if c in dataset.column_names]
    if not clash:
        return dataset
    if overwrite:
        logger.warning(f"--overwrite: replacing existing column(s) {clash}")
        return dataset.remove_columns(clash)
    logger.error(
        f"Output column(s) {clash} already exist in the input dataset "
        f"(columns: {dataset.column_names})."
    )
    logger.error("Choose a different --output-column, or pass --overwrite to replace them.")
    sys.exit(1)


def load_schema_arg(value: str) -> Dict[str, Any]:
    """Resolve --schema (inline JSON, a URL, or a file path) into a JSON Schema dict."""
    text = value.strip()
    if text.startswith(("http://", "https://")):
        logger.info(f"Loading schema from URL: {text}")
        text = urlopen(text).read().decode("utf-8")  # noqa: S310
    elif not text.startswith("{"):
        # Looks like a path (inline JSON would start with "{"); read it if it exists.
        if os.path.isfile(text):
            logger.info(f"Loading schema from file: {text}")
            with open(text) as f:
                text = f.read()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse --schema as JSON (tried URL/path/inline): {e}"
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError("--schema must be a JSON object (a JSON Schema).")
    return parsed


def cell_to_bytes(cell: Any) -> bytes:
    """Normalize an HF dataset cell (image or document) to raw file bytes.

    Handles decoded PIL images (Image feature), {"bytes"/"path"} dicts, raw bytes
    (e.g. a binary PDF column), and string paths/URLs.
    """
    if isinstance(cell, Image.Image):
        buf = io.BytesIO()
        cell.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return cell["bytes"]
        if cell.get("path"):
            with open(cell["path"], "rb") as f:
                return f.read()
        raise ValueError(
            f"Unsupported image/document dict (no bytes/path): {list(cell)}"
        )
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, str):
        if cell.startswith(("http://", "https://")):
            return urlopen(cell).read()  # noqa: S310
        with open(cell, "rb") as f:
            return f.read()
    raise ValueError(f"Unsupported cell type: {type(cell)}")


def load_document_images(
    load_file, cell: Any, page_range: Optional[str]
) -> List[Image.Image]:
    """Render one dataset cell into the page images lift expects.

    Reuses lift's own `load_file`, which auto-detects PDF vs image by content
    (pypdfium2 for PDFs, with the model's DPI/min-dim and page-range handling).
    """
    data = cell_to_bytes(cell)
    # load_file detects type from content, so the temp file needs no extension.
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        config = {"page_range": page_range} if page_range else {}
        return load_file(path, config)
    finally:
        os.unlink(path)


def pil_to_data_uri(img: Image.Image) -> str:
    """PNG data URI for an OpenAI-format image content block."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def parse_json_output(text: str) -> Tuple[Optional[Any], bool]:
    """Return (parsed, ok). Strips ```json fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    try:
        return json.loads(stripped), True
    except (json.JSONDecodeError, ValueError):
        return None, False


# --- HF backend (lift-pdf package, in-process Transformers) ---
def make_hf_processor(schema: Dict[str, Any], max_tokens: Optional[int]):
    """Load lift via the package's HF backend; return a batch-processing closure."""
    from lift.model import InferenceManager
    from lift.model.schema import BatchInputItem

    logger.info("Loading lift via Transformers (method=hf)...")
    manager = InferenceManager(method="hf")

    def process(image_lists: List[List[Image.Image]]) -> List[DocResult]:
        items = [
            BatchInputItem(images=imgs, schema=schema, prompt_type="direct")
            for imgs in image_lists
        ]
        results = manager.generate(items, max_output_tokens=max_tokens)
        return [(r.extraction, bool(r.error), r.raw) for r in results]

    return process


# --- vLLM backend (offline LLM() engine + structured outputs) ---
def build_guided_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Reproduce lift's vLLM guided-decoding schema: JSON Schema -> pydantic ->
    json_schema with every leaf made nullable (so absent fields can be null,
    matching lift's own server-side behavior)."""
    from json_schema_to_pydantic import create_model
    from lift.model.vllm import make_properties_nullable

    schema_model = create_model(schema)
    json_schema = schema_model.model_json_schema()
    make_properties_nullable(json_schema)
    return json_schema


def make_sampling_params(json_schema: Dict[str, Any], max_tokens: int):
    """SamplingParams with structured JSON output, across vLLM API versions.

    lift uses greedy-ish decoding (temperature 0.0, top_p 0.1).
    """
    from vllm import SamplingParams

    # vLLM >= 0.12
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            temperature=0.0,
            top_p=0.1,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=json_schema),
        )
    except (ImportError, TypeError):
        pass
    # Older vLLM
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            temperature=0.0,
            top_p=0.1,
            max_tokens=max_tokens,
            guided_decoding=GuidedDecodingParams(json=json_schema),
        )
    except (ImportError, TypeError):
        pass
    logger.warning(
        "Structured output unavailable in this vLLM version; relying on lift's "
        "training to emit valid JSON."
    )
    return SamplingParams(temperature=0.0, top_p=0.1, max_tokens=max_tokens)


def make_vllm_processor(
    schema: Dict[str, Any],
    model: str,
    max_tokens: Optional[int],
    max_model_len: int,
    gpu_memory_utilization: float,
    max_images_per_doc: int,
):
    """Load lift into vLLM's offline engine; return a batch-processing closure."""
    try:
        from vllm import LLM
    except ImportError as e:
        raise RuntimeError(
            "--method vllm needs vLLM. Run on the vllm/vllm-openai image: "
            "--image vllm/vllm-openai --python /usr/bin/python3 "
            "-e PYTHONPATH=/usr/local/lib/python3.12/dist-packages"
        ) from e
    from lift.model.util import scale_to_fit
    from lift.prompts import PROMPT_MAPPING

    json_schema = build_guided_schema(schema)
    prompt = PROMPT_MAPPING["direct"].replace("{schema}", json.dumps(schema, indent=2))

    logger.info("Loading lift via vLLM offline engine (method=vllm)...")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": max_images_per_doc},
        # lift's own server-side image bounds, applied by the offline processor too.
        mm_processor_kwargs={"min_pixels": 3136, "max_pixels": 861696},
    )
    sampling_params = make_sampling_params(
        json_schema, max_tokens or DEFAULT_MAX_TOKENS
    )

    def process(image_lists: List[List[Image.Image]]) -> List[DocResult]:
        messages = []
        for imgs in image_lists:
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_data_uri(scale_to_fit(img))},
                }
                for img in imgs
            ]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        outputs = llm.chat(
            messages, sampling_params, chat_template_content_format="openai"
        )
        results: List[DocResult] = []
        for o in outputs:
            raw = o.outputs[0].text
            parsed, ok = parse_json_output(raw)
            results.append((parsed if ok else None, not ok, raw))
        return results

    return process


def create_dataset_card(
    source_dataset: str,
    model: str,
    method: str,
    schema: Dict[str, Any],
    num_samples: int,
    n_valid: int,
    source_column: str,
    is_pdf: bool,
    page_range: Optional[str],
    output_column: str,
    split: str,
    processing_time: str,
) -> str:
    """Build the output dataset card documenting the lift run."""
    schema_block = json.dumps(schema, indent=2)
    input_kind = "PDF documents" if is_pdf else "images"
    col_desc = "PDF" if is_pdf else "image"
    if page_range:
        col_desc += f", pages {page_range}"
    backend_desc = (
        "vLLM offline engine" if method == "vllm" else "Transformers (lift-pdf)"
    )
    return f"""---
tags:
- ocr
- structured-extraction
- document-processing
- lift
- json
- uv-script
- generated
---

# lift structured extraction on {source_dataset}

Schema-constrained JSON extracted from {input_kind} in
[{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using
[lift](https://huggingface.co/{model}) (9B, Qwen3.5-based) by Datalab, via the
[`lift-pdf`](https://github.com/datalab-to/lift) package.

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Backend**: `{method}` ({backend_desc})
- **Input column**: `{source_column}` ({col_desc})
- **Output column**: `{output_column}` (JSON string per row)
- **Split**: `{split}`
- **Samples**: {num_samples:,}
- **Valid JSON**: {n_valid:,} / {num_samples:,}
- **Processing time**: {processing_time}
- **Date**: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

### Extraction Schema

```json
{schema_block}
```

## License note

lift's code is Apache-2.0, but the model **weights** use a modified OpenRAIL-M
license: free for research, personal use, and startups under $5M funding/revenue,
restricted from competitive use against Datalab's API. See the
[model card](https://huggingface.co/{model}).

## Dataset Structure

Original columns plus:
- `{output_column}`: lift output (JSON string; raw text kept on parse failure)
- `inference_info`: JSON list tracking models applied to this dataset

Generated with [UV Scripts](https://huggingface.co/uv-scripts).
"""


def main(
    input_dataset: str,
    output_dataset: str,
    schema_arg: str,
    image_column: str = "image",
    pdf_column: Optional[str] = None,
    output_column: str = "extraction",
    overwrite: bool = False,
    method: str = "hf",
    page_range: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
    batch_size: int = 8,
    max_tokens: Optional[int] = None,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.9,
    max_images_per_doc: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    private: bool = False,
    config: Optional[str] = None,
    create_pr: bool = False,
    hf_token: Optional[str] = None,
    verbose: bool = False,
) -> None:
    # Unlock full Xet bandwidth for the 9B (~19GB) model download (repo convention).
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    check_cuda_availability()
    start_time = datetime.now(timezone.utc)

    HF_TOKEN = hf_token or os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)

    schema = load_schema_arg(schema_arg)

    # lift reads the checkpoint from env (pydantic-settings) at import time; set it first.
    os.environ["MODEL_CHECKPOINT"] = model

    # Import lift only after env is set so settings pick up the right checkpoint.
    from lift import resolve_schema
    from lift.input import load_file

    schema = resolve_schema(schema)  # validates and normalizes
    fields = list(schema.get("properties", {}).keys())

    source_column = pdf_column or image_column
    is_pdf = pdf_column is not None
    # vLLM caps images per prompt at init; PDFs need headroom for multiple pages.
    if max_images_per_doc is None:
        max_images_per_doc = 30 if is_pdf else 1

    logger.info(f"Model: {model}  Backend: {method}")
    logger.info(f"Schema top-level fields: {fields}")

    logger.info(f"Loading dataset: {input_dataset} (split={split})")
    dataset = load_dataset(input_dataset, split=split)
    if source_column not in dataset.column_names:
        logger.error(
            f"Column '{source_column}' not found. Available: {dataset.column_names}"
        )
        sys.exit(1)

    # Fail fast if the output column would collide with an existing input column
    dataset = ensure_output_columns_free(dataset, [output_column], overwrite=overwrite)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    logger.info(f"Processing {len(dataset)} documents from column '{source_column}'")

    if method == "vllm":
        process_batch = make_vllm_processor(
            schema,
            model,
            max_tokens,
            max_model_len,
            gpu_memory_utilization,
            max_images_per_doc,
        )
    else:
        process_batch = make_hf_processor(schema, max_tokens)

    extractions: List[Optional[str]] = [None] * len(dataset)
    error_flags: List[bool] = [True] * len(dataset)

    chunks = list(partition_all(batch_size, range(len(dataset))))
    for chunk in tqdm(chunks, desc="Extracting"):
        chunk = list(chunk)
        rendered: Dict[int, List[Image.Image]] = {}
        for i in chunk:
            try:
                rendered[i] = load_document_images(
                    load_file, dataset[i][source_column], page_range
                )
            except Exception as e:
                logger.warning(f"Row {i}: failed to load document: {e}")
                extractions[i] = f"[LIFT LOAD ERROR] {e}"
                error_flags[i] = True
        if not rendered:
            continue

        idxs = list(rendered.keys())
        try:
            results = process_batch([rendered[i] for i in idxs])
        except Exception as e:
            logger.error(f"Batch generate failed: {e}")
            for i in idxs:
                extractions[i] = "[LIFT GENERATE ERROR]"
                error_flags[i] = True
            continue

        for i, (parsed, err, raw) in zip(idxs, results):
            if parsed is not None and not err:
                extractions[i] = json.dumps(parsed, ensure_ascii=False)
                error_flags[i] = False
            else:
                extractions[i] = raw if raw else "[LIFT EMPTY OUTPUT]"
                error_flags[i] = True

    n_valid = sum(not f for f in error_flags)
    logger.info(f"Valid JSON: {n_valid}/{len(dataset)}")

    dataset = dataset.add_column(output_column, extractions)

    inference_entry = {
        "model": model,
        "model_name": "lift",
        "column_name": output_column,
        "task": "schema-constrained extraction",
        "backend": method,
        "fields": fields,
        "page_range": page_range,
        "parse_error_rate": (len(dataset) - n_valid) / len(dataset)
        if len(dataset)
        else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": "lift-extract.py",
    }
    if "inference_info" in dataset.column_names:

        def update_info(example):
            try:
                existing = (
                    json.loads(example["inference_info"])
                    if example["inference_info"]
                    else []
                )
            except (json.JSONDecodeError, TypeError):
                existing = []
            existing.append(inference_entry)
            return {"inference_info": json.dumps(existing)}

        dataset = dataset.map(update_info)
    else:
        dataset = dataset.add_column(
            "inference_info", [json.dumps([inference_entry])] * len(dataset)
        )

    processing_time = (
        f"{(datetime.now(timezone.utc) - start_time).total_seconds() / 60:.1f} min"
    )

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
                create_pr=create_pr,
                **({"config_name": config} if config else {}),
                commit_message=f"Add lift extraction results ({len(dataset)} samples)"
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
                logger.error("All upload attempts failed. Results are lost.")
                sys.exit(1)

    try:
        card = DatasetCard(
            create_dataset_card(
                source_dataset=input_dataset,
                model=model,
                method=method,
                schema=schema,
                num_samples=len(dataset),
                n_valid=n_valid,
                source_column=source_column,
                is_pdf=is_pdf,
                page_range=page_range,
                output_column=output_column,
                split=split,
                processing_time=processing_time,
            )
        )
        card.push_to_hub(output_dataset, token=HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not push dataset card: {e}")

    logger.info("Done! lift extraction complete.")
    logger.info(f"Dataset: https://huggingface.co/datasets/{output_dataset}")
    logger.info(f"Processing time: {processing_time}")

    if verbose:
        import importlib.metadata

        logger.info("--- Resolved package versions ---")
        pkgs = ["lift-pdf", "transformers", "torch", "datasets", "pillow", "openai"]
        if method == "vllm":
            pkgs.append("vllm")
        for pkg in pkgs:
            try:
                logger.info(f"  {pkg}=={importlib.metadata.version(pkg)}")
            except importlib.metadata.PackageNotFoundError:
                logger.info(f"  {pkg}: not installed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("lift — schema-constrained JSON extraction from images & PDFs (9B)")
        print("\nUsage:")
        print("  uv run lift-extract.py INPUT OUTPUT --schema SCHEMA [options]")
        print("\nExamples:")
        print("  # image column -> JSON")
        print("  uv run lift-extract.py my-images my-fields \\")
        print(
            '    --schema \'{"type":"object","properties":{"title":{"type":"string"}}}\''
        )
        print("\n  # multi-page PDFs -> JSON (one extraction per document)")
        print(
            "  uv run lift-extract.py my-pdfs my-fields --pdf-column pdf --page-range 0-5 \\"
        )
        print("    --schema schema.json")
        print("\n  --schema accepts inline JSON, a URL, or a file path.")
        print(
            "  --method hf (default) | vllm (offline LLM engine; needs the vllm image)"
        )
        print("\nFor full help: uv run lift-extract.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Schema-constrained JSON extraction from images & PDFs using datalab-to/lift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backends (both in-process, single command):
  --method hf     Transformers via lift-pdf (default). Simplest; default image.
  --method vllm   vLLM offline LLM() engine with structured outputs. Faster on
                  large jobs. Needs the vllm/vllm-openai image.

Input (one document per row):
  --image-column COL   one image per row   (default: image)
  --pdf-column COL     PDF bytes per row    (multi-page; honors --page-range)
""",
    )
    parser.add_argument(
        "input_dataset", help="Input dataset ID from the Hugging Face Hub"
    )
    parser.add_argument(
        "output_dataset", help="Output dataset ID for the Hugging Face Hub"
    )
    parser.add_argument(
        "--schema",
        required=True,
        help="JSON Schema: inline JSON, a URL, or a file path",
    )
    parser.add_argument(
        "--image-column", default="image", help="Image column (default: image)"
    )
    parser.add_argument(
        "--pdf-column",
        default=None,
        help="PDF column (bytes/path). Mutually exclusive with --image-column.",
    )
    parser.add_argument(
        "--output-column",
        default="extraction",
        help="Output column (default: extraction)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output column if it already exists in the input dataset "
        "(default: error out to avoid clobbering an existing column).",
    )
    parser.add_argument(
        "--method",
        choices=["hf", "vllm"],
        default="hf",
        help="Inference backend (default: hf)",
    )
    parser.add_argument(
        "--page-range",
        default=None,
        help="Pages to extract from PDFs, e.g. '0-5,7' (PDF column only)",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Limit number of documents (for testing)"
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle before sampling"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed (default: 42)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Documents per generate() call (default: 8; lower for big multi-page PDFs)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=f"Max output tokens (default: lift's {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="vLLM context length (default: 32768; raise for long multi-page PDFs)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory fraction (default: 0.9)",
    )
    parser.add_argument(
        "--max-images-per-doc",
        type=int,
        default=None,
        help="vLLM images-per-prompt cap (default: 1 for images, 30 for PDFs)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config/subset name when pushing (for benchmarking backends in one repo)",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Push as a pull request instead of directly (for parallel benchmarking)",
    )
    parser.add_argument("--hf-token", help="Hugging Face API token (or set HF_TOKEN)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved package versions after processing",
    )

    args = parser.parse_args()

    if args.pdf_column and args.image_column != "image":
        parser.error("--image-column and --pdf-column are mutually exclusive.")

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        schema_arg=args.schema,
        image_column=args.image_column,
        pdf_column=args.pdf_column,
        output_column=args.output_column,
        overwrite=args.overwrite,
        method=args.method,
        page_range=args.page_range,
        split=args.split,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_images_per_doc=args.max_images_per_doc,
        model=args.model,
        private=args.private,
        config=args.config,
        create_pr=args.create_pr,
        hf_token=args.hf_token,
        verbose=args.verbose,
    )
