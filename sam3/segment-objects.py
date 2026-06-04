#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=5.4.0",
#     "datasets",
#     "huggingface-hub[hf_transfer]",
#     "pillow",
#     "torch",
#     "torchvision",
#     "accelerate",
# ]
# ///

"""
Segment objects in images using Meta's SAM3 (Segment Anything Model 3).

This script processes images from a HuggingFace dataset and produces pixel-level
segmentation masks for objects matching a text prompt. Outputs either per-instance
binary masks or a combined semantic segmentation map.

Examples:
    # Segment photographs in historical newspapers (instance masks)
    uv run segment-objects.py \\
        davanstrien/newspapers-with-images-after-photography \\
        my-username/newspapers-segmented \\
        --class-name photograph

    # Segment with semantic map output (single image per sample)
    uv run segment-objects.py \\
        wildlife-images \\
        wildlife-segmented \\
        --class-name animal \\
        --output-format semantic-mask

    # Include bounding boxes alongside masks
    uv run segment-objects.py \\
        input-dataset output-dataset \\
        --class-name table \\
        --include-boxes

    # Test on small subset
    uv run segment-objects.py input output \\
        --class-name table \\
        --max-samples 10

    # Run on HF Jobs with GPU
    hf jobs uv run --flavor a100-large \\
        -s HF_TOKEN=HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/sam3/raw/main/segment-objects.py \\
        input-dataset output-dataset \\
        --class-name photograph

Note: To segment multiple object types, run the script multiple times with different
      --class-name values.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import torch
from datasets import ClassLabel, Dataset, Sequence, Value, load_dataset
from datasets import Image as ImageFeature
from huggingface_hub import DatasetCard, login
from PIL import Image
from transformers import Sam3Model, Sam3Processor

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    logger.error("CUDA is not available. This script requires a GPU.")
    logger.error("For cloud execution, use HF Jobs with --flavor l4x1 or similar.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment objects in images using SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input_dataset", help="Input HuggingFace dataset ID (e.g., 'username/dataset')"
    )
    parser.add_argument(
        "output_dataset", help="Output HuggingFace dataset ID (e.g., 'username/output')"
    )

    parser.add_argument(
        "--class-name",
        required=True,
        help="Object class to segment (e.g., 'photograph', 'animal', 'table')",
    )
    parser.add_argument(
        "--output-format",
        default="semantic-mask",
        choices=["instance-masks", "semantic-mask"],
        help="Output format: 'instance-masks' (one binary mask per object) or "
        "'semantic-mask' (single image, pixel value = instance ID). Default: semantic-mask",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score for detections (default: 0.5)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold for mask binarization (default: 0.5)",
    )
    parser.add_argument(
        "--include-boxes",
        action="store_true",
        help="Also include bounding boxes in output",
    )

    parser.add_argument(
        "--image-column",
        default="image",
        help="Name of the column containing images (default: 'image')",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to process (default: 'train')"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)",
    )
    parser.add_argument(
        "--model",
        default="facebook/sam3",
        help="SAM3 model ID (default: 'facebook/sam3')",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model precision (default: 'bfloat16')",
    )

    parser.add_argument(
        "--private", action="store_true", help="Make output dataset private"
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (default: uses HF_TOKEN env var or cached token)",
    )

    return parser.parse_args()


def masks_to_semantic_map(masks: torch.Tensor) -> Image.Image:
    """Combine per-instance binary masks into a single semantic segmentation map.

    Pixel values: 0=background, 1=first instance, 2=second instance, etc.
    Later instances take priority in overlapping regions.
    """
    if len(masks) == 0:
        return Image.new("L", (1, 1), 0)

    h, w = masks.shape[1], masks.shape[2]
    seg_map = np.zeros((h, w), dtype=np.uint8)

    for i, mask in enumerate(masks):
        binary = mask.cpu().numpy().astype(bool)
        seg_map[binary] = i + 1  # 0 is background

    return Image.fromarray(seg_map, mode="L")


def masks_to_instance_images(masks: torch.Tensor) -> list[Image.Image]:
    """Convert per-instance mask tensors to a list of binary PIL Images."""
    images = []
    for mask in masks:
        binary = (mask.cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(binary, mode="L"))
    return images


def process_batch(
    batch: dict[str, list[Any]],
    image_column: str,
    class_name: str,
    processor: Sam3Processor,
    model: Sam3Model,
    confidence_threshold: float,
    mask_threshold: float,
    output_format: str,
    include_boxes: bool,
) -> dict[str, list]:
    """Process a batch of images and return segmentation masks."""
    images = batch[image_column]

    pil_images = []
    for img in images:
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        pil_images.append(img)

    try:
        inputs = processor(
            images=pil_images,
            text=[class_name] * len(pil_images),
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=confidence_threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

    except Exception as e:
        logger.warning(f"Failed to process batch: {e}")
        return _empty_batch_result(len(pil_images), output_format, include_boxes)

    batch_result: dict[str, list] = {}

    if output_format == "semantic-mask":
        batch_result["segmentation_map"] = []
        batch_result["num_instances"] = []
    else:
        batch_result["segmentation_masks"] = []

    batch_result["scores"] = []
    batch_result["category"] = []
    if include_boxes:
        batch_result["boxes"] = []

    for result in results:
        masks = result.get("masks", torch.tensor([]))
        scores = result.get("scores", torch.tensor([]))
        boxes = result.get("boxes", torch.tensor([]))

        scores_np = scores.cpu().float().numpy() if len(scores) > 0 else np.array([])
        score_list = [float(s) for s in scores_np]
        category_list = [0] * len(score_list)

        if output_format == "semantic-mask":
            batch_result["segmentation_map"].append(masks_to_semantic_map(masks))
            batch_result["num_instances"].append(len(score_list))
        else:
            batch_result["segmentation_masks"].append(
                masks_to_instance_images(masks) if len(masks) > 0 else []
            )

        batch_result["scores"].append(score_list)
        batch_result["category"].append(category_list)

        if include_boxes:
            if len(boxes) > 0:
                boxes_np = boxes.cpu().float().numpy()
                box_list = []
                for box in boxes_np:
                    x1, y1, x2, y2 = box
                    box_list.append(
                        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    )
                batch_result["boxes"].append(box_list)
            else:
                batch_result["boxes"].append([])

    return batch_result


def _empty_batch_result(
    n: int, output_format: str, include_boxes: bool
) -> dict[str, list]:
    result: dict[str, list] = {}
    if output_format == "semantic-mask":
        result["segmentation_map"] = [Image.new("L", (1, 1), 0)] * n
        result["num_instances"] = [0] * n
    else:
        result["segmentation_masks"] = [[]] * n
    result["scores"] = [[]] * n
    result["category"] = [[]] * n
    if include_boxes:
        result["boxes"] = [[]] * n
    return result


def load_and_validate_dataset(
    dataset_id: str,
    split: str,
    image_column: str,
    max_samples: int | None = None,
    shuffle: bool = False,
    hf_token: str | None = None,
) -> Dataset:
    logger.info(f"Loading dataset: {dataset_id} (split: {split})")

    try:
        dataset = load_dataset(dataset_id, split=split, token=hf_token)
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_id}': {e}")
        sys.exit(1)

    if image_column not in dataset.column_names:
        logger.error(f"Column '{image_column}' not found in dataset")
        logger.error(f"Available columns: {dataset.column_names}")
        sys.exit(1)

    if shuffle:
        logger.info("Shuffling dataset")
        dataset = dataset.shuffle()

    if max_samples is not None:
        logger.info(f"Limiting to {max_samples} samples")
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def create_dataset_card(
    source_dataset: str,
    model: str,
    class_name: str,
    output_format: str,
    num_samples: int,
    total_detections: int,
    images_with_detections: int,
    processing_time: str,
    confidence_threshold: float,
    mask_threshold: float,
    include_boxes: bool,
) -> str:
    from datetime import datetime

    detection_rate = (
        (images_with_detections / num_samples * 100) if num_samples > 0 else 0
    )
    avg_detections = total_detections / num_samples if num_samples > 0 else 0

    format_desc = (
        "per-instance binary masks"
        if output_format == "instance-masks"
        else "semantic segmentation maps"
    )

    return f"""---
tags:
- image-segmentation
- sam3
- segment-anything
- segmentation-masks
- uv-script
- generated
---

# Image Segmentation: {class_name.title()} using SAM3

This dataset contains **{format_desc}** for **{class_name}** segmented in images from [{source_dataset}](https://huggingface.co/datasets/{source_dataset}) using Meta's SAM3.

**Generated using**: [uv-scripts/sam3](https://huggingface.co/datasets/uv-scripts/sam3) segmentation script

## Statistics

- **Objects Segmented**: {class_name}
- **Total Instances**: {total_detections:,}
- **Images with Detections**: {images_with_detections:,} / {num_samples:,} ({detection_rate:.1f}%)
- **Average Instances per Image**: {avg_detections:.2f}
- **Output Format**: {output_format}

## Processing Details

- **Source Dataset**: [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
- **Model**: [{model}](https://huggingface.co/{model})
- **Processing Date**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
- **Processing Time**: {processing_time}
- **Confidence Threshold**: {confidence_threshold}
- **Mask Threshold**: {mask_threshold}
- **Includes Bounding Boxes**: {"Yes" if include_boxes else "No"}

## Reproduction

```bash
uv run https://huggingface.co/datasets/uv-scripts/sam3/raw/main/segment-objects.py \\
    {source_dataset} \\
    <output-dataset> \\
    --class-name {class_name} \\
    --output-format {output_format} \\
    --confidence-threshold {confidence_threshold} \\
    --mask-threshold {mask_threshold}{"  --include-boxes" if include_boxes else ""}
```

---

Generated with [UV Scripts](https://huggingface.co/uv-scripts)
"""


def main():
    args = parse_args()

    class_name = args.class_name.strip()
    if not class_name:
        logger.error("Invalid --class-name argument. Provide a class name.")
        sys.exit(1)

    logger.info("SAM3 Image Segmentation")
    logger.info(f"  Input: {args.input_dataset}")
    logger.info(f"  Output: {args.output_dataset}")
    logger.info(f"  Class: {class_name}")
    logger.info(f"  Format: {args.output_format}")
    logger.info(f"  Confidence threshold: {args.confidence_threshold}")
    logger.info(f"  Batch size: {args.batch_size}")

    if args.hf_token:
        login(token=args.hf_token)
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))

    dataset = load_and_validate_dataset(
        args.input_dataset,
        args.split,
        args.image_column,
        args.max_samples,
        args.shuffle,
        args.hf_token,
    )

    logger.info(f"Loading SAM3 model: {args.model}")
    try:
        processor = Sam3Processor.from_pretrained(args.model)
        model = Sam3Model.from_pretrained(
            args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto"
        )
        logger.info(f"Model loaded on {model.device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Ensure the model exists and you have access permissions")
        sys.exit(1)

    # Build output features
    new_features = dataset.features.copy()
    if args.output_format == "semantic-mask":
        new_features["segmentation_map"] = ImageFeature()
        new_features["num_instances"] = Value("int32")
    else:
        new_features["segmentation_masks"] = Sequence(ImageFeature())

    new_features["scores"] = Sequence(Value("float32"))
    new_features["category"] = Sequence(ClassLabel(names=[class_name]))

    if args.include_boxes:
        new_features["boxes"] = Sequence(Sequence(Value("float32"), length=4))

    logger.info("Processing images...")
    start_time = time.time()
    processed_dataset = dataset.map(
        lambda batch: process_batch(
            batch,
            args.image_column,
            class_name,
            processor,
            model,
            args.confidence_threshold,
            args.mask_threshold,
            args.output_format,
            args.include_boxes,
        ),
        batched=True,
        batch_size=args.batch_size,
        features=new_features,
        desc="Segmenting objects",
    )
    end_time = time.time()
    processing_time_str = f"{(end_time - start_time) / 60:.1f} minutes"

    # Calculate statistics
    if args.output_format == "semantic-mask":
        total_detections = sum(processed_dataset["num_instances"])
        images_with_detections = sum(
            1 for n in processed_dataset["num_instances"] if n > 0
        )
    else:
        total_detections = sum(
            len(masks) for masks in processed_dataset["segmentation_masks"]
        )
        images_with_detections = sum(
            1 for masks in processed_dataset["segmentation_masks"] if len(masks) > 0
        )

    logger.info("Segmentation complete!")
    logger.info(f"  Total instances: {total_detections}")
    logger.info(
        f"  Images with detections: {images_with_detections}/{len(processed_dataset)}"
    )

    logger.info(f"Pushing to HuggingFace Hub: {args.output_dataset}")
    try:
        processed_dataset.push_to_hub(args.output_dataset, private=args.private)
        logger.info(
            f"Dataset available at: https://huggingface.co/datasets/{args.output_dataset}"
        )
    except Exception as e:
        logger.error(f"Failed to push to hub: {e}")
        logger.info("Saving locally as backup...")
        processed_dataset.save_to_disk("./output_dataset")
        logger.info("Saved to ./output_dataset")
        sys.exit(1)

    logger.info("Creating dataset card...")
    card_content = create_dataset_card(
        source_dataset=args.input_dataset,
        model=args.model,
        class_name=class_name,
        output_format=args.output_format,
        num_samples=len(processed_dataset),
        total_detections=total_detections,
        images_with_detections=images_with_detections,
        processing_time=processing_time_str,
        confidence_threshold=args.confidence_threshold,
        mask_threshold=args.mask_threshold,
        include_boxes=args.include_boxes,
    )
    card = DatasetCard(card_content)
    card.push_to_hub(args.output_dataset, token=args.hf_token or os.getenv("HF_TOKEN"))
    logger.info("Dataset card created and pushed!")


if __name__ == "__main__":
    main()
