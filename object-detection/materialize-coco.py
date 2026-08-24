#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=4.0",
#   "huggingface_hub>=1.0",
#   "pillow",
# ]
# ///
"""Materialize a COCO directory tree FROM the canonical parquet, in-job, on ephemeral disk.

Some trainers (RF-DETR and friends) refuse HF datasets and demand the canonical COCO 2017
layout: annotations/instances_train2017.json + train2017/*.jpg. Never hand-assemble or
upload that tree -- generate it from the parquet on the training node instead. A generated
tree cannot reference images that are not there, which kills the referenced-vs-uploaded
mismatch class outright (it caused three paid job failures in one measured run).

    # inside the training job, before the trainer starts:
    uv run materialize-coco.py --data hf://buckets/<ns>/<training-bucket>/dataset --out /tmp/coco

    # or from a dataset repo produced by embed-bucket-images.py:
    uv run materialize-coco.py --data <ns>/<training-dataset> --out /tmp/coco

Boxes are converted yolo-normalized -> COCO xywh pixels (pass --bbox-format coco_xywh if your
parquet already stores pixels). masks_rle, when present, is carried through as COCO RLE
segmentation (RF-DETR-class trainers accept RLE natively).
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

SPLIT_DIR = {"train": "train2017", "validation": "val2017"}


def to_xywh(bbox, w, h, fmt):
    if fmt == "coco_xywh":
        return [float(v) for v in bbox]
    cx, cy, bw, bh = bbox  # yolo normalised
    return [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data",
        required=True,
        help="dataset repo id, or hf://buckets/... prefix holding <split>.parquet",
    )
    p.add_argument(
        "--out", required=True, help="output dir (ephemeral disk, e.g. /tmp/coco)"
    )
    p.add_argument("--bbox-format", default="yolo", choices=["yolo", "coco_xywh"])
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = p.parse_args()

    out = Path(args.out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        if args.data.startswith("hf://"):
            ds = load_dataset(
                "parquet",
                data_files=f"{args.data.rstrip('/')}/{split}.parquet",
                split="train",
            )
        else:
            ds = load_dataset(args.data, split=split)
        assert "image" in ds.column_names, (
            "no image column — run embed-bucket-images.py first"
        )

        cat_feature = ds.features["objects"]["category"].feature
        names = getattr(cat_feature, "names", None) or ["object"]
        img_dir = out / SPLIT_DIR.get(split, split)
        img_dir.mkdir(exist_ok=True)

        images, annotations, ann_id = [], [], 1
        for row in ds.with_format(None):
            iid = int(row["image_id"])
            fname = f"{iid}.jpg"
            row["image"].convert("RGB").save(img_dir / fname, "JPEG", quality=95)
            w, h = int(row["width"]), int(row["height"])
            images.append({"id": iid, "file_name": fname, "width": w, "height": h})
            rles = json.loads(row["masks_rle"]) if row.get("masks_rle") else []
            for i, bbox in enumerate(row["objects"]["bbox"]):
                x, y, bw, bh = to_xywh(bbox, w, h, args.bbox_format)
                ann = {
                    "id": ann_id,
                    "image_id": iid,
                    "category_id": int(row["objects"]["category"][i]) + 1,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
                if i < len(rles):
                    ann["segmentation"] = rles[i]
                annotations.append(ann)
                ann_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
        }
        jpath = out / "annotations" / f"instances_{SPLIT_DIR.get(split, split)}.json"
        jpath.write_text(json.dumps(coco))

        n_files = len(list(img_dir.glob("*.jpg")))
        assert n_files == len(images), (
            f"{split}: {n_files} files != {len(images)} referenced"
        )
        print(
            f"{split}: {len(images)} images / {len(annotations)} annotations -> {img_dir} + {jpath.name}"
        )


if __name__ == "__main__":
    main()
