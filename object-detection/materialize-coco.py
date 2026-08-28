#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=4.0",
#   "huggingface_hub>=1.27",  # hf://buckets in HfFileSystem (1.6) + prefix-collision fix (1.27)
#   "pillow",
#   "numpy",
#   "pycocotools>=2.0.11",
# ]
# ///
"""Materialize a COCO directory tree FROM the canonical parquet, in-job, on ephemeral disk.

Some trainers (RF-DETR and friends) refuse HF datasets and demand the canonical COCO 2017
layout: annotations/instances_train2017.json + train2017/*.jpg. Never hand-assemble or
upload that tree -- generate it from the parquet with this script instead. A generated
tree cannot reference images that are not there, which kills the referenced-vs-uploaded
mismatch class outright (it caused three paid job failures in one measured run).

    # inside the training job, before the trainer starts:
    uv run materialize-coco.py --data hf://buckets/<ns>/<training-bucket>/dataset --out /tmp/coco

    # or from a dataset repo produced by embed-bucket-images.py:
    uv run materialize-coco.py --data <ns>/<training-dataset> --out /tmp/coco

    # training more than once? generate ONCE onto a bucket mount and let later jobs reuse it:
    #   hf jobs run ... -v hf://buckets/<ns>/<training-bucket>:/data ...
    uv run materialize-coco.py --data /data/dataset --out /data/coco

A split whose tree is already complete (annotations file present, image count matches) is
reused, not regenerated; --force rebuilds it.

Boxes are converted yolo-normalized -> COCO xywh pixels (pass --bbox-format coco_xywh if your
parquet already stores pixels). masks_rle, when present, is carried through as COCO RLE
segmentation (RF-DETR-class trainers accept RLE natively).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image as PILImage
from pycocotools import mask as mask_utils

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
        help="dataset repo id, hf://buckets/... prefix, or local directory holding <split>.parquet",
    )
    p.add_argument(
        "--out", required=True, help="output dir (ephemeral disk, e.g. /tmp/coco)"
    )
    p.add_argument("--bbox-format", default="yolo", choices=["yolo", "coco_xywh"])
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument(
        "--force",
        action="store_true",
        help="rebuild a split even if its tree is complete",
    )
    args = p.parse_args()

    out = Path(args.out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        img_dir = out / SPLIT_DIR.get(split, split)
        jpath = out / "annotations" / f"instances_{SPLIT_DIR.get(split, split)}.json"
        if jpath.exists() and not args.force:
            referenced = len(json.loads(jpath.read_text())["images"])
            present = len(list(img_dir.glob("*.jpg")))
            if referenced and referenced == present:
                print(
                    f"{split}: reusing complete tree ({present} images) at {img_dir} — pass --force to rebuild"
                )
                continue
            print(
                f"{split}: tree incomplete ({present} files vs {referenced} referenced) — rebuilding"
            )

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
        img_dir.mkdir(exist_ok=True)

        images, annotations, ann_id = [], [], 1
        for row in ds.with_format(None):
            iid = int(row["image_id"])
            fname = f"{iid}.jpg"
            im = row["image"].convert("RGB")
            im.save(img_dir / fname, "JPEG", quality=95)
            # dims from the DECODED image, never metadata columns: error rows carry
            # width/height=None, and the saved JPEG is the frame everything must match
            w, h = im.size
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
                    rle = rles[i]
                    # masks live in the INFERENCE frame, which diverges from the image
                    # frame whenever the teacher thumbnailed -- resize before writing
                    if rle["size"] != [h, w]:
                        seg = mask_utils.decode(
                            {**rle, "counts": rle["counts"].encode()}
                        )
                        seg = np.asarray(
                            PILImage.fromarray(seg).resize((w, h), PILImage.NEAREST)
                        )
                        enc = mask_utils.encode(np.asfortranarray(seg))
                        rle = {"size": [h, w], "counts": enc["counts"].decode("ascii")}
                    ann["segmentation"] = rle
                annotations.append(ann)
                ann_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
        }
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
