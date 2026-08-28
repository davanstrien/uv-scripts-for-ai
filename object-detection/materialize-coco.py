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

A split is reused, not regenerated, when its tree is complete AND was built from the same
labels: the annotations file carries a fingerprint of (image ids, boxes), so a corrected
dataset -- the step-6 loop, same images, new labels -- rebuilds automatically. --force rebuilds
regardless. Rows whose image cannot be decoded (teacher error rows, truncated files) are skipped
and counted, never allowed to kill the job.

Boxes are converted yolo-normalized -> COCO xywh pixels (pass --bbox-format coco_xywh if your
parquet already stores pixels). masks_rle, when present, is carried through as COCO RLE
segmentation (RF-DETR-class trainers accept RLE natively).
"""

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image as PILImage
from pycocotools import mask as mask_utils

SPLIT_DIR = {"train": "train2017", "validation": "val2017"}


def to_xywh(bbox, w, h, fmt):
    if fmt == "coco_xywh":
        return [float(v) for v in bbox]
    cx, cy, bw, bh = bbox  # yolo normalised
    return [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]


def label_fingerprint(ds):
    """Hash of (image_id, boxes) for every row -- changes when labels change, not when bytes do."""
    labels = ds.select_columns(["image_id", "objects"]).with_format(None)
    items = sorted(
        (
            int(row["image_id"]),
            [[round(float(v), 6) for v in b] for b in row["objects"]["bbox"]],
        )
        for row in labels
    )
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()


def load_split(data, split):
    if data.startswith("hf://"):
        return load_dataset(
            "parquet", data_files=f"{data.rstrip('/')}/{split}.parquet", split="train"
        )
    return load_dataset(data, split=split)


def tree_is_reusable(jpath, img_dir, fingerprint):
    if not jpath.exists():
        return False, "no tree yet"
    coco = json.loads(jpath.read_text())
    stamped = coco.get("provenance", {}).get("fingerprint")
    if stamped != fingerprint:
        return False, "labels changed since the tree was built"
    referenced = len(coco["images"])
    present = len(list(img_dir.glob("*.jpg")))
    if not referenced or referenced != present:
        return False, f"tree incomplete ({present} files vs {referenced} referenced)"
    return True, f"complete tree ({present} images), same labels"


def decode_image(raw):
    """raw is the undecoded {bytes, path} struct (or None for error rows)."""
    if not raw or not raw.get("bytes"):
        return None
    try:
        im = PILImage.open(io.BytesIO(raw["bytes"]))
        im.load()
        return im.convert("RGB")
    except Exception:  # noqa: BLE001 -- any decode failure means "skip this row"
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data",
        required=True,
        help="dataset repo id, hf://buckets/... prefix, or local directory holding <split>.parquet",
    )
    p.add_argument(
        "--out",
        required=True,
        help="output dir (ephemeral disk, or a bucket mount to reuse across jobs)",
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
        ds = load_split(args.data, split)
        assert "image" in ds.column_names, (
            "no image column — run embed-bucket-images.py first"
        )
        img_dir = out / SPLIT_DIR.get(split, split)
        jpath = out / "annotations" / f"instances_{SPLIT_DIR.get(split, split)}.json"

        fingerprint = label_fingerprint(ds)
        reusable, why = tree_is_reusable(jpath, img_dir, fingerprint)
        if reusable and not args.force:
            print(f"{split}: reusing {why} at {img_dir} — pass --force to rebuild")
            continue
        print(f"{split}: building ({'--force' if args.force else why})")
        # a rebuild starts from nothing: stale JPEGs from an older tree would fail the
        # files == referenced assert below after all the decode work
        shutil.rmtree(img_dir, ignore_errors=True)
        jpath.unlink(missing_ok=True)
        img_dir.mkdir()

        cat_feature = ds.features["objects"]["category"].feature
        names = getattr(cat_feature, "names", None) or ["object"]

        # undecoded bytes so a corrupt image is OUR decision to skip, not a crash inside datasets
        rows = ds.cast_column("image", HFImage(decode=False)).with_format(None)
        images, annotations, ann_id, skipped = [], [], 1, []
        for row in rows:
            iid = int(row["image_id"])
            im = None if row.get("error") else decode_image(row["image"])
            if im is None:
                skipped.append(iid)
                continue
            fname = f"{iid}.jpg"
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
            "provenance": {
                "source": args.data,
                "fingerprint": fingerprint,
                "skipped_image_ids": skipped,
            },
        }
        jpath.write_text(json.dumps(coco))

        n_files = len(list(img_dir.glob("*.jpg")))
        assert n_files == len(images), (
            f"{split}: {n_files} files != {len(images)} referenced"
        )
        note = (
            f" (skipped {len(skipped)} undecodable/error rows, e.g. {skipped[:3]})"
            if skipped
            else ""
        )
        print(
            f"{split}: {len(images)} images / {len(annotations)} annotations -> {img_dir} + {jpath.name}{note}"
        )


if __name__ == "__main__":
    main()
