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
"""Render detection overlays from a dataset in this directory's schema -- and PROVE they rendered.

Draws boxes (and masks, when masks_rle is present) over the embedded images and writes PNGs.
Before reporting success it pixel-diffs every render against its source image: a page with
instances whose render is identical to the source means the overlay silently failed (alpha
bugs, empty mask lists, wrong-column reads -- all observed in real runs, twice shown to a
human as "done"). Any blank render exits nonzero and names the file.

    uv run render-detections.py <ns>/<teacher-or-training-dataset> --limit 10 --out previews/
    uv run render-detections.py "hf://buckets/<ns>/<bucket>/dataset/train.parquet" --out previews/
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image, ImageDraw

COLORS = [
    (255, 210, 0),
    (80, 200, 120),
    (90, 160, 255),
    (230, 90, 80),
    (200, 120, 220),
    (255, 150, 50),
]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "data", help="dataset repo id, or a parquet path/glob (hf:// or local)"
    )
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--out", default="previews")
    p.add_argument("--bbox-format", default="yolo", choices=["yolo", "coco_xywh"])
    p.add_argument("--no-masks", action="store_true")
    p.add_argument(
        "--min-pixels",
        type=int,
        default=1,
        help="a page with instances whose render changed fewer pixels than this is BLANK "
        "(default 1: any drawn pixel proves the overlay; a 50x50 box on a 3000px scan is real)",
    )
    args = p.parse_args()

    if "://" in args.data or args.data.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=args.data, split="train")
    else:
        ds = load_dataset(args.data, split=args.split)
    assert "image" in ds.column_names, (
        "no image column in this dataset — nothing to render over"
    )
    ds = ds.select(range(min(args.limit, len(ds))))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blank, rendered, skipped = [], 0, []
    # undecoded bytes: a corrupt image or an error row is skipped, not a crash inside datasets
    for row in ds.cast_column("image", HFImage(decode=False)).with_format(None):
        raw = row["image"]
        try:
            src = (
                Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
                if raw and raw.get("bytes")
                else None
            )
        except Exception:  # noqa: BLE001 -- any decode failure means "skip this row"
            src = None
        if src is None or row.get("error"):
            skipped.append(row["image_id"])
            continue
        im = src.copy()
        w, h = im.size
        n = len(row["objects"]["bbox"])

        if not args.no_masks and row.get("masks_rle"):
            from pycocotools import mask as mask_utils

            overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
            for i, rle in enumerate(json.loads(row["masks_rle"])):
                seg = mask_utils.decode({**rle, "counts": rle["counts"].encode()})
                if seg.shape != (h, w):
                    seg = np.asarray(Image.fromarray(seg).resize((w, h), Image.NEAREST))
                r, g, b = COLORS[i % len(COLORS)]
                tint = np.zeros((h, w, 4), np.uint8)
                tint[seg > 0] = (r, g, b, 110)
                overlay = Image.alpha_composite(overlay, Image.fromarray(tint))
            im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(im)
        for i, bbox in enumerate(row["objects"]["bbox"]):
            if args.bbox_format == "yolo":
                cx, cy, bw, bh = bbox
                box = [
                    (cx - bw / 2) * w,
                    (cy - bh / 2) * h,
                    (cx + bw / 2) * w,
                    (cy + bh / 2) * h,
                ]
            else:
                x, y, bw, bh = bbox
                box = [x, y, x + bw, y + bh]
            draw.rectangle(box, outline=COLORS[i % len(COLORS)], width=max(3, w // 400))

        name = f"{row['image_id']}_{n}inst.png"
        im.save(out / name)

        # ---- the point of this script: prove the overlay exists ----
        changed = int(np.any(np.asarray(src) != np.asarray(im), axis=-1).sum())
        if n > 0 and changed < args.min_pixels:
            blank.append(name)
        elif n > 0:
            rendered += 1
        print(
            f"{name}: {n} instances, {changed} pixels changed ({changed / (w * h):.2%})"
        )

    if blank:
        sys.exit(
            f"BLANK RENDERS ({len(blank)}): {blank} — overlays did not draw; do not show these to a human."
        )
    if skipped:
        print(f"skipped {len(skipped)} undecodable/error rows, e.g. {skipped[:3]}")
    if rendered == 0:
        sys.exit(
            "No page with instances was rendered — nothing verified; increase --limit."
        )
    print(f"OK: {rendered} non-empty renders verified against source pixels -> {out}/")


if __name__ == "__main__":
    main()
