#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=4.0",
#   "pillow",
#   "numpy",
#   "pycocotools>=2.0.11",
# ]
# ///
"""Free, local, ~20 s: prove the plumbing scripts still work BEFORE a paid job depends on them.

Builds a 3-page synthetic dataset in this directory's schema -- one page with two instances whose
masks are stored in a 2x-thumbnailed inference frame (the frame-mismatch bug seen in a real run),
one page with a box and no mask, one empty page -- then runs the plumbing scripts on it and checks
their output, not just their exit codes:

    materialize-coco.py   COCO tree: file count == referenced count, masks resized to the image
                          frame, every mask's pixel extent agrees with its box
    render-detections.py  overlays drawn (pixel-verified), the empty page not flagged as blank

Run it after cloning, after bumping a dependency, and before submitting any job that uses these
scripts. Exit 0 = green. Anything else names the failing check.

    uv run smoke-test.py

Not covered: embed-bucket-images.py (needs a real bucket for the image join; run it on a --limit
slice of your own bucket instead).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from datasets import Image as HFImage
from PIL import Image
from pycocotools import mask as mask_utils

HERE = Path(__file__).resolve().parent
IMAGE_W, IMAGE_H = 800, 1000  # the image frame
INFER_W, INFER_H = 400, 500  # the teacher's inference frame (thumbnailed 2x)


def synthetic_page(seed):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 10, (IMAGE_H, IMAGE_W, 1), dtype=np.uint8)
    arr = np.full((IMAGE_H, IMAGE_W, 3), 245, np.uint8) - noise
    return Image.fromarray(arr)


def rle_in_inference_frame(x0, y0, x1, y1):
    m = np.zeros((INFER_H, INFER_W), np.uint8)
    m[y0:y1, x0:x1] = 1
    encoded = mask_utils.encode(np.asfortranarray(m))
    return {"size": [INFER_H, INFER_W], "counts": encoded["counts"].decode("ascii")}


def yolo_box(x0, y0, x1, y1):
    """Pixel corners in the IMAGE frame -> yolo-normalised [cx, cy, w, h]."""
    return [
        (x0 + x1) / 2 / IMAGE_W,
        (y0 + y1) / 2 / IMAGE_H,
        (x1 - x0) / IMAGE_W,
        (y1 - y0) / IMAGE_H,
    ]


# Ground truth in image-frame pixels: (x0, y0, x1, y1). Masks are the same rectangles, but
# stored at half resolution so the scripts must resize them.
PAGE1_BOXES = [(100, 100, 300, 400), (450, 600, 750, 900)]
PAGE2_BOXES = [(200, 200, 600, 500)]


def build_fixture(out_dir):
    rows = [
        {
            "image_id": 1,
            "image": synthetic_page(1),
            "width": IMAGE_W,
            "height": IMAGE_H,
            "objects": {
                "bbox": [yolo_box(*b) for b in PAGE1_BOXES],
                "category": [0, 0],
            },
            "masks_rle": json.dumps(
                [rle_in_inference_frame(*[v // 2 for v in b]) for b in PAGE1_BOXES]
            ),
        },
        {
            "image_id": 2,
            "image": synthetic_page(2),
            "width": IMAGE_W,
            "height": IMAGE_H,
            "objects": {"bbox": [yolo_box(*b) for b in PAGE2_BOXES], "category": [0]},
            "masks_rle": None,
        },
        {
            "image_id": 3,
            "image": synthetic_page(3),
            "width": IMAGE_W,
            "height": IMAGE_H,
            "objects": {"bbox": [], "category": []},
            "masks_rle": None,
        },
    ]
    features = Features(
        {
            "image_id": Value("int64"),
            "image": HFImage(),
            "width": Value("int64"),
            "height": Value("int64"),
            "objects": {
                "bbox": Sequence(Sequence(Value("float32"))),
                "category": Sequence(ClassLabel(names=["illustration"])),
            },
            "masks_rle": Value("string"),
        }
    )
    ds = Dataset.from_list(rows, features=features)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out_dir / "train.parquet"))
    ds.select([0]).to_parquet(str(out_dir / "validation.parquet"))


def run(script, *argv):
    # `uv run` so each script resolves its OWN PEP 723 header -- a bad dependency line in a
    # child script is exactly the kind of breakage this test exists to catch
    cmd = ["uv", "run", "--quiet", str(HERE / script), *map(str, argv)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        sys.exit(f"FAIL: {script} exited {proc.returncode}")
    return proc.stdout


def check(condition, message):
    if not condition:
        sys.exit(f"FAIL: {message}")


def check_coco(coco_dir):
    ann_path = coco_dir / "annotations" / "instances_train2017.json"
    coco = json.loads(ann_path.read_text())
    n_files = len(list((coco_dir / "train2017").glob("*.jpg")))
    check(n_files == len(coco["images"]) == 3, "train2017 file count != referenced images")
    check(coco["categories"] == [{"id": 1, "name": "illustration"}], "categories wrong")

    by_image = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    check(len(by_image.get(1, [])) == 2, "page 1 should have 2 annotations")
    check(len(by_image.get(2, [])) == 1, "page 2 should have 1 annotation")
    check(3 not in by_image, "empty page 3 should have no annotations")

    for ann, (x0, y0, x1, y1) in zip(by_image[1], PAGE1_BOXES):
        bx, by, bw, bh = [round(v) for v in ann["bbox"]]
        check((bx, by, bw, bh) == (x0, y0, x1 - x0, y1 - y0), f"bbox mismatch: {ann['bbox']}")
        seg = ann.get("segmentation")
        check(seg is not None, "page 1 annotation lost its mask")
        check(seg["size"] == [IMAGE_H, IMAGE_W], f"mask not resized to image frame: {seg['size']}")
        m = mask_utils.decode({**seg, "counts": seg["counts"].encode()})
        ys, xs = np.where(m)
        extent = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        check(extent == (x0, y0, x1, y1), f"mask extent {extent} != box {(x0, y0, x1, y1)}")
    check("segmentation" not in by_image[2][0], "page 2 has no mask but got a segmentation")
    print("OK materialize-coco: 3 files, 3 annotations, masks resized 500x400 -> 1000x800 and aligned to boxes")


def check_render(stdout, previews):
    names = sorted(p.name for p in previews.glob("*.png"))
    check(names == ["1_2inst.png", "2_1inst.png", "3_0inst.png"], f"unexpected previews: {names}")
    check("OK: 2 non-empty renders verified" in stdout, "render did not verify 2 pages")
    print("OK render-detections: 2 overlays pixel-verified, empty page not flagged")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = tmp / "data"
        build_fixture(data)
        print(f"fixture: 3 pages -> {data}")

        run("materialize-coco.py", "--data", data, "--out", tmp / "coco")
        check_coco(tmp / "coco")

        out = run("render-detections.py", data / "train.parquet", "--out", tmp / "previews")
        check_render(out, tmp / "previews")

    print("SMOKE TEST GREEN")


if __name__ == "__main__":
    main()
