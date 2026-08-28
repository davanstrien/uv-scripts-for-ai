#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=4.0",
#   "pillow",
#   "numpy",
#   "pyarrow>=18",
#   "pycocotools>=2.0.11",
# ]
# ///
"""Free, local, ~30 s: prove the plumbing scripts still work BEFORE a paid job depends on them.

Builds what a teacher pass leaves behind -- annotations-only parquet parts in the
falcon-perception-bucket.py schema, a source directory of page images, a gold slice -- for four
pages: one with two instances whose masks are stored in a 2x-thumbnailed inference frame (the
frame-mismatch bug seen in a real run), one with a box and no mask, one empty, one teacher ERROR
row whose file is not a decodable image. Then it runs the plumbing chain on it and checks the
OUTPUT of each step, not just the exit code:

    embed-bucket-images.py   error row dropped, gold page excluded (and asserted), images joined
                             in, one write of train.parquet + validation.parquet; also the
                             --embed-images path (parts already carry the bytes -> no --src) and
                             --keep-errors (the undecodable row must survive to the next step)
    materialize-coco.py      COCO tree: file count == referenced count, masks resized to the
                             image frame, every mask's pixel extent agrees with its box, the
                             undecodable row skipped not crashed; a second run reuses the tree;
                             the same images with CORRECTED labels rebuild it (fingerprint);
                             --force over a tree holding a stale JPEG rebuilds clean
    render-detections.py     overlays drawn (pixel-verified), the empty page not flagged as
                             blank, the undecodable row skipped

Run it after cloning, after bumping a dependency, and before submitting any job that uses these
scripts. Exit 0 = green. Anything else names the failing check.

    uv run smoke-test.py

Not covered: the teacher pass itself (falcon-perception-bucket.py needs a GPU); run it on a
--limit slice of your own bucket.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from PIL import Image
from pycocotools import mask as mask_utils

HERE = Path(__file__).resolve().parent
IMAGE_W, IMAGE_H = 800, 1000  # the image frame
INFER_W, INFER_H = 400, 500  # the teacher's inference frame (thumbnailed 2x)

# Ground truth in image-frame pixels: (x0, y0, x1, y1). Masks are the same rectangles, but
# stored at half resolution so the scripts must resize them.
PAGE1_BOXES = [(100, 100, 300, 400), (450, 600, 750, 900)]
PAGE2_BOXES = [(200, 200, 600, 500)]
PAGE2_CORRECTED = [
    (200, 200, 600, 500),
    (50, 700, 250, 950),
]  # the step-6 loop found one more
GOLD_IMAGE_ID = 3  # the empty page is held out as "gold" and must never reach train/val
ERROR_IMAGE_ID = 4  # the teacher could not open this file; its bytes are not an image

# mirrors SCHEMA in falcon-perception-bucket.py
PARTS_SCHEMA = pa.schema(
    [
        ("__source_key", pa.string()),
        ("image_id", pa.int64()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        (
            "objects",
            pa.struct(
                [
                    ("bbox", pa.list_(pa.list_(pa.float32()))),
                    ("category", pa.list_(pa.int64())),
                    ("area", pa.list_(pa.float32())),
                    ("rectangularity", pa.list_(pa.float32())),
                ]
            ),
        ),
        ("n_instances", pa.int32()),
        ("masks_rle", pa.string()),
        ("query", pa.string()),
        ("gen_seconds", pa.float32()),
        ("error", pa.string()),
    ]
)
IMAGE_FIELD = pa.field(
    "image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])
)


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


def teacher_row(image_id, boxes, with_masks, error=None):
    bbox = [yolo_box(*b) for b in boxes]
    masks = (
        [rle_in_inference_frame(*[v // 2 for v in b]) for b in boxes]
        if with_masks
        else []
    )
    return {
        "__source_key": f"pages/{image_id}.jpg",
        "image_id": image_id,
        "width": None if error else IMAGE_W,
        "height": None if error else IMAGE_H,
        "objects": {
            "bbox": bbox,
            "category": [0] * len(bbox),
            "area": [b[2] * b[3] for b in bbox],
            "rectangularity": [1.0] * len(bbox),
        },
        "n_instances": len(bbox),
        "masks_rle": json.dumps(masks),
        "query": "illustration",
        "gen_seconds": 0.1,
        "error": error,
    }


def build_teacher_output(root):
    """parts/ (annotations-only), parts-corrected/, parts-embedded/ (with bytes), pages/, gold.parquet"""
    pages = root / "pages" / "pages"
    pages.mkdir(parents=True)
    rows = [
        teacher_row(1, PAGE1_BOXES, with_masks=True),
        teacher_row(2, PAGE2_BOXES, with_masks=False),
        teacher_row(GOLD_IMAGE_ID, [], with_masks=False),
        teacher_row(
            ERROR_IMAGE_ID, [], with_masks=False, error="OSError: truncated file"
        ),
    ]
    blobs = {}
    for row in rows:
        path = pages / f"{row['image_id']}.jpg"
        if row["error"]:
            path.write_bytes(b"this is not a jpeg")
        else:
            synthetic_page(row["image_id"]).save(path, "JPEG", quality=95)
        blobs[row["image_id"]] = path.read_bytes()

    (root / "parts").mkdir()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=PARTS_SCHEMA),
        root / "parts" / "part-a.parquet",
    )

    corrected = [
        teacher_row(2, PAGE2_CORRECTED, with_masks=False) if r["image_id"] == 2 else r
        for r in rows
    ]
    (root / "parts-corrected").mkdir()
    pq.write_table(
        pa.Table.from_pylist(corrected, schema=PARTS_SCHEMA),
        root / "parts-corrected" / "part-a.parquet",
    )

    # --embed-images parts: the error row carries image=None (the teacher never read its bytes)
    embedded = [
        {
            **r,
            "image": None
            if r["error"]
            else {"bytes": blobs[r["image_id"]], "path": None},
        }
        for r in rows
    ]
    (root / "parts-embedded").mkdir()
    pq.write_table(
        pa.Table.from_pylist(embedded, schema=PARTS_SCHEMA.append(IMAGE_FIELD)),
        root / "parts-embedded" / "part-a.parquet",
    )

    gold = pa.table({"image_id": pa.array([GOLD_IMAGE_ID], pa.int64())})
    pq.write_table(gold, root / "gold.parquet")


def run(script, *argv):
    # `uv run` so each script resolves its OWN PEP 723 header -- a bad dependency line in a
    # child script is exactly the kind of breakage this test exists to catch
    cmd = ["uv", "run", "--quiet", str(HERE / script), *map(str, argv)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        sys.exit(f"FAIL: {script} exited {proc.returncode}")
    return proc.stdout


def check(condition, message):
    if not condition:
        sys.exit(f"FAIL: {message}")


def check_embedded(out_dir, label, expect_ids=(1, 2)):
    files = sorted(p.name for p in out_dir.glob("*.parquet"))
    check(files == ["train.parquet", "validation.parquet"], f"{label}: wrote {files}")
    ds = load_dataset("parquet", data_files=str(out_dir / "*.parquet"), split="train")
    ids = sorted(ds["image_id"])
    check(
        ids == list(expect_ids),
        f"{label}: expected pages {list(expect_ids)}, got {ids}",
    )
    check("image" in ds.column_names, f"{label}: no image column")
    check(
        type(ds.features["image"]).__name__ == "Image",
        f"{label}: image column is not an Image feature",
    )
    names = ds.features["objects"]["category"].feature.names
    check(names == ["illustration"], f"{label}: category names {names}")
    print(f"OK embed-bucket-images ({label}): pages {ids}, Image column, one write")


def load_tree(coco_dir):
    """All splits merged: which page lands in train vs val depends on the shuffle, the checks don't."""
    images, anns, files, skipped = {}, [], 0, set()
    for split_dir in ("train2017", "val2017"):
        ann_path = coco_dir / "annotations" / f"instances_{split_dir}.json"
        coco = json.loads(ann_path.read_text())
        n_files = len(list((coco_dir / split_dir).glob("*.jpg")))
        check(
            n_files == len(coco["images"]),
            f"{split_dir}: {n_files} files != {len(coco['images'])} referenced",
        )
        check(
            coco["categories"] == [{"id": 1, "name": "illustration"}],
            "categories wrong",
        )
        files += n_files
        images.update({im["id"]: im for im in coco["images"]})
        anns.extend(coco["annotations"])
        skipped |= set(coco["provenance"]["skipped_image_ids"])
    return images, anns, files, skipped


def check_coco(coco_dir, page2_boxes=PAGE2_BOXES, expect_skipped=()):
    images, annotations, n_files, skipped = load_tree(coco_dir)
    check(
        sorted(images) == [1, 2], f"tree holds pages {sorted(images)}, expected [1, 2]"
    )
    check(n_files == 2, f"{n_files} JPEGs in the tree, expected 2")
    check(
        skipped == set(expect_skipped),
        f"skipped ids {skipped}, expected {set(expect_skipped)}",
    )
    by_image = {}
    for ann in annotations:
        by_image.setdefault(ann["image_id"], []).append(ann)
    check(GOLD_IMAGE_ID not in by_image, "gold page leaked into the COCO tree")
    check(ERROR_IMAGE_ID not in by_image, "error page leaked into the COCO tree")
    for image_id, boxes in ((1, PAGE1_BOXES), (2, page2_boxes)):
        anns = by_image.get(image_id, [])
        check(
            len(anns) == len(boxes),
            f"page {image_id}: {len(anns)} annotations, expected {len(boxes)}",
        )
        for ann, (x0, y0, x1, y1) in zip(anns, boxes):
            bx, by, bw, bh = [round(v) for v in ann["bbox"]]
            check(
                (bx, by, bw, bh) == (x0, y0, x1 - x0, y1 - y0),
                f"bbox mismatch: {ann['bbox']}",
            )
            if image_id == 1:
                seg = ann.get("segmentation")
                check(seg is not None, "page 1 annotation lost its mask")
                check(
                    seg["size"] == [IMAGE_H, IMAGE_W],
                    f"mask not resized to image frame: {seg['size']}",
                )
                m = mask_utils.decode({**seg, "counts": seg["counts"].encode()})
                ys, xs = np.where(m)
                extent = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                check(
                    extent == (x0, y0, x1, y1),
                    f"mask extent {extent} != box {(x0, y0, x1, y1)}",
                )
            else:
                check(
                    "segmentation" not in ann,
                    "page 2 has no mask but got a segmentation",
                )
    print(
        "OK materialize-coco: files == referenced, masks resized 500x400 -> 1000x800 and aligned to boxes, "
        f"page 2 has {len(page2_boxes)} box(es)"
    )


def check_render(stdout, previews):
    names = sorted(p.name for p in previews.glob("*.png"))
    check(
        names == ["1_2inst.png", "2_1inst.png", "3_0inst.png"],
        f"unexpected previews: {names}",
    )
    check("OK: 2 non-empty renders verified" in stdout, "render did not verify 2 pages")
    check(
        "skipped 1 undecodable/error rows" in stdout,
        "render did not skip the undecodable page",
    )
    print(
        "OK render-detections: 2 overlays pixel-verified, empty page not flagged, undecodable page skipped"
    )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_teacher_output(tmp)
        print(f"fixture: 4 pages of teacher output -> {tmp}")

        # 1. annotations-only parts + local source dir (the default teacher output):
        #    error row dropped by default, gold page excluded
        run(
            "embed-bucket-images.py",
            "--parts",
            tmp / "parts" / "part-*.parquet",
            "--src",
            tmp / "pages",
            "--gold",
            tmp / "gold.parquet",
            "--out",
            tmp / "dataset",
            "--val-frac",
            "0.5",
            "--chunk",
            "1",
        )
        check_embedded(tmp / "dataset", "annotations-only parts + --src")

        # 2. parts that already carry the bytes (teacher ran with --embed-images): no --src
        run(
            "embed-bucket-images.py",
            "--parts",
            tmp / "parts-embedded" / "part-*.parquet",
            "--gold",
            tmp / "gold.parquet",
            "--out",
            tmp / "dataset-embedded",
            "--val-frac",
            "0.5",
        )
        check_embedded(tmp / "dataset-embedded", "--embed-images parts, no --src")

        # 3. --keep-errors: the undecodable page must reach the next step, which must survive it
        run(
            "embed-bucket-images.py",
            "--parts",
            tmp / "parts" / "part-*.parquet",
            "--src",
            tmp / "pages",
            "--gold",
            tmp / "gold.parquet",
            "--out",
            tmp / "dataset-kept",
            "--val-frac",
            "0.34",
            "--keep-errors",
        )
        check_embedded(
            tmp / "dataset-kept", "--keep-errors", expect_ids=(1, 2, ERROR_IMAGE_ID)
        )

        # 4. COCO tree (undecodable row skipped, not crashed), then a second run must reuse it
        run(
            "materialize-coco.py", "--data", tmp / "dataset-kept", "--out", tmp / "coco"
        )
        check_coco(tmp / "coco", expect_skipped=(ERROR_IMAGE_ID,))
        again = run(
            "materialize-coco.py", "--data", tmp / "dataset-kept", "--out", tmp / "coco"
        )
        check(
            again.count("reusing complete tree") == 2,
            f"second run rebuilt instead of reusing:\n{again}",
        )
        print("OK materialize-coco: second run reused both complete trees")

        # 5. same images, CORRECTED labels (the step-6 loop) -> the tree must rebuild, not reuse
        run(
            "embed-bucket-images.py",
            "--parts",
            tmp / "parts-corrected" / "part-*.parquet",
            "--src",
            tmp / "pages",
            "--gold",
            tmp / "gold.parquet",
            "--out",
            tmp / "dataset-corrected",
            "--val-frac",
            "0.5",
        )
        rebuilt = run(
            "materialize-coco.py",
            "--data",
            tmp / "dataset-corrected",
            "--out",
            tmp / "coco",
        )
        # the fingerprint is per split: only the split holding page 2 must rebuild, the other may reuse
        check(
            "labels changed" in rebuilt,
            f"corrected labels did not trigger a rebuild:\n{rebuilt}",
        )
        check_coco(tmp / "coco", page2_boxes=PAGE2_CORRECTED)
        print(
            "OK materialize-coco: corrected labels rebuilt the tree (fingerprint), new box present"
        )

        # 6. --force over a tree holding a stale JPEG must come back clean
        (tmp / "coco" / "train2017" / "999.jpg").write_bytes(b"stale")
        run(
            "materialize-coco.py",
            "--data",
            tmp / "dataset-corrected",
            "--out",
            tmp / "coco",
            "--force",
        )
        check(
            not (tmp / "coco" / "train2017" / "999.jpg").exists(),
            "--force left a stale JPEG in the tree",
        )
        check_coco(tmp / "coco", page2_boxes=PAGE2_CORRECTED)
        print("OK materialize-coco: --force cleared the stale file and rebuilt clean")

        # 7. overlays over the raw teacher pages (all 4, incl. the empty and the undecodable one)
        run(
            "embed-bucket-images.py",
            "--parts",
            tmp / "parts" / "part-*.parquet",
            "--src",
            tmp / "pages",
            "--out",
            tmp / "all",
            "--val-frac",
            "0.25",
            "--keep-errors",
        )
        out = run(
            "render-detections.py", tmp / "all" / "*.parquet", "--out", tmp / "previews"
        )
        check_render(out, tmp / "previews")

    print("SMOKE TEST GREEN")


if __name__ == "__main__":
    main()
