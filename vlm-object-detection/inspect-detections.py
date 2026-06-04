# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=4.0.0", "huggingface-hub", "pillow", "matplotlib"]
# ///
"""
Local visual inspection for the output of qwen3vl-detect.py.

Given an output dataset (produced by qwen3vl-detect.py) and optionally its
source dataset (with ground-truth bboxes), renders side-by-side PNGs of
predicted detections vs ground truth, one per row.

Output: a /tmp/<slug>-viz/ directory of PNGs and a per-row text summary
(detection counts, label histograms, bbox value ranges).

Usage:
    uv run inspect-detections.py OUTPUT_DATASET [--split SPLIT] [--source SOURCE_DATASET]

If --source is provided, the script will also load the source dataset, pull
ground-truth `objects` (bbox + category), auto-discover class names from the
ClassLabel feature, and overlay GT boxes on a second panel for comparison.
"""

import argparse
import json
import os
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from datasets import load_dataset


def render_box(ax, x1, y1, x2, y2, color, lw=1.2, label=None, fontsize=7):
    rect = mpatches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=lw
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x1,
            y1,
            label,
            fontsize=fontsize,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=color,
                alpha=0.75,
                edgecolor="none",
            ),
            va="bottom",
            ha="left",
        )


def discover_class_names(ds) -> list[str]:
    """Pull class names from an `objects.category` or `objects.category_id`
    ClassLabel feature on a HF dataset. Returns [] if not found."""
    feats = ds.features
    if "objects" not in feats:
        return []
    obj_feat = feats["objects"]
    # objects may be List[Dict[...]] or Sequence(Dict[...]) — both expose `feature`
    inner = getattr(obj_feat, "feature", None) or obj_feat
    if not hasattr(inner, "keys"):
        return []
    for key in ("category", "category_id"):
        if key in inner:
            cat_feat = inner[key]
            # Sequence(ClassLabel) or ClassLabel
            cl = getattr(cat_feat, "feature", None) or cat_feat
            names = getattr(cl, "names", None)
            if names:
                return list(names)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Qwen3-VL detection output side-by-side with ground truth."
    )
    parser.add_argument("output_dataset", help="HF dataset ID with detections column")
    parser.add_argument(
        "--split", default="train", help="Split of output_dataset (default: train)"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional source dataset with ground-truth `objects` for GT overlay panel.",
    )
    parser.add_argument(
        "--source-split", default=None, help="Split of source dataset (default: same as --split)"
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="Render only the first N rows"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: /tmp/<output-slug>-viz/)",
    )
    args = parser.parse_args()

    slug = args.output_dataset.split("/")[-1]
    out_dir = args.out_dir or f"/tmp/{slug}-viz"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.output_dataset} split={args.split}…")
    ds = load_dataset(args.output_dataset, split=args.split)
    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    print(f"rows={len(ds)}  cols={ds.column_names}")

    # qwen3vl-detect preserves the source `objects` column on each output row,
    # so GT is co-located with detections (no shuffled-index join needed).
    # The source dataset (if provided) is used only to recover ClassLabel names
    # since push_to_hub strips the ClassLabel typing on round-trip.
    gt_names: list[str] = []
    if args.source:
        gt_split = args.source_split or args.split
        print(f"Loading source {args.source} split={gt_split} for class names…")
        gt_src = load_dataset(args.source, split=gt_split)
        gt_names = discover_class_names(gt_src)
        print(f"GT classes ({len(gt_names)}): {gt_names}")
    elif "objects" in ds.column_names:
        gt_names = discover_class_names(ds)
        if gt_names:
            print(f"GT classes from output features: {gt_names}")

    for i in range(len(ds)):
        row = ds[i]
        img = row["image"]
        W, H = img.size
        info = json.loads(row["inference_info"])
        dets = row["detections"]

        # GT extraction from the output row itself (qwen3vl-detect preserves
        # all input columns including `objects`). Handles both list-of-dicts
        # and dict-of-lists shapes.
        gt_cats, gt_bbox = [], []
        if "objects" in row and row["objects"] is not None:
            objs = row["objects"]
            if isinstance(objs, dict):
                gt_cats = objs.get("category", objs.get("category_id", []))
                gt_bbox = objs.get("bbox", [])
            else:  # list of dicts
                for o in objs:
                    if "category" in o:
                        gt_cats.append(o["category"])
                    elif "category_id" in o:
                        gt_cats.append(o["category_id"])
                    gt_bbox.append(o["bbox"])

        bbox_vals = [v for d in dets for v in d["bbox"]]
        bbox_max = max(bbox_vals) if bbox_vals else 0
        bbox_min = min(bbox_vals) if bbox_vals else 0
        qwen_hist = Counter(d["label"] for d in dets)
        gt_hist = Counter(
            gt_names[c] if 0 <= c < len(gt_names) else f"#{c}" for c in gt_cats
        ) if gt_names else Counter()

        print(
            f"\n=== row {i}  image_id={row.get('image_id', '?')}  orig={W}x{H}  "
            f"inference_image_size={info['image_size']} ==="
        )
        if gt_cats:
            print(f"  GT: {len(gt_cats)} objects  {dict(gt_hist)}")
        print(f"  Qwen: {len(dets)} detections  {dict(qwen_hist)}")
        print(f"  Qwen bbox range: [{bbox_min:.0f}, {bbox_max:.0f}]  (image {W}x{H})")

        # Render
        n_panels = 2 if gt_cats else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 12))
        if n_panels == 1:
            axes = [axes]
        for ax in axes:
            ax.imshow(img)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_aspect("equal")
            ax.axis("off")

        # v1.1+ output stores bbox in pixel coords; v1 stored 0-1000 normalised.
        needs_denorm = bbox_max <= 1001 and max(W, H) > 1001
        sx = (W / 1000.0) if needs_denorm else 1.0
        sy = (H / 1000.0) if needs_denorm else 1.0
        title_extra = "(0-1000 → pixels)" if needs_denorm else "(pixel coords)"

        axes[0].set_title(
            f"Detections (n={len(dets)}) {title_extra}  range [{bbox_min:.0f}, {bbox_max:.0f}]",
            fontsize=11,
        )
        for d in dets:
            b = d["bbox"]
            if len(b) == 4:
                render_box(axes[0], b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy, "#E03030", lw=1.0)
        for d in dets[:10]:
            b = d["bbox"]
            if len(b) == 4:
                render_box(
                    axes[0],
                    b[0] * sx,
                    b[1] * sy,
                    b[2] * sx,
                    b[3] * sy,
                    "#E03030",
                    lw=1.4,
                    label=d["label"][:18],
                    fontsize=8,
                )

        if gt_cats and len(axes) > 1:
            palette = [
                "#1f78b4", "#33a02c", "#ff7f00", "#6a3d9a", "#b15928",
                "#e31a1c", "#fb9a99", "#a6cee3", "#b2df8a", "#fdbf6f",
                "#cab2d6", "#ffff99",
            ]
            color_by_name: dict[str, str] = {}
            axes[1].set_title(f"Ground truth (n={len(gt_cats)})", fontsize=11)
            for c, b in zip(gt_cats, gt_bbox):
                if len(b) != 4:
                    continue
                name = gt_names[c] if 0 <= c < len(gt_names) else f"#{c}"
                color = color_by_name.setdefault(name, palette[len(color_by_name) % len(palette)])
                x, y, w_, h_ = b  # COCO xywh
                render_box(axes[1], x, y, x + w_, y + h_, color, lw=2.0, label=name[:18], fontsize=8)

        plt.tight_layout()
        path = f"{out_dir}/row{i}_id{row.get('image_id', i)}.png"
        plt.savefig(path, dpi=70, bbox_inches="tight")
        plt.close()

    print(f"\nDone. {len(ds)} rows. Open: {out_dir}")


if __name__ == "__main__":
    main()
