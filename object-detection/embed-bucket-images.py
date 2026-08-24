#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=4.0",
#   "huggingface_hub>=1.0",
#   "pyarrow>=18",
# ]
# ///
"""Build the canonical training parquet from a bucket teacher pass -> ONE final-schema push.

All image bytes are held in RAM while building (fine to ~10k typical page scans; for much
bigger corpora run in chunks with --limit or on a high-RAM Jobs flavor).

Takes the annotations-only parquet parts that falcon-perception-bucket.py wrote, joins the
image bytes back in from the source bucket, excludes (and ASSERTS the exclusion of) a gold
slice, splits train/validation, and pushes everything in a single final-schema write --
either to a dataset repo (one push_to_hub, never a second) or as train.parquet /
validation.parquet in a bucket for `materialize-coco.py` / direct `load_dataset` use.

    uv run embed-bucket-images.py \
      --parts "hf://buckets/<ns>/<teacher-out>/part-*.parquet" \
      --src <ns>/<source-bucket> \
      --gold <ns>/<gold-dataset> \
      --out <ns>/<training-dataset> --private

    # bucket output instead of a dataset repo:
    ... --out hf://buckets/<ns>/<training-bucket>/dataset

Why this exists (both failure modes measured): the bucket path's output has no image column,
so every agent hand-writes this join; and staging an intermediate push then re-pushing a
different schema to the same repo id leaves stale repo features -> load_dataset CastError.
This script builds the final schema in memory and writes exactly once.
"""

import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
from datasets import ClassLabel, DatasetDict, Image, Sequence, load_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--parts",
        required=True,
        help='parquet glob, e.g. "hf://buckets/ns/out/part-*.parquet"',
    )
    p.add_argument(
        "--src",
        required=True,
        help="source image bucket, e.g. ns/pages (keys = __source_key)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="dataset repo id, or hf://buckets/... prefix for parquet files",
    )
    p.add_argument(
        "--gold",
        default=None,
        help="gold dataset repo id (or parquet glob) to exclude, by image_id",
    )
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit", type=int, default=None, help="debug: cap rows AFTER gold exclusion"
    )
    p.add_argument(
        "--drop-errors",
        action="store_true",
        help="drop rows whose teacher pass errored",
    )
    p.add_argument(
        "--allow-gold-disjoint",
        action="store_true",
        help="permit a gold set that shares no image_id with this corpus (a genuinely different corpus)",
    )
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    try:
        ds = load_dataset("parquet", data_files=args.parts, split="train")
    except FileNotFoundError:
        raise SystemExit(
            f"no parquet files match {args.parts!r} — check the glob and bucket path"
        )
    total = len(ds)
    if total == 0:
        raise SystemExit(f"{args.parts!r} matched files but they contain 0 rows")
    if args.drop_errors:
        ds = ds.filter(lambda r: not r["error"])
    if not isinstance(ds.features["objects"]["category"].feature, ClassLabel):
        feats = ds.features.copy()
        feats["objects"] = dict(
            feats["objects"]
        )  # copy() is shallow; don't mutate ds.features
        feats["objects"]["category"] = Sequence(ClassLabel(names=[ds[0]["query"]]))
        ds = ds.cast(feats)

    # ---- gold exclusion, asserted on the stable image_id (never on path-shaped keys) ----
    if args.gold:
        gold_files = (
            args.gold
            if "://" in args.gold
            else f"hf://datasets/{args.gold}/data/train-*.parquet"
        )
        gold_ids = set(
            load_dataset("parquet", data_files=gold_files, split="train")["image_id"]
        )
        before = len(ds)
        ds = ds.filter(lambda r: r["image_id"] not in gold_ids)
        removed = before - len(ds)
        overlap = gold_ids & set(ds["image_id"])
        assert not overlap, (
            f"gold exclusion FAILED: {len(overlap)} gold ids remain, e.g. {sorted(overlap)[:3]}"
        )
        if removed == 0 and gold_ids and not args.allow_gold_disjoint:
            raise SystemExit(
                "gold exclusion matched 0 rows — the gold set and this corpus share no image_id. "
                "That usually means the ids were built from different key prefixes. If this corpus "
                "really is disjoint from the gold set, pass --allow-gold-disjoint."
            )
        print(f"gold: excluded {removed} rows; {len(gold_ids)} gold ids, overlap now 0")

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # ---- join image bytes back in from the source bucket ----
    def fetch(key):
        with fsspec.open(f"hf://buckets/{args.src}/{key}", "rb") as f:
            return f.read()

    keys = ds["__source_key"]
    with ThreadPoolExecutor(args.workers) as ex:
        blobs = list(ex.map(fetch, keys))
    bad = [k for k, b in zip(keys, blobs) if not b]
    assert not bad, f"{len(bad)} images fetched empty, e.g. {bad[:3]}"
    ds = ds.add_column(
        "image", [{"bytes": b, "path": None} for b in blobs]
    ).cast_column("image", Image())

    # ---- split, then ONE write ----
    parts = ds.train_test_split(test_size=args.val_frac, seed=args.seed)
    out = DatasetDict({"train": parts["train"], "validation": parts["test"]})
    print(
        f"rows: {total} read -> {len(ds)} kept -> train {len(out['train'])} / validation {len(out['validation'])}"
    )

    if args.out.startswith("hf://buckets/"):
        with tempfile.TemporaryDirectory() as td:
            for split, d in out.items():
                local = Path(td) / f"{split}.parquet"
                d.to_parquet(local)
                subprocess.run(
                    ["hf", "cp", str(local), f"{args.out.rstrip('/')}/{split}.parquet"],
                    check=True,
                )
        print(f"wrote train.parquet + validation.parquet -> {args.out}")
    else:
        out.push_to_hub(args.out, private=args.private)
        print(f"pushed -> https://huggingface.co/datasets/{args.out}")


if __name__ == "__main__":
    main()
