#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=4.0",
#   "huggingface_hub>=1.27",  # hf://buckets in HfFileSystem (1.6) + prefix-collision fix (1.27)
#   "pillow",  # datasets encodes Image examples through PIL in the generator path
#   "pyarrow>=18",
# ]
# ///
"""Build the canonical training parquet from a bucket teacher pass -> ONE final-schema push.

Takes the parquet parts that falcon-perception-bucket.py wrote, joins the image bytes back in
from the source bucket (skipped when the parts already carry an `image` column, i.e. the teacher
ran with --embed-images), excludes (and ASSERTS the exclusion of) a gold slice, splits
train/validation, and writes everything in a single final-schema write -- either to a dataset
repo (one push_to_hub, never a second) or as train.parquet / validation.parquet in a bucket for
`materialize-coco.py` / direct `load_dataset` use.

Image bytes are fetched in chunks through a dataset generator, so RAM stays bounded to one
chunk (--chunk, default 256 images) however large the corpus is; the Arrow cache on disk holds
the rest.

    uv run embed-bucket-images.py \\
      --parts "hf://buckets/<ns>/<teacher-out>/part-*.parquet" \\
      --src <ns>/<source-bucket> \\
      --gold <ns>/<gold-dataset> \\
      --out <ns>/<training-dataset> --private

    # bucket output instead of a dataset repo:
    ... --out hf://buckets/<ns>/<training-bucket>/dataset

    # local everything (smoke tests, a laptop-sized corpus): --src is a directory holding the
    # keys as relative paths, --out an absolute or ./relative directory
    ... --parts "./parts/part-*.parquet" --src ./pages --gold ./gold.parquet --out ./dataset

Why this exists (both failure modes measured): the bucket path's output has no image column
by default, so every agent hand-writes this join; and staging an intermediate push then
re-pushing a different schema to the same repo id leaves stale repo features ->
load_dataset CastError. This script builds the final schema in memory and writes exactly once.
"""

import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
from datasets import ClassLabel, Dataset, DatasetDict, Image, Sequence, load_dataset


def is_local_path(s):
    # explicit prefixes only: a repo id like ns/name must never be mistaken for a directory
    # that happens to exist in the cwd
    return s.startswith(("/", "./", "../", "~"))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--parts",
        required=True,
        help='parquet glob, e.g. "hf://buckets/ns/out/part-*.parquet" (or a local glob)',
    )
    p.add_argument(
        "--src",
        default=None,
        help="source image bucket, e.g. ns/pages (keys = __source_key), or a local directory; "
        "not needed when the parts already carry an image column",
    )
    p.add_argument(
        "--out",
        required=True,
        help="dataset repo id, hf://buckets/... prefix, or a local directory for parquet files",
    )
    p.add_argument(
        "--gold",
        default=None,
        help="gold dataset repo id (or parquet glob / local path) to exclude, by image_id",
    )
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit", type=int, default=None, help="debug: cap rows AFTER gold exclusion"
    )
    p.add_argument(
        "--keep-errors",
        action="store_true",
        help="keep rows whose teacher pass errored (dropped by default: they have no usable image)",
    )
    p.add_argument(
        "--allow-gold-disjoint",
        action="store_true",
        help="permit a gold set that shares no image_id with this corpus (a genuinely different corpus)",
    )
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--chunk", type=int, default=256, help="images fetched per generator chunk"
    )
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
    if args.src and args.src.startswith("hf://buckets/"):
        args.src = args.src[len("hf://buckets/") :]
    if (
        not (args.out.startswith("hf://buckets/") or is_local_path(args.out))
        and "/" not in args.out
    ):
        raise SystemExit(
            f"--out {args.out!r}: a dataset repo id needs a namespace (ns/name)"
        )
    if "error" in ds.column_names and not args.keep_errors:
        n_err = sum(1 for e in ds["error"] if e)
        if n_err:
            ds = ds.filter(lambda r: not r["error"])
            print(f"dropped {n_err} teacher error rows (--keep-errors to keep them)")
    if not isinstance(ds.features["objects"]["category"].feature, ClassLabel):
        feats = ds.features.copy()
        feats["objects"]["category"] = Sequence(ClassLabel(names=[ds[0]["query"]]))
        ds = ds.cast(feats)

    # ---- gold exclusion, asserted on the stable image_id (never on path-shaped keys) ----
    if args.gold:
        if "://" in args.gold or is_local_path(args.gold):
            gold = load_dataset("parquet", data_files=args.gold, split="train")
        else:
            gold = load_dataset(args.gold, split="train")
        gold_ids = set(gold["image_id"])
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

    # ---- join image bytes back in from the source (unless the parts already carry them) ----
    if "image" in ds.column_names:
        print(
            "parts already carry an image column (teacher ran with --embed-images) — no fetch"
        )
        ds = ds.cast_column("image", Image())
    else:
        if not args.src:
            raise SystemExit(
                "parts have no image column — pass --src <bucket or local dir>"
            )
        src_dir = Path(args.src).expanduser() if is_local_path(args.src) else None

        def fetch(key):
            if src_dir is not None:
                return (src_dir / key).read_bytes()
            with fsspec.open(f"hf://buckets/{args.src}/{key}", "rb") as f:
                return f.read()

        plain = ds.with_format(None)
        features = plain.features.copy()
        features["image"] = Image()

        def rows_with_images():
            # one chunk of bytes in RAM at a time; datasets streams the yielded rows to
            # its Arrow cache on disk, so corpus size never sets the RAM ceiling
            for start in range(0, len(plain), args.chunk):
                chunk = plain[start : start + args.chunk]  # dict of column -> list
                keys = chunk["__source_key"]
                with ThreadPoolExecutor(args.workers) as ex:
                    blobs = list(ex.map(fetch, keys))
                bad = [k for k, b in zip(keys, blobs) if not b]
                assert not bad, f"{len(bad)} images fetched empty, e.g. {bad[:3]}"
                for i, blob in enumerate(blobs):
                    row = {col: chunk[col][i] for col in chunk}
                    row["image"] = {"bytes": blob, "path": None}
                    yield row

        ds = Dataset.from_generator(rows_with_images, features=features)

    # ---- split, then ONE write ----
    if args.val_frac <= 0 or len(ds) < 2:
        out = DatasetDict({"train": ds})
        print(
            f"rows: {total} read -> {len(ds)} kept -> train only (no validation split; "
            "materialize-coco.py then needs --splits train)"
        )
    else:
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
        print(f"wrote {' + '.join(f'{s}.parquet' for s in out)} -> {args.out}")
    elif is_local_path(args.out):
        out_dir = Path(args.out).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, d in out.items():
            d.to_parquet(out_dir / f"{split}.parquet")
        print(f"wrote {' + '.join(f'{s}.parquet' for s in out)} -> {out_dir}")
    else:
        out.push_to_hub(args.out, private=args.private)
        print(f"pushed -> https://huggingface.co/datasets/{args.out}")


if __name__ == "__main__":
    main()
