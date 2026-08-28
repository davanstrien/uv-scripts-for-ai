#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "falcon-perception>=1.0.0",
#   # tarball not git+: some GPU images have no `git` for uv to shell out to
#   "bucketbag @ https://github.com/davanstrien/bucketbag/archive/refs/tags/v0.3.1.tar.gz",
#   "pyarrow>=18",
#   "pycocotools>=2.0.11",
# ]
# ///
"""Falcon-Perception over a whole HF bucket, resumable.

    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN \
        https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception-bucket.py \
        --src biglam/bl-images --prefix full/embellishments \
        --out davanstrien/bl-masks --query illustration

Input  : bucketbag batched_files — bounded scratch, files deleted as the loop advances
Engine : PagedInferenceEngine (CUDA, continuous batching)
Output : one parquet per batch -> out bucket; resume via completed_keys(__source_key)

Kill it at any point and re-run the same command. Done keys are skipped.

Output is parquet parts in a BUCKET, not a dataset repo — that is what makes the
run resumable (`completed_keys` reads the done-set back from `__source_key`).
To hand the result to the rest of this directory, publish it once at the end:

    from datasets import ClassLabel, Sequence, load_dataset
    ds = load_dataset("parquet", data_files="hf://buckets/<namespace>/<bucket>/part-*.parquet",
                      split="train")
    feats = ds.features.copy()  # parquet stores category as bare ints; name the class
    feats["objects"]["category"] = Sequence(ClassLabel(names=[ds[0]["query"]]))
    ds.cast(feats).push_to_hub("<namespace>/<dataset>")  # a dataset repo, distinct from the bucket

    uv run validate-hf-dataset.py <namespace>/<dataset> --bbox-format yolo

By default the parts carry `width`/`height` but no `image` column: the images stay in
the source bucket, the parts stay small and resumable, and `embed-bucket-images.py`
joins the bytes back in. Pass --embed-images to write the source bytes into each part
instead (an `image` column datasets decodes directly) -- storage is cheap and it saves
the join's re-fetch of every image; the cost is a copy of the corpus in the output bucket.

GOTCHAS (all measured, none in the model card):
  * --query is a CLASS NAME. "illustration" works; "the illustration, excluding
    captions" returns nothing.
  * torch.compile breaks on per-image dynamic shapes -> compile is OFF here.
  * engine_config_for_gpu() sizes from the GPU and ignores host RAM; the 15 GB
    flavors (t4-small, a10g-small) get OOMKilled (exit 137) before processing
    anything -- pick >15 GB `ram` from `hf jobs hardware --json`.
    cudagraph is off by default here for the same reason.
  * xy in the output is the NORMALISED CENTRE, not a corner.
"""

import argparse
import hashlib
import io
import json
import time

import pyarrow as pa
import pyarrow.parquet as pq
from bucketbag import batched_files, boost, completed_keys, iter_keys, put_files
from pycocotools import mask as mask_utils


def stable_id(key):
    """Deterministic int64 image_id from the source key (COCO consumers need an int;
    a hash, not a running index, keeps ids identical across resumed runs)."""
    return int.from_bytes(hashlib.blake2b(str(key).encode(), digest_size=8).digest(), "big") >> 1

# Same YOLO column layout as falcon-perception.py, so both outputs validate with
# `validate-hf-dataset.py --bbox-format yolo` and can be concatenated.
# `__source_key` is bucketbag's resume column — the name is load-bearing.
SCHEMA = pa.schema([
    ("__source_key", pa.string()),
    ("image_id", pa.int64()),  # int, not str: COCO-style trainers tensorise it
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("objects", pa.struct([
        ("bbox", pa.list_(pa.list_(pa.float32()))),   # yolo: cx, cy, w, h normalised
        ("category", pa.list_(pa.int64())),           # single class per run; the class NAME
                                                      # is the `query` column — cast to
                                                      # ClassLabel at publish (see docstring)
        ("area", pa.list_(pa.float32())),
        ("rectangularity", pa.list_(pa.float32())),   # triage proxy — no confidence score exists
    ])),
    ("n_instances", pa.int32()),
    ("masks_rle", pa.string()),
    ("query", pa.string()),
    ("gen_seconds", pa.float32()),
    ("error", pa.string()),
])


def pair_bboxes(raw):
    boxes, cur = [], {}
    for e in raw:
        if not isinstance(e, dict):
            continue
        cur.update(e)
        if all(k in cur for k in ("x", "y", "h", "w")):
            boxes.append(dict(cur)); cur = {}
    return boxes


def serialise(rows, fmt, schema=SCHEMA):
    if fmt == "jsonl":
        return "\n".join(json.dumps(r) for r in rows) + "\n"
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), buf, compression="zstd")
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="source bucket, e.g. biglam/bl-images")
    p.add_argument("--prefix", default=None, help="bucket prefix, e.g. full/embellishments")
    p.add_argument("--out", required=True, help="output bucket")
    p.add_argument("--query", default="illustration", help="a CLASS NAME, not an instruction")
    p.add_argument("--task", default="segmentation", choices=["segmentation", "detection"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-dim", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--batch-n", type=int, default=32, help="files per bucketbag batch")
    p.add_argument("--max-bytes", type=int, default=2 * 2**30)
    p.add_argument("--cudagraph", action="store_true", help="opt IN; off by default (host OOM)")
    p.add_argument("--format", default="parquet", choices=["parquet", "jsonl"])
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--embed-images", action="store_true",
                   help="also write the source image bytes into each part (see docstring)")
    args = p.parse_args()
    if args.embed_images and args.format == "jsonl":
        raise SystemExit("--embed-images writes raw image bytes, which jsonl cannot carry; use --format parquet.")
    schema = SCHEMA
    if args.embed_images:
        schema = SCHEMA.append(pa.field("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])))

    if args.format == "jsonl" and not args.no_resume:
        # completed_keys only reads the done-set back from .parquet parts, so jsonl
        # output silently reprocesses EVERYTHING on every re-run.
        raise SystemExit("--format jsonl is not resumable; pass --no-resume to run it anyway.")

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:  # falcon-perception pins torch off-darwin, so it may be absent
        has_cuda = False
    if not has_cuda:
        raise SystemExit(
            "This script needs a CUDA GPU (PagedInferenceEngine). "
            "For MLX/CPU-capable runs use falcon-perception.py instead."
        )

    boost()  # raise xet small-file concurrency — the whole point on many small objects

    from huggingface_hub import HfApi

    # first run: the out bucket may not exist yet — completed_keys 404s on a
    # missing bucket, killing the job before anything happens
    HfApi().create_bucket(args.out, private=True, exist_ok=True)

    done = set() if args.no_resume else completed_keys(args.out)
    print(f"{len(done)} keys already done", flush=True)

    # objects=True yields BucketFile (with .size), so max_bytes is honoured.
    # Needs bucketbag >= 0.3.0: before that, string keys made batched_files drop
    # max_bytes silently and run unbounded against RAM-tmpfs scratch.
    keys = [
        f for f in iter_keys(args.src, prefix=args.prefix, objects=True)
        if f.path.lower().endswith((".jpg", ".jpeg", ".png")) and f.path not in done
    ]
    if args.limit:
        keys = keys[: args.limit]
    print(f"{len(keys)} keys to process", flush=True)
    if not keys:
        raise SystemExit(
            f"0 keys matched under {args.src}/{args.prefix or ''} — this script reads only "
            ".jpg/.jpeg/.png (convert JPEG 2000 / TIFF first), and already-done keys are skipped "
            "(pass --no-resume to redo)."
        )

    from falcon_perception import PERCEPTION_MODEL_ID, build_prompt_for_task, load_and_prepare_model, setup_torch_config
    from falcon_perception.data import ImageProcessor
    from falcon_perception.paged_inference import (
        PagedInferenceEngine, SamplingParams, Sequence, engine_config_for_gpu,
    )

    setup_torch_config()
    t = time.perf_counter()
    model, tokenizer, _ = load_and_prepare_model(
        hf_model_id=PERCEPTION_MODEL_ID, dtype="bfloat16", compile=False,  # compile breaks on dynamic shapes
    )
    print(f"model loaded in {time.perf_counter() - t:.1f}s", flush=True)

    cfg = engine_config_for_gpu(max_image_size=args.max_dim, dtype=model.dtype)
    print(f"paged config: {cfg}", flush=True)
    engine = PagedInferenceEngine(
        model, tokenizer, ImageProcessor(patch_size=16, merge_size=1),
        max_seq_length=8192, capture_cudagraph=args.cudagraph, **cfg,
    )
    sp = SamplingParams(
        args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.end_of_query_token_id],
        coord_dedup_threshold=0.01,
    )
    prompt = build_prompt_for_task(args.query, args.task)

    n, gen_total, t_all, batch_i = 0, 0.0, time.perf_counter(), 0
    for batch in batched_files(args.src, keys=keys, n=args.batch_n, max_bytes=args.max_bytes):
        # NOTE: never hold a LoadedItem past its batch — convert eagerly.
        pairs = []
        for it in batch:
            try:
                img = it.image.convert("RGB")  # convert() forces the load off disk
                orig_size = img.size  # SOURCE dims -- the images downstream tools decode
                if max(img.size) > args.max_dim * 2:
                    img.thumbnail((args.max_dim * 2, args.max_dim * 2))
                raw = it.bytes if args.embed_images else None  # read before the batch is deleted
                pairs.append((str(it.key), img, orig_size, raw))
            except Exception as e:
                pairs.append((str(it.key), e, None, None))

        good = [(k, im, sz, raw) for k, im, sz, raw in pairs if not isinstance(im, Exception)]
        seqs = [
            Sequence(text=prompt, image=im, min_image_size=256,
                     max_image_size=args.max_dim, request_idx=i, task=args.task)
            for i, (_, im, _, _) in enumerate(good)
        ]
        t0 = time.perf_counter()
        if seqs:
            engine.generate(seqs, sampling_params=sp)
        dt = time.perf_counter() - t0
        gen_total += dt

        rows = []
        for (key, im, orig_size, raw), seq in zip(good, seqs):
            aux = seq.output_aux
            boxes = pair_bboxes(aux.bboxes_raw)
            masks = list(aux.masks_rle)
            for m in masks:
                if isinstance(m.get("counts"), bytes):
                    m["counts"] = m["counts"].decode()
            # width/height are the SOURCE image's dims: boxes are normalised (frame-free),
            # and downstream pixel conversions run against the untouched bucket images.
            W, H = orig_size
            bbox, area, rect = [], [], []
            for i, b in enumerate(boxes):
                bbox.append([b["x"], b["y"], b["w"], b["h"]])  # yolo: cx, cy, w, h normalised
                a = b["w"] * b["h"]
                area.append(a)
                r = 0.0
                if i < len(masks):  # rectangularity — the only triage signal; no score exists
                    try:
                        m = masks[i]
                        if isinstance(m.get("counts"), str):
                            m = {**m, "counts": m["counts"].encode()}
                        # box area measured in the MASK's own frame (rle size) — mixing
                        # frames skews r
                        mh, mw = (m.get("size") or [H, W])[:2]
                        r = min(float(mask_utils.area(m)) / max(a * mw * mh, 1.0), 1.0)
                    except Exception:
                        r = 0.0
                rect.append(r)
            row = {
                "__source_key": key, "image_id": stable_id(key), "width": W, "height": H,
                "objects": {"bbox": bbox, "category": [0] * len(bbox),
                            "area": area, "rectangularity": rect},
                "n_instances": len(bbox), "masks_rle": json.dumps(masks),
                "query": args.query, "gen_seconds": dt / max(len(seqs), 1), "error": None,
            }
            if args.embed_images:
                row["image"] = {"bytes": raw, "path": None}
            rows.append(row)
        for key, err, _, _ in [t for t in pairs if isinstance(t[1], Exception)]:
            # a durable error row, never a gap — and it counts as done so it is
            # not retried forever on every re-run. objects is EMPTY, not null:
            # a null struct crashes validate-hf-dataset.py after publish.
            rows.append({k: None for k in SCHEMA.names} | {
                "__source_key": key, "image_id": stable_id(key), "query": args.query,
                "objects": {"bbox": [], "category": [], "area": [], "rectangularity": []},
                "n_instances": 0, "masks_rle": "[]",
                "error": f"{type(err).__name__}: {err}",
            })

        # part name derives from batch CONTENT, not a run-local counter: a resumed run's
        # counter restarts at 0 and put_files overwrites, silently destroying the first
        # run's parts. A content-derived name is stable per batch and collision-free
        # across resumes (a re-run of the same batch overwrites its own part, idempotent).
        ext = "jsonl" if args.format == "jsonl" else "parquet"
        part = hashlib.blake2b(rows[0]["__source_key"].encode(), digest_size=6).hexdigest()
        put_files([(f"part-{part}.{ext}", serialise(rows, args.format, schema))], args.out)
        n += len(rows); batch_i += 1
        rate = n / (time.perf_counter() - t_all)
        print(f"batch {batch_i}: {len(rows)} rows ({dt / max(len(seqs), 1):.2f}s/img)  "
              f"total {n}  {rate:.2f} img/s", flush=True)

    wall = time.perf_counter() - t_all
    print(f"\n{n} images in {wall:.1f}s ({gen_total:.1f}s generation) | {n / wall:.2f} img/s", flush=True)
    if n:
        print(f"extrapolation: 100k images ≈ {wall / n * 100_000 / 3600:.1f} GPU-hours end-to-end", flush=True)


main()
