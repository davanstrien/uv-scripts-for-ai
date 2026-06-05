# Buckets: mutable storage for data that changes a lot

The default recipe pattern is **a Hub dataset in, a Hub dataset out** — versioned and easy to hand off between recipes. Reach for a **[storage bucket](https://huggingface.co/docs/hub/storage-buckets)** when data churns: incremental/streaming writes, checkpoints, scratch space, or one-file-per-item output. Buckets are S3-like mutable object storage on the Hub — overwrite or delete in place, no commit overhead.

## Dataset or bucket?

| Use a **dataset** | Use a **bucket** |
|---|---|
| One read, one write (recipe handoff) | Frequent / incremental writes, mutable scratch |
| Versioned, shareable result | Checkpoints, partial results, one file per page |
| The common case | Large or resumable jobs; data rewritten often |

## Mount a bucket in a Job

`hf jobs uv run` mounts volumes with `-v hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]` — datasets and spaces are read-only, **buckets are read+write**:

```bash
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  -v hf://buckets/<user>/<bucket>:/mnt \
  https://huggingface.co/datasets/uv-scripts/ocr/raw/main/glm-ocr-bucket.py \
  /mnt/input /mnt/output
```

The `ocr` family already ships bucket-aware recipes — **`glm-ocr-bucket.py`** and **`falcon-ocr-bucket.py`** read images/PDFs from a mounted bucket and write one `.md` per page. Read the recipe's `--help` for exact arguments.

## Write to a bucket from Python (fsspec)

A recipe that writes incrementally can use `hf://buckets/…` paths directly:

```python
batch_ds.to_parquet(f"hf://buckets/{user}/{bucket}/shard-{i:05d}.parquet")
```

Read the shards back and `push_to_hub` a clean dataset as the final step if you also want a versioned result.

## Manage buckets (hf CLI)

```bash
hf buckets create <user>/<bucket>
hf buckets list <user>/<bucket> --tree
```

More: the [storage buckets docs](https://huggingface.co/docs/hub/storage-buckets) and the `hf-cli` skill (`hf buckets --help`).
