# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "sentence-transformers>=3.0.0",
#     "torch",
#     "numpy",
#     "pyarrow",
#     "pylance",
#     "huggingface-hub",
# ]
# ///
"""
Embed a Hugging Face dataset and push it back as a Lance vector index — a Hub dataset that
IS a searchable vector database. Anyone you share it with can vector-search it over `hf://`
without downloading it:

    import lance
    ds = lance.dataset("hf://datasets/your-name/my-vecdb/vecdb.lance")   # opens fast, no download
    hits = ds.to_table(nearest={"column": "vector", "q": query_vector, "k": 5})

Best for share-and-search over a corpus; for high-QPS serving, pull the dataset local first.

    hf jobs uv run --flavor l4x1 -s HF_TOKEN embed-to-lance.py \\
        stanfordnlp/imdb your-name/imdb-vecdb --column text --model BAAI/bge-base-en-v1.5 --private
"""
import argparse, logging, os, shutil, time
import numpy as np
import pyarrow as pa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed-to-lance")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dataset")
    ap.add_argument("output_repo")
    ap.add_argument("--column", default="text")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    import torch
    import lance
    from datasets import load_dataset
    from huggingface_hub import HfApi, login
    from sentence_transformers import SentenceTransformer

    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    ds = load_dataset(args.input_dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    texts = [t if isinstance(t, str) and t.strip() else " " for t in ds[args.column]]
    n = len(texts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    if getattr(model, "max_seq_length", None):
        model.max_seq_length = min(model.max_seq_length, args.max_seq_len)
    dim = model.get_sentence_embedding_dimension()

    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    log.info(f"embedded {n} rows in {time.perf_counter()-t0:.1f}s, dim={dim}")

    tbl = pa.table({
        "id": pa.array(range(n), pa.int64()),
        "text": pa.array([t[:2000] for t in texts]),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1), pa.float32()), dim),
    })
    local = "vecdb.lance"
    if os.path.exists(local):
        shutil.rmtree(local)
    lds = lance.write_dataset(tbl, local, mode="overwrite")
    try:
        parts = max(1, min(256, int(np.sqrt(n))))
        lds.create_index("vector", index_type="IVF_PQ", num_partitions=parts,
                         num_sub_vectors=max(1, dim // 16))
        log.info(f"built IVF_PQ index (partitions={parts})")
    except Exception as e:
        log.warning(f"index build skipped ({repr(e)[:120]}); flat search still works over hf://")

    api = HfApi()
    api.create_repo(args.output_repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=local, path_in_repo="vecdb.lance",
                      repo_id=args.output_repo, repo_type="dataset")
    log.info(f"✅ vector DB at hf://datasets/{args.output_repo}/vecdb.lance "
             f"(search it with lance.dataset(...).to_table(nearest=...))")


if __name__ == "__main__":
    main()
