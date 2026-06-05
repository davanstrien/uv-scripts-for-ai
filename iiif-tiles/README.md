---
viewer: false
tags:
  - uv-script
  - iiif
  - glam
  - deep-zoom
  - static-tiles
---

# IIIF Static Tiles from HF Storage Buckets

Generate [IIIF Image API 3.0](https://iiif.io/api/image/3.0/) Level 0 static tiles from images and serve them via Hugging Face Storage Buckets — no image server required.

Drop images into a bucket, run one command, and get deep-zoom viewing in any IIIF viewer (Mirador, Universal Viewer, OpenSeadragon).

## Demo

[View in Mirador](https://projectmirador.org/embed/?iiif-content=https://huggingface.co/buckets/davanstrien/iiif-tiles-streaming/resolve/manifest.json) — 6 pages from the Wellcome Collection, served entirely from an HF Storage Bucket.

## How it works

```
Source images (bucket or local)
  → tile_iiif.py (pyvips generates tile pyramid)
    → Output bucket (static files served via HF CDN)
      → Any IIIF viewer (deep zoom, pan, browse)
```

The script generates a complete IIIF Level 0 tile set: a directory tree of pre-rendered JPEG tiles at multiple zoom levels, plus `info.json` descriptors and a IIIF Presentation v3 `manifest.json`. Since everything is static files, any CDN or file server works — no dynamic image server needed.

In bucket-to-bucket mode, images are streamed through in small batches (download → tile → upload → cleanup) so local storage stays minimal regardless of collection size.

## Quick start

```bash
# Bucket to bucket (recommended for large collections)
uv run tile_iiif.py \
    --source-bucket myorg/source-images \
    --output-bucket myorg/iiif-tiles

# From a local directory → HF Bucket
uv run tile_iiif.py \
    --source-dir ./my-scans \
    --output-bucket myorg/iiif-tiles \
    --collection-name "My Collection"

# Local only (for testing)
uv run tile_iiif.py \
    --source-dir ./my-scans \
    --output-dir ./tiles
# Then: cd tiles && python -m http.server
```

No system dependencies — `pyvips[binary]` bundles libvips in the pip wheel.

## Run on HF Jobs

Process large collections without tying up your machine:

```bash
hf jobs uv run tile_iiif.py \
    --source-bucket myorg/source-images \
    --output-bucket myorg/iiif-tiles \
    --collection-name "Historic Manuscripts"
```

## View the results

Once tiles are in a bucket, open the manifest in any IIIF viewer:

- **Mirador**: `https://projectmirador.org/embed/?iiif-content=https://huggingface.co/buckets/myorg/iiif-tiles/resolve/manifest.json`
- **Universal Viewer**: `https://uv-v4.netlify.app/#?manifest=https://huggingface.co/buckets/myorg/iiif-tiles/resolve/manifest.json`

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source-bucket` | — | HF bucket with source images (e.g., `org/source`) |
| `--source-dir` | — | Local directory with source images |
| `--output-bucket` | — | HF bucket for tiles (e.g., `org/tiles`) |
| `--output-dir` | — | Local directory for tiles |
| `--tile-size` | 512 | Tile size in pixels |
| `--base-url` | auto | Base URL for IIIF ids |
| `--collection-name` | "IIIF Collection" | Label for the manifest |
| `--workers` | 3 | Concurrent workers for bucket-to-bucket mode |

Supported image formats: JPEG, TIFF, PNG, WebP.

## Performance

Bucket-to-bucket with 6 images (2411x3372 each), generating 54 tiles per image:

| Workers | Time | Local storage |
|---------|------|---------------|
| 1 | 14.3s | ~10MB (1 image + tiles) |
| 3 | 6.7s | ~30MB (3 images + tiles) |

Per image: ~0.6s download, ~0.2s tile, ~1.4s upload. The bottleneck is upload I/O, so concurrent workers overlap network time effectively.

## What is IIIF Level 0?

[IIIF](https://iiif.io/) (International Image Interoperability Framework) is a set of APIs for serving and annotating images, widely used by libraries, archives, and museums. Level 0 means all tiles are pre-generated static files — no dynamic image server needed. Viewers request tiles that already exist on disk (or in a bucket).

This covers the primary use case: deep-zoom viewing of high-resolution scans. What you don't get (vs. a full IIIF server) is arbitrary cropping, rotation, or format conversion on the fly — but for browsing collections, Level 0 is all you need.

## Why HF Storage Buckets?

- **Free hosting** for public buckets
- **Global CDN** with signed URLs
- **CORS support** for browser-based IIIF viewers
- **No infrastructure to maintain** — just static files
- **Streaming processing** — handles large collections without landing everything locally
