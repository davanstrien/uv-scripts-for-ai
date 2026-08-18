---
name: detection-bootstrap
description: Bootstrap an object-detection dataset and a small trained detector from images that have NO labels — zero-shot label with Falcon-Perception, validate, convert, then fine-tune a compact Apache-licensed model, all on Hugging Face Jobs. Use when you have an image collection and want a detector but no annotations exist.
---

# Bootstrap a detector from unlabeled images

The loop: **zero-shot teacher labels → validate → convert → train a small student → evaluate → publish.**
Every step is a self-contained UV script from `uv-scripts/object-detection` on the Hugging Face Hub, or a
`hf jobs` command. `--help` works on every script. Set

```
RAW=https://huggingface.co/datasets/uv-scripts/object-detection/raw/main
```

## 1. Sense-check the class name before spending GPU money (~free)

Falcon-Perception queries are **class names, not instructions**:

- `--query photograph` works. `--query "the photographs, excluding captions"` returns **nothing**
  (measured: 0.01 instances/image). Never write instruction-style queries.
- **One class per run.** A combined query returned 6 instances where single-class runs found 24.
  N classes = N runs; merge afterwards (outputs share `image_id`).

Check cheaply on 3 images before any full pass. The teacher (Falcon-Perception) is a **0.6B model,
~1.3 GB download** — it runs on a CUDA GPU (fast), Apple Silicon (MLX backend auto-selected, ~6 s/image),
or plain CPU (slow, but fine for 3 images). Run the check wherever is practical for you:

```
# locally, if your machine can:
uv run $RAW/falcon-perception.py --dataset <USER>/<IMAGES> --limit 3 \
  --query photograph --preview

# or the same check as a small job (previews don't persist on Jobs — push a tiny dataset instead):
hf jobs uv run --flavor t4-small --secrets HF_TOKEN \
  $RAW/falcon-perception.py --dataset <USER>/<IMAGES> --limit 3 \
  --query photograph --out <USER>/<NAME>-check --private
```

Judge the result before scaling up:
- **If you can view images**, look at the rendered previews (or the pushed check dataset) — are the
  right things boxed?
- **If you can't**, compare instance counts across candidate queries (`stats-hf-dataset.py` below works
  on a pushed check dataset): ~0 instances/image means the class name is wrong for this material —
  try a synonym (`photograph` / `illustration` / `figure` / `cartoon`). Suspiciously many (> ~10/image)
  usually means the query is matching layout blocks, not pictures.

(Falcon-Perception has a custom architecture, so it can't be served as an OpenAI-compatible endpoint —
iterate via the batch script. If you swap in a teacher that vLLM can serve, a temporary hot server on
Jobs is the faster way to iterate on queries: see
[Serve Models on Jobs](https://huggingface.co/docs/hub/jobs-serving).)

## 2. Teacher pass on Jobs (~$1 per 1k images)

```
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h \
  $RAW/falcon-perception.py --dataset <USER>/<IMAGES> \
  --query photograph --out <USER>/<NAME>-photograph --private
```

- ~0.4–0.6 s/image on `a10g-large`. Do **not** use `a10g-small` — the engine sizes itself from the GPU,
  ignores host RAM, and gets OOM-killed (exit 137) before processing anything.
- One job per class (step 1's rule). Merge per-class outputs by concatenating the `objects` entries of
  rows with the same `image_id`.
- Output schema: `objects.bbox` in **YOLO format** (normalized center x, y, w, h), `objects.category`,
  `objects.area`, `objects.rectangularity`, plus `image`, `image_id`, `width`, `height`.
- There are **no confidence scores** (the model has none). `rectangularity` (mask area ÷ box area) is the
  triage proxy: values near 0 are usually junk, ~0.785 is a circle, ~1.0 a full rectangle.
- Submit with `--detach` (returns the job id immediately), then block on completion with
  `hf jobs wait <id> [<id> ...] --timeout 2h` — it exits 0 only if every job succeeded, so it
  chains cleanly into the next step. `hf jobs logs <id>` / `hf jobs inspect <id>` for progress and errors.
- A job can sit in SCHEDULING for a long time while the flavor queue drains — that is a queue, not a
  failure. **Don't resubmit**: a second copy racing to the same `--out` just doubles the bill. If you do
  switch flavor, cancel the queued copy first (`hf jobs cancel <id>`).
- For images in a storage bucket instead of a dataset, use `$RAW/falcon-perception-bucket.py` (resumable).

## 3. Validate the labels (free, local)

```
uv run $RAW/validate-hf-dataset.py <USER>/<NAME>-photograph --bbox-format yolo
uv run $RAW/stats-hf-dataset.py    <USER>/<NAME>-photograph --bbox-format yolo
```

Expect **VALID** with 0 out-of-bounds and 0 zero-area boxes. `W001` warnings on empty images are normal —
they are true negatives (pages with no instance), and they are useful training signal; keep them.
Drop obvious junk before training: degenerate slivers (extreme aspect ratio + tiny area) and near-duplicate
boxes (IoU > 0.9 within one image).

## 4. Convert YOLO → COCO for training (free, local)

Trainers expect COCO `xywh` pixels; the teacher emits YOLO normalized. One command:

```
uv run $RAW/convert-hf-dataset.py <USER>/<NAME>-photograph <USER>/<NAME>-coco \
  --from yolo --to coco_xywh
```

## 5. Train a small detector (~$1–2 on Jobs)

Fine-tune a **compact, permissively licensed** detector: **D-FINE** or **RT-DETRv2** (both Apache-2.0,
in `transformers`). Do **not** reach for ultralytics/YOLO weights — they are **AGPL**, which you cannot
ship from most projects.

The **`huggingface-vision-trainer`** skill covers this end to end (dataset validation, augmentation,
mAP eval, Hub persistence) — install it with `hf skills add huggingface-vision-trainer` if you don't
have it, and follow its object-detection path with the `<USER>/<NAME>-coco` dataset from step 4.
Before training, split off a held-out slice (e.g. 10% of images) and never train on it.

Other trainers work too — the dataset is plain COCO. [RF-DETR](https://github.com/roboflow/rf-detr)
(Apache-2.0, DINOv2 backbone) is a good starter, and its Seg variant can learn from the teacher's
`masks_rle` masks. Decode them like this — each RLE lives in its own frame, which never matches the
recorded width/height:

```python
import json, numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

for rle in json.loads(row["masks_rle"]):
    seg = mask_utils.decode({**rle, "counts": rle["counts"].encode()})  # frame = rle["size"]
    if seg.shape != (row["height"], row["width"]):
        seg = np.asarray(Image.fromarray(seg).resize((row["width"], row["height"]), Image.NEAREST))
```

Pin the package version — the API is young — and if the segmentation training path resists, ship
detection-only and say so rather than burning the day on it.

## 6. Evaluate honestly

- Report mAP on the held-out slice. Be clear about what it measures: **agreement with the teacher**,
  not accuracy against human truth — no human labels exist in this loop.
- The student can at best match its teacher (measured on a comparable loop: student 97.4% vs teacher
  95.0% human-acceptable on the same sample). The point of distilling is **throughput and cost**
  (~10–100× cheaper per image than the teacher), not accuracy gains.
- Spot-check ~20 predictions visually before calling it done.

## 7. Publish with honest provenance

Push the model and dataset (private first). The cards must state: labels are **zero-shot weak labels**
from Falcon-Perception (name the script + date), which filters ran, and that **recall is unmeasured**
unless you measured it against an independent source. Say what the model is for and what it was
trained on. A model trained this way is a starting point for human correction, not a ground-truth system.
