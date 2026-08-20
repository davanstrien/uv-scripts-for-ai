---
name: detection-bootstrap
description: Bootstrap an object-detection dataset and a small trained detector from images that have NO labels — zero-shot label with Falcon-Perception, validate, convert, then fine-tune a compact Apache-licensed model, all on Hugging Face Jobs. Runs fully autonomously or with human review checkpoints. Use when you have an image collection and want a detector but no annotations exist.
---

# Bootstrap a detector from unlabeled images

The loop: **zero-shot teacher labels → validate → convert → train a small student → evaluate → publish.**
Every step is a self-contained UV script from
[`uv-scripts/object-detection`](https://huggingface.co/datasets/uv-scripts/object-detection) on the
Hugging Face Hub, or a `hf jobs` command. `--help` works on every script.

## Check if a human in the loop

You can use the approach outlined in this skill with or without a human in the loop.

- **With a human in the loop** (better models): show them the step-1 previews — "is the teacher boxing
  the right things?" is the highest-value question, and its fix is the cheapest (a better query, about $2 to
  re-run the teacher). Then train on a small slice first (500–1k images, about $1) and show 20 rendered
  predictions before spending on the full corpus. If corrections are worth collecting at volume, run a
  review pass with `review-detections.py` (keyboard accept/reject in the browser — quick mode for
  whole-image verdicts in random order with quotable rates, boxes mode for per-box rejects; pushes a
  `review` column), fold corrections in and retrain (about $1). Diff the corrected set against the first pass (`diff-hf-datasets.py`) to measure how
  good the zero-shot pass actually was.
- **Autonomously** (headless): don't pause for review — use the numeric proxies, and say **unreviewed**
  in the final report and model card.

## 1. Sense-check the class name before spending GPU money (free)

Falcon-Perception queries are **class names, not instructions** ("photograph" works; "the photographs,
excluding captions" returns nothing), and **one class per run** (combined queries collapse — run per
class and merge on `image_id`). Model details: `hf models card tiiuae/Falcon-Perception`.

Check cheaply on 3 images before any full pass. The teacher (Falcon-Perception) is a **0.6B model,
1.3 GB download** — it runs on a CUDA GPU (fast), Apple Silicon (MLX backend auto-selected, about 6 s/image),
or plain CPU (slow, but fine for 3 images). Run the check wherever is practical for you:

```
# locally, if your machine can:
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py \
  --dataset <USER>/<IMAGES> --limit 3 --query photograph --preview

# or the same check as a small job (previews don't persist on Jobs — push a tiny dataset instead).
# l4x1 is the cheapest flavor that fits the engine (see step 2's flavor rule):
hf jobs uv run --flavor l4x1 --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py \
  --dataset <USER>/<IMAGES> --limit 3 --query photograph --out <USER>/<NAME>-check --private
```

Judge the result before scaling up:
- **If you can view images**, look at the rendered previews (or the pushed check dataset) — are the
  right things boxed?
- **If you can't**, compare instance counts across candidate queries (`stats-hf-dataset.py` below works
  on a pushed check dataset): near-zero instances/image means the class name is wrong for this material —
  try a synonym (`photograph` / `illustration` / `figure` / `cartoon`). Suspiciously many (more than about 10/image)
  *can* mean the query is matching layout blocks — but dense plates genuinely carry 10–20 figures
  (measured on encyclopaedia plates), so counts are a fallback signal only; previews are the judge.
- Measured on real material, previews judged:

  | material | worked | partial | dud |
  |---|---|---|---|
  | historic newspaper pages (b/w scans) | `photograph`, `illustration` | | |
  | book / encyclopaedia plates | `illustration` (incl. dense multi-figure plates) | `caption` (good on true plates, grabs whole text columns on text-heavy pages) | `figure` (0 hits on the same pages) |
- **No vision at all?** A vision-capable subagent can judge the previews if you can spawn one;
  otherwise tell the user the check ran unviewed.

(Falcon-Perception has a custom architecture, so it can't be served as an OpenAI-compatible endpoint —
iterate via the batch script. If you swap in a teacher that vLLM can serve, a temporary hot server on
Jobs is the faster way to iterate on queries: see
[Serve Models on Jobs](https://huggingface.co/docs/hub/jobs-serving).)

## 2. Teacher pass on Jobs

```
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h \
  https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception.py \
  --dataset <USER>/<IMAGES> --query photograph --out <USER>/<NAME>-photograph --private
```

- Flavor rule (all three failures measured): the engine needs a **24 GB-VRAM GPU** (16 GB T4s
  CUDA-OOM during prefill) and **more than 15 GB host RAM** (the engine sizes itself from the GPU
  and ignores host RAM, so `t4-small` and `a10g-small` are OOMKilled before the first image).
  `hf jobs hardware --json` lists every flavor's `ram`, accelerator and price — `l4x1` is the
  cheapest fit (fine for the step-1 check); `a10g-large` is faster for a corpus pass.
- One job per class (step 1's rule). Every run labels its boxes `category` 0 in a single-name
  `ClassLabel`, so a naive concat collapses the classes — renumber each run to its index in a
  combined `ClassLabel` when merging. Rows align on `image_id` (every run contains every image):

  ```python
  from datasets import ClassLabel, Sequence, load_dataset

  names = ["illustration", "map"]
  parts = [load_dataset(f"<USER>/<NAME>-{n}", split="train") for n in names]
  extra = [dict(zip(ds["image_id"], ds["objects"])) for ds in parts[1:]]

  def merge(row):
      o = {k: list(v) for k, v in row["objects"].items()}
      for i, run in enumerate(extra, start=1):
          r = run[row["image_id"]]
          o["bbox"] += r["bbox"]; o["area"] += r["area"]
          o["rectangularity"] += r["rectangularity"]
          o["category"] += [i] * len(r["bbox"])
      return {"objects": o, "n_instances": len(o["bbox"])}

  feats = parts[0].features.copy()
  feats["objects"]["category"] = Sequence(ClassLabel(names=names))
  merged = parts[0].map(merge, features=feats)
  ```

  (`masks_rle` concatenates the same way if you need the masks.)
- Output schema: `objects.bbox` in **YOLO format** (normalized center x, y, w, h), `objects.category`
  (a `ClassLabel` named after the query), `objects.area`, `objects.rectangularity`, plus `image`,
  `image_id`, `width`, `height`.
- There are **no confidence scores** (the model has none). `rectangularity` (mask area ÷ box area) is the
  triage proxy: values near 0 are usually junk, 0.785 is a circle, 1.0 a full rectangle.
- Submit with `--detach` (returns the job id immediately), then block on completion with
  `hf jobs wait <id> [<id> ...] --timeout 2h` — it exits 0 only if every job succeeded, so it
  chains cleanly into the next step. `hf jobs logs <id>` / `hf jobs inspect <id>` for progress and errors.
- A job can sit in SCHEDULING while the flavor queue drains — that is a queue, not a failure.
  **Don't resubmit**: a second copy racing to the same `--out` just doubles the bill. If you do
  switch (`hf jobs hardware` for alternatives), cancel the queued copy first (`hf jobs cancel <id>`).
- For images in a [storage bucket](https://huggingface.co/docs/hub/storage-buckets) instead of a
  dataset, use `falcon-perception-bucket.py` — it writes resumable parquet parts back to a bucket
  (kill and re-run the same command; done keys are skipped):

  ```
  hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h --detach \
    https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/falcon-perception-bucket.py \
    --src <namespace>/<bucket> --prefix <path/under/bucket> \
    --out <namespace>/<out-bucket> --query illustration
  ```

  Publish once at the end so the parts feed the rest of this loop (parquet stores `category` as
  bare ints; the cast attaches the class name):

  ```python
  from datasets import ClassLabel, Sequence, load_dataset
  ds = load_dataset("parquet", data_files="hf://buckets/<namespace>/<out-bucket>/part-*.parquet",
                    split="train")
  feats = ds.features.copy()
  feats["objects"]["category"] = Sequence(ClassLabel(names=[ds[0]["query"]]))
  ds.cast(feats).push_to_hub("<namespace>/<dataset>")
  ```

## 3. Validate the labels (free, local)

```
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/validate-hf-dataset.py \
  <USER>/<NAME>-photograph --bbox-format yolo
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/stats-hf-dataset.py \
  <USER>/<NAME>-photograph --bbox-format yolo
```

Expect **VALID** with 0 out-of-bounds and 0 zero-area boxes. `W001` warnings on empty images are normal —
they are true negatives (pages with no instance), and they are useful training signal; keep them.
Drop obvious junk before training: degenerate slivers (extreme aspect ratio + tiny area) and near-duplicate
boxes (IoU > 0.9 within one image).

## 4. Convert YOLO → COCO for training (free, local)

Trainers expect COCO `xywh` pixels; the teacher emits YOLO normalized. One command:

```
uv run https://huggingface.co/datasets/uv-scripts/object-detection/raw/main/convert-hf-dataset.py \
  <USER>/<NAME>-photograph <USER>/<NAME>-coco --from yolo --to coco_xywh
```

## 5. Train a small detector

The measured default (a clean-context validation run of this skill): fine-tune
[`ustc-community/dfine-small-coco`](https://huggingface.co/ustc-community/dfine-small-coco)
(D-FINE small, 10.4M params, Apache-2.0, in `transformers`) on the step-4 COCO dataset.
On an 800-image corpus, 30 epochs on a `t4-medium` Job took 48 min and about $0.35. Training
needs only a T4: step 2's 24 GB-VRAM rule is the teacher's engine, not the student's.

The **`huggingface-vision-trainer`** skill runs the training end to end (dataset validation,
augmentation, mAP eval, Hub persistence) — install it with `hf skills add huggingface-vision-trainer`
if you don't have it, and follow its object-detection path with the `<USER>/<NAME>-coco` dataset and
the settings above. Hold out the validation split — and the step-6 gold slice — BEFORE training,
and never train on either.

Other trainers work — the dataset is plain COCO. [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r18vd)
is a comparable compact Apache-2.0 pick; [RF-DETR](https://github.com/roboflow/rf-detr) (Apache-2.0,
DINOv2 backbone) is a good starter, and its Seg variant can learn from the teacher's `masks_rle`
masks. Check the license fits the use — `hf models card <id>` shows it; flag restrictive licenses
(e.g. ultralytics/YOLO is AGPL) to the user rather than deciding for them. Explore further:
[transformers object-detection models](https://huggingface.co/models?pipeline_tag=object-detection&library=transformers&sort=trending) ·
[ultralytics-library models](https://huggingface.co/models?library=ultralytics).

Decode `masks_rle` like this — each RLE lives in its own frame, which never matches the
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

## 6. Evaluate honestly

- Report mAP on the held-out slice. Be clear about what it measures: **agreement with the teacher**,
  not accuracy against human truth — no human labels exist in this loop unless you make some (next
  bullet).
- **Gold slice** (with a human in the loop): hold out about 100 random images BEFORE training, and have
  the human verify every box on them with `review-detections.py --mode boxes --order random`, then
  correct any misses (the tool flags them with M; drawing the missing boxes is manual for now).
  Then report TWO numbers: mAP vs teacher labels AND mAP vs the human gold. They
  differ, and the gap is the finding — in the validation run of this skill: 0.84 vs teacher labels
  but 0.44 vs human gold, both mAP@50 on held-out pages. That gap is the teacher's systematic
  divergence from human annotators, which teacher-agreement alone cannot see.
- The student can at best match its teacher (measured on a comparable loop: student 97.4% vs teacher
  95.0% human-acceptable on the same sample). The point of distilling is **throughput and cost**
  (10–100× cheaper per image than the teacher), not accuracy gains.
- Evaluate with the model card's decode contract, and write that contract INTO the card (input
  padding, score handling — with one class use the raw logit/sigmoid, never softmax). This is
  load-bearing: a standard decode against a padded-square model measured 0.03 mAP where the
  documented decode measured 10× higher. (Evaluating locally on Apple Silicon: pass the trainer's
  eval a CPU device — the COCO eval path uses float64, which MPS lacks.)
- Spot-check 20 or so predictions visually before calling it done — or, if running without a human and you
  cannot view images, state prominently in the report that the model is **unreviewed**.
- It can make sense to run this process in a loop: predict → review (a human, or a vision-capable
  agent, via `review-detections.py`) → retrain on the corrections → review again, until the acceptance
  rate stops improving.

## 7. Publish with honest provenance

Push the model and dataset — ask the user whether public or private; if you can't ask, default to
private and say so. The cards must state: labels are **zero-shot weak labels** from Falcon-Perception
(name the script + date), which filters ran, and that **recall is unmeasured** unless you measured it
against an independent source. Say what the model is for and what it was trained on. A model trained
this way is a first pass. The review loop above is how it gets better.
