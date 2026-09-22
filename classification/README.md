---
viewer: false
tags: [uv-script, classification, fine-tuning, few-shot, zero-shot, setfit, gliner2, vllm, structured-outputs, hf-jobs]
---

# Classification Scripts

Text classification on [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) — both directions:

| Script | What it does |
|--------|--------------|
| [`train-classifier.py`](#fine-tune-a-classifier-train-classifierpy) | **Fine-tune** an encoder into a classifier (default: [LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M)) and push it to the Hub |
| [`train-setfit.py`](#few-shot-with-setfit-train-setfitpy) | **Few-shot** train a classifier from 8-64 labels per class with [SetFit](https://github.com/huggingface/setfit) — runs on CPU or GPU |
| [`train-gliner2.py`](#zero-shot-first-then-fine-tune-gliner2) | **Fine-tune** [GLiNER2](https://github.com/fastino-ai/GLiNER2), a small model (74M–287M) that already classifies zero-shot, and report the zero-shot score next to the fine-tuned one |
| [`classify-gliner2.py`](#zero-shot-first-then-fine-tune-gliner2) | **Label a dataset** with GLiNER2: zero-shot from label names, or with a `train-gliner2.py` model |
| [`classify-dataset.py`](#zero-shot-classification-classify-datasetpy) | **Zero-shot** classify a dataset with an instruction LLM (SmolLM3 + vLLM, structured outputs) |
| `classify-dataset-sglang.py` | Zero-shot variant on SGLang (reasoning-aware `<think>` models) |

Pick by how many labels you have:

| Labels you have | Use | Hardware |
|---|---|---|
| none | `classify-gliner2.py --labels ...` for a cheap first pass; `classify-dataset.py` when the task needs an LLM's reasoning | small GPU (CPU works at ~1.4 rows/s); GPU |
| ~8-64 per class | `train-setfit.py` | CPU supported; GPU for faster training |
| a few hundred to a few thousand | `train-gliner2.py`, which also shows you what zero-shot already gets | small GPU (`t4-small`) |
| a few thousand or more | `train-classifier.py` | GPU |

The rungs chain: bootstrap labels with `classify-dataset.py`, review them, then train a small
dedicated model on what you kept.

## Fine-tune a classifier (`train-classifier.py`)

Fine-tunes a text-classification encoder on any Hub dataset and pushes the trained model
back to the Hub — download, train, evaluate, push, and reload-verify in one job.

- **Default model**: [LiquidAI/LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M) — a bidirectional encoder that beats ModernBERT-base on GLUE/SuperGLUE and handles 8,192-token documents. Any Hub encoder works via `--model` (ModernBERT, BERT, DeBERTa, …).
- **Single-label and multi-label**, auto-detected from the label column (`ClassLabel`/string/int → cross-entropy; list of labels → BCE + per-label threshold tuning).
- **Round-trippable artifacts**: standard architectures produce standard models; encoders without a classification head (like LFM2.5) get a generic mean-pooling head pushed as custom code, so `AutoModelForSequenceClassification.from_pretrained(..., trust_remote_code=True)` always works.

```bash
# single-label (ag_news has a ClassLabel column)
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-classifier.py \
  fancyzhx/ag_news username/news-classifier

# multi-label (go_emotions has a list-of-labels column)
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-classifier.py \
  google-research-datasets/go_emotions username/emotion-classifier --label-column labels
```

Key options: `--model`, `--max-length` (512 default; up to 8192 with
`--gradient-checkpointing` and a small `--batch-size` on a10g/a100), `--epochs`, `--lr`,
`--batch-size`, `--max-samples` (smoke runs), `--eval-split` (auto-detects
validation/test, or holds out 10% of train). Run `uv run train-classifier.py --help` for all.

### Worked example: classify dataset cards by task

[`davanstrien/dataset-cards-with-task-categories`](https://huggingface.co/datasets/davanstrien/dataset-cards-with-task-categories)
contains 21k Hub dataset cards (frontmatter stripped) labelled with their `task_categories`
metadata — a real multi-label task over long documents:

```bash
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-classifier.py \
  davanstrien/dataset-cards-with-task-categories username/dataset-card-task-classifier \
  --label-column labels --max-length 1024 --batch-size 8 --grad-accum 2
```

The output model predicts likely task categories from a card's prose — e.g. for suggesting
metadata on datasets that lack it.

### Training a standard encoder instead

`--model answerdotai/ModernBERT-base` (or any encoder with a native classification head)
produces a plain, vLLM-servable model — pair it with
[`uv-scripts/vllm`](https://huggingface.co/datasets/uv-scripts/vllm)'s
`classify-dataset.py` for large-scale batch inference with the model you just trained.

## Few-shot with SetFit (`train-setfit.py`)

Trains a [SetFit](https://github.com/huggingface/setfit) classifier from a handful of labelled
examples per class. SetFit finetunes a sentence-transformer body on contrastive pairs, then fits a
logistic regression head on the resulting embeddings.

**Runs on CPU or GPU.** CPU is practical for small few-shot experiments. Use a GPU for faster
training, particularly with larger models, longer texts or more classes. The same model and
training settings work on either; the recipe uses the available accelerator automatically.

- **Default body**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (22M), chosen for CPU speed. Swap it with `--body-model`.
- **Evaluation split** follows the same precedence as `train-classifier.py`: `--eval-split` if given, else `validation`, else `test`, else a stratified carve-out of `--eval-fraction` from train.
- **Single-label only.** A multi-label column exits with a pointer to `train-classifier.py`.
- **Metrics match `train-classifier.py`** (accuracy + macro F1). Match evaluation rows and preprocessing when comparing runs.
- **`--num-samples`** sets labelled examples per class (default 8). **`--sampling-strategy`** controls contrastive pairing: `oversampling` (default), `undersampling`, `unique`.
- **Every run reports a majority baseline.** The run warns when accuracy fails to beat it, or the gain is below five percentage points. That fixed threshold is a review heuristic, not a measured noise level or significance test.
- **It estimates training time before starting.** The script times forward/backward passes on actual texts and hardware, then refuses training projected above `--max-minutes` (default 60). Setup, evaluation and upload take additional time. A measurement error can skip this guard; use Jobs `--timeout` to enforce a wall-clock limit.
- **Rows with missing or blank labels or texts are dropped**, with a count. Missing labels include `ClassLabel`'s `-1` sentinel and numeric NaN; plain integer `-1` remains a valid class. Splits with no usable labelled text, fewer than two observed training classes, and missing or non-string text columns exit before model loading.
- **`--private` verifies the output repository is private before training.** If the destination already exists publicly, choose a new repo or change its visibility first.

```bash
# 8 labels per class, on CPU
hf jobs uv run --flavor cpu-basic --timeout 20m --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py \
  fancyzhx/ag_news username/ag-news-setfit --num-samples 8

# Same model and training settings on a GPU for faster training
hf jobs uv run --flavor t4-small --timeout 20m --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py \
  fancyzhx/ag_news username/ag-news-setfit-gpu --num-samples 8
```

### Choosing another body or longer context

`--body-model` accepts a Sentence Transformer checkpoint. Set `--max-seq-length` within that
model's supported context window; increasing it cannot extend a model's native limit or restore
text already shortened during dataset preparation. Longer sequences can need a smaller
`--batch-size` or more GPU memory. The recipe measures training cost on the selected hardware.

Follow the body's task-prefix instructions when preparing inputs. For example,
[`nomic-ai/modernbert-embed-base`](https://huggingface.co/nomic-ai/modernbert-embed-base)
uses Nomic's task prefixes: classification inputs should begin with `classification: `.
Include the same prefix during training, evaluation and inference. The recipe does not add it
automatically. Retain the original texts and the preprocessing details with the model.

### Measured

8 labels per class, seed 42, evaluated on each dataset's own held-out split (capped at 500
examples, 1000 for banking77):

| Dataset | Classes | Labels used | Body | Flavor | Training | Accuracy | Macro F1 |
|---|---|---|---|---|---|---|---|
| [`SetFit/enron_spam`](https://huggingface.co/datasets/SetFit/enron_spam) | 2 | 16 | MiniLM-L6 | `cpu-basic` | 78s | 0.924 | 0.924 |
| [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | 4 | 32 | MiniLM-L6 | `cpu-basic` | 118s | 0.804 | 0.807 |
| [`legacy-datasets/banking77`](https://huggingface.co/datasets/legacy-datasets/banking77) | 77 | 616 | MiniLM-L6 | `t4-small` | 18s | 0.803 | 0.789 |
| [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) | 6 | 48 | MiniLM-L6 | `cpu-basic` | 216s | 0.370 | 0.325 |

**Single seed each — these do not rank models or predict your dataset.** Few-shot results vary
substantially with which examples happen to get sampled; SetFit's own benchmarks report mean and
standard deviation across ten seeds for exactly this reason. Run your own task before trusting
any of these numbers.

### Compare more than the majority baseline

The `emotion` run reached **0.370** accuracy against a **0.352** majority baseline. Other
single-seed body-model runs reached 0.418 (`bge-small`) and 0.410 (`paraphrase-mpnet-base-v2`).
These results call for further evaluation; they do not establish a limit on the task or method.

SetFit's [zero-shot guide](https://huggingface.co/docs/setfit/how_to/zero_shot) reports **0.591**
on emotion using BGE and training examples templated from the class names. It uses a different
evaluation setup from the table above, so this is motivation for a matched comparison rather
than a controlled comparison with this recipe. Templated training needs no labeled documents,
but still uses compute.

For your task, compare with a simple baseline such as TF-IDF plus logistic regression using
the same training and evaluation rows. A zero-shot comparison can also be useful when class
names describe the task well. Use repeated seeds and appropriate task metrics before drawing
conclusions from small accuracy differences. This recipe trains and evaluates a supervised
classifier; built-in templated zero-shot training is a separate possible extension.

### Real-world data: a worked failure

`biglam/hansard_speech` (2.7M parliamentary speeches, predicting `party` from `speech`) is the
case where none of the convenient properties hold, and it is instructive precisely because it
produces no score:

- **No held-out split**, so the eval set has to be carved from train — the numbers stop being
  comparable to anything published.
- **~9.5% of rows have a blank `party`**, which without the drop trains an `""` class.
- **28 parties after cleaning, nine of which cannot supply 8 examples** (`Respect` 4,
  `Independent SDP` 2, `Change UK` 1). The requested eight-example budget cannot be met for those classes.
- **1,878 steps at ~11s/step on CPU** — the script refuses it, projecting well past an hour.

On completed runs, the model card discloses a carved evaluation split, per-class training counts
and classes below the requested sample count. Dropped-row counts and measured truncation are
reported in the logs; retain those logs alongside the model when documenting data preparation.

### Many classes: watch the pair count

SetFit trains on pairs drawn from every combination of training examples, so the pair count grows
with the **square** of the training-set size — which is `--num-samples` x number of classes. The
script logs the estimate before training starts:

| Dataset | Strategy | Pairs | Steps |
|---|---|---|---|
| ag_news (4 classes x 8) | `oversampling` (default) | 768 | 48 |
| banking77 (77 classes x 8) | `oversampling` (default) | 374,528 | 23,408 |
| banking77 (77 classes x 8) | `undersampling` | 4,312 | 270 |

At 77 classes the default would take roughly 15 hours on `cpu-basic`; `--sampling-strategy
undersampling` finished in 18 seconds on a T4 in the recorded run. The script reports the pair
and step counts, then measures step time to check `--max-minutes`. When it refuses training,
it suggests undersampling where applicable and estimates whether that would fit the budget.

> **Note**: a SetFit model is a sentence-transformer body plus a scikit-learn head. Load it with
> `SetFitModel.from_pretrained(repo)`, not `AutoModelForSequenceClassification`.

---

# Zero-shot classification (`classify-dataset.py`)

GPU-accelerated text classification for Hugging Face datasets with guaranteed valid outputs through structured generation. Powered by SmolLM3-3B's advanced reasoning capabilities.

## Zero-shot first, then fine-tune (GLiNER2)

[GLiNER2](https://github.com/fastino-ai/GLiNER2) is a small encoder (default
[`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1), 287M, multilingual,
Apache-2.0) that reads the label names as part of its input. So it classifies with no training,
and it fine-tunes on a `t4-small` in a few minutes. Two scripts:

- **`train-gliner2.py`** scores the base model zero-shot, fine-tunes it on your labels, and scores
  it again on the same held-out rows. The model card reports both, next to the majority-class
  floor, so you can see what the labels bought you.
- **`classify-gliner2.py`** labels a whole dataset. Pass `--labels` for zero-shot, or `--model`
  with a `train-gliner2.py` output — the tasks and labels are read from the model repo.

```bash
# fine-tune: British Library book titles -> Fiction / Non-fiction
hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py \
  biglam/blbooksgenre username/gliner2-blbooks-genre \
  --dataset-config title_genre_classifiction --text-column title

# label a dataset with that model
hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-gliner2.py \
  biglam/blbooksgenre username/blbooks-genre-predictions \
  --dataset-config title_genre_classifiction --text-column title --model username/gliner2-blbooks-genre

# or skip training: zero-shot from label names
hf jobs uv run --flavor t4-small --timeout 1h --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-gliner2.py \
  fancyzhx/ag_news username/ag-news-topics --split test \
  --labels World Sports Business "Science and technology" --task-name topic
```

- **Single-label and multi-label**, auto-detected from the label column (a list per row is multi-label). An empty list is kept as a valid "none of these" answer.
- **Several tasks in one model.** Repeat `--label-column` and each column becomes a task; the model answers all of them in one pass. `classify-gliner2.py` then writes one `predicted_<task>` and one `predicted_<task>_confidence` column per task.
- **Evaluation split and metrics match `train-classifier.py` and `train-setfit.py`** (`--eval-split`, else `validation`, else `test`, else a carve-out; accuracy + macro F1), so the rungs are comparable. Multi-label tasks report micro/macro F1 and exact match.
- **Label names are part of the prompt.** Real names (`Fiction`, `Sports`) work; integer codes make zero-shot meaningless, and the script warns. Brackets are stripped from label names because GLiNER2 rejects them at inference.
- **It does not train on nothing.** The GLiNER2 trainer catches a CUDA out-of-memory error, skips the batch and carries on, so an undersized GPU looks like a healthy job that produces an untrained model. After 5 out-of-memory steps the script restarts itself at a quarter of the batch size, with 4× the gradient accumulation, so the effective batch size stays the same. It goes down to batch size 1, then stops before anything is pushed. The model card's reproduce command records the batch size that worked. Memory grows with batch size × number of labels × text length.
- **The launch config ships with the script.** Both scripts carry a [`[tool.hf-jobs]` header](https://huggingface.co/docs/hub/jobs-configuration#define-the-launch-config-in-the-script) (`t4-small`, a 1 hour timeout, the `HF_TOKEN` secret). With `hf` CLI 1.32 or newer, `hf jobs uv run <script-url> <args>` is enough, and `--dry-run` shows what it resolves to. Flags still win, and the examples here keep them so they also work on older CLIs — which ignore the header and stop the Job after 30 minutes, before the model is pushed. Pass `--timeout` explicitly (the examples use `1h`; a large run needs more) whenever you cannot be sure which CLI launches the job.
- **Outputs are private by default.** `train-gliner2.py` creates a private model repo and `classify-gliner2.py` a private dataset; pass `--public` to opt out. If the target repo already exists and is public, both scripts stop before doing any work. (`--private` is still accepted, and does nothing.)
- **Local files, several eval splits, exported predictions.** Instead of a Hub dataset, `train-gliner2.py` can read JSON Lines files, for example from a bucket mounted with `-v`:

  ```bash
  hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
    -v hf://buckets/username/my-bucket:/bucket \
    https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py \
    --train-file /bucket/train.jsonl \
    --eval-file calibration=/bucket/calibration.jsonl --eval-file development=/bucket/development.jsonl \
    --labels-file /bucket/labels.json --label-column labels --label-augmentation off \
    --no-push --output-dir /bucket/runs/gliner2 --export-predictions /bucket/runs/gliner2/predictions
  ```

  - `--eval-file NAME=PATH` (repeatable): each file is an eval split, scored zero-shot and fine-tuned, in full and in file order.
  - `--labels-file`: a JSON list or one label per line. It fixes the label set and its order for training, zero-shot and evaluation; a label in the data that is not in the file stops the run.
  - `--export-predictions DIR`: writes `DIR/{base,finetuned}-<split>/predictions.jsonl`, one line per row: `{"row": i, "probabilities": {task: {label: p}}, "logits": {task: {label: logit}}}` with every label. Probabilities are gliner2's (softmax for a single-label task, a sigmoid per label for multi-label); logits are the raw per-label scores. Exporting also switches off the `--max-eval-samples` cap for a Hub eval split.
  - `--no-push`: no Hub repo is created or written; the model stays in `--output-dir/final`.
  - `--label-augmentation off`: gliner2's trainer by default renames the labels to "label 1", "label 2", ... in half of the training rows and drops up to half of the labels (`upstream`). With a fixed label set that is always scored in full, `off` trains on the real, complete label set every time; label-order shuffling stays on.
  - A `run_manifest.json` (all arguments except the token, the label-augmentation config, precision, device and package versions, then the results) is written to `--output-dir`, the export directory and the model folder.
- **Pick the GPU by label count and text length.** A `t4-small` fit short texts with 2, 4 and 28 labels at the default batch size. The 56-label TREC task and 2,000-character IMDB reviews both fell back to batch size 4, and IMDB did so on the A10G too. `a10g-small` (24 GB) trains about 2.3× faster and fit the 56-label TREC run at the default batch size, in 509s against 1,761s on the T4 at batch size 4. `--precision auto` uses bf16 on Ampere or newer GPUs (A10G, L4) and fp32 on a T4; on the A10G bf16 and fp32 ran at the same speed (62s and 59s), so bf16 there buys memory, not time.
- **Texts are truncated** to `--max-text-chars` (default 2000), with a count.
- **It is a GLiNER2 checkpoint**, loaded with `gliner2.classification.Classifier.from_pretrained(repo)`, not `AutoModelForSequenceClassification`. `gliner2` pins `transformers<5`; the script's own environment keeps that from mattering.

### Measured

On `t4-small` unless stated, single seed, default learning rates. "Zero-shot" and "fine-tuned" are scored on the same held-out rows.

| Dataset | Task | Labels | Train rows × epochs | Train time | Metric | Majority floor | Zero-shot | Fine-tuned |
|---|---|---|---|---|---|---|---|---|
| [`biglam/blbooksgenre`](https://huggingface.co/datasets/biglam/blbooksgenre) (book titles) | single-label | 2 | 1,562 × 5 | 141s | accuracy | 0.747 | 0.767 (0.753–0.782) | **0.907** (0.897–0.925) |
| [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | single-label | 4 | 2,000 × 2 | 125s | accuracy | 0.268 | 0.718 | **0.852** |
| [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | multi-label | 28 | 2,000 × 2 | 216s | micro F1 | — | 0.265 | **0.464** |
| [`SetFit/TREC-QC`](https://huggingface.co/datasets/SetFit/TREC-QC), two tasks in one model | single-label ×2 | 6 + 50 | 5,452 × 3 | 1,761s | accuracy | 0.276 / 0.246 | 0.542 / 0.468 | **0.954 / 0.876** |
| same, on `a10g-small`, default batch size, bf16 | single-label ×2 | 6 + 50 | 5,452 × 3 | 509s | accuracy | 0.276 / 0.246 | not run | **0.944 / 0.872** |
| [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (reviews; 15% truncated at 2,000 characters) | single-label | 2 | 1,000 × 1 | 122s | accuracy | 0.500 | 0.777 | **0.840** |

On the T4, TREC and IMDB ran out of memory at the default batch size of 16, and the script restarted
them at batch size 4 with 4 gradient accumulation steps. Before that fallback existed, the TREC run
"completed" with 1,006 of 1,020 steps skipped and scored 0.576 / 0.484, barely above zero-shot. Many
labels are also slow: TREC trained at 9 rows/s against 55 rows/s for the 2-label task, because every
label is part of the input. For hundreds of labels, use `train-classifier.py`. Prediction was not the
limit: the 300 IMDB reviews were scored at batch size 32 on the T4 with no fallback.

The same BL books run (seed 42) scored 0.925 on a T4, 0.937 on an A10G in fp32 and 0.931 on an A10G in
bf16 — one or two eval rows apart, so hardware and precision are not a way to gain accuracy, but they
are one more thing to hold constant when you compare runs.

Two upstream options are deliberately absent: in `gliner2` 2.0.0, gradient checkpointing crashes
with the 2.5 models, and a LoRA run trained but its final checkpoint did not load for scoring (LoRA
also did not fix the 56-label out-of-memory case on a T4).

The BL books row is the mean and range of five seeds; the other rows are one seed each. Read the
range before you compare two runs: `--seed` also picks the carve-out rows, and the zero-shot model
never changes, so its 3-point spread is what 174 eval rows do to the number on their own. A
difference smaller than that between two runs is not a result. Use a dataset with a fixed
`--eval-split`, and as many eval rows as you can get, when you want to compare runs.

The ag_news, go_emotions and TREC rows are deliberately small runs (capped training rows, 2–3 epochs) that
test the script, not tuned results. The BL books row trains on the full 1,562 titles; its eval is a 10%
carve-out, so it is not comparable with published numbers for that dataset.

## 🚀 Quick Start

```bash
# Classify IMDB reviews
uv run classify-dataset.py \
  --input-dataset stanfordnlp/imdb \
  --column text \
  --labels "positive,negative" \
  --output-dataset user/imdb-classified
```

That's it! No installation, no setup - just `uv run`.

## 📋 Requirements

- **GPU Required**: Uses GPU-accelerated inference
- Python 3.10+
- UV (will handle all dependencies automatically)
- vLLM >= 0.6.6

## 🎯 Features

- **Guaranteed valid outputs** using structured generation with guided decoding
- **Zero-shot classification** without training data required
- **GPU-optimized** for maximum throughput and efficiency
- **Default model**: [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) - a fast 3B model with native thinking capabilities (`<think>` tags)
- **Robust text handling** with preprocessing and validation
- **Automatic progress tracking** and detailed statistics
- **Direct Hub integration** - read and write datasets seamlessly
- **Label descriptions** support for providing context to improve accuracy
- **Reasoning mode** for interpretable classifications with thinking traces
- **JSON output parsing** for reliable extraction from reasoning mode
- **Optimized batching** with vLLM's automatic batch processing
- **Multiple guided backends** - supports outlines, xgrammar, and more

## 💻 Usage

### Basic Classification

```bash
uv run classify-dataset.py \
  --input-dataset <dataset-id> \
  --column <text-column> \
  --labels <comma-separated-labels> \
  --output-dataset <output-id>
```

### Arguments

**Required:**

- `--input-dataset`: Hugging Face dataset ID (e.g., `stanfordnlp/imdb`, `user/my-dataset`)
- `--column`: Name of the text column to classify
- `--labels`: Comma-separated classification labels (e.g., `"spam,ham"`)
- `--output-dataset`: Where to save the classified dataset

**Optional:**

- `--model`: Model to use (default: **`HuggingFaceTB/SmolLM3-3B`** - a fast 3B parameter model)
- `--label-descriptions`: Provide descriptions for each label to improve classification accuracy
- `--enable-reasoning`: Enable reasoning mode with thinking traces (adds reasoning column)
- `--split`: Dataset split to process (default: `train`)
- `--max-samples`: Limit samples for testing
- `--shuffle`: Shuffle dataset before selecting samples (useful for random sampling)
- `--shuffle-seed`: Random seed for shuffling (default: 42)
- `--temperature`: Generation temperature (default: 0.1)
- `--guided-backend`: Backend for guided decoding (default: `outlines`)
- `--hf-token`: Hugging Face token (or use `HF_TOKEN` env var)

### Label Descriptions

Provide context for your labels to improve classification accuracy:

```bash
uv run classify-dataset.py \
  --input-dataset user/support-tickets \
  --column content \
  --labels "bug,feature,question,other" \
  --label-descriptions "bug:something is broken,feature:request for new functionality,question:asking for help,other:anything else" \
  --output-dataset user/tickets-classified
```

The model uses these descriptions to better understand what each label represents, leading to more accurate classifications.

### Reasoning Mode

Enable thinking traces for interpretable classifications:

```bash
uv run classify-dataset.py \
  --input-dataset stanfordnlp/imdb \
  --column text \
  --labels "positive,negative,neutral" \
  --enable-reasoning \
  --output-dataset user/imdb-with-reasoning
```

When `--enable-reasoning` is used:
- The model generates step-by-step reasoning using SmolLM3's thinking capabilities
- Output includes three columns: `classification`, `reasoning`, and `parsing_success`
- Final answer must be in JSON format: `{"label": "chosen_label"}`
- Useful for understanding complex classification decisions
- Trade-off: Slower but more interpretable

## 📊 Examples

### Sentiment Analysis

```bash
uv run classify-dataset.py \
  --input-dataset stanfordnlp/imdb \
  --column text \
  --labels "positive,negative" \
  --output-dataset user/imdb-sentiment
```

### Support Ticket Classification

```bash
# Run on HF Jobs with SmolLM3-3B (default)
hf jobs uv run \
  --flavor l4x1 \
  --image vllm/vllm-openai:latest \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset.py \
  --input-dataset user/support-tickets \
  --column content \
  --labels "bug,feature_request,question,other" \
  --label-descriptions "bug:code or product not working as expected,feature_request:asking for new functionality,question:seeking help or clarification,other:general comments or feedback" \
  --output-dataset user/tickets-classified
```

### News Categorization

```bash
# Using SmolLM3-3B for efficient news classification
hf jobs uv run \
  --flavor l4x1 \
  --image vllm/vllm-openai:latest \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset.py \
  --input-dataset ag_news \
  --column text \
  --labels "world,sports,business,tech" \
  --output-dataset user/ag-news-categorized
```

### Complex Classification with Reasoning

```bash
# SmolLM3's thinking mode for nuanced feedback analysis
hf jobs uv run \
  --flavor l4x1 \
  --image vllm/vllm-openai:latest \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset.py \
  --input-dataset user/customer-feedback \
  --column text \
  --labels "very_positive,positive,neutral,negative,very_negative" \
  --label-descriptions "very_positive:extremely satisfied,positive:generally satisfied,neutral:mixed feelings,negative:dissatisfied,very_negative:extremely dissatisfied" \
  --enable-reasoning \
  --output-dataset user/feedback-analyzed
```

This combines label descriptions with reasoning mode for maximum interpretability.

### ArXiv ML Research Classification

Classify academic papers into machine learning research areas:

```bash
# Fast classification with random sampling
uv run classify-dataset.py \
  --input-dataset librarian-bots/arxiv-metadata-snapshot \
  --column abstract \
  --labels "llm,computer_vision,reinforcement_learning,optimization,theory,other" \
  --label-descriptions "llm:language models and NLP,computer_vision:image and video processing,reinforcement_learning:RL and decision making,optimization:training and efficiency,theory:theoretical ML foundations,other:other ML topics" \
  --output-dataset user/arxiv-ml-classified \
  --split "train[:10000]" \
  --max-samples 100 \
  --shuffle

# With reasoning for nuanced classification
hf jobs uv run \
  --flavor l4x1 \
  --image vllm/vllm-openai:latest \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset.py \
  --input-dataset librarian-bots/arxiv-metadata-snapshot \
  --column abstract \
  --labels "multimodal,agents,reasoning,safety,efficiency" \
  --label-descriptions "multimodal:vision-language and cross-modal models,agents:autonomous agents and tool use,reasoning:reasoning and planning systems,safety:alignment and safety research,efficiency:model optimization and deployment" \
  --enable-reasoning \
  --output-dataset user/arxiv-frontier-research \
  --split "train[:1000]" \
  --max-samples 50
```

The reasoning mode is particularly valuable for academic abstracts where papers often span multiple topics and require careful analysis to determine the primary focus.

## 🚀 Running on HF Jobs

Optimized for [Hugging Face Jobs](https://huggingface.co/docs/hub/spaces-gpu-jobs) (requires Pro subscription or Team/Enterprise organization):
```bash
# Run on L4 GPU with vLLM image
hf jobs uv run \
  --flavor l4x1 \
  --image vllm/vllm-openai:latest \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset.py \
  --input-dataset stanfordnlp/imdb \
  --column text \
  --labels "positive,negative" \
  --output-dataset user/imdb-classified
```

### GPU Flavors
- `l4x1`: **Recommended starting point** - great for SmolLM3
- `a10g-large`: More memory for larger batches or 7B+ models
- `a100-large`: Maximum performance for demanding workloads

## 🔧 Advanced Usage

### Random Sampling

When working with ordered datasets, use `--shuffle` with `--max-samples` to get a representative sample:

```bash
# Get 50 random reviews instead of the first 50
uv run classify-dataset.py \
  --input-dataset stanfordnlp/imdb \
  --column text \
  --labels "positive,negative" \
  --output-dataset user/imdb-sample \
  --max-samples 50 \
  --shuffle \
  --shuffle-seed 123  # For reproducibility
```

This is especially important for:
- Chronologically ordered datasets (news, papers, social media)
- Pre-sorted datasets (by rating, category, etc.)
- Testing on diverse samples before processing the full dataset

### Using Different Models

By default, this script uses **[HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)** - a state-of-the-art 3B parameter model specifically designed for efficient inference. SmolLM3 features:
- Native thinking capabilities with `<think>` tags for step-by-step reasoning
- Excellent performance on classification tasks
- Fast inference speed (50-100 texts/second on A10)
- Low memory footprint allowing larger batch sizes

While you can use other models, SmolLM3 is recommended for its balance of quality, speed, and reasoning capabilities:

```bash
# Larger model for complex classification
uv run classify-dataset.py \
  --input-dataset user/legal-docs \
  --column text \
  --labels "contract,patent,brief,memo,other" \
  --output-dataset user/legal-classified \
  --model Qwen/Qwen2.5-7B-Instruct
```

### Large Datasets

vLLM automatically handles batching for optimal performance. For very large datasets, it will process efficiently without manual intervention:

```bash
uv run classify-dataset.py \
  --input-dataset user/huge-dataset \
  --column text \
  --labels "A,B,C" \
  --output-dataset user/huge-classified
```

## 📈 Performance

- **SmolLM3-3B (default)**: ~50-100 texts/second on A10
- **7B models**: ~20-50 texts/second on A10
- vLLM automatically optimizes batching for best throughput
- Performance scales with GPU memory and compute capability

## 🤝 How It Works

1. **vLLM**: Provides efficient GPU batch inference with automatic batching
2. **Guided Decoding**: Uses outlines backend to guarantee valid label outputs
3. **Structured Generation**: Constrains model outputs to exact label choices
4. **UV**: Handles all dependencies automatically

The script loads your dataset, preprocesses texts, classifies each one with guaranteed valid outputs, then saves the results as a new column in the output dataset.

## 🐛 Troubleshooting

### CUDA Not Available

This script requires a GPU. Run it on:

- A machine with NVIDIA GPU
- HF Jobs (recommended)
- Cloud GPU instances

### Out of Memory

- Use a smaller model
- Use a larger GPU (e.g., a100-large)

### Invalid/Skipped Texts

- Texts shorter than 3 characters are skipped
- Empty or None values are marked as invalid
- Very long texts are truncated to 4000 characters

### Classification Quality

- With guided decoding, outputs are guaranteed to be valid labels
- For better results, use clear and distinct label names
- Try the `reasoning` prompt style for complex classifications
- Use a larger model for nuanced tasks

### vLLM Version Issues

If you see `ImportError: cannot import name 'GuidedDecodingParams'`:

- Your vLLM version is too old (requires >= 0.6.6)
- The script specifies the correct version in its dependencies
- UV should automatically install the correct version

## 🔬 Advanced Workflows

For complex real-world workflows that integrate UV scripts with the Python HF Jobs API, see the [ArXiv ML Trends example](examples/arxiv-workflow/). This demonstrates:

- **Multi-stage pipelines**: Data preparation → GPU classification → Analysis
- **Python API orchestration**: Using `run_uv_job()` to manage GPU jobs programmatically
- **Production patterns**: Error handling, parallel execution, and incremental updates
- **Cost optimization**: Choosing appropriate compute resources for each task

```python
# Example: Submit a classification job via Python API
from huggingface_hub import run_uv_job

job = run_uv_job(
    script="https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-dataset.py",
    args=["--input-dataset", "my/dataset", "--labels", "A,B,C"],
    flavor="l4x1",
    image="vllm/vllm-openai:latest"
)
result = job.wait()
```

## 📝 License

This script is provided as-is for use with the UV Scripts organization.
