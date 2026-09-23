---
viewer: false
tags: [uv-script, classification, fine-tuning, few-shot, zero-shot, setfit, gliner2, vllm, structured-outputs, hf-jobs]
---

# Classification Scripts

Text classification on [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) — both directions:

| Script | What it does |
|--------|--------------|
| [`classify-gliner2.py`](#zero-shot-first-then-fine-tune-gliner2) | **Label a dataset** with GLiNER2: zero-shot from label names, or with a `train-gliner2.py` model |
| [`train-gliner2.py`](#zero-shot-first-then-fine-tune-gliner2) | **Fine-tune** [GLiNER2](https://github.com/fastino-ai/GLiNER2), a small model (74M–287M) that already classifies zero-shot, and report the zero-shot score next to the fine-tuned one |
| [`train-setfit.py`](#few-shot-with-setfit-train-setfitpy) | **Few-shot** train a classifier from 8-64 labels per class with [SetFit](https://github.com/huggingface/setfit) — runs on CPU or GPU |
| [`train-classifier.py`](#fine-tune-a-classifier-train-classifierpy) | **Fine-tune** an encoder into a classifier (default: [LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M)) and push it to the Hub |
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

## Zero-shot first, then fine-tune (GLiNER2)

Label a dataset with your own list of labels, see how far zero-shot gets you, then fine-tune a
small model on your labels, in minutes and for a few cents on one GPU. The result is a model
that returns a label and a probability for every row, and is small enough to run on a CPU.

[GLiNER2](https://github.com/fastino-ai/GLiNER2) is a small encoder that reads the label names
as part of its input, so it classifies with no training at all, and fine-tuning teaches it what
your labels mean in your data. (For entity extraction with the original GLiNER library, see
[`uv-scripts/gliner`](https://huggingface.co/datasets/uv-scripts/gliner).) Two scripts:

- **`train-gliner2.py`** scores the base model zero-shot, fine-tunes it on your labels, and
  scores it again on the same held-out rows. The model card reports both next to the
  majority-class floor, so you can see what the labels bought you.
- **`classify-gliner2.py`** labels a whole dataset. Pass `--labels` for zero-shot, or `--model`
  with a `train-gliner2.py` output; the tasks and labels are read from the model repo.
  Labels are passed as separate words; quote a label with spaces:
  `--labels World Sports Business "Science and technology"`. A fine-tuned model is passed
  by repo id: `--model username/gliner2-blbooks-genre`.

### Quick start

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

The second command labels the same rows the model was trained on, so it shows the workflow, not
the model's accuracy; the held-out scores are on the model card. For real use, point it at data
the model has not seen. Outputs are **private by default** (`--public` to opt out).

These commands use the default base model, `fastino/gliner2.5-multi-v1` (multilingual). For
English text, `--base-model fastino/gliner2.5-base-v1` is smaller and faster;
`gliner2.5-small-v1` is the fastest and loses about 4 points on the 52-tag example. See [Choosing a model size](#choosing-a-model-size).

### What it buys you

| Dataset | Task | Labels | Train rows × epochs | Train time | Train cost | Metric | Majority floor | Zero-shot | Fine-tuned |
|---|---|---|---|---|---|---|---|---|---|
| [`biglam/blbooksgenre`](https://huggingface.co/datasets/biglam/blbooksgenre) (book titles) | single-label | 2 | 1,562 × 5 | 141s | $0.02 | accuracy | 0.747 | 0.767 (0.753–0.782) | **0.907** (0.897–0.925) |
| [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | single-label | 4 | 2,000 × 2 | 125s | $0.01 | accuracy | 0.268 | 0.718 | **0.852** |
| [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions) | multi-label | 28 | 2,000 × 2 | 216s | $0.02 | micro F1 | — | 0.265 | **0.464** |
| [`SetFit/TREC-QC`](https://huggingface.co/datasets/SetFit/TREC-QC), two tasks in one model | single-label ×2 | 6 + 50 | 5,452 × 3 | 1,761s | $0.20 | accuracy | 0.276 / 0.246 | 0.542 / 0.468 | **0.954 / 0.876** |
| same, on `a10g-small`, default batch size, bf16 | single-label ×2 | 6 + 50 | 5,452 × 3 | 509s | $0.14 | accuracy | 0.276 / 0.246 | not run | **0.944 / 0.872** |
| [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (reviews; 15% truncated at 2,000 characters) | single-label | 2 | 1,000 × 1 | 122s | $0.01 | accuracy | 0.500 | 0.777 | **0.840** |
| Hub dataset task tags ([worked example](#worked-example-tagging-hub-datasets)), `--base-model fastino/gliner2.5-base-v1 --label-augmentation off`, on `rtx-pro-6000` | single-label choice from a fixed set | 52 | 16,000 × 5 | 17 min | ~$1.50 | top-1 in the owner's tags | 0.320 (always "text-generation") | 0.102 | **0.690** (2 seeds: 0.695 / 0.686) |

Most rows are single, deliberately small runs that test the script, not tuned results. Seed ranges,
out-of-memory history and GPU comparisons are in [GLINER2-NOTES.md](GLINER2-NOTES.md).

### Choosing a model size

| Base model                             | Params | Use it when                                | Hub-tags top-1 | CPU latency per row (free Space, 2 vCPU) | GPU (L4, fp16) |
| -------------------------------------- | ------ | ------------------------------------------ | -------------- | ---------------------------------------- | -------------- |
| `fastino/gliner2.5-small-v1`           | 74M    | speed matters most                         | 0.653          | ~0.3 s                                   | ~19 ms         |
| `fastino/gliner2.5-base-v1`            | 194M   | English text; the best accuracy per second | 0.690          | ~0.7–1 s                                 | ~19 ms         |
| `fastino/gliner2.5-multi-v1` (default) | 287M   | non-English or mixed-language text         | not measured   | —                                        | —              |

On a GPU, base and small are equally fast per row; the difference only shows on a CPU.

For speed on a CPU, use plain fp32 PyTorch; see the notes for what did not work (int8, ONNX).

### Larger or fixed label sets

For tens of labels that are always scored together (a taxonomy, a fixed tag list), and for data
you keep in a bucket instead of a Hub dataset:

```bash
hf jobs uv run --flavor a10g-small --timeout 2h --secrets HF_TOKEN \
  -v hf://buckets/username/my-bucket:/bucket \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py \
  --train-file /bucket/train.jsonl \
  --eval-file calibration=/bucket/calibration.jsonl --eval-file development=/bucket/development.jsonl \
  --labels-file /bucket/labels.json --label-column labels --label-augmentation off \
  --base-model fastino/gliner2.5-base-v1 \
  --no-push --output-dir /bucket/runs/gliner2 --export-predictions /bucket/runs/gliner2/predictions
```

- `--labels-file` fixes the label set and its order for training, zero-shot and evaluation, so a
  label that is rare or missing in the training data is still an option.
- `--label-augmentation off`: gliner2's trainer by default renames labels to "label 1", "label 2",
  … in half the rows and drops up to half of them, which helps a general zero-shot model. With a
  fixed label set it cost 2–3 points of top-1 on the 52-tag example.
- `--eval-file NAME=PATH` (repeatable) scores each split in full and in file order.
- `--export-predictions` writes every row's probability and raw logit for every label, so you can
  fit your own temperature or thresholds on one split and check them on another.
- `--no-push` keeps the model in `--output-dir` instead of creating a Hub repo.


### Worked example: tagging Hub datasets

A GLiNER2.5-base model fine-tuned with this script on 16,000 Hub datasets suggests task tags for
a dataset from its column names and first row, among the 52 tags the Hub offers. Its first
suggestion matches one of the owner's tags 69% of the time on 3,000 newer datasets from owners it
never saw. Owners' tags are a noisy target, so the true rate is higher.
[Model](https://huggingface.co/davanstrien/hub-task-tagger-gliner2.5-base) · [Demo](https://huggingface.co/spaces/davanstrien/hub-task-tagger) · [Notes on how it was trained](GLINER2-NOTES.md#a-larger-label-set-52-hub-task-tags)

How it was set up, if you want to do something similar with your own label list:

- **Input text:** the dataset's column names and types, then its first row, built from the dataset
  viewer's preview and cut to about 370 tokens. Keep the exact same builder for training and
  prediction; a small difference in the text is a different input.
- **Labels:** a fixed `--labels-file` of 52 tags, multi-label (`labels` is a list per row), with
  `--label-augmentation off`.
- **Split by time and owner:** the evaluation rows are newer datasets from owners who are not in
  the training data, so the score is not inflated by near-duplicate datasets from the same owner.
- **Two eval files** (`--eval-file calibration=… --eval-file development=…`) and
  `--export-predictions`: thresholds are chosen on one file and checked on the other.

### Good to know

- **Single-label and multi-label**, auto-detected from the label column (a list per row is multi-label). An empty list is kept as a valid "none of these" answer.
- **Several tasks in one model.** Repeat `--label-column`; each column becomes a task, answered in one pass.
- **Label names are part of the prompt.** Real names (`Fiction`, `Sports`) work; integer codes make zero-shot meaningless, and the script warns.
- **Evaluation split and metrics match `train-classifier.py` and `train-setfit.py`**, so the rungs are comparable.
- **Out of GPU memory, it restarts at a smaller batch size** instead of quietly training on nothing, and stops before pushing if even batch size 1 fails. Memory grows with batch size × number of labels × text length; many labels or long texts want an `a10g-small`.
- **Always pass `--timeout`.** The scripts carry a `[tool.hf-jobs]` header (t4-small, 1 hour), but older `hf` CLIs ignore it and stop the Job after 30 minutes.
- **It is a GLiNER2 checkpoint**, loaded with `gliner2.classification.Classifier.from_pretrained(repo)`. `gliner2` pins `transformers<5`, which keeps `huggingface_hub` below 1.0 inside the Job; your local `hf` CLI is a separate install.

Tested commands with their results, and the findings and dead ends behind these defaults, are in
[GLINER2-NOTES.md](GLINER2-NOTES.md).

## Few-shot with SetFit (`train-setfit.py`)

An alternative when you have only a handful of labelled examples per class (8-64) and want a sentence-transformer model.

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

---

# Zero-shot classification (`classify-dataset.py`)

GPU-accelerated text classification for Hugging Face datasets with guaranteed valid outputs through structured generation. Powered by SmolLM3-3B's advanced reasoning capabilities.

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
