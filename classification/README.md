---
viewer: false
tags: [uv-script, classification, fine-tuning, few-shot, setfit, vllm, structured-outputs, hf-jobs]
---

# Classification Scripts

Text classification on [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) — both directions:

| Script | What it does |
|--------|--------------|
| [`train-classifier.py`](#fine-tune-a-classifier-train-classifierpy) | **Fine-tune** an encoder into a classifier (default: [LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M)) and push it to the Hub |
| [`train-setfit.py`](#few-shot-with-setfit-train-setfitpy) | **Few-shot** train a classifier from 8-64 labels per class with [SetFit](https://github.com/huggingface/setfit) — runs on CPU |
| [`classify-dataset.py`](#zero-shot-classification-classify-datasetpy) | **Zero-shot** classify a dataset with an instruction LLM (SmolLM3 + vLLM, structured outputs) |
| `classify-dataset-sglang.py` | Zero-shot variant on SGLang (reasoning-aware `<think>` models) |

Pick by how many labels you have:

| Labels you have | Use | Hardware |
|---|---|---|
| none | `classify-dataset.py` to bootstrap labels, or for one-off jobs | GPU |
| ~8-64 per class | `train-setfit.py` | CPU |
| a few thousand | `train-classifier.py` | GPU |

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
logistic regression head on the resulting embeddings — no GPU required.

- **Default body**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (22M), chosen for CPU speed. Swap it with `--body-model`.
- **Metrics match `train-classifier.py`** (accuracy + macro F1 on a held-out split), so the two rungs are directly comparable.
- **`--num-samples`** sets labelled examples per class (default 8). **`--sampling-strategy`** controls contrastive pairing: `oversampling` (default), `undersampling`, `unique`.

```bash
# 8 labels per class, on CPU
hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py \
  fancyzhx/ag_news username/ag-news-setfit --num-samples 8

# larger body on a GPU for the accuracy ceiling
hf jobs uv run --flavor t4-small --secrets HF_TOKEN \
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py \
  fancyzhx/ag_news username/ag-news-setfit \
  --body-model sentence-transformers/paraphrase-mpnet-base-v2
```

### Measured on `ag_news`

8 labels per class, 500 held-out eval examples, seed 42:

| Body model | Flavor | Training | Accuracy | Macro F1 | Total job |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` (22M) | `cpu-basic` | 204s | 0.826 | 0.819 | 285s |
| `paraphrase-mpnet-base-v2` (110M) | `t4-small` | 32s | 0.862 | 0.857 | 205s |
| `paraphrase-mpnet-base-v2` (110M) | `cpu-basic` | 915s | — | — | 1015s |

Single seed on one dataset. Few-shot results vary substantially with which examples get sampled,
so treat these as indicative — SetFit's own benchmarks report mean and standard deviation across
several seeds.

> **Note**: a SetFit model is a sentence-transformer body plus a scikit-learn head. Load it with
> `SetFitModel.from_pretrained(repo)`, not `AutoModelForSequenceClassification`.

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
