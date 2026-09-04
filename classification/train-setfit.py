# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "setfit>=1.2.0",
#     "datasets>=4.0.0",
#     "scikit-learn",
#     "huggingface-hub",
#     "torch",
#     "transformers",
# ]
# ///
"""
Few-shot text classification with SetFit — train on 8-64 labelled examples per class, on CPU.

SetFit fine-tunes a sentence-transformer body with contrastive pairs, then fits a logistic
regression head on the embeddings. With a handful of examples per class it reaches a useful
classifier in minutes without a GPU, which makes it the cheap middle rung between zero-shot
LLM labelling and a full encoder fine-tune (`train-classifier.py`).

Run on HF Jobs (cpu-basic is enough; no GPU needed):

    hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py \\
        fancyzhx/ag_news username/ag-news-setfit \\
        --num-samples 8

Metrics match `train-classifier.py` (accuracy + macro F1 on a held-out split) so the two are
directly comparable at equal eval settings.

NOTE: a SetFit model is a sentence-transformer body plus a scikit-learn head. It loads with
`SetFitModel.from_pretrained(repo)`, NOT `AutoModelForSequenceClassification`.
"""

import argparse
import logging
import os
import sys
import time
from collections import Counter

# tqdm reads TQDM_DISABLE when it is imported, so this must be set before any third-party import
# pulls tqdm in — setting it later has no effect. Jobs logs have no TTY, so progress bars arrive
# as hundreds of carriage-return frames that bury the lines you actually want.
os.environ.setdefault("TQDM_DISABLE", "1")

import datasets
import torch
import transformers
from datasets import ClassLabel, Dataset, Value, load_dataset
from huggingface_hub import HfApi, ModelCard, login
from huggingface_hub.utils import disable_progress_bars
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import accuracy_score, f1_score



def configure_logging() -> logging.Logger:
    """Keep Jobs logs readable.

    `basicConfig(level=INFO)` sets the ROOT logger, which switches on every library's INFO
    output — on Jobs that means one line per HTTP request. Root stays at WARNING here and only
    this script's logger is verbose. Progress bars are disabled because Jobs logs have no TTY:
    tqdm's carriage-return frames arrive as hundreds of near-identical lines.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    disable_progress_bars()
    transformers.utils.logging.disable_progress_bar()
    if hasattr(datasets, "disable_progress_bars"):
        datasets.disable_progress_bars()

    script_logger = logging.getLogger("train-setfit")
    script_logger.setLevel(logging.INFO)
    return script_logger


logger = configure_logging()

# MiniLM-L6 is the default because it makes the CPU path viable: ~4.4x faster than
# paraphrase-mpnet-base-v2 on cpu-basic (207s vs 915s for 8 examples/class on ag_news). It is
# NOT chosen on accuracy — on ag_news's test split the two scored 0.804 and 0.788 respectively,
# a gap well inside single-seed few-shot noise. Pass --body-model to try a larger body.
SCRIPT_URL = (
    "https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py"
)

DEFAULT_BODY = "sentence-transformers/all-MiniLM-L6-v2"


def check_label_column(dataset: Dataset, label_column: str) -> None:
    """Fail early and clearly on a missing or multi-label column."""
    if label_column not in dataset.column_names:
        sys.exit(
            f"Label column '{label_column}' not found. Columns are: {dataset.column_names}. "
            "Pass --label-column."
        )

    # A Sequence/list feature carries an inner `feature`; that is the multi-label shape.
    feature = dataset.features.get(label_column)
    if getattr(feature, "feature", None) is not None:
        sys.exit(
            f"Label column '{label_column}' is multi-label (a list per row). "
            "train-setfit.py is single-label only — use train-classifier.py, which "
            "auto-detects multi-label and tunes per-label thresholds."
        )
    if isinstance(dataset[label_column][0], list):
        sys.exit(
            f"Label column '{label_column}' holds lists (multi-label). "
            "Use train-classifier.py instead."
        )


def normalise_label_column(dataset: Dataset, label_column: str) -> Dataset:
    """Make the label column safe for SetFit's positional label mapping.

    SetFit maps an integer prediction through `model.labels` BY POSITION, so integers are only
    safe when they really are indices — which is true for a ClassLabel column and nothing else.
    Any other integer column holds arbitrary values (1-indexed, sparse, or negative), so it is
    stringified and the head learns the label text directly. Without this, a -1/0/1 column
    decodes through Python's negative indexing and silently mislabels everything while the
    metrics still look correct.
    """
    feature = dataset.features.get(label_column)
    if isinstance(feature, ClassLabel):
        return dataset
    # cast_column, not map: map re-casts its output back to the column's EXISTING feature, so
    # returning str() from a map over an int64 column silently converts straight back to int64.
    return dataset.cast_column(label_column, Value("string"))


def resolve_label_names(dataset: Dataset, label_column: str) -> list[str]:
    """Return the class names, in the order SetFit should map predictions through."""
    feature = dataset.features.get(label_column)
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    return sorted(set(dataset[label_column]))


def pick_eval_split(dataset_id, config, train_split, requested):
    """Resolve which split to evaluate on, matching train-classifier.py's precedence."""
    if requested:
        if requested == train_split:
            sys.exit(
                f"--eval-split and --train-split are both '{requested}'. Evaluating on the "
                "training data would report a meaningless score."
            )
        return requested

    # Same auto-detect order as the sibling, so both scripts evaluate on the same split by
    # default and their reported metrics really are comparable.
    available = datasets.get_dataset_split_names(dataset_id, config)
    for candidate in ("validation", "test"):
        if candidate in available and candidate != train_split:
            logger.info("Using the '%s' split for evaluation.", candidate)
            return candidate
    return None


def split_train_eval(dataset_id, config, train_split, eval_split, eval_fraction, seed, label_column):
    """Load the train split, and either the named eval split or a stratified carve-out."""
    train_data = load_dataset(dataset_id, config, split=train_split)

    if eval_split:
        eval_data = load_dataset(dataset_id, config, split=eval_split)
        return train_data, eval_data

    logger.info("No eval split found; carving %.0f%% off the train split.", eval_fraction * 100)
    # Stratify when the labels are typed, so a rare class cannot vanish from a small carve.
    feature = train_data.features.get(label_column)
    stratify = label_column if isinstance(feature, ClassLabel) else None
    parts = train_data.train_test_split(
        test_size=eval_fraction, seed=seed, stratify_by_column=stratify
    )
    return parts["train"], parts["test"]


def evaluate(model, eval_data, text_column, label_column, label_names) -> dict:
    """Predict on the eval set and report the same metrics as train-classifier.py."""
    # None in the text column would crash model.encode after training has already been paid for.
    texts = [str(text) for text in eval_data[text_column]]
    gold_raw = list(eval_data[label_column])

    # The model predicts label NAMES. Decode the gold side using the EVAL set's own feature —
    # a named --eval-split can order its ClassLabel differently from the train split, and
    # decoding through train-derived names would silently score against the wrong table.
    feature = eval_data.features.get(label_column)
    if isinstance(feature, ClassLabel):
        gold = [feature.int2str(int(value)) for value in gold_raw]
    else:
        gold = [str(value) for value in gold_raw]

    started = time.time()
    predictions = model.predict(texts)
    elapsed = time.time() - started

    # SetFit returns a tensor for int labels and a list for string labels.
    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()

    return {
        "accuracy": round(accuracy_score(gold, predictions), 4),
        "f1_macro": round(f1_score(gold, predictions, average="macro", zero_division=0), 4),
        "eval_examples": len(gold),
        "predict_seconds": round(elapsed, 1),
    }


def build_reproduce_command(args) -> str:
    """Rebuild the exact invocation, so the card's command produces the card's model.

    Only non-default flags are appended, keeping the command short while staying faithful.
    """
    flavor = "t4-small" if torch.cuda.is_available() else "cpu-basic"
    parts = [
        f"hf jobs uv run --flavor {flavor} --secrets HF_TOKEN \\",
        f"  {SCRIPT_URL} \\",
        f"  {args.input_dataset} {args.output_repo} \\",
    ]

    flags = []
    if args.body_model != DEFAULT_BODY:
        flags.append(f"--body-model {args.body_model}")
    if args.dataset_config:
        flags.append(f"--dataset-config {args.dataset_config}")
    if args.text_column != "text":
        flags.append(f"--text-column {args.text_column}")
    if args.label_column != "label":
        flags.append(f"--label-column {args.label_column}")
    if args.train_split != "train":
        flags.append(f"--train-split {args.train_split}")
    if args.eval_split:
        flags.append(f"--eval-split {args.eval_split}")
    if args.num_samples != 8:
        flags.append(f"--num-samples {args.num_samples}")
    if args.num_epochs != 1:
        flags.append(f"--num-epochs {args.num_epochs}")
    if args.batch_size != 16:
        flags.append(f"--batch-size {args.batch_size}")
    if args.max_seq_length != 256:
        flags.append(f"--max-seq-length {args.max_seq_length}")
    if args.sampling_strategy != "oversampling":
        flags.append(f"--sampling-strategy {args.sampling_strategy}")
    if args.seed != 42:
        flags.append(f"--seed {args.seed}")

    # --num-samples is the defining knob, so always show it even at its default.
    if f"--num-samples {args.num_samples}" not in flags:
        flags.insert(0, f"--num-samples {args.num_samples}")

    parts.append("  " + " ".join(flags))
    return "\n".join(parts)


def build_card(args, label_names, metrics, per_class, train_seconds) -> str:
    """Model card following the uv-scripts conventions (org credit, Jobs claim gated on JOB_ID)."""
    on_jobs = os.environ.get("JOB_ID") is not None
    provenance = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) "
        "with [`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification)."
        if on_jobs
        else "Produced with [`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification)."
    )

    tags = ["setfit", "text-classification", "few-shot", "uv-script"]
    if on_jobs:
        tags.append("hf-jobs")
    tag_lines = "\n".join(f"- {tag}" for tag in tags)

    metric_lines = "\n".join(f"| {name} | {value} |" for name, value in metrics.items())
    train_size = sum(per_class.values())
    counts = ", ".join(f"`{name}`: {count}" for name, count in per_class.items())

    return f"""---
tags:
{tag_lines}
library_name: setfit
pipeline_tag: text-classification
base_model: {args.body_model}
datasets:
- {args.input_dataset}
---

# {args.output_repo.split("/")[-1]}

Few-shot text classifier trained with [SetFit](https://github.com/huggingface/setfit) on
**up to {args.num_samples} examples per class** ({train_size} training examples in total) from
[`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}).

{provenance}

## Results

| Metric | Value |
|---|---|
{metric_lines}
| training seconds | {round(train_seconds, 1)} |

## Training examples per class

{counts}

## Labels

{", ".join(f"`{name}`" for name in label_names)}

## Use it

```python
from setfit import SetFitModel

model = SetFitModel.from_pretrained("{args.output_repo}")
model.predict(["some text to classify"])
```

## Reproduction

Produced by [`train-setfit.py`]({SCRIPT_URL}) from
[`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification):

```bash
{build_reproduce_command(args)}
```
"""


def main(args) -> None:
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("No HF token. Pass --hf-token or run with --secrets HF_TOKEN.")
    login(token=token)

    # Prove we can write the output repo BEFORE paying for training. A permissions failure
    # after trainer.train() costs the whole run and leaves no artifact behind.
    HfApi(token=token).create_repo(
        args.output_repo, repo_type="model", private=args.private, exist_ok=True
    )

    logger.info("Loading %s", args.input_dataset)
    eval_split = pick_eval_split(
        args.input_dataset, args.dataset_config, args.train_split, args.eval_split
    )
    train_pool, eval_data = split_train_eval(
        args.input_dataset,
        args.dataset_config,
        args.train_split,
        eval_split,
        args.eval_fraction,
        args.seed,
        args.label_column,
    )

    check_label_column(train_pool, args.label_column)
    train_pool = normalise_label_column(train_pool, args.label_column)
    eval_data = normalise_label_column(eval_data, args.label_column)

    label_names = resolve_label_names(train_pool, args.label_column)
    logger.info("Found %d classes: %s", len(label_names), label_names)
    if len(label_names) < 2:
        sys.exit(f"Only one class found in '{args.label_column}'. A classifier needs two or more.")

    if args.max_eval_samples and len(eval_data) > args.max_eval_samples:
        eval_data = eval_data.shuffle(seed=args.seed).select(range(args.max_eval_samples))
        logger.info("Capped eval set at %d examples.", args.max_eval_samples)

    # sample_dataset calls .to_pandas() on the whole pool, which would OOM a cpu-basic job on a
    # multi-million-row dataset. Capping first keeps the CPU path viable; the cap samples
    # randomly, so a very rare class can be thinned by it.
    if args.max_train_pool and len(train_pool) > args.max_train_pool:
        train_pool = train_pool.shuffle(seed=args.seed).select(range(args.max_train_pool))
        logger.info("Capped train pool at %d rows before sampling.", args.max_train_pool)

    train_data = sample_dataset(
        train_pool, label_column=args.label_column, num_samples=args.num_samples, seed=args.seed
    )
    # sample_dataset takes AT MOST num_samples per class, so report what was actually drawn.
    per_class = dict(sorted(Counter(train_data[args.label_column]).items()))
    logger.info("Sampled %d training examples; per class: %s", len(train_data), per_class)

    logger.info("Loading body model %s", args.body_model)
    model = SetFitModel.from_pretrained(args.body_model, labels=label_names)
    model.model_body.max_seq_length = args.max_seq_length

    # SetFit picks the accelerator itself; report what it chose so a run's logs are self-describing.
    if torch.cuda.is_available():
        logger.info("DEVICE: cuda (%s)", torch.cuda.get_device_name(0))
    else:
        logger.info("DEVICE: cpu")
    logger.info("DEVICE: body model is on %s", model.model_body.device)

    training_args = TrainingArguments(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        report_to="none",
        show_progress_bar=False,
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        column_mapping={args.text_column: "text", args.label_column: "label"},
    )

    started = time.time()
    trainer.train()
    train_seconds = time.time() - started
    logger.info("Training finished in %.1fs", train_seconds)

    metrics = evaluate(model, eval_data, args.text_column, args.label_column, label_names)
    logger.info("Metrics: %s", metrics)

    logger.info("Pushing to %s (private=%s)", args.output_repo, args.private)
    model.push_to_hub(args.output_repo, private=args.private, token=token)
    card = build_card(args, label_names, metrics, per_class, train_seconds)
    ModelCard(card).push_to_hub(args.output_repo, token=token)

    logger.info("Verifying reload from the Hub")
    reloaded = SetFitModel.from_pretrained(args.output_repo, token=token)
    sample_texts = eval_data[args.text_column][:4]
    logger.info("Reloaded predictions: %s", reloaded.predict(sample_texts))
    logger.info("Done: https://huggingface.co/%s", args.output_repo)


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot text classification with SetFit")
    parser.add_argument("input_dataset", help="Input dataset ID")
    parser.add_argument("output_repo", help="Output model repo ID (username/model-name)")
    parser.add_argument("--body-model", default=DEFAULT_BODY, help=f"Sentence-transformer body (default: {DEFAULT_BODY})")
    parser.add_argument("--dataset-config", help="Dataset config name")
    parser.add_argument("--text-column", default="text", help="Text column (default: text)")
    parser.add_argument("--label-column", default="label", help="Label column (default: label)")
    parser.add_argument("--train-split", default="train", help="Train split (default: train)")
    parser.add_argument(
        "--eval-split",
        help="Eval split. Default: validation, else test, else carve --eval-fraction off "
             "train. A slice such as train[:10%] is NOT checked for overlap with training.",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Eval fraction if no eval split (default: 0.1)")
    parser.add_argument("--max-eval-samples", type=int, default=2000, help="Cap eval examples (default: 2000)")
    parser.add_argument(
        "--max-train-pool", type=int, default=200_000,
        help="Cap the pool before per-class sampling (default: 200000)",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Labelled examples per class (default: 8)")
    parser.add_argument("--num-epochs", type=int, default=1, help="Epochs (default: 1)")
    parser.add_argument("--sampling-strategy", default="oversampling",
                        choices=["oversampling", "undersampling", "unique"],
                        help="Contrastive pair sampling (default: oversampling)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Max sequence length (default: 256)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--private", action="store_true", help="Make the output model repo private")
    parser.add_argument("--hf-token", help="HF token (or set HF_TOKEN)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
