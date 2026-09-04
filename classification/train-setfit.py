# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "setfit>=1.2.0",
#     "datasets>=4.0.0",
#     "scikit-learn",
#     "huggingface-hub",
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

# tqdm reads TQDM_DISABLE when it is imported, so this must be set before any third-party import
# pulls tqdm in — setting it later has no effect. Jobs logs have no TTY, so progress bars arrive
# as hundreds of carriage-return frames that bury the lines you actually want.
os.environ.setdefault("TQDM_DISABLE", "1")

import datasets
import torch
import transformers
from datasets import Dataset, load_dataset
from huggingface_hub import ModelCard, login
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

# MiniLM-L6 trains ~4.4x faster than paraphrase-mpnet-base-v2 on cpu-basic (207s vs 915s for
# 8 examples/class on ag_news) for 3.6pp less accuracy (0.826 vs 0.862). The CPU-first default
# matters more here than the ceiling; pass --body-model with a GPU flavor for the ceiling.
DEFAULT_BODY = "sentence-transformers/all-MiniLM-L6-v2"


def resolve_label_names(dataset: Dataset, label_column: str) -> list[str]:
    """Return human-readable class names, falling back to stringified label values."""
    feature = dataset.features.get(label_column)
    if hasattr(feature, "names"):
        return list(feature.names)
    distinct = sorted(set(dataset[label_column]))
    return [str(value) for value in distinct]


def split_train_eval(dataset_id, config, train_split, eval_split, eval_fraction, seed):
    """Load the train split, and either the named eval split or a carved-out fraction."""
    train_data = load_dataset(dataset_id, config, split=train_split)

    if eval_split:
        eval_data = load_dataset(dataset_id, config, split=eval_split)
        return train_data, eval_data

    logger.info("No --eval-split given; carving %.0f%% off the train split.", eval_fraction * 100)
    parts = train_data.train_test_split(test_size=eval_fraction, seed=seed)
    return parts["train"], parts["test"]


def evaluate(model, eval_data, text_column, label_column, label_names) -> dict:
    """Predict on the eval set and report the same metrics as train-classifier.py."""
    texts = eval_data[text_column]
    gold_raw = list(eval_data[label_column])

    # The model was loaded with `labels=`, so it predicts label NAMES, while a ClassLabel column
    # holds integer indices. Normalise the gold side to names so both sides are the same type.
    gold = [label_names[value] if isinstance(value, int) else value for value in gold_raw]

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


def build_card(args, label_names, metrics, train_size, train_seconds) -> str:
    """Model card following the uv-scripts conventions (org credit, Jobs claim gated on JOB_ID)."""
    on_jobs = os.environ.get("JOB_ID") is not None
    provenance = (
        "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) "
        "with [`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification)."
        if on_jobs
        else "Produced with [`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification)."
    )
    metric_lines = "\n".join(
        f"| {name} | {value} |" for name, value in metrics.items()
    )
    return f"""---
tags:
- setfit
- text-classification
- few-shot
- uv-script
{"- hf-jobs" if on_jobs else ""}
library_name: setfit
base_model: {args.body_model}
---

# {args.output_repo.split("/")[-1]}

Few-shot text classifier trained with [SetFit](https://github.com/huggingface/setfit) on
**{args.num_samples} examples per class** ({train_size} training examples total) from
[`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}).

{provenance}

## Results

| Metric | Value |
|---|---|
{metric_lines}
| training seconds | {round(train_seconds, 1)} |

## Labels

{", ".join(f"`{name}`" for name in label_names)}

## Use it

```python
from setfit import SetFitModel

model = SetFitModel.from_pretrained("{args.output_repo}")
model.predict(["some text to classify"])
```

## Reproduce

```bash
hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN \\
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-setfit.py \\
  {args.input_dataset} {args.output_repo} --num-samples {args.num_samples}
```
"""


def main(args) -> None:
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("No HF token. Pass --hf-token or run with --secrets HF_TOKEN.")
    login(token=token)

    logger.info("Loading %s", args.input_dataset)
    train_pool, eval_data = split_train_eval(
        args.input_dataset,
        args.dataset_config,
        args.train_split,
        args.eval_split,
        args.eval_fraction,
        args.seed,
    )

    label_names = resolve_label_names(train_pool, args.label_column)
    logger.info("Found %d classes: %s", len(label_names), label_names)

    if args.max_eval_samples and len(eval_data) > args.max_eval_samples:
        eval_data = eval_data.shuffle(seed=args.seed).select(range(args.max_eval_samples))
        logger.info("Capped eval set at %d examples.", args.max_eval_samples)

    train_data = sample_dataset(
        train_pool, label_column=args.label_column, num_samples=args.num_samples, seed=args.seed
    )
    logger.info("Sampled %d training examples (%d per class).", len(train_data), args.num_samples)

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
    card = build_card(args, label_names, metrics, len(train_data), train_seconds)
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
    parser.add_argument("--eval-split", help="Eval split (default: carve --eval-fraction off train)")
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Eval fraction if no eval split (default: 0.1)")
    parser.add_argument("--max-eval-samples", type=int, default=2000, help="Cap eval examples (default: 2000)")
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
