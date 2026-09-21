# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "gliner2[train]==2.0.0",
#     "protobuf",
#     "sentencepiece",
#     "datasets>=4.0.0",
#     "scikit-learn",
#     "huggingface-hub",
# ]
# ///
"""
Fine-tune GLiNER2 into a text classifier — a ~300M model that already works zero-shot.

GLiNER2 reads the label names as part of its input, so it classifies with no training at all.
This script measures that zero-shot score first, fine-tunes on your labels, then measures
again on the same held-out rows. The model card reports both numbers, so you can see what the
labels bought you. One model can answer several questions at once: pass --label-column more
than once and each column becomes a task.

Run on HF Jobs (t4-small is enough for a few thousand short texts):

    hf jobs uv run --flavor t4-small --secrets HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py \\
        biglam/blbooksgenre username/gliner2-blbooks-genre \\
        --dataset-config title_genre_classifiction --text-column title

Jobs stop after 30 minutes by default and the model is pushed at the end, so add `--timeout 1h`
for larger datasets or tasks with many labels (56 labels x 5,452 rows x 3 epochs took 31 minutes).

Metrics match `train-classifier.py` and `train-setfit.py` (accuracy + macro F1 on a held-out
split), so the three are directly comparable at equal eval settings.

NOTE: the output is a GLiNER2 checkpoint. It loads with
`gliner2.classification.Classifier.from_pretrained(repo)`, NOT `AutoModelForSequenceClassification`.
Apply it to a whole dataset with the sibling `classify-gliner2.py`.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

# tqdm reads TQDM_DISABLE when it is imported, so this must be set before any third-party import
# pulls tqdm in. Jobs logs have no TTY, so progress bars arrive as hundreds of near-identical lines.
# (The gliner2 trainer passes disable=False to its own bar, so its training bar still prints.)
os.environ.setdefault("TQDM_DISABLE", "1")

import datasets
import torch
from datasets import ClassLabel, Dataset, load_dataset
from gliner2 import AutoExtractor
from gliner2.classification import (
    ClassificationConfig,
    ClassificationSchema,
    Classifier,
)
from gliner2.training.data import Classification, InputExample
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from huggingface_hub import HfApi, login
from huggingface_hub.utils import disable_progress_bars
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def configure_logging() -> logging.Logger:
    """Keep Jobs logs readable: root at WARNING, only this script's logger at INFO."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    disable_progress_bars()
    if hasattr(datasets, "disable_progress_bars"):
        datasets.disable_progress_bars()

    script_logger = logging.getLogger("train-gliner2")
    script_logger.setLevel(logging.INFO)
    return script_logger


logger = configure_logging()

SCRIPT_URL = (
    "https://huggingface.co/datasets/uv-scripts/classification/raw/main/train-gliner2.py"
)
DEFAULT_BASE_MODEL = "fastino/gliner2.5-multi-v1"

# The file this script adds to the model repo. It records the tasks and label names the model
# was trained on, so classify-gliner2.py can rebuild the same schema without any flags.
SCHEMA_FILENAME = "classification_schema.json"

# Give up after this many out-of-memory training steps. See StopOnRepeatedOOM.
MAX_OOM_STEPS = 5

# GLiNER2 puts label names into the model prompt verbatim. Its inference schema rejects these
# strings, but its trainer accepts them — so a label like "manuscripts (documents)" trains
# without complaint and then cannot be predicted. We clean labels once, before either side.
RESERVED_MARKERS = ("[P]", "[L]", "[C]", "[E]", "[R]", "[DESCRIPTION]", "[EXAMPLE]", "[OUTPUT]")


def clean_label(label: str) -> str:
    """Make one label name safe for the GLiNER2 prompt."""
    cleaned = label.replace("(", " ").replace(")", " ")
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        sys.exit(f"Label {label!r} is empty after cleaning. Rename it in the dataset.")
    for marker in RESERVED_MARKERS:
        if marker in cleaned:
            sys.exit(
                f"Label {label!r} contains {marker!r}, which GLiNER2 reserves for its prompt. "
                "Rename it in the dataset."
            )
    return cleaned


def is_multi_label_column(dataset: Dataset, column: str) -> bool:
    """A list-valued column is a multi-label task."""
    feature = dataset.features.get(column)
    # A Sequence/list feature carries an inner `feature`.
    return getattr(feature, "feature", None) is not None


def label_feature(dataset: Dataset, column: str):
    """Return the ClassLabel that types this column, or None if the labels are plain values."""
    feature = dataset.features.get(column)
    inner = getattr(feature, "feature", None)
    if isinstance(feature, ClassLabel):
        return feature
    if isinstance(inner, ClassLabel):
        return inner
    return None


def decode_label(value, class_label) -> str:
    """Turn one raw label value into its cleaned name."""
    if class_label is not None:
        return clean_label(class_label.int2str(int(value)))
    return clean_label(str(value))


def decode_column(dataset: Dataset, column: str) -> list:
    """Return the gold labels for a column: a name per row, or a sorted list of names per row.

    Decoding uses this split's OWN feature. A named --eval-split can order its ClassLabel
    differently from the train split, and decoding through the train names would silently
    score against the wrong table.
    """
    class_label = label_feature(dataset, column)
    multi = is_multi_label_column(dataset, column)
    decoded = []
    for value in dataset[column]:
        if multi:
            names = {decode_label(item, class_label) for item in (value or [])}
            decoded.append(sorted(names))
        else:
            decoded.append(decode_label(value, class_label))
    return decoded


def drop_unlabelled_rows(dataset: Dataset, columns: list, text_column: str, split_name: str) -> Dataset:
    """Remove rows with no text, or with a missing or blank single-label value.

    An empty LIST in a multi-label column is kept: "none of these labels" is a valid answer,
    and the model needs to see it to learn when to select nothing.
    """
    # In a ClassLabel column, -1 is the Hub convention for "no label" (common in test splits).
    typed_columns = [column for column in columns if isinstance(dataset.features.get(column), ClassLabel)]

    def is_usable(example) -> bool:
        text = example[text_column]
        if text is None or not str(text).strip():
            return False
        for column in columns:
            value = example[column]
            if isinstance(value, list):
                continue
            if value is None:
                return False
            if isinstance(value, str) and not value.strip():
                return False
            if column in typed_columns and value < 0:
                return False
        return True

    kept = dataset.filter(is_usable)
    dropped = len(dataset) - len(kept)
    if dropped:
        logger.warning(
            "Dropped %d %s rows with no text or a missing label (%d remain).",
            dropped, split_name, len(kept),
        )
    return kept


def pick_eval_split(dataset_id, config, train_split, requested):
    """Resolve which split to evaluate on, matching train-classifier.py's precedence."""
    if requested:
        if requested == train_split:
            sys.exit(
                f"--eval-split and --train-split are both '{requested}'. Evaluating on the "
                "training data would report a meaningless score."
            )
        return requested

    available = datasets.get_dataset_split_names(dataset_id, config)
    for candidate in ("validation", "test"):
        if candidate in available and candidate != train_split:
            logger.info("Using the '%s' split for evaluation.", candidate)
            return candidate
    return None


def split_train_eval(args, eval_split, first_label_column):
    """Load the train split, and either the named eval split or a carve-out of train."""
    train_data = load_dataset(args.input_dataset, args.dataset_config, split=args.train_split)

    if eval_split:
        eval_data = load_dataset(args.input_dataset, args.dataset_config, split=eval_split)
        return train_data, eval_data

    logger.info("No eval split found; carving %.0f%% off the train split.", args.eval_fraction * 100)
    # Stratify when the first label column is typed, so a rare class cannot vanish from a small carve.
    feature = train_data.features.get(first_label_column)
    stratify = first_label_column if isinstance(feature, ClassLabel) else None
    try:
        parts = train_data.train_test_split(
            test_size=args.eval_fraction, seed=args.seed, stratify_by_column=stratify
        )
    except ValueError as error:
        # Stratification needs at least two members of every class, so it fails on exactly the
        # singleton classes it is meant to protect. An unstratified split is worse but usable.
        logger.warning("Could not stratify the carve-out (%s). Using an unstratified split.", error)
        parts = train_data.train_test_split(test_size=args.eval_fraction, seed=args.seed)
    return parts["train"], parts["test"]


def prepare_texts(dataset: Dataset, text_column: str, max_text_chars: int, split_name: str) -> list:
    """Return the text for each row, truncated to max_text_chars."""
    texts = []
    truncated = 0
    for value in dataset[text_column]:
        text = str(value)
        if len(text) > max_text_chars:
            text = text[:max_text_chars]
            truncated += 1
        texts.append(text)
    if truncated:
        logger.warning(
            "Truncated %d of %d %s texts to %d characters. Raise --max-text-chars if the label "
            "depends on text past that point.",
            truncated, len(texts), split_name, max_text_chars,
        )
    return texts


def build_tasks(train_data: Dataset, label_columns: list, task_names: list) -> list:
    """Describe one classification task per label column.

    GLiNER2 reads the task name as part of its prompt, next to the label names, so --task-name
    lets you call the task "genre" rather than "label". Do not expect much from it: on BL book
    titles the zero-shot accuracy was 0.79 with "label" and 0.78 with "genre".

    The label list comes from the TRAIN split. A label that appears only in the eval split
    cannot be predicted, and evaluate() counts it as an error rather than hiding it.
    """
    tasks = []
    for column, task_name in zip(label_columns, task_names):
        multi = is_multi_label_column(train_data, column)
        class_label = label_feature(train_data, column)
        if class_label is not None:
            labels = [clean_label(name) for name in class_label.names]
        else:
            seen = set()
            for value in decode_column(train_data, column):
                if multi:
                    seen.update(value)
                else:
                    seen.add(value)
            labels = sorted(seen)

        if class_label is not None:
            raw_names = list(class_label.names)
        else:
            raw_names = []
            for value in train_data[column]:
                raw_names.extend(value or [] if multi else [value])
            raw_names = sorted({str(name) for name in raw_names})
        renamed = {name: clean_label(name) for name in raw_names if clean_label(name) != name}
        if renamed:
            logger.warning(
                "Column '%s': GLiNER2 does not allow brackets in label names, so the model will "
                "predict the cleaned names: %s", column, renamed,
            )

        if len(set(labels)) != len(labels):
            sys.exit(f"Column '{column}': two labels are identical after cleaning: {labels}")
        if len(labels) < 2:
            sys.exit(f"Column '{column}' has fewer than two labels: {labels}")
        if all(label.lstrip("-").isdigit() for label in labels):
            logger.warning(
                "Column '%s' has numeric labels %s. GLiNER2 reads label NAMES, so the zero-shot "
                "score will be meaningless and fine-tuning starts from nothing. A ClassLabel or "
                "string column with real names will do better.",
                column, labels[:6],
            )
        tasks.append({"name": clean_label(task_name), "column": column, "labels": labels, "multi_label": multi})
        logger.info(
            "Task '%s' (column '%s'): %d labels, %s.",
            task_name, column, len(labels), "multi-label" if multi else "single-label",
        )
    return tasks


def build_training_examples(texts: list, gold_by_task: dict, tasks: list) -> list:
    examples = []
    for row, text in enumerate(texts):
        classifications = []
        for task in tasks:
            classifications.append(
                Classification(
                    task=task["name"],
                    labels=task["labels"],
                    true_label=gold_by_task[task["name"]][row],
                    # Only auto-inferred when a row has 2+ true labels, so state it.
                    multi_label=task["multi_label"],
                )
            )
        examples.append(InputExample(text=text, classifications=classifications))
    return examples


def build_schema(tasks: list) -> ClassificationSchema:
    schema = ClassificationSchema()
    for task in tasks:
        if task["multi_label"]:
            schema.multi(task["name"], task["labels"])
        else:
            schema.single(task["name"], task["labels"])
    return schema


class TrainingOutOfMemory(Exception):
    """Raised when the GPU keeps running out of memory during training."""


class StopOnRepeatedOOM(logging.Handler):
    """Abort training when the gliner2 trainer keeps hitting CUDA out-of-memory.

    The gliner2 trainer catches an OOM, skips that batch, logs a warning and carries on. On a
    GPU that is too small for the batch, EVERY step is skipped: the job runs to the end, looks
    healthy, and produces a model that never trained. (Seen on t4-small with 56 labels at batch
    size 16: 1,006 of 1,020 steps skipped.) The trainer has no option to raise instead, so this
    handler watches its log. An exception raised in emit() propagates out of the trainer's own
    logger.warning() call, which stops trainer.train().
    """

    def __init__(self, limit: int):
        super().__init__(level=logging.WARNING)
        self.limit = limit
        self.oom_steps = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "OOM at step" not in record.getMessage():
            return
        self.oom_steps += 1
        if self.oom_steps >= self.limit:
            raise TrainingOutOfMemory()


def evaluate(model_path: str, texts: list, gold_by_task: dict, tasks: list, batch_size: int) -> dict:
    """Load a GLiNER2 checkpoint, predict every task in one pass, and score each task."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # from_pretrained(device=...) does not move the weights in gliner2 2.0.0; .to() does.
    classifier = Classifier.from_pretrained(model_path).to(device=device).eval()

    started = time.time()
    results = classifier.batch_classify(
        texts, build_schema(tasks), config=ClassificationConfig(batch_size=batch_size)
    )
    elapsed = time.time() - started

    metrics = {}
    for task in tasks:
        name = task["name"]
        gold = gold_by_task[name]
        if task["multi_label"]:
            predicted = [sorted(result.selected(name)) for result in results]
            # Labels seen only in eval still count: the binarizer covers the union.
            all_labels = sorted(set(task["labels"]) | {label for row in gold for label in row})
            binarizer = MultiLabelBinarizer(classes=all_labels)
            gold_matrix = binarizer.fit_transform(gold)
            predicted_matrix = binarizer.transform(predicted)
            metrics[name] = {
                "f1_micro": round(f1_score(gold_matrix, predicted_matrix, average="micro", zero_division=0), 4),
                "f1_macro": round(f1_score(gold_matrix, predicted_matrix, average="macro", zero_division=0), 4),
                "exact_match": round(accuracy_score(gold_matrix, predicted_matrix), 4),
            }
        else:
            predicted = [result.value(name) for result in results]
            # The majority-class rate is the floor any classifier must clear to be worth having.
            majority = Counter(gold).most_common(1)[0][1] / len(gold)
            metrics[name] = {
                "accuracy": round(accuracy_score(gold, predicted), 4),
                "f1_macro": round(f1_score(gold, predicted, average="macro", zero_division=0), 4),
                "majority_baseline": round(majority, 4),
            }

    # Release the GPU before training starts.
    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"tasks": metrics, "eval_examples": len(texts), "predict_seconds": round(elapsed, 1)}


def jobs_flavor() -> str:
    """Return the Jobs hardware flavor, or "" when it is not known.

    Jobs sets ACCELERATOR to the flavor on some hardware ("a10g-small", "l4x1") but to a bare
    "gpu" on others (seen on t4-small), and a bare "gpu" is not a valid --flavor.
    """
    hardware = os.environ.get("ACCELERATOR") or ""
    looks_like_flavor = "-" in hardware or any(character.isdigit() for character in hardware)
    return hardware if looks_like_flavor else ""


def build_reproduce_command(args) -> str:
    """Rebuild the exact invocation, so the card's command produces the card's model."""
    flavor = jobs_flavor() or "t4-small"
    parts = [
        f"hf jobs uv run --flavor {flavor} --secrets HF_TOKEN \\",
        f"  {SCRIPT_URL} \\",
        f"  {args.input_dataset} {args.output_repo}",
    ]
    flags = []
    if args.dataset_config:
        flags.append(f"--dataset-config {args.dataset_config}")
    if args.text_column != "text":
        flags.append(f"--text-column {args.text_column}")
    if args.label_column != ["label"]:
        for column in args.label_column:
            flags.append(f"--label-column {column}")
    if args.task_name != args.label_column:
        for task_name in args.task_name:
            flags.append(f"--task-name {task_name}")
    if args.base_model != DEFAULT_BASE_MODEL:
        flags.append(f"--base-model {args.base_model}")
    if args.train_split != "train":
        flags.append(f"--train-split {args.train_split}")
    if args.eval_split:
        flags.append(f"--eval-split {args.eval_split}")
    # These change which rows are trained on or scored, so a command without them reproduces
    # a different model and a different number.
    if args.eval_fraction != 0.1:
        flags.append(f"--eval-fraction {args.eval_fraction}")
    if args.max_train_samples:
        flags.append(f"--max-train-samples {args.max_train_samples}")
    if args.max_eval_samples != 2000:
        flags.append(f"--max-eval-samples {args.max_eval_samples}")
    if args.max_text_chars != 2000:
        flags.append(f"--max-text-chars {args.max_text_chars}")
    if args.epochs != 5:
        flags.append(f"--epochs {args.epochs}")
    if args.batch_size != 16:
        flags.append(f"--batch-size {args.batch_size}")
    if args.grad_accum != 1:
        flags.append(f"--grad-accum {args.grad_accum}")
    if args.encoder_lr != 1e-5:
        flags.append(f"--encoder-lr {args.encoder_lr}")
    if args.task_lr != 5e-4:
        flags.append(f"--task-lr {args.task_lr}")
    if args.seed != 42:
        flags.append(f"--seed {args.seed}")
    if args.skip_zero_shot:
        flags.append("--skip-zero-shot")
    if args.private:
        flags.append("--private")

    if flags:
        parts[-1] += " \\"
        parts.append("  " + " ".join(flags))
    return "\n".join(parts)


def results_table(tasks: list, zero_shot, fine_tuned) -> str:
    """One row per task and metric, with the zero-shot score next to the fine-tuned one."""
    lines = ["| Task | Metric | Zero-shot | Fine-tuned |", "|---|---|---|---|"]
    for task in tasks:
        name = task["name"]
        for metric, value in fine_tuned["tasks"][name].items():
            if metric == "majority_baseline":
                continue
            before = zero_shot["tasks"][name][metric] if zero_shot else "not run"
            lines.append(f"| `{name}` | {metric} | {before} | **{value}** |")
    return "\n".join(lines)


def build_card(args, tasks, zero_shot, fine_tuned, train_size, train_seconds, eval_split, oom_steps) -> str:
    """Model card following the uv-scripts conventions (org credit, Jobs claim gated on JOB_ID)."""
    on_jobs = os.environ.get("JOB_ID") is not None
    hardware = jobs_flavor()
    if on_jobs:
        provenance = "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        if hardware:
            provenance += f" (`{hardware}`)"
    else:
        provenance = "Produced"
    provenance += " with [`uv-scripts/classification`](https://huggingface.co/datasets/uv-scripts/classification)."

    tags = ["gliner2", "text-classification", "uv-script"]
    if on_jobs:
        tags.append("hf-jobs")
    tag_lines = "\n".join(f"- {tag}" for tag in tags)

    caveats = []
    if not eval_split:
        caveats.append(
            f"**No held-out split existed, so {args.eval_fraction:.0%} was carved out of train.** "
            "These numbers are not comparable with published results on this dataset."
        )
    for task in tasks:
        if not task["multi_label"]:
            floor = fine_tuned["tasks"][task["name"]]["majority_baseline"]
            caveats.append(
                f"`{task['name']}`: always answering the most common label scores `{floor}` accuracy. "
                "Read the accuracy against that floor."
            )
    if oom_steps:
        caveats.append(
            f"**{oom_steps} training step(s) were skipped** because the GPU ran out of memory. "
            "The model saw less data than the example count above suggests."
        )
    caveats.append("Single seed. Small differences between runs are not evidence of anything.")
    caveat_block = "\n".join(f"- {caveat}" for caveat in caveats)

    label_lines = []
    for task in tasks:
        kind = "multi-label" if task["multi_label"] else "single-label"
        names = ", ".join(f"`{label}`" for label in task["labels"])
        label_lines.append(f"- **`{task['name']}`** ({kind}): {names}")
    label_block = "\n".join(label_lines)

    schema_lines = ["schema = ClassificationSchema()"]
    for task in tasks:
        method = "multi" if task["multi_label"] else "single"
        schema_lines.append(f"schema.{method}({task['name']!r}, {task['labels']!r})")
    schema_code = "\n".join(schema_lines)
    first_task = tasks[0]
    read_result = "selected" if first_task["multi_label"] else "value"

    return f"""---
tags:
{tag_lines}
library_name: gliner2
pipeline_tag: text-classification
base_model: {args.base_model}
datasets:
- {args.input_dataset}
---

# {args.output_repo.split("/")[-1]}

[GLiNER2](https://github.com/fastino-ai/GLiNER2) text classifier, fine-tuned from
[`{args.base_model}`](https://huggingface.co/{args.base_model}) on {train_size} examples from
[`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}).

{provenance}

## Results

Scored on {fine_tuned["eval_examples"]} held-out examples. "Zero-shot" is the base model given only
the label names, before any training, on the same examples.

{results_table(tasks, zero_shot, fine_tuned)}

Training took {round(train_seconds)} seconds.

## Read this before trusting the numbers

{caveat_block}

## Tasks and labels

{label_block}

## Use it

```python
# pip install "gliner2[local]==2.0.0" protobuf sentencepiece
from gliner2.classification import ClassificationSchema, Classifier

classifier = Classifier.from_pretrained("{args.output_repo}").eval()

{schema_code}

result = classifier.batch_classify(["some text to classify"], schema)[0]
print(result.{read_result}({first_task["name"]!r}), result.confidence({first_task["name"]!r}))
```

To label a whole Hub dataset with this model:

```bash
hf jobs uv run --flavor t4-small --secrets HF_TOKEN \\
  https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-gliner2.py \\
  <input-dataset> <output-dataset> --model {args.output_repo} --text-column {args.text_column}
```

## Reproduction

Produced by [`train-gliner2.py`]({SCRIPT_URL}) from
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

    if not torch.cuda.is_available():
        if not args.allow_cpu:
            sys.exit(
                "No GPU found. GLiNER2 fine-tuning needs one: run with `--flavor t4-small` on HF "
                "Jobs. Pass --allow-cpu to run anyway (only sensible with a tiny --max-train-samples)."
            )
        logger.warning("No GPU found; training on CPU because --allow-cpu was passed.")

    # Prove we can write the output repo BEFORE paying for training.
    api = HfApi(token=token)
    api.create_repo(args.output_repo, repo_type="model", private=args.private, exist_ok=True)

    logger.info("Loading %s", args.input_dataset)
    eval_split = pick_eval_split(args.input_dataset, args.dataset_config, args.train_split, args.eval_split)
    train_data, eval_data = split_train_eval(args, eval_split, args.label_column[0])

    for column in [args.text_column] + args.label_column:
        if column not in train_data.column_names:
            sys.exit(f"Column '{column}' not found. Columns are: {train_data.column_names}.")

    train_data = drop_unlabelled_rows(train_data, args.label_column, args.text_column, "train")
    eval_data = drop_unlabelled_rows(eval_data, args.label_column, args.text_column, "eval")

    if args.max_train_samples and len(train_data) > args.max_train_samples:
        train_data = train_data.shuffle(seed=args.seed).select(range(args.max_train_samples))
    if len(eval_data) > args.max_eval_samples:
        eval_data = eval_data.shuffle(seed=args.seed).select(range(args.max_eval_samples))
    logger.info("Train examples: %d. Eval examples: %d.", len(train_data), len(eval_data))

    tasks = build_tasks(train_data, args.label_column, args.task_name)
    train_texts = prepare_texts(train_data, args.text_column, args.max_text_chars, "train")
    eval_texts = prepare_texts(eval_data, args.text_column, args.max_text_chars, "eval")
    train_gold = {task["name"]: decode_column(train_data, task["column"]) for task in tasks}
    eval_gold = {task["name"]: decode_column(eval_data, task["column"]) for task in tasks}
    logger.info("Example input: %s", train_texts[0][:300])

    zero_shot = None
    if not args.skip_zero_shot:
        logger.info("Scoring the base model zero-shot, before any training.")
        zero_shot = evaluate(args.base_model, eval_texts, eval_gold, tasks, args.eval_batch_size)
        logger.info("Zero-shot: %s", json.dumps(zero_shot["tasks"]))

    examples = build_training_examples(train_texts, train_gold, tasks)
    model = AutoExtractor.from_pretrained(args.base_model)
    config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        seed=args.seed,
        # We score the held-out split ourselves, before and after, with the same code.
        eval_strategy="no",
        # Mixed precision is on by default in gliner2; fp32 avoids fp16 overflow on T4.
        fp16=False,
        logging_steps=20,
    )
    logger.info("Training for %d epochs on %d examples.", args.epochs, len(examples))
    started = time.time()
    oom_guard = StopOnRepeatedOOM(limit=MAX_OOM_STEPS)
    logging.getLogger("gliner2.training.trainer").addHandler(oom_guard)
    try:
        ExtractorTrainer(model, config).train(train_data=examples)
    except TrainingOutOfMemory:
        smaller = max(1, args.batch_size // 4)
        sys.exit(
            f"Stopped: the GPU ran out of memory on {MAX_OOM_STEPS} training steps, so nothing was "
            f"pushed. Memory grows with batch size x number of labels x text length. Try "
            f"`--batch-size {smaller} --grad-accum {args.grad_accum * (args.batch_size // smaller)}` "
            "(same effective batch), a lower --max-text-chars, or a larger --flavor such as a10g-small."
        )
    train_seconds = time.time() - started
    logger.info("Training took %.0f seconds.", train_seconds)
    if oom_guard.oom_steps:
        logger.warning(
            "%d training step(s) were skipped after running out of GPU memory. The model trained "
            "on the rest. Lower --batch-size to avoid this.", oom_guard.oom_steps,
        )

    # Release the training copy, then score the checkpoint that will actually be uploaded.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final_dir = os.path.join(args.output_dir, "final")
    fine_tuned = evaluate(final_dir, eval_texts, eval_gold, tasks, args.eval_batch_size)
    logger.info("Fine-tuned: %s", json.dumps(fine_tuned["tasks"]))

    schema_record = {
        "text_column": args.text_column,
        "tasks": [
            {"name": task["name"], "labels": task["labels"], "multi_label": task["multi_label"]}
            for task in tasks
        ],
    }
    with open(os.path.join(final_dir, SCHEMA_FILENAME), "w") as handle:
        json.dump(schema_record, handle, indent=2)
    card = build_card(
        args, tasks, zero_shot, fine_tuned, len(examples), train_seconds, eval_split, oom_guard.oom_steps
    )
    with open(os.path.join(final_dir, "README.md"), "w") as handle:
        handle.write(card)

    api.upload_folder(repo_id=args.output_repo, folder_path=final_dir, repo_type="model")
    logger.info("Pushed to https://huggingface.co/%s", args.output_repo)
    print("SUMMARY_JSON " + json.dumps({"zero_shot": zero_shot, "fine_tuned": fine_tuned, "train_seconds": round(train_seconds)}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dataset", help="Input dataset ID")
    parser.add_argument("output_repo", help="Output model repo ID (username/model-name)")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help=f"GLiNER2 checkpoint to start from (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--dataset-config", help="Dataset config name")
    parser.add_argument("--text-column", default="text", help="Text column (default: text)")
    parser.add_argument(
        "--label-column", action="append",
        help="Label column (default: label). Repeat for several tasks in one model. A column of "
        "lists is treated as multi-label.",
    )
    parser.add_argument(
        "--task-name", action="append",
        help="Name of the task, one per --label-column in the same order (default: the column "
        "name). The model reads it as part of its prompt, and it names the output columns.",
    )
    parser.add_argument("--train-split", default="train", help="Train split (default: train)")
    parser.add_argument("--eval-split", help="Eval split (default: validation or test if present, else a carve-out of train)")
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="Eval fraction if no eval split (default: 0.1)")
    parser.add_argument("--max-train-samples", type=int, help="Cap training examples (smoke runs)")
    parser.add_argument("--max-eval-samples", type=int, default=2000, help="Cap eval examples (default: 2000)")
    parser.add_argument("--max-text-chars", type=int, default=2000, help="Truncate texts to this many characters (default: 2000)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Prediction batch size (default: 32)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--encoder-lr", type=float, default=1e-5, help="Encoder learning rate (default: 1e-5)")
    parser.add_argument("--task-lr", type=float, default=5e-4, help="Task-head learning rate (default: 5e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument("--skip-zero-shot", action="store_true", help="Skip the zero-shot score of the base model")
    parser.add_argument("--allow-cpu", action="store_true", help="Train without a GPU (slow)")
    parser.add_argument("--output-dir", default="./output", help="Local checkpoint directory (default: ./output)")
    parser.add_argument("--private", action="store_true", help="Make the output model repo private")
    parser.add_argument("--hf-token", help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()
    if not args.label_column:
        args.label_column = ["label"]
    if not args.task_name:
        args.task_name = list(args.label_column)
    if len(args.task_name) != len(args.label_column):
        parser.error("Pass one --task-name per --label-column, in the same order.")
    if len(set(args.task_name)) != len(args.task_name):
        parser.error("Each --task-name must be different.")
    return args


if __name__ == "__main__":
    main(parse_args())
