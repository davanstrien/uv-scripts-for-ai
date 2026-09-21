# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "gliner2[local]==2.0.0",
#     "protobuf",
#     "sentencepiece",
#     "datasets>=4.0.0",
#     "huggingface-hub",
# ]
#
# [tool.hf-jobs]
# flavor = "t4-small"
# timeout = "1h"
# secrets = ["HF_TOKEN"]
# ///
"""
Classify a text column of a Hub dataset with GLiNER2 — zero-shot, or with your fine-tuned model.

GLiNER2 is a ~300M encoder that reads the label names as part of its input. That gives two ways
to use this script:

1. Zero-shot: pass the label names with --labels. No training and no LLM. A t4-small does about
   33 rows/s; cpu-basic works but manages about 1.4 rows/s, so keep CPU for a few hundred rows.
2. Fine-tuned: pass --model with a repo produced by `train-gliner2.py`. The tasks and labels are
   read from the model repo, so no --labels flag is needed.

Zero-shot on HF Jobs:

    hf jobs uv run --flavor t4-small --secrets HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-gliner2.py \\
        fancyzhx/ag_news username/ag-news-gliner2 \\
        --labels World Sports Business "Science and technology" --max-samples 1000

With a fine-tuned model:

    hf jobs uv run --flavor t4-small --secrets HF_TOKEN \\
        https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-gliner2.py \\
        biglam/blbooks-parquet username/blbooks-genre-predictions \\
        --model username/gliner2-blbooks-genre --text-column title

Output: the original columns, plus `predicted_<task>` (a label, or a list of labels for a
multi-label task) and `predicted_<task>_confidence` for every task.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

os.environ.setdefault("TQDM_DISABLE", "1")

import datasets
import torch
from datasets import load_dataset
from gliner2.classification import (
    ClassificationConfig,
    ClassificationSchema,
    Classifier,
)
from huggingface_hub import DatasetCard, hf_hub_download, login
from huggingface_hub.utils import EntryNotFoundError, disable_progress_bars


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

    script_logger = logging.getLogger("classify-gliner2")
    script_logger.setLevel(logging.INFO)
    return script_logger


logger = configure_logging()

SCRIPT_URL = (
    "https://huggingface.co/datasets/uv-scripts/classification/raw/main/classify-gliner2.py"
)
DEFAULT_MODEL = "fastino/gliner2.5-multi-v1"

# Written into the model repo by train-gliner2.py: the tasks and labels the model was trained on.
SCHEMA_FILENAME = "classification_schema.json"

# GLiNER2 puts label names into the model prompt verbatim and rejects these strings.
FORBIDDEN_IN_LABELS = ("(", ")", "[P]", "[L]", "[C]", "[E]", "[R]", "[DESCRIPTION]", "[EXAMPLE]", "[OUTPUT]")


def check_labels(labels: list) -> None:
    for label in labels:
        for token in FORBIDDEN_IN_LABELS:
            if token in label:
                sys.exit(
                    f"Label {label!r} contains {token!r}, which GLiNER2 does not allow in a label "
                    "name. Rephrase it, for example with a dash instead of brackets."
                )
    if len(set(labels)) != len(labels):
        sys.exit(f"--labels contains a duplicate: {labels}")
    if len(labels) < 2:
        sys.exit("Pass at least two --labels.")


def load_trained_tasks(model_id: str):
    """Read the tasks that train-gliner2.py recorded in the model repo, or return None."""
    local_file = os.path.join(model_id, SCHEMA_FILENAME)
    if os.path.isfile(local_file):
        path = local_file
    elif os.path.isdir(model_id):
        return None
    else:
        try:
            path = hf_hub_download(model_id, SCHEMA_FILENAME)
        except EntryNotFoundError:
            return None
    with open(path) as handle:
        return json.load(handle)["tasks"]


def resolve_tasks(args) -> list:
    """Decide which tasks to run: --labels wins, otherwise the model repo's recorded tasks."""
    if args.labels:
        check_labels(args.labels)
        return [{"name": args.task_name, "labels": args.labels, "multi_label": args.multi_label}]

    tasks = load_trained_tasks(args.model)
    if tasks is None:
        sys.exit(
            f"No --labels given, and '{args.model}' has no {SCHEMA_FILENAME}. Pass the label "
            "names with --labels, or use a model trained with train-gliner2.py."
        )
    logger.info("Using the %d task(s) recorded in %s.", len(tasks), args.model)
    return tasks


def build_schema(tasks: list) -> ClassificationSchema:
    schema = ClassificationSchema()
    for task in tasks:
        if task["multi_label"]:
            schema.multi(task["name"], task["labels"])
        else:
            schema.single(task["name"], task["labels"])
    return schema


def label_counts_table(tasks: list, counts_by_task: dict, total: int) -> str:
    lines = ["| Task | Label | Rows | Share |", "|---|---|---|---|"]
    for task in tasks:
        for label, count in counts_by_task[task["name"]].most_common():
            lines.append(f"| `{task['name']}` | {label} | {count} | {count / total:.1%} |")
    return "\n".join(lines)


# The smallest Jobs flavor for each GPU, keyed by a fragment of the GPU's name. "L40" comes
# before "L4" because the first match wins.
GPU_NAME_TO_FLAVOR = {"T4": "t4-small", "A10G": "a10g-small", "L40": "l40sx1", "L4": "l4x1", "A100": "a100-large"}


def jobs_flavor() -> str:
    """Return the Jobs hardware flavor, or "" when it is not known.

    The docs say ACCELERATOR holds the flavor ("a10g-small"). On the t4-small and a10g-small
    jobs that tested this script it held a bare "gpu", which is not a valid --flavor. So use
    ACCELERATOR when it looks like a flavor, and otherwise name the smallest flavor that has
    this GPU. A larger flavor of the same GPU reproduces the same result.
    """
    hardware = os.environ.get("ACCELERATOR") or ""
    looks_like_flavor = "-" in hardware or any(character.isdigit() for character in hardware)
    if looks_like_flavor:
        return hardware
    if not torch.cuda.is_available():
        return ""
    gpu_name = torch.cuda.get_device_name(0)
    for fragment, flavor in GPU_NAME_TO_FLAVOR.items():
        if fragment in gpu_name:
            return flavor
    return ""


def build_reproduce_command(args) -> str:
    flavor = jobs_flavor() or "t4-small"
    parts = [
        f"hf jobs uv run --flavor {flavor} --secrets HF_TOKEN \\",
        f"  {SCRIPT_URL} \\",
        f"  {args.input_dataset} {args.output_dataset}",
    ]
    flags = []
    if args.model != DEFAULT_MODEL:
        flags.append(f"--model {args.model}")
    if args.labels:
        quoted = " ".join(f'"{label}"' if " " in label else label for label in args.labels)
        flags.append(f"--labels {quoted}")
        if args.task_name != "label":
            flags.append(f"--task-name {args.task_name}")
        if args.multi_label:
            flags.append("--multi-label")
    if args.dataset_config:
        flags.append(f"--dataset-config {args.dataset_config}")
    if args.text_column != "text":
        flags.append(f"--text-column {args.text_column}")
    if args.split != "train":
        flags.append(f"--split {args.split}")
    if args.max_samples:
        flags.append(f"--max-samples {args.max_samples}")
    if args.max_text_chars != 2000:
        flags.append(f"--max-text-chars {args.max_text_chars}")
    if args.private:
        flags.append("--private")
    if flags:
        parts[-1] += " \\"
        parts.append("  " + " ".join(flags))
    return "\n".join(parts)


def build_card(args, tasks, counts_by_task, total, seconds, zero_shot: bool) -> str:
    """Dataset card with the canonical uv-scripts provenance stamp."""
    on_jobs = os.environ.get("JOB_ID") is not None
    hardware = jobs_flavor()
    if on_jobs:
        origin = "Produced on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)"
        if hardware:
            origin += f" (`{hardware}`)"
    else:
        origin = "Generated"

    tags = ["uv-script", "gliner2", "text-classification"]
    if on_jobs:
        tags.append("hf-jobs")
    tag_lines = "\n".join(f"- {tag}" for tag in tags)

    if zero_shot:
        how = (
            "The model was used **zero-shot**: it was given only the label names and has never "
            "seen labelled examples of this task. Treat the labels as a first pass to review, "
            "not as ground truth."
        )
    else:
        how = (
            "The model was fine-tuned for these tasks. Its model card reports the held-out "
            "scores. They apply only where this data resembles the training data."
        )

    column_lines = []
    for task in tasks:
        kind = "list of labels" if task["multi_label"] else "one label"
        column_lines.append(f"- `predicted_{task['name']}`: {kind} from {task['labels']}")
        note = " (empty when no label was selected)" if task["multi_label"] else ""
        column_lines.append(f"- `predicted_{task['name']}_confidence`: model confidence in [0, 1]{note}")
    column_block = "\n".join(column_lines)

    return f"""---
tags:
{tag_lines}
---

# {args.output_dataset.split("/")[-1]}

[`{args.input_dataset}`](https://huggingface.co/datasets/{args.input_dataset}) (split `{args.split}`,
{total} rows) with the `{args.text_column}` column classified by
[`{args.model}`](https://huggingface.co/{args.model}), a [GLiNER2](https://github.com/fastino-ai/GLiNER2) model.

{how}

## Added columns

{column_block}

Texts were truncated to {args.max_text_chars} characters before classification.
The confidence is not calibrated. Check it against a labelled sample before you use it as a filter.

## Label distribution

{label_counts_table(tasks, counts_by_task, total)}

Classified {total} rows in {round(seconds)} seconds ({total / max(seconds, 1e-9):.0f} rows/s).

## Reproduction

{origin} with the [`classify-gliner2.py`]({SCRIPT_URL}) recipe from [uv-scripts](https://huggingface.co/uv-scripts). Run it yourself:

```bash
{build_reproduce_command(args)}
```
"""


def main(args) -> None:
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("No HF token. Pass --hf-token or run with --secrets HF_TOKEN.")
    login(token=token)

    tasks = resolve_tasks(args)
    for task in tasks:
        logger.info("Task '%s': %s", task["name"], task["labels"])

    logger.info("Loading %s (split %s)", args.input_dataset, args.split)
    dataset = load_dataset(args.input_dataset, args.dataset_config, split=args.split)
    if args.text_column not in dataset.column_names:
        sys.exit(f"Text column '{args.text_column}' not found. Columns are: {dataset.column_names}.")
    for task in tasks:
        for column in (f"predicted_{task['name']}", f"predicted_{task['name']}_confidence"):
            if column in dataset.column_names:
                sys.exit(f"The dataset already has a '{column}' column. Pass a different --task-name.")
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
    logger.info("Rows to classify: %d", len(dataset))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No GPU found; classifying on CPU. Expect about 1-2 rows per second on cpu-basic.")
    # from_pretrained(device=...) does not move the weights in gliner2 2.0.0; .to() does.
    classifier = Classifier.from_pretrained(args.model).to(device=device).eval()
    schema = build_schema(tasks)
    config = ClassificationConfig(batch_size=args.batch_size)

    counts_by_task = {task["name"]: Counter() for task in tasks}
    empty_texts = 0

    def classify_batch(batch: dict) -> dict:
        nonlocal empty_texts
        texts = []
        for value in batch[args.text_column]:
            text = "" if value is None else str(value)
            if not text.strip():
                empty_texts += 1
                # The model needs some input; a missing text gets a prediction we then blank out.
                text = "-"
            texts.append(text[: args.max_text_chars])

        results = classifier.batch_classify(texts, schema, config=config)

        new_columns = {}
        for task in tasks:
            name = task["name"]
            predictions = []
            confidences = []
            for value, result in zip(batch[args.text_column], results):
                if value is None or not str(value).strip():
                    predictions.append([] if task["multi_label"] else None)
                    confidences.append(None)
                    continue
                if task["multi_label"]:
                    labels = list(result.selected(name))
                    predictions.append(labels)
                    counts_by_task[name].update(labels or ["(none selected)"])
                else:
                    label = result.value(name)
                    predictions.append(label)
                    counts_by_task[name][label] += 1
                confidence = result.confidence(name)
                confidences.append(None if confidence is None else float(confidence))
            new_columns[f"predicted_{name}"] = predictions
            new_columns[f"predicted_{name}_confidence"] = confidences
        return new_columns

    started = time.time()
    # One map batch holds several model batches, so progress is logged at a useful rate.
    dataset = dataset.map(classify_batch, batched=True, batch_size=args.batch_size * 8)
    seconds = time.time() - started
    logger.info("Classified %d rows in %.0f seconds.", len(dataset), seconds)
    if empty_texts:
        logger.warning("%d rows had no text and were left unlabelled.", empty_texts)
    for task in tasks:
        logger.info("Task '%s' distribution: %s", task["name"], dict(counts_by_task[task["name"]].most_common(10)))

    dataset.push_to_hub(args.output_dataset, private=args.private)
    card = build_card(args, tasks, counts_by_task, len(dataset), seconds, zero_shot=bool(args.labels))
    DatasetCard(card).push_to_hub(args.output_dataset, repo_type="dataset")
    logger.info("Pushed to https://huggingface.co/datasets/%s", args.output_dataset)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dataset", help="Input dataset ID")
    parser.add_argument("output_dataset", help="Output dataset ID (username/dataset-name)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"GLiNER2 model: a base checkpoint for zero-shot, or a train-gliner2.py output (default: {DEFAULT_MODEL})")
    parser.add_argument("--labels", nargs="+", help="Label names for zero-shot classification. Overrides the tasks recorded in the model repo.")
    parser.add_argument("--task-name", default="label", help="Name of the --labels task; sets the output column names (default: label)")
    parser.add_argument("--multi-label", action="store_true", help="With --labels: allow several labels, or none, per text")
    parser.add_argument("--dataset-config", help="Dataset config name")
    parser.add_argument("--text-column", default="text", help="Text column (default: text)")
    parser.add_argument("--split", default="train", help="Split to classify (default: train)")
    parser.add_argument("--max-samples", type=int, help="Classify only the first N rows")
    parser.add_argument("--max-text-chars", type=int, default=2000, help="Truncate texts to this many characters (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=32, help="Model batch size (default: 32)")
    parser.add_argument("--private", action="store_true", help="Make the output dataset private")
    parser.add_argument("--hf-token", help="HF token (or set HF_TOKEN)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
