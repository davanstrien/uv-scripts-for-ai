# SetFit script: findings and behaviour details

Notes behind `train-setfit.py`. The README has what you need to run it; this file records the
longer findings and the full behaviour details.

## Compare more than the majority baseline

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

## Real-world data: a worked failure

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

## Many classes: watch the pair count

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

## Choosing another body or longer context

`--body-model` accepts a Sentence Transformer checkpoint. Set `--max-seq-length` within that
model's supported context window; increasing it cannot extend a model's native limit or restore
text already shortened during dataset preparation. Longer sequences can need a smaller
`--batch-size` or more GPU memory. The recipe measures training cost on the selected hardware.

Follow the body's task-prefix instructions when preparing inputs. For example,
[`nomic-ai/modernbert-embed-base`](https://huggingface.co/nomic-ai/modernbert-embed-base)
uses Nomic's task prefixes: classification inputs should begin with `classification: `.
Include the same prefix during training, evaluation and inference. The recipe does not add it
automatically. Retain the original texts and the preprocessing details with the model.

## Behaviour details

- **Evaluation split** follows the same precedence as `train-classifier.py`: `--eval-split` if given, else `validation`, else `test`, else a stratified carve-out of `--eval-fraction` from train.
- **Metrics match `train-classifier.py`** (accuracy + macro F1). Match evaluation rows and preprocessing when comparing runs.
- **`--num-samples`** sets labelled examples per class (default 8). **`--sampling-strategy`** controls contrastive pairing: `oversampling` (default), `undersampling`, `unique`.
- **Every run reports a majority baseline.** The run warns when accuracy fails to beat it, or the gain is below five percentage points. That fixed threshold is a review heuristic, not a measured noise level or significance test.
- **It estimates training time before starting.** The script times forward/backward passes on actual texts and hardware, then refuses training projected above `--max-minutes` (default 60). Setup, evaluation and upload take additional time. A measurement error can skip this guard; use Jobs `--timeout` to enforce a wall-clock limit.
- **Rows with missing or blank labels or texts are dropped**, with a count. Missing labels include `ClassLabel`'s `-1` sentinel and numeric NaN; plain integer `-1` remains a valid class. Splits with no usable labelled text, fewer than two observed training classes, and missing or non-string text columns exit before model loading.
- **`--private` verifies the output repository is private before training.** If the destination already exists publicly, choose a new repo or change its visibility first.
