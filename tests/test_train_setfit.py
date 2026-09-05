"""Data preflight regressions; no Hub access or model downloads required.

Run with the recipe dependencies and pytest installed:
    python -m pytest tests/test_train_setfit.py
"""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
from datasets import ClassLabel, Dataset, Features, Value

SCRIPT = Path(__file__).resolve().parents[1] / "classification" / "train-setfit.py"
spec = importlib.util.spec_from_file_location("train_setfit", SCRIPT)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def typed_dataset(labels):
    return Dataset.from_dict(
        {"text": [f"document {i}" for i in range(len(labels))], "label": labels},
        features=Features({"text": Value("string"), "label": ClassLabel(names=["a", "b", "c"])}),
    )


def test_missing_classlabel_is_dropped_without_remapping():
    cleaned = recipe.prepare_split(typed_dataset([0, -1, 2, None]), "text", "label", "train")
    assert list(cleaned["label"]) == [0, 2]
    assert recipe.resolve_label_names(cleaned, "label") == ["a", "b", "c"]


def test_plain_negative_integer_is_a_real_class():
    data = Dataset.from_dict({"text": ["a", "b", "c"], "label": [-1, 0, 1]})
    assert list(recipe.prepare_split(data, "text", "label", "train")["label"]) == ["-1", "0", "1"]


def test_nan_is_removed_before_string_cast():
    data = Dataset.from_dict({"text": ["a", "b", "c"], "label": [1.0, float("nan"), 2.0]})
    cleaned = recipe.prepare_split(data, "text", "label", "train")
    assert len(cleaned) == 2
    assert "nan" not in cleaned["label"]


@pytest.mark.parametrize("labels", [[], [None, None], ["", " "]])
def test_empty_or_unlabelled_split_is_rejected(labels):
    data = Dataset.from_dict({"text": ["document"] * len(labels), "label": labels})
    with pytest.raises(SystemExit, match="empty|No labelled rows"):
        recipe.prepare_split(data, "text", "label", "eval")


@pytest.mark.parametrize("text", [None, "", " ", 12])
def test_invalid_text_is_rejected_before_training(text):
    data = Dataset.from_dict({"text": [text], "label": ["a"]})
    with pytest.raises(SystemExit, match="Clean the text column"):
        recipe.prepare_split(data, "text", "label", "eval")


@pytest.mark.parametrize("column", ["text", "label"])
def test_missing_columns_are_actionable(column):
    data = Dataset.from_dict({"text": ["document"], "label": ["a"]}).remove_columns(column)
    with pytest.raises(SystemExit, match=f"--{column}-column"):
        recipe.prepare_split(data, "text", "label", "eval")


def test_multilabel_eval_is_rejected():
    data = Dataset.from_dict({"text": ["document"], "label": [["a", "b"]]})
    with pytest.raises(SystemExit, match="multi-label"):
        recipe.prepare_split(data, "text", "label", "eval")


@pytest.mark.parametrize("labels", [["a", "a", "b", "b"], [-1, -1, 2, 2]])
def test_plain_labels_get_stratified_disjoint_carve(monkeypatch, labels):
    data = Dataset.from_dict({"text": ["one", "two", "three", "four"], "label": labels})
    monkeypatch.setattr(recipe, "load_dataset", lambda *args, **kwargs: data)
    train, evaluation = recipe.split_train_eval("fixture", None, "train", None, 0.5, 2, "label")
    assert len(set(train["label"])) == len(set(evaluation["label"])) == 2
    assert set(train["text"]).isdisjoint(evaluation["text"])
    assert set(train.features["label"].names) == {str(label) for label in labels}


def test_carve_drops_missing_classlabel_before_stratifying(monkeypatch):
    data = typed_dataset([0, 0, 2, 2, -1, None])
    monkeypatch.setattr(recipe, "load_dataset", lambda *args, **kwargs: data)
    train, evaluation = recipe.split_train_eval("fixture", None, "train", None, 0.5, 2, "label")
    assert len(train) == len(evaluation) == 2
    assert set(train["label"]) == set(evaluation["label"]) == {0, 2}


def test_one_observed_class_stops_before_model_loading(monkeypatch):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "fixture", "user/model", "--hf-token", "fake"])
    monkeypatch.setattr(recipe, "login", Mock())
    monkeypatch.setattr(recipe, "HfApi", Mock())
    monkeypatch.setattr(recipe, "pick_eval_split", lambda *args: "test")
    monkeypatch.setattr(recipe, "split_train_eval", lambda *args: (typed_dataset([2, 2]), typed_dataset([0, 2])))
    load_model = Mock()
    monkeypatch.setattr(recipe.SetFitModel, "from_pretrained", load_model)
    with pytest.raises(SystemExit, match="two observed classes"):
        recipe.main(recipe.parse_args())
    load_model.assert_not_called()


def test_evaluation_decodes_its_own_classlabel_table():
    evaluation = typed_dataset([2, 0])
    model = Mock()
    model.predict.return_value = ["c", "a"]
    assert recipe.evaluate(model, evaluation, "text", "label")["accuracy"] == 1.0


def test_help_renders_slice_example(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as result:
        recipe.parse_args()
    assert result.value.code == 0
    assert "train[:10%]" in capsys.readouterr().out
