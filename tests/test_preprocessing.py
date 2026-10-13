import pandas as pd
import pytest

from app.preprocessing import clean_text, load_dataset, require_text, split_by_source, tokenize


def write_csv(tmp_path, rows, columns=("source_text", "translated_text", "reference_text", "human_score")):
    path = tmp_path / "data.csv"
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)
    return path


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello \n  world  ") == "hello world"


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Hello,  World!") == ["hello", "world"]


@pytest.mark.parametrize("text", ["", "   ", "?!..."])
def test_require_text_rejects_text_without_words(text):
    with pytest.raises(ValueError):
        require_text(text)


def test_load_dataset_requires_all_columns(tmp_path):
    path = write_csv(tmp_path, [["a", "b", 0.5]], columns=("source_text", "translated_text", "human_score"))
    with pytest.raises(ValueError, match="reference_text"):
        load_dataset(path)


def test_load_dataset_drops_invalid_rows(tmp_path):
    path = write_csv(tmp_path, [["a b", "c d", "c d", 0.9], ["", "c d", "c d", 0.5], ["a", "b", "c", "bad"]])
    with pytest.warns(UserWarning, match="Dropped 2"):
        df = load_dataset(path)
    assert len(df) == 1


def test_load_dataset_rejects_scores_outside_range(tmp_path):
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_dataset(write_csv(tmp_path, [["a", "b", "c", 7]]))


def test_split_keeps_each_source_in_one_split():
    df = pd.DataFrame(
        {"source_text": [f"source {i // 2}" for i in range(40)], "translated_text": "t", "reference_text": "r", "human_score": 0.5}
    )
    parts = split_by_source(df)
    sources = [set(part["source_text"]) for part in parts]
    assert sources[0].isdisjoint(sources[1]) and sources[0].isdisjoint(sources[2]) and sources[1].isdisjoint(sources[2])
    assert sum(len(part) for part in parts) == len(df)
