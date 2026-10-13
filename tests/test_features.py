import pytest

from app.features import FEATURE_NAMES, REFERENCE_BASED, edit_distance, extract_features, ter, to_array, wer


def test_edit_distance_counts_substitution_insertion_deletion():
    assert edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    assert edit_distance([], ["a", "b"]) == 2


def test_wer_known_examples():
    assert wer("the cat sat", "the cat sat") == 0
    assert wer("the cat sat", "the cat sat on the mat") == 0.5


def test_ter_known_examples():
    assert ter("the cat sat", "the cat sat") == 0
    assert ter("the cat sat", "the cat sat on the mat") == pytest.approx(0.5)


def test_metrics_ignore_case_and_punctuation():
    assert wer("The cat, sat!", "the cat sat") == 0


def test_features_without_reference_skip_reference_metrics():
    features = extract_features("one two three four", "uno dos")
    assert features["length_ratio"] == 0.5
    assert not set(REFERENCE_BASED) & set(features)


def test_features_with_reference_cover_every_feature_name():
    features = extract_features("the cat", "el gato", "el gato negro")
    assert set(features) == set(FEATURE_NAMES)
    assert features["reference_length_diff"] == -1


def test_unique_token_ratio_detects_repetition():
    assert extract_features("a b c d", "hola hola hola hola")["unique_token_ratio"] == 0.25


@pytest.mark.parametrize("source, translation, reference", [("", "x", None), ("x", "...", None), ("x", "y", "?!")])
def test_features_reject_text_without_words(source, translation, reference):
    with pytest.raises(ValueError):
        extract_features(source, translation, reference)


def test_to_array_rejects_non_finite_values():
    row = dict.fromkeys(FEATURE_NAMES, 1.0) | {"wer": float("nan")}
    with pytest.raises(ValueError):
        to_array([row])
