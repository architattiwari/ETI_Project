import numpy as np
from sacrebleu.metrics import TER

from app.preprocessing import tokenize

REFERENCE_FREE = ["source_tokens", "translation_tokens", "length_ratio", "avg_word_length", "unique_token_ratio"]
REFERENCE_BASED = ["wer", "ter", "reference_length_diff"]
FEATURE_NAMES = REFERENCE_FREE + REFERENCE_BASED

_ter = TER()


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, word_a in enumerate(a, 1):
        current = [i]
        for j, word_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word_a != word_b)))
        previous = current
    return previous[-1]


def wer(hypothesis: str, reference: str) -> float:
    reference_tokens = tokenize(reference)
    return edit_distance(tokenize(hypothesis), reference_tokens) / len(reference_tokens)


def ter(hypothesis: str, reference: str) -> float:
    """Translation Edit Rate (edits + block shifts / reference length) from sacrebleu, as a fraction."""
    hypothesis, reference = " ".join(tokenize(hypothesis)), " ".join(tokenize(reference))
    return _ter.sentence_score(hypothesis, [reference]).score / 100


def extract_features(source: str, translation: str, reference: str | None = None) -> dict[str, float]:
    source_tokens, translation_tokens = tokenize(source), tokenize(translation)
    if not source_tokens or not translation_tokens:
        raise ValueError("source and translation must contain at least one word")

    features = {
        "source_tokens": len(source_tokens),
        "translation_tokens": len(translation_tokens),
        "length_ratio": len(translation_tokens) / len(source_tokens),
        "avg_word_length": sum(map(len, translation_tokens)) / len(translation_tokens),
        "unique_token_ratio": len(set(translation_tokens)) / len(translation_tokens),
    }
    if reference is not None:
        reference_tokens = tokenize(reference)
        if not reference_tokens:
            raise ValueError("reference must contain at least one word")
        features |= {
            "wer": wer(translation, reference),
            "ter": ter(translation, reference),
            "reference_length_diff": len(translation_tokens) - len(reference_tokens),
        }
    return features


def dataset_features(df) -> list[dict[str, float]]:
    return [
        extract_features(source, translation, reference)
        for source, translation, reference in zip(df["source_text"], df["translated_text"], df["reference_text"])
    ]


def to_array(rows: list[dict[str, float]]) -> np.ndarray:
    array = np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Features contain NaN or infinite values")
    return array
