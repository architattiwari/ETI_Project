import re
import warnings

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
TEXT_COLUMNS = ["source_text", "translated_text", "reference_text"]
REQUIRED_COLUMNS = TEXT_COLUMNS + ["human_score"]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def require_text(text: str) -> str:
    cleaned = clean_text(text)
    if not tokenize(cleaned):
        raise ValueError("text must contain at least one word")
    return cleaned


def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str).map(clean_text)
    df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")

    has_words = df[TEXT_COLUMNS].apply(lambda col: col.map(tokenize).map(bool)).all(axis=1)
    valid = has_words & df["human_score"].notna()
    if not valid.all():
        warnings.warn(f"Dropped {(~valid).sum()} rows with empty text or a non-numeric score")
    df = df[valid].reset_index(drop=True)

    if not df["human_score"].between(0, 1).all():
        raise ValueError("human_score must be between 0 and 1")
    return df


def split_by_source(df: pd.DataFrame, seed: int = SEED):
    """Train/validation/test split (70/15/15 by source sentence) so no source appears in two splits."""
    train_idx, rest_idx = next(
        GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed).split(df, groups=df["source_text"])
    )
    rest = df.iloc[rest_idx]
    val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed).split(rest, groups=rest["source_text"])
    )
    return df.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]
