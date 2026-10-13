import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from app.features import FEATURE_NAMES, dataset_features, to_array
from app.preprocessing import SEED, split_by_source

HIDDEN_UNITS = {"ETI-Baseline": [8], "ETI-5": [64, 32, 16, 8]}
MIN_ROWS, MIN_SOURCES = 30, 10
TRAIN_HINT = "Train the models first: python -m scripts.train"


class ModelsNotTrainedError(RuntimeError):
    pass


def artifacts_dir() -> Path:
    default = Path(__file__).resolve().parent.parent / "artifacts"
    return Path(os.environ.get("ETI_ARTIFACTS_DIR", default))


def build_model(hidden_units: list[int]):
    # TensorFlow is imported lazily so feature extraction and most tests don't pay its start-up cost.
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(len(FEATURE_NAMES),)),
            *[tf.keras.layers.Dense(units, activation="relu") for units in hidden_units],
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train(df, directory: Path | None = None) -> tuple[int, int, int]:
    if len(df) < MIN_ROWS or df["source_text"].nunique() < MIN_SOURCES:
        raise ValueError(
            f"Need at least {MIN_ROWS} valid rows and {MIN_SOURCES} distinct source sentences "
            f"to train; got {len(df)} rows and {df['source_text'].nunique()} sources."
        )
    import tensorflow as tf

    directory = directory or artifacts_dir()
    directory.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = split_by_source(df)
    x_train, x_val = (to_array(dataset_features(part)) for part in (train_df, val_df))
    scaler = StandardScaler().fit(x_train)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    for name, units in HIDDEN_UNITS.items():
        model = build_model(units)
        model.fit(
            scaler.transform(x_train),
            train_df["human_score"].to_numpy(),
            validation_data=(scaler.transform(x_val), val_df["human_score"].to_numpy()),
            epochs=300,
            batch_size=8,
            verbose=0,
            callbacks=[early_stop],
        )
        model.save(directory / f"{name}.keras")

    joblib.dump(scaler, directory / "scaler.joblib")
    test_df.to_csv(directory / "test_set.csv", index=False)
    return len(train_df), len(val_df), len(test_df)


def load_artifacts(directory: Path | None = None):
    directory = directory or artifacts_dir()
    model_paths = {name: directory / f"{name}.keras" for name in HIDDEN_UNITS}
    scaler_path = directory / "scaler.joblib"
    if not scaler_path.exists() or not all(path.exists() for path in model_paths.values()):
        raise ModelsNotTrainedError(f"No trained models found in {directory}. {TRAIN_HINT}")

    import tensorflow as tf

    return joblib.load(scaler_path), {name: tf.keras.models.load_model(path) for name, path in model_paths.items()}


def predict(rows: list[dict[str, float]], scaler, models) -> dict[str, np.ndarray]:
    x = scaler.transform(to_array(rows))
    return {name: model.predict(x, verbose=0)[:, 0] for name, model in models.items()}
