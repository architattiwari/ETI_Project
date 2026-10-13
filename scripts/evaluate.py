import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.features import dataset_features
from app.models import ModelsNotTrainedError, artifacts_dir, load_artifacts, predict
from app.preprocessing import load_dataset

SMALL_TEST_SET = 200


def main() -> None:
    try:
        scaler, models = load_artifacts()
    except ModelsNotTrainedError as error:
        sys.exit(str(error))
    test_path = artifacts_dir() / "test_set.csv"
    if not test_path.exists():
        sys.exit(f"{test_path} not found. Run python -m scripts.train first.")

    test_df = load_dataset(test_path)
    predictions = predict(dataset_features(test_df), scaler, models)
    truth = test_df["human_score"].to_numpy()

    lines = [
        "# ETI evaluation report",
        "",
        f"Held-out test rows: {len(test_df)} (split by source sentence; target: human_score, 0-1)",
        "",
        "| Model | MAE | MSE | RMSE |",
        "|---|---|---|---|",
    ]
    for name, predicted in predictions.items():
        mse = mean_squared_error(truth, predicted)
        lines.append(f"| {name} | {mean_absolute_error(truth, predicted):.4f} | {mse:.4f} | {np.sqrt(mse):.4f} |")
    if len(test_df) < SMALL_TEST_SET:
        lines += ["", f"**Warning:** only {len(test_df)} test rows. These numbers are too noisy to show that one model is better."]

    report = "\n".join(lines)
    (artifacts_dir() / "evaluation_report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
