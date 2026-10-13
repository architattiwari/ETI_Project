import argparse
from pathlib import Path

from app.models import artifacts_dir, train
from app.preprocessing import load_dataset

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "sample_translations.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ETI-Baseline and ETI-5 on a CSV dataset.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="CSV with source_text, translated_text, reference_text, human_score")
    args = parser.parse_args()

    train_rows, val_rows, test_rows = train(load_dataset(args.data))
    print(f"Trained on {train_rows} rows, validated on {val_rows}, held out {test_rows} for testing.")
    print(f"Artifacts saved to {artifacts_dir()}. Next: python -m scripts.evaluate")


if __name__ == "__main__":
    main()
