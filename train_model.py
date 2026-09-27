import argparse
from pathlib import Path

import pandas as pd

from src.detector import DATA_PATH, MODEL_PATH, train_and_save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fake news detection model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="CSV file with text and label columns.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Output path for the trained joblib model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    _, metrics = train_and_save_model(df, args.model)
    print("Model trained successfully")
    print(f"Rows: {metrics['total_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Model: {args.model}")


if __name__ == "__main__":
    main()
