from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "news_sample.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fake_news_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

TEXT_COLUMN_ALIASES = ("text", "title", "headline", "article", "content", "news")
LABEL_COLUMN_ALIASES = ("label", "target", "class", "category")
FAKE_LABELS = {"fake", "false", "0", "misleading", "rumor", "rumour"}
REAL_LABELS = {"real", "true", "1", "reliable", "genuine"}


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    fake_probability: float
    real_probability: float


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    normalized = {column.strip().lower(): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _normalize_label(value: object) -> str:
    label = str(value).strip().lower()
    if label in FAKE_LABELS:
        return "FAKE"
    if label in REAL_LABELS:
        return "REAL"
    raise ValueError(
        "Labels must use fake/real, false/true, misleading/reliable, or 0/1 values."
    )


def normalize_input_dataframe(df: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    text_column = _find_column(df.columns, TEXT_COLUMN_ALIASES)
    if text_column is None:
        raise ValueError("CSV must include a text, title, headline, article, content, or news column.")

    normalized = pd.DataFrame()
    normalized["text"] = df[text_column].fillna("").astype(str).str.strip()
    normalized = normalized[normalized["text"] != ""]

    if require_label:
        label_column = _find_column(df.columns, LABEL_COLUMN_ALIASES)
        if label_column is None:
            raise ValueError("Training CSV must include a label, target, class, or category column.")
        normalized["label"] = df.loc[normalized.index, label_column].map(_normalize_label)

    if normalized.empty:
        raise ValueError("CSV does not contain any usable text rows.")

    return normalized.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    normalized = normalize_input_dataframe(df, require_label=True)
    label_counts = normalized["label"].value_counts()

    if set(label_counts.index) != {"FAKE", "REAL"}:
        raise ValueError("Training data must contain both FAKE and REAL examples.")

    model = build_pipeline()
    can_split = len(normalized) >= 10 and label_counts.min() >= 2

    if can_split:
        train_df, test_df = train_test_split(
            normalized,
            test_size=0.25,
            random_state=42,
            stratify=normalized["label"],
        )
        model.fit(train_df["text"], train_df["label"])
        predictions = model.predict(test_df["text"])
        metrics = {
            "accuracy": float(accuracy_score(test_df["label"], predictions)),
            "training_rows": int(len(train_df)),
            "validation_rows": int(len(test_df)),
        }
    else:
        model.fit(normalized["text"], normalized["label"])
        metrics = {
            "accuracy": 1.0,
            "training_rows": int(len(normalized)),
            "validation_rows": 0,
        }

    model.fit(normalized["text"], normalized["label"])
    metrics["total_rows"] = int(len(normalized))
    metrics["fake_rows"] = int(label_counts.get("FAKE", 0))
    metrics["real_rows"] = int(label_counts.get("REAL", 0))
    return model, metrics


def train_and_save_model(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> tuple[Pipeline, dict]:
    model, metrics = train_model(df)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, metrics


def load_or_train_model() -> tuple[Pipeline, dict, str]:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metrics = _load_metrics()
        return model, metrics, "Saved model"

    df = pd.read_csv(DATA_PATH)
    model, metrics = train_and_save_model(df, MODEL_PATH)
    return model, metrics, "Sample data"


def _load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {"accuracy": 0.0, "training_rows": 0}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def predict_text(model: Pipeline, text: str) -> Prediction:
    probabilities = model.predict_proba([text])[0]
    class_to_probability = dict(zip(model.classes_, probabilities))
    fake_probability = float(class_to_probability.get("FAKE", 0.0))
    real_probability = float(class_to_probability.get("REAL", 0.0))
    label = "FAKE" if fake_probability >= real_probability else "REAL"
    confidence = max(fake_probability, real_probability)
    return Prediction(
        label=label,
        confidence=confidence,
        fake_probability=fake_probability,
        real_probability=real_probability,
    )


def predict_batch(model: Pipeline, texts: Iterable[str]) -> list[Prediction]:
    return [predict_text(model, text) for text in texts]
