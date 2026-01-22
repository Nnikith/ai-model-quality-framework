from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


@dataclass
class V1Artifacts:
    model_path: str
    vectorizer_path: str
    metrics_path: str


def load_processed_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame, split: str) -> Tuple[list[str], list[int]]:
    sub = df[df["split"] == split]
    return sub["text"].tolist(), sub["label"].astype(int).tolist()


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    # These require both classes present; guard for edge cases
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = None
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["pr_auc"] = None

    return metrics


def train_v1(
    dataset_path: str | Path = "data/processed/isot.parquet",
    eval_config_path: str | Path = "configs/eval.yaml",
    out_dir: str | Path = "artifacts/models/v1",
) -> V1Artifacts:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = yaml.safe_load(Path(eval_config_path).read_text())
    threshold = float(eval_cfg["thresholding"]["default_threshold"])

    df = load_processed_dataset(dataset_path)

    x_train, y_train = split_xy(df, "train")
    x_val, y_val = split_xy(df, "val")
    x_test, y_test = split_xy(df, "test")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )

    Xtr = vectorizer.fit_transform(x_train)
    Xva = vectorizer.transform(x_val)
    Xte = vectorizer.transform(x_test)

    model = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(Xtr, y_train)

    val_prob = model.predict_proba(Xva)[:, 1]
    test_prob = model.predict_proba(Xte)[:, 1]

    val_metrics = compute_metrics(y_val, val_prob, threshold=threshold)
    test_metrics = compute_metrics(y_test, test_prob, threshold=threshold)

    report = {
        "model_version": "v1",
        "dataset_path": str(dataset_path),
        "threshold": threshold,
        "val": val_metrics,
        "test": test_metrics,
    }

    model_path = out_dir / "model.joblib"
    vec_path = out_dir / "vectorizer.joblib"
    metrics_path = Path("artifacts/reports/eval_metrics_v1.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    pd.Series(report).to_json(metrics_path, indent=2)

    return V1Artifacts(
        model_path=str(model_path),
        vectorizer_path=str(vec_path),
        metrics_path=str(metrics_path),
    )
