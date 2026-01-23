from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Reuse v1 helpers to keep behavior consistent (splits, metrics, safety transforms)
from fakenews.models.train_v1 import (
    load_processed_dataset,
    split_xy,
    _safe_transform,
    compute_metrics,
)


@dataclass
class V2Artifacts:
    model_path: str
    vectorizer_path: str
    metrics_path: str


def train_v2(
    dataset_path: str | Path = "data/processed/isot.parquet",
    model_config_path: str | Path = "configs/model_v2.yaml",
    eval_config_path: str | Path = "configs/eval.yaml",
    out_dir: str | Path = "artifacts/models/v2",
    report_path: str | Path = "artifacts/reports/eval_metrics_v2.json",
) -> V2Artifacts:
    """
    Train Model v2: char n-gram TF-IDF + Logistic Regression.

    Saves:
      - artifacts/models/v2/model.joblib
      - artifacts/models/v2/vectorizer.joblib
      - artifacts/reports/eval_metrics_v2.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    model_cfg = yaml.safe_load(Path(model_config_path).read_text())
    eval_cfg = yaml.safe_load(Path(eval_config_path).read_text())
    threshold = float(eval_cfg["thresholding"]["default_threshold"])

    df = load_processed_dataset(dataset_path)

    x_train, y_train = split_xy(df, "train")
    x_val, y_val = split_xy(df, "val")
    x_test, y_test = split_xy(df, "test")

    if len(x_train) == 0:
        raise ValueError("Training split is empty; cannot train model.")

    # ---- Small dataset fallbacks (same philosophy as v1) ----
    # In CI we sometimes only have train/test or train only.
    if len(x_val) == 0 and len(x_test) > 0:
        x_val, y_val = x_test, y_test
    if len(x_test) == 0 and len(x_val) > 0:
        x_test, y_test = x_val, y_val
    if len(x_val) == 0 and len(x_test) == 0:
        # Last resort: carve out a tiny holdout from train
        if len(x_train) < 2:
            raise ValueError("Not enough samples to create a holdout split.")
        x_val, y_val = x_train[:1], y_train[:1]
        x_test, y_test = x_train[:1], y_train[:1]
        x_train, y_train = x_train[1:], y_train[1:]

    # ---- Vectorizer: char n-gram TF-IDF ----
    vec_cfg = model_cfg["vectorizer"]

    # min_df: ignore features that appear in fewer than N documents.
    # For tiny CI datasets, use min_df_ci to avoid empty vocab.
    min_df_default = int(vec_cfg.get("min_df_default", 2))
    min_df_ci = int(vec_cfg.get("min_df_ci", 1))
    min_df = min_df_default if len(x_train) >= 10 else min_df_ci

    vectorizer = TfidfVectorizer(
        analyzer=vec_cfg.get("analyzer", "char_wb"),
        ngram_range=tuple(vec_cfg.get("ngram_range", [3, 5])),
        max_features=int(vec_cfg.get("max_features", 100000)),
        min_df=min_df,
        strip_accents=vec_cfg.get("strip_accents", "unicode"),
        lowercase=bool(vec_cfg.get("lowercase", True)),
    )

    Xtr = vectorizer.fit_transform(x_train)
    Xva = _safe_transform(vectorizer, x_val)
    Xte = _safe_transform(vectorizer, x_test)

    # ---- Classifier: Logistic Regression ----
    clf_cfg = model_cfg["classifier"]
    model = LogisticRegression(
        max_iter=int(clf_cfg.get("max_iter", 3000)),
        class_weight=clf_cfg.get("class_weight", "balanced"),
        random_state=int(clf_cfg.get("random_state", 42)),
    )
    model.fit(Xtr, y_train)

    report: Dict[str, Any] = {"model_version": "v2"}

    if Xva is None or len(y_val) == 0:
        report["val"] = None
    else:
        val_prob = model.predict_proba(Xva)[:, 1]
        report["val"] = compute_metrics(y_val, val_prob, threshold=threshold)

    if Xte is None or len(y_test) == 0:
        report["test"] = None
    else:
        test_prob = model.predict_proba(Xte)[:, 1]
        report["test"] = compute_metrics(y_test, test_prob, threshold=threshold)

    model_path = out_dir / "model.joblib"
    vec_path = out_dir / "vectorizer.joblib"
    metrics_path = Path(report_path)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    # Write JSON metrics in a simple, stable way
    pd.Series(report).to_json(metrics_path)

    return V2Artifacts(
        model_path=str(model_path),
        vectorizer_path=str(vec_path),
        metrics_path=str(metrics_path),
    )
