from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import json

import numpy as np
import joblib


@dataclass
class PredictionDriftResult:
    passed: bool
    warnings: List[str]
    stats: Dict[str, Any]


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """
    Population Stability Index for distributions.
    Higher = more drift. Rules of thumb:
      <0.1 no drift, 0.1-0.25 moderate, >0.25 significant
    """
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.quantile(expected, quantiles)
    # Make edges strictly increasing
    breakpoints[0] = 0.0
    breakpoints[-1] = 1.0

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_dist = exp_counts / max(1, exp_counts.sum())
    act_dist = act_counts / max(1, act_counts.sum())

    exp_dist = np.clip(exp_dist, eps, 1.0)
    act_dist = np.clip(act_dist, eps, 1.0)

    return float(np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist)))


def predict_proba_batch(
    texts: List[str],
    *,
    model_dir: str | Path = "artifacts/models/v1",
) -> np.ndarray:
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    vectorizer = joblib.load(model_dir / "vectorizer.joblib")

    X = vectorizer.transform(texts)
    probs = model.predict_proba(X)[:, 1]
    return probs.astype(float)


def detect_prediction_drift(
    baseline_probs: np.ndarray,
    current_probs: np.ndarray,
    *,
    psi_warn: float = 0.10,
    psi_fail: float = 0.25,
) -> PredictionDriftResult:
    warnings: List[str] = []
    psi = _psi(baseline_probs, current_probs, bins=10)

    stats: Dict[str, Any] = {
        "psi": psi,
        "baseline": {
            "count": int(len(baseline_probs)),
            "mean": float(np.mean(baseline_probs)),
            "p50": float(np.percentile(baseline_probs, 50)),
            "p90": float(np.percentile(baseline_probs, 90)),
        },
        "current": {
            "count": int(len(current_probs)),
            "mean": float(np.mean(current_probs)),
            "p50": float(np.percentile(current_probs, 50)),
            "p90": float(np.percentile(current_probs, 90)),
        },
    }

    if psi >= psi_fail:
        warnings.append(f"PSI indicates significant drift: psi={psi:.3f} (fail >= {psi_fail:.2f})")
    elif psi >= psi_warn:
        warnings.append(f"PSI indicates moderate drift: psi={psi:.3f} (warn >= {psi_warn:.2f})")

    passed = psi < psi_fail
    return PredictionDriftResult(passed=passed, warnings=warnings, stats=stats)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
