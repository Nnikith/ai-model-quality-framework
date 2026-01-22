from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


REQUIRED_COLUMNS = {"id", "text", "label", "source", "split"}
VALID_SPLITS = {"train", "val", "test"}
VALID_LABELS = {0, 1}


@dataclass
class ValidationResult:
    passed: bool
    errors: List[str]
    stats: Dict[str, Any]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _label_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    dist: Dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            dist[split] = {"count": 0, "label_0": 0, "label_1": 0}
            continue
        vc = sub["label"].value_counts(dropna=False).to_dict()
        dist[split] = {
            "count": int(len(sub)),
            "label_0": int(vc.get(0, 0)),
            "label_1": int(vc.get(1, 0)),
        }
    return dist


def validate_dataframe(
    df: pd.DataFrame,
    *,
    dataset_name: str = "unknown",
    report_path: Optional[str | Path] = None,
) -> ValidationResult:
    errors: List[str] = []
    stats: Dict[str, Any] = {}

    meta: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_columns": sorted(REQUIRED_COLUMNS),
    }

    # ---- Schema checks ----
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")

    # Stop early if schema is broken
    if errors:
        result = ValidationResult(False, errors, stats, meta)
        if report_path:
            _write_report(result, report_path)
        return result

    # ---- Basic counts ----
    stats["rows_total"] = int(len(df))
    stats["rows_train"] = int((df["split"] == "train").sum())
    stats["rows_val"] = int((df["split"] == "val").sum())
    stats["rows_test"] = int((df["split"] == "test").sum())

    # ---- ID checks ----
    stats["id_unique"] = bool(df["id"].is_unique)
    if not stats["id_unique"]:
        errors.append("Column 'id' must contain unique values")

    # ---- Label checks ----
    unique_labels = set(df["label"].unique())
    stats["labels_unique"] = sorted([int(x) for x in unique_labels if pd.notna(x)])
    if not unique_labels.issubset(VALID_LABELS):
        errors.append("Column 'label' must be binary {0,1}")

    # ---- Text checks ----
    null_text = int(df["text"].isna().sum())
    empty_text = int((df["text"].astype(str).str.strip() == "").sum())
    stats["null_text_count"] = null_text
    stats["empty_text_count"] = empty_text

    if null_text > 0:
        errors.append(f"Found {null_text} null text values")
    if empty_text > 0:
        errors.append(f"Found {empty_text} empty text values")

    # ---- Split checks ----
    invalid_splits = set(df["split"].unique()) - VALID_SPLITS
    stats["splits_unique"] = sorted([str(x) for x in df["split"].unique()])
    if invalid_splits:
        errors.append(f"Invalid split values found: {sorted(invalid_splits)}")

    # ---- Duplicate checks ----
    # exact duplicates on text are a useful early warning signal
    dup_text = int(df["text"].astype(str).duplicated().sum())
    stats["duplicate_text_count"] = dup_text

    # ---- Label distribution ----
    stats["label_distribution"] = _label_distribution(df)

    passed = len(errors) == 0
    result = ValidationResult(passed, errors, stats, meta)

    if report_path:
        _write_report(result, report_path)

    return result


def _write_report(result: ValidationResult, report_path: str | Path) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use pandas for stable JSON writing without adding more deps
    pd.Series(result.to_dict()).to_json(path, indent=2)
