from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import pandas as pd


REQUIRED_COLUMNS = {"id", "text", "label", "source", "split"}
VALID_SPLITS = {"train", "val", "test"}
VALID_LABELS = {0, 1}


@dataclass
class ValidationResult:
    passed: bool
    errors: List[str]
    stats: Dict[str, int]


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    errors: List[str] = []
    stats: Dict[str, int] = {}

    # ---- Schema checks ----
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")

    # Stop early if schema is broken
    if errors:
        return ValidationResult(False, errors, stats)

    # ---- ID checks ----
    if not df["id"].is_unique:
        errors.append("Column 'id' must contain unique values")

    # ---- Label checks ----
    if not set(df["label"].unique()).issubset(VALID_LABELS):
        errors.append("Column 'label' must be binary {0,1}")

    # ---- Text checks ----
    null_text = df["text"].isna().sum()
    empty_text = (df["text"].astype(str).str.strip() == "").sum()

    if null_text > 0:
        errors.append(f"Found {null_text} null text values")
    if empty_text > 0:
        errors.append(f"Found {empty_text} empty text values")

    # ---- Split checks ----
    invalid_splits = set(df["split"].unique()) - VALID_SPLITS
    if invalid_splits:
        errors.append(f"Invalid split values found: {sorted(invalid_splits)}")

    # ---- Stats (for reports) ----
    stats["rows_total"] = len(df)
    stats["rows_train"] = int((df["split"] == "train").sum())
    stats["rows_val"] = int((df["split"] == "val").sum())
    stats["rows_test"] = int((df["split"] == "test").sum())

    passed = len(errors) == 0
    return ValidationResult(passed, errors, stats)
