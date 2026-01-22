from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class IsotPaths:
    true_csv: Path
    fake_csv: Path


def load_isot(true_csv: str | Path, fake_csv: str | Path) -> pd.DataFrame:
    """Load ISOT-style True/Fake CSVs and return a unified raw dataframe."""
    true_path = Path(true_csv)
    fake_path = Path(fake_csv)

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    # Standardize columns we expect (common: title, text, subject, date)
    for df in (df_true, df_fake):
        if "text" not in df.columns:
            raise ValueError("ISOT CSV must contain a 'text' column")
        if "title" not in df.columns:
            df["title"] = ""
        if "subject" not in df.columns:
            df["subject"] = None
        if "date" not in df.columns:
            df["date"] = None

    df_true["label"] = 0  # real
    df_fake["label"] = 1  # fake

    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Normalize types
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["subject"] = df["subject"].where(pd.notna(df["subject"]), None)

    # Try parsing date; if not parseable, keep as None
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["date"] = parsed.where(parsed.notna(), None)

    # Construct final text: title + "\n\n" + text (title optional)
    df["text"] = (df["title"].astype(str).str.strip() + "\n\n" + df["text"].astype(str).str.strip()).str.strip()

    # Add source field
    df["source"] = "kaggle_isot_fake_and_real_news"

    return df


def add_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic IDs based on row order and label."""
    df = df.copy()
    # Deterministic (stable given same input ordering)
    df["id"] = [f"isot_{i:07d}" for i in range(len(df))]
    return df


def stratified_split(
    df: pd.DataFrame,
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    """Add a 'split' column with train/val/test splits.

    Uses stratification when possible. For very small datasets where stratification
    is not feasible (e.g., not enough samples per class), falls back to non-stratified
    splitting to keep the pipeline testable and robust.
    """
    from sklearn.model_selection import train_test_split

    if round(train_size + val_size + test_size, 10) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    df = df.copy()

    def can_stratify(y: pd.Series) -> bool:
        counts = y.value_counts()
        # Need at least 2 samples per class for stratified splitting
        return (counts.min() >= 2) and (counts.shape[0] >= 2)

    # First split: train vs temp
    stratify_y = df["label"] if can_stratify(df["label"]) else None
    train_df, temp_df = train_test_split(
        df, train_size=train_size, stratify=stratify_y, random_state=seed
    )

    # Second split: val vs test from temp
    remaining = 1.0 - train_size
    test_frac_of_temp = test_size / remaining

    stratify_temp = temp_df["label"] if can_stratify(temp_df["label"]) else None
    val_df, test_df = train_test_split(
        temp_df, test_size=test_frac_of_temp, stratify=stratify_temp, random_state=seed
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    out = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return out


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical schema dataframe."""
    cols = ["id", "text", "label", "source", "subject", "date", "split"]
    # Ensure missing optional columns exist
    if "subject" not in df.columns:
        df["subject"] = None
    if "date" not in df.columns:
        df["date"] = None
    return df[cols].copy()


def write_outputs(
    df: pd.DataFrame,
    *,
    processed_path: str | Path,
    manifest_path: str | Path,
) -> None:
    processed_path = Path(processed_path)
    manifest_path = Path(manifest_path)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Store as parquet for efficiency if available; fallback to csv
    try:
        df.to_parquet(processed_path, index=False)
        stored_as = "parquet"
    except Exception:
        # If user doesn't have pyarrow installed, fallback
        csv_path = processed_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        stored_as = "csv"
        processed_path = csv_path

    manifest = {
        "rows_total": int(len(df)),
        "rows_train": int((df["split"] == "train").sum()),
        "rows_val": int((df["split"] == "val").sum()),
        "rows_test": int((df["split"] == "test").sum()),
        "stored_as": stored_as,
        "processed_path": str(processed_path),
    }
    pd.Series(manifest).to_json(manifest_path, indent=2)
