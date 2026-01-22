from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    df["text"] = (
        df["title"].astype(str).str.strip()
        + "\n\n"
        + df["text"].astype(str).str.strip()
    ).str.strip()

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
    is not feasible, falls back to non-stratified splitting.

    Robustness goals:
    - Avoid sklearn errors on tiny datasets (e.g., holdout size < #classes)
    - Support val_size == 0.0 or test_size == 0.0 without doing a second split
    - Support train_size == 1.0
    """
    from sklearn.model_selection import train_test_split

    if round(train_size + val_size + test_size, 10) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    df = df.copy()

    def can_stratify(y: pd.Series) -> bool:
        counts = y.value_counts()
        # Need at least 2 classes and at least 2 samples per class
        return (counts.shape[0] >= 2) and (counts.min() >= 2)

    # If everything goes to train, avoid train_test_split entirely
    if train_size == 1.0:
        return df.assign(split="train")

    n = len(df)
    classes = int(df["label"].nunique())

    # Sklearn requires each split in stratified splitting to have at least one
    # sample from each class. A necessary condition is: holdout_size >= #classes.
    holdout_size = int(round(n * (1.0 - train_size)))
    if holdout_size < classes:
        stratify_y = None
    else:
        stratify_y = df["label"] if can_stratify(df["label"]) else None

    # First split: train vs temp
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=stratify_y,
        random_state=seed,
    )

    # If no remaining data (shouldn't happen unless train_size==1.0), still be safe
    if len(temp_df) == 0:
        return train_df.assign(split="train")

    # If val_size == 0, everything remaining becomes test
    if val_size == 0.0 and test_size > 0.0:
        return pd.concat(
            [train_df.assign(split="train"), temp_df.assign(split="test")],
            ignore_index=True,
        )

    # If test_size == 0, everything remaining becomes val
    if test_size == 0.0 and val_size > 0.0:
        return pd.concat(
            [train_df.assign(split="train"), temp_df.assign(split="val")],
            ignore_index=True,
        )

    # Otherwise split temp into val/test
    remaining = 1.0 - train_size
    test_frac_of_temp = test_size / remaining

    # For tiny temp sets, stratification may still fail. Only stratify if the temp
    # set can support both splits containing at least one sample per class.
    # A conservative sufficient check: temp size >= 2 * #classes and per-class >= 2.
    if len(temp_df) < max(2 * classes, 4):
        stratify_temp = None
    else:
        stratify_temp = temp_df["label"] if can_stratify(temp_df["label"]) else None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_frac_of_temp,
        stratify=stratify_temp,
        random_state=seed,
    )

    return pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical schema dataframe."""
    cols = ["id", "text", "label", "source", "subject", "date", "split"]
    df = df.copy()
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
