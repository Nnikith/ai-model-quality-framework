from __future__ import annotations

from pathlib import Path

import yaml

from fakenews.data.ingest_isot import load_isot, add_ids, stratified_split, to_canonical


def test_ci_sample_ingestion_produces_train_and_test_splits() -> None:
    cfg_path = Path("configs/data_ci.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    splits = cfg["splits"]

    df_raw = load_isot("data/raw/sample/true.csv", "data/raw/sample/fake.csv")
    df_raw = add_ids(df_raw)

    df_split = stratified_split(
        df_raw,
        train_size=float(splits["train_size"]),
        val_size=float(splits["val_size"]),
        test_size=float(splits["test_size"]),
        seed=int(splits["random_seed"]),
    )

    df_can = to_canonical(df_split)

    assert "split" in df_can.columns
    assert (df_can["split"] == "train").sum() > 0
    assert (df_can["split"] == "test").sum() > 0

    # Monitoring assumes text exists and labels are binary ints
    assert df_can["text"].astype(str).str.len().min() > 0
    assert set(df_can["label"].unique()).issubset({0, 1})
