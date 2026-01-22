from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fakenews.data.ingest_isot import (
    load_isot,
    add_ids,
    stratified_split,
    to_canonical,
    write_outputs,
)
from fakenews.data.validate import validate_dataframe


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Kaggle ISOT Fake/Real News dataset into canonical format."
    )
    parser.add_argument("--true-csv", default="data/raw/isot/True.csv")
    parser.add_argument("--fake-csv", default="data/raw/isot/Fake.csv")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config yaml")
    parser.add_argument(
        "--out", default="data/processed/isot.parquet", help="Output processed dataset path"
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/reports/split_manifest.json",
        help="Split manifest JSON path",
    )
    parser.add_argument(
        "--report",
        default="artifacts/reports/data_validation.json",
        help="Validation report JSON path",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    splits = cfg["splits"]

    df_raw = load_isot(args.true_csv, args.fake_csv)
    df_raw = add_ids(df_raw)

    train_size = float(splits["train_size"])
    val_size = float(splits["val_size"])
    test_size = float(splits["test_size"])

    if val_size == 0.0 and test_size == 0.0:
        df_split = df_raw.copy()
        df_split["split"] = "train"
    else:
        df_split = stratified_split(
            df_raw,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=int(splits["random_seed"]),
        )

    df_can = to_canonical(df_split)

    # Validate and write report
    result = validate_dataframe(
        df_can, dataset_name=cfg["dataset"]["name"], report_path=args.report
    )
    if not result.passed:
        print("❌ Data validation failed:")
        for e in result.errors:
            print(f"  - {e}")
        print(f"Validation report written to: {args.report}")
        return 1

    write_outputs(df_can, processed_path=args.out, manifest_path=args.manifest)
    print("✅ Ingestion complete.")
    print(f"Processed dataset written to: {args.out}")
    print(f"Manifest written to: {args.manifest}")
    print(f"Validation report written to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
