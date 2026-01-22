from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from fakenews.models.train_v1 import train_v1
from fakenews.evaluation.gates import check_v1_gates


def main() -> int:
    parser = argparse.ArgumentParser(description="Train v1 baseline model (TF-IDF + Logistic Regression).")
    parser.add_argument("--dataset", default="data/processed/isot.parquet")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--out-dir", default="artifacts/models/v1")
    args = parser.parse_args()

    artifacts = train_v1(dataset_path=args.dataset, eval_config_path=args.eval_config, out_dir=args.out_dir)

    report = pd.read_json(artifacts.metrics_path, typ="series").to_dict()
    eval_cfg = yaml.safe_load(Path(args.eval_config).read_text())

    gate = check_v1_gates(report, eval_cfg)
    if not gate.passed:
        print("❌ Model v1 gates FAILED:")
        for f in gate.failures:
            print(f"  - {f}")
        print(f"Metrics report: {artifacts.metrics_path}")
        return 1

    print("✅ Model v1 training complete and gates PASSED.")
    print(f"Model: {artifacts.model_path}")
    print(f"Vectorizer: {artifacts.vectorizer_path}")
    print(f"Metrics: {artifacts.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
