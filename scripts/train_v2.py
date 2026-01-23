from __future__ import annotations

import argparse

import pandas as pd

from fakenews.models.train_v2 import train_v2


def main() -> int:
    p = argparse.ArgumentParser(
        description="Train v2 model (char n-gram TF-IDF + Logistic Regression)."
    )
    p.add_argument("--dataset", default="data/processed/isot.parquet")
    p.add_argument("--model-config", default="configs/model_v2.yaml")
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--out-dir", default="artifacts/models/v2")
    p.add_argument("--report-path", default="artifacts/reports/eval_metrics_v2.json")
    args = p.parse_args()

    artifacts = train_v2(
        dataset_path=args.dataset,
        model_config_path=args.model_config,
        eval_config_path=args.eval_config,
        out_dir=args.out_dir,
        report_path=args.report_path,
    )

    # Pretty print the report location and what got written
    report = pd.read_json(artifacts.metrics_path, typ="series").to_dict()
    print("✅ v2 training complete.")
    print(f"Model:      {artifacts.model_path}")
    print(f"Vectorizer: {artifacts.vectorizer_path}")
    print(f"Metrics:    {artifacts.metrics_path}")
    print("")
    print("Metrics summary:")
    print(report)

    # (We will add evaluation gates after we confirm training works end-to-end)
    # Keeping this script simple for now helps avoid breaking CI.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
