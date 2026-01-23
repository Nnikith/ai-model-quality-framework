from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from fakenews.evaluation.gates import check_v2_gates
from fakenews.models.train_v2 import train_v2


def _load_report(path: Path) -> Dict[str, Any]:
    # train_v2 writes a JSON via pd.Series.to_json -> easiest stable read is Series
    return pd.read_json(path, typ="series").to_dict()


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

    eval_cfg = yaml.safe_load(Path(args.eval_config).read_text())

    report_v2 = _load_report(Path(artifacts.metrics_path))

    # Optional: compare to v1 if available
    v1_report_path = Path("artifacts/reports/eval_metrics_v1.json")
    report_v1 = _load_report(v1_report_path) if v1_report_path.exists() else None

    gate = check_v2_gates(report_v2=report_v2, eval_cfg=eval_cfg, report_v1=report_v1)

    print("✅ v2 training complete.")
    print(f"Model:      {artifacts.model_path}")
    print(f"Vectorizer: {artifacts.vectorizer_path}")
    print(f"Metrics:    {artifacts.metrics_path}")
    print("")
    print("Metrics summary:")
    print(report_v2)

    if not gate.passed:
        print("\n❌ v2 evaluation gates FAILED:")
        for f in gate.failures:
            print(f"  - {f}")
        return 1

    print("\n✅ v2 evaluation gates PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
