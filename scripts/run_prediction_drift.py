from __future__ import annotations

import argparse

import pandas as pd

from fakenews.monitoring.prediction_drift import (
    predict_proba_batch,
    detect_prediction_drift,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate prediction drift report (baseline vs current)."
    )
    parser.add_argument("--dataset", default="data/processed/isot.parquet")
    parser.add_argument("--baseline-split", default="train")
    parser.add_argument("--current-split", default="test")
    parser.add_argument("--model-dir", default="artifacts/models/v1")
    parser.add_argument("--out", default="artifacts/monitoring/pred_drift_report.json")
    args = parser.parse_args()

    df = (
        pd.read_parquet(args.dataset)
        if str(args.dataset).endswith(".parquet")
        else pd.read_csv(args.dataset)
    )

    base_texts = df[df["split"] == args.baseline_split]["text"].tolist()
    cur_texts = df[df["split"] == args.current_split]["text"].tolist()

    base_probs = predict_proba_batch(base_texts, model_dir=args.model_dir)
    cur_probs = predict_proba_batch(cur_texts, model_dir=args.model_dir)

    res = detect_prediction_drift(base_probs, cur_probs)

    report = {
        "dataset": str(args.dataset),
        "baseline_split": args.baseline_split,
        "current_split": args.current_split,
        "model_dir": str(args.model_dir),
        "passed": res.passed,
        "warnings": res.warnings,
        "stats": res.stats,
    }

    write_json(args.out, report)

    if res.warnings:
        print("⚠️ Prediction drift warnings:")
        for w in res.warnings:
            print(f"  - {w}")
    else:
        print("✅ No prediction drift warnings.")

    print(f"Report written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
