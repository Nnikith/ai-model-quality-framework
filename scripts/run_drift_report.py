from __future__ import annotations

import argparse
import pandas as pd

from fakenews.monitoring.drift import detect_data_drift, write_drift_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a data drift report (baseline vs current)."
    )
    parser.add_argument("--dataset", default="data/processed/isot.parquet")
    parser.add_argument("--baseline-split", default="train")
    parser.add_argument("--current-split", default="test")  # demo: compare test vs train
    parser.add_argument("--out", default="artifacts/monitoring/drift_report.json")
    args = parser.parse_args()

    df = (
        pd.read_parquet(args.dataset)
        if str(args.dataset).endswith(".parquet")
        else pd.read_csv(args.dataset)
    )

    base = df[df["split"] == args.baseline_split]["text"].tolist()
    cur = df[df["split"] == args.current_split]["text"].tolist()

    res = detect_data_drift(base, cur)

    report = {
        "dataset": str(args.dataset),
        "baseline_split": args.baseline_split,
        "current_split": args.current_split,
        "passed": res.passed,
        "warnings": res.warnings,
        "stats": res.stats,
    }

    write_drift_report(args.out, report)

    if res.passed:
        print("✅ Drift check passed (no warnings).")
    else:
        print("⚠️ Drift warnings detected:")
        for w in res.warnings:
            print(f"  - {w}")
    print(f"Report written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
