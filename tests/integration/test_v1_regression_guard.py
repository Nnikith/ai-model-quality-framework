import json
from pathlib import Path

import pandas as pd
import yaml

from fakenews.evaluation.gates import check_v1_gates


def test_v1_regression_guard_against_baseline():
    """
    Ensures our current v1 evaluation does not regress severely vs a stored baseline.
    This is a lightweight quality gate suitable for CI once eval artifacts exist.
    """
    eval_path = Path("artifacts/reports/eval_metrics_v1.json")
    baseline_path = Path("artifacts/reports/baselines/v1_test_metrics.json")
    cfg_path = Path("configs/eval.yaml")

    # If artifacts don't exist (e.g., CI without training), skip.
    if not eval_path.exists() or not baseline_path.exists():
        return

    report = pd.read_json(eval_path, typ="series").to_dict()
    baseline = json.loads(baseline_path.read_text())
    cfg = yaml.safe_load(cfg_path.read_text())

    gate = check_v1_gates(report, cfg)
    assert gate.passed, f"V1 gates failed unexpectedly: {gate.failures}"

    max_drop = float(cfg["gates"]["v1_baseline"]["max_f1_drop_vs_previous"])
    current_f1 = float(report["test"]["f1"])
    baseline_f1 = float(baseline["f1"])

    assert (baseline_f1 - current_f1) <= max_drop, (
        f"F1 regression too large: baseline={baseline_f1:.6f}, current={current_f1:.6f}, "
        f"allowed_drop={max_drop:.6f}"
    )
