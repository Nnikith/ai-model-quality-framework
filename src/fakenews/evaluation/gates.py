from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class GateResult:
    passed: bool
    failures: List[str]


def check_v1_gates(report: Dict[str, Any], eval_cfg: Dict[str, Any]) -> GateResult:
    failures: List[str] = []
    gates = eval_cfg["gates"]["v1_baseline"]

    test_metrics = report.get("test", {})
    f1 = test_metrics.get("f1")
    pr_auc = test_metrics.get("pr_auc")

    if f1 is None or f1 < float(gates["min_f1"]):
        failures.append(f"Gate failed: test f1 {f1} < min_f1 {gates['min_f1']}")

    if pr_auc is None or pr_auc < float(gates["min_pr_auc"]):
        failures.append(f"Gate failed: test pr_auc {pr_auc} < min_pr_auc {gates['min_pr_auc']}")

    return GateResult(passed=(len(failures) == 0), failures=failures)


def check_v2_gates(
    report_v2: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    report_v1: Dict[str, Any] | None = None,
) -> GateResult:
    """
    v2 improved gates:
    - must meet min_f1 and min_pr_auc
    - optionally must improve over v1 by a minimum margin (if v1 report provided)
    """
    failures: List[str] = []
    gates = eval_cfg["gates"]["v2_improved"]

    test_metrics_v2 = report_v2.get("test", {}) or {}
    f1_v2 = test_metrics_v2.get("f1")
    pr_auc_v2 = test_metrics_v2.get("pr_auc")

    if f1_v2 is None or f1_v2 < float(gates["min_f1"]):
        failures.append(f"Gate failed: v2 test f1 {f1_v2} < min_f1 {gates['min_f1']}")

    if pr_auc_v2 is None or pr_auc_v2 < float(gates["min_pr_auc"]):
        failures.append(
            f"Gate failed: v2 test pr_auc {pr_auc_v2} < min_pr_auc {gates['min_pr_auc']}"
        )

    # Improvement check (only if we have a v1 report to compare against)
    min_improve = gates.get("min_improvement_over_v1_f1", None)
    if min_improve is not None and report_v1 is not None:
        test_metrics_v1 = report_v1.get("test", {}) or {}
        f1_v1 = test_metrics_v1.get("f1")

        if f1_v1 is None or f1_v2 is None:
            failures.append("Gate failed: cannot compare v2 vs v1 f1 (missing metric).")
        else:
            delta = float(f1_v2) - float(f1_v1)
            if delta < float(min_improve):
                failures.append(
                    f"Gate failed: v2 f1 improvement {delta:.4f} < min_improvement_over_v1_f1 {min_improve}"
                )

    return GateResult(passed=(len(failures) == 0), failures=failures)
