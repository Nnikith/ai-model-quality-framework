from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


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
