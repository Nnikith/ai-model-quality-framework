from fakenews.evaluation.gates import check_v1_gates


def test_v1_gate_passes_when_metrics_meet_thresholds():
    report = {"test": {"f1": 0.9, "pr_auc": 0.9}}
    cfg = {"gates": {"v1_baseline": {"min_f1": 0.7, "min_pr_auc": 0.7}}}
    result = check_v1_gates(report, cfg)
    assert result.passed is True
    assert result.failures == []


def test_v1_gate_fails_when_below_thresholds():
    report = {"test": {"f1": 0.5, "pr_auc": 0.6}}
    cfg = {"gates": {"v1_baseline": {"min_f1": 0.7, "min_pr_auc": 0.7}}}
    result = check_v1_gates(report, cfg)
    assert result.passed is False
    assert len(result.failures) == 2
