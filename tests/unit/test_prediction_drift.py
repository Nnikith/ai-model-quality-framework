import numpy as np

from fakenews.monitoring.prediction_drift import detect_prediction_drift


def test_prediction_drift_low_when_distributions_similar():
    baseline = np.random.RandomState(0).beta(2, 5, size=1000)
    current = np.random.RandomState(1).beta(2, 5, size=1000)

    res = detect_prediction_drift(baseline, current, psi_warn=0.5, psi_fail=0.8)
    assert res.passed is True


def test_prediction_drift_high_when_distributions_shifted():
    baseline = np.random.RandomState(0).beta(2, 5, size=2000)
    current = np.random.RandomState(1).beta(8, 2, size=2000)  # shifted toward 1

    res = detect_prediction_drift(baseline, current, psi_warn=0.1, psi_fail=0.25)
    assert res.passed is False or len(res.warnings) > 0
