from fakenews.monitoring.drift import detect_data_drift


def test_no_drift_for_similar_texts():
    baseline = ["hello world"] * 100
    current = ["hello world"] * 100

    res = detect_data_drift(baseline, current)
    assert res.passed is True
    assert res.warnings == []


def test_drift_detects_length_shift():
    baseline = ["short text"] * 100
    current = [("very long text " * 200).strip()] * 100

    res = detect_data_drift(baseline, current, length_mean_shift_pct_warn=0.2)
    assert res.passed is False
    assert any("mean shifted" in w for w in res.warnings)


def test_drift_detects_token_shift():
    baseline = ["alpha beta gamma"] * 100
    current = ["x y z"] * 100

    res = detect_data_drift(baseline, current, top_token_jaccard_warn=0.9, top_k_tokens=3)
    assert res.passed is False
    assert any("token overlap" in w.lower() for w in res.warnings)
