from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakenews.serving.api import create_app


def _client():
    app = create_app(model_dir=Path("artifacts/models/v1"))
    return TestClient(app)


def _skip_if_model_not_loaded(client: TestClient):
    h = client.get("/health")
    if not h.json().get("model_loaded", False):
        pytest.skip("Model artifacts not present; skipping robustness tests.")


def test_predict_probability_range():
    client = _client()
    _skip_if_model_not_loaded(client)

    r = client.post("/predict", json={"text": "A basic test sentence."})
    assert r.status_code == 200
    p = r.json()["probability_fake"]
    assert 0.0 <= p <= 1.0


def test_invariance_case_and_punctuation_small_change():
    """
    Expect probability to not swing wildly for minor perturbations.
    This is not a strict mathematical guarantee; we use a tolerance.
    """
    client = _client()
    _skip_if_model_not_loaded(client)

    base = "Breaking news: the economy is showing signs of recovery according to reports."
    variants = [
        base,
        base.upper(),
        base.lower(),
        base.replace(":", " - "),
        base + "!!!",
        "  " + base + "  ",
    ]

    probs = []
    for t in variants:
        r = client.post("/predict", json={"text": t})
        assert r.status_code == 200
        probs.append(r.json()["probability_fake"])

    # Tolerance: max difference <= 0.20 for these mild perturbations
    assert max(probs) - min(probs) <= 0.20

@pytest.mark.xfail(
    reason="Known limitation of TF-IDF v1 baseline: sensitive to tokenization and typos. Expected to improve in v2.",
    strict=False,
)
def test_invariance_minor_typos():
    client = _client()
    _skip_if_model_not_loaded(client)

    base = "Scientists report a new discovery in renewable energy technology."
    typo = "Scientsits repot a new discovry in reneweble enegy technlogy."

    r1 = client.post("/predict", json={"text": base})
    r2 = client.post("/predict", json={"text": typo})
    assert r1.status_code == 200 and r2.status_code == 200

    p1 = r1.json()["probability_fake"]
    p2 = r2.json()["probability_fake"]

    # We still measure it; the xfail documents that v1 is not robust to typos.
    assert abs(p1 - p2) <= 0.30


def test_ood_gibberish_handling():
    """
    Gibberish should still return a valid probability and not crash.
    (We don't enforce a specific label yet.)
    """
    client = _client()
    _skip_if_model_not_loaded(client)

    gib = "asdjkhqweoiu zxcmnqweoiu 123123 !!! ??? qweqwe"
    r = client.post("/predict", json={"text": gib})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_fake"] <= 1.0
    assert body["label"] in (0, 1)


def test_extremely_long_input_does_not_crash():
    client = _client()
    _skip_if_model_not_loaded(client)

    long_text = "news " * 20000  # large but should remain manageable
    r = client.post("/predict", json={"text": long_text})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_fake"] <= 1.0
