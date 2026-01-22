from pathlib import Path

from fastapi.testclient import TestClient

from fakenews.serving.api import create_app


def test_health_endpoint_reports_model_status():
    app = create_app(model_dir=Path("artifacts/models/v1"))
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_contract_success():
    app = create_app(model_dir=Path("artifacts/models/v1"))
    client = TestClient(app)

    payload = {"text": "This is a normal news-like sentence.", "request_id": "req-123"}
    r = client.post("/predict", json=payload)
    assert r.status_code in (200, 503)  # 503 if model artifacts not present in CI

    if r.status_code == 200:
        body = r.json()
        assert body["request_id"] == "req-123"
        assert body["label"] in (0, 1)
        assert 0.0 <= body["probability_fake"] <= 1.0
        assert body["model_version"] == "v1"


def test_predict_rejects_empty_text():
    app = create_app(model_dir=Path("artifacts/models/v1"))
    client = TestClient(app)

    r = client.post("/predict", json={"text": "   "})
    # If model isn't loaded, API returns 503; otherwise should be 400
    assert r.status_code in (400, 503)
