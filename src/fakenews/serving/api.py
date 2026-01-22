from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_MODEL_DIR = Path("artifacts/models/v1")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="News text to classify")
    request_id: Optional[str] = Field(
        default=None, description="Optional client-provided request id"
    )


class PredictResponse(BaseModel):
    request_id: Optional[str]
    label: int
    probability_fake: float
    model_version: str
    model_dir: str


@dataclass
class ModelBundle:
    model: object
    vectorizer: object
    model_version: str
    model_dir: str


def load_bundle(model_dir: Path = DEFAULT_MODEL_DIR) -> ModelBundle:
    model_path = model_dir / "model.joblib"
    vec_path = model_dir / "vectorizer.joblib"

    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    return ModelBundle(
        model=model,
        vectorizer=vectorizer,
        model_version="v1",
        model_dir=str(model_dir),
    )


def create_app(model_dir: Path = DEFAULT_MODEL_DIR) -> FastAPI:
    app = FastAPI(title="AI Model Quality Framework API", version="0.1.0")

    bundle: Optional[ModelBundle] = None
    load_error: Optional[str] = None

    try:
        bundle = load_bundle(model_dir=model_dir)
    except Exception as e:
        load_error = str(e)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": bundle is not None,
            "model_version": getattr(bundle, "model_version", None),
            "error": load_error,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if bundle is None:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text must be non-empty")

        # Vectorize
        X = bundle.vectorizer.transform([text])

        # Predict probability(fake=1)
        prob_fake = float(bundle.model.predict_proba(X)[0, 1])
        label = 1 if prob_fake >= 0.5 else 0

        return PredictResponse(
            request_id=req.request_id,
            label=label,
            probability_fake=prob_fake,
            model_version=bundle.model_version,
            model_dir=bundle.model_dir,
        )

    return app
