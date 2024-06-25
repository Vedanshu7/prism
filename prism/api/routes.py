from __future__ import annotations
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel as PydanticModel

router = APIRouter()


class PredictRequest(PydanticModel):
    features: dict[str, Any]
    return_proba: bool = False


class PredictResponse(PydanticModel):
    prediction: Any
    probability: Optional[list[float]] = None
    model_name: str


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    import pandas as pd
    X = pd.DataFrame([req.features])

    prediction = model.predict(X).tolist()[0]
    probability = None
    if req.return_proba:
        proba = model.predict_proba(X)
        if proba is not None:
            probability = proba[0].tolist()

    return PredictResponse(
        prediction=prediction,
        probability=probability,
        model_name=model.__class__.__name__,
    )


@router.get("/model/info")
def model_info(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        return {"status": "no model loaded"}
    return {
        "model":    model.__class__.__name__,
        "params":   model.get_params(),
        "features": model.feature_names_,
    }


@router.get("/health")
def health():
    return {"status": "ok"}
