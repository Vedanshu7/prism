from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Feature values as key-value pairs")
    return_proba: bool = Field(False, description="Whether to return class probabilities")

    model_config = {"json_schema_extra": {"example": {"features": {"age": 25, "income": 50000}}}}


class PredictResponse(BaseModel):
    prediction: Any
    probability: Optional[list[float]] = None
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
