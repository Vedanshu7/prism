from __future__ import annotations
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prism.api.routes import router

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        from prism.models.base import BaseModel
        _model = BaseModel.load(model_path)
        app.state.model = _model
    yield
    _model = None


app = FastAPI(
    title="Prism Model Server",
    description="FastAPI endpoint for ML model predictions",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
