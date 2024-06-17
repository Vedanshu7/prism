from __future__ import annotations
import logging
from typing import Any

import yaml

from prism.pipeline.steps import (
    LoadDataStep, PreprocessStep, TrainStep, EvaluateStep, SaveStep, PipelineStep
)

logger = logging.getLogger(__name__)


class PipelineBuilder:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._steps: list[PipelineStep] = []

    @classmethod
    def from_config(cls, path: str) -> "PipelineBuilder":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def build(self) -> "PipelineBuilder":
        self._steps = [
            LoadDataStep(self.config),
            PreprocessStep(self.config),
            TrainStep(self.config),
            EvaluateStep(self.config),
            SaveStep(self.config),
        ]
        return self

    def run(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        for step in self._steps:
            logger.info(f"Running step: {step.name}")
            ctx = step.run(ctx)
        return ctx
