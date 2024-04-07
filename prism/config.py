from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class Config:
    data_path: str = "data/input.csv"
    model_path: str = "outputs/model.pkl"
    log_level: str = "INFO"
    random_state: int = 42
    test_size: float = 0.2
    target_column: str = "target"
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def ensure_output_dir(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig(Config):
    experiment_name: str = "default"
    models_to_try: list[str] = field(default_factory=lambda: ["random_forest", "gradient_boosting"])
    cv_folds: int = 5
    metric: str = "f1_weighted"
    hyperparameter_search: bool = False
    search_n_iter: int = 20
