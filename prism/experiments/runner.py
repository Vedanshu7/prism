from __future__ import annotations
import logging
from typing import Any

import pandas as pd

from prism.data.loader import DataLoader
from prism.data.preprocessor import Preprocessor
from prism.evaluation.evaluator import Evaluator
from prism.experiments.tracker import ExperimentTracker
from prism.models.registry import get_model_class
from prism.models.trainer import Trainer

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, tracking_dir: str = "mlruns"):
        self.tracker = ExperimentTracker(tracking_dir=tracking_dir)

    def run_experiment(self, config: dict[str, Any]) -> dict[str, Any]:
        loader     = DataLoader(random_state=config.get("random_state", 42))
        preprocessor = Preprocessor()
        trainer    = Trainer(random_state=config.get("random_state", 42))
        evaluator  = Evaluator(task=config.get("task", "classification"))

        run = self.tracker.start_run(experiment=config.get("experiment_name", "default"))
        self.tracker.log_params(config)

        try:
            X_train, X_test, y_train, y_test = loader.load_split(
                config["data_path"], config["target_column"],
                test_size=config.get("test_size", 0.2)
            )

            X_train = preprocessor.handle_missing_values(X_train)
            X_test  = preprocessor.handle_missing_values(X_test)
            X_train = preprocessor.encode_categoricals(X_train)
            X_test  = preprocessor.encode_categoricals(X_test)

            model_name = config.get("model", "random_forest")
            model_cls  = get_model_class(model_name)
            model = model_cls(random_state=config.get("random_state", 42))

            trainer.train(model, X_train, y_train)
            results = evaluator.evaluate(model, X_test, y_test)
            self.tracker.log_metrics(results["metrics"])

            self.tracker.end_run("finished")
            return results
        except Exception as e:
            self.tracker.end_run("failed")
            raise
