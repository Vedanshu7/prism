from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class PipelineStep(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class LoadDataStep(PipelineStep):
    def run(self, ctx):
        from prism.data.loader import DataLoader
        loader = DataLoader(random_state=self.config.get("random_state", 42))
        X_train, X_test, y_train, y_test = loader.load_split(
            self.config["data_path"],
            self.config["target_column"],
            self.config.get("test_size", 0.2),
        )
        ctx.update({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test})
        return ctx


class PreprocessStep(PipelineStep):
    def run(self, ctx):
        from prism.data.preprocessor import Preprocessor
        p = Preprocessor()
        ctx["X_train"] = p.handle_missing_values(ctx["X_train"])
        ctx["X_test"]  = p.handle_missing_values(ctx["X_test"])
        ctx["X_train"] = p.encode_categoricals(ctx["X_train"])
        ctx["X_test"]  = p.encode_categoricals(ctx["X_test"])
        ctx["preprocessor"] = p
        return ctx


class TrainStep(PipelineStep):
    def run(self, ctx):
        from prism.models.registry import get_model_class
        from prism.models.trainer import Trainer
        model_cls = get_model_class(self.config.get("model", "random_forest"))
        model = model_cls(random_state=self.config.get("random_state", 42))
        trainer = Trainer()
        trainer.train(model, ctx["X_train"], ctx["y_train"])
        ctx["model"] = model
        ctx["trainer"] = trainer
        return ctx


class EvaluateStep(PipelineStep):
    def run(self, ctx):
        from prism.evaluation.evaluator import Evaluator
        evaluator = Evaluator(task=self.config.get("task", "classification"))
        results = evaluator.evaluate(ctx["model"], ctx["X_test"], ctx["y_test"])
        ctx["results"] = results
        return ctx


class SaveStep(PipelineStep):
    def run(self, ctx):
        output = self.config.get("model_path", "outputs/model.pkl")
        ctx["model"].save(output)
        ctx["model_path"] = output
        return ctx
