from __future__ import annotations
import json
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Run:
    def __init__(self, experiment: str, run_id: str):
        self.experiment = experiment
        self.run_id     = run_id
        self.params:    dict[str, Any] = {}
        self.metrics:   dict[str, float] = {}
        self.artifacts: list[str] = []
        self.start_time = time.time()
        self.end_time:  Optional[float] = None
        self.status     = "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id":     self.run_id,
            "experiment": self.experiment,
            "params":     self.params,
            "metrics":    self.metrics,
            "artifacts":  self.artifacts,
            "duration":   (self.end_time or time.time()) - self.start_time,
            "status":     self.status,
        }


class ExperimentTracker:
    def __init__(self, tracking_dir: str = "mlruns"):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self._current_run: Optional[Run] = None

    def start_run(self, experiment: str = "default") -> Run:
        run_id = str(uuid.uuid4())[:8]
        self._current_run = Run(experiment=experiment, run_id=run_id)
        logger.info(f"Started run {run_id} in experiment '{experiment}'")
        return self._current_run

    def log_params(self, params: dict[str, Any]) -> None:
        if self._current_run:
            self._current_run.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._current_run:
            self._current_run.metrics.update(metrics)

    def log_artifact(self, path: str) -> None:
        if self._current_run:
            self._current_run.artifacts.append(path)

    def end_run(self, status: str = "finished") -> Optional[Run]:
        if not self._current_run:
            return None
        self._current_run.end_time = time.time()
        self._current_run.status   = status
        self._save_run(self._current_run)
        run = self._current_run
        self._current_run = None
        return run

    def _save_run(self, run: Run) -> None:
        exp_dir = self.tracking_dir / run.experiment
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / f"{run.run_id}.json", "w") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

    def list_runs(self, experiment: str = "default") -> list[dict]:
        exp_dir = self.tracking_dir / experiment
        if not exp_dir.exists():
            return []
        runs = []
        for f in exp_dir.glob("*.json"):
            with open(f) as fp:
                runs.append(json.load(fp))
        return sorted(runs, key=lambda r: r.get("duration", 0))
