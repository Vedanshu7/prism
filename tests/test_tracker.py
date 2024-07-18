import time
import pytest
from prism.experiments.tracker import ExperimentTracker


def test_start_and_end_run(tmp_path):
    tracker = ExperimentTracker(tracking_dir=str(tmp_path))
    run = tracker.start_run("test_exp")
    assert run.status == "running"
    assert run.experiment == "test_exp"

    tracker.log_params({"lr": 0.01, "n_estimators": 100})
    tracker.log_metrics({"accuracy": 0.95, "f1": 0.94})
    tracker.log_artifact("outputs/model.pkl")

    ended = tracker.end_run("finished")
    assert ended.status == "finished"
    assert ended.params["lr"] == 0.01
    assert ended.metrics["accuracy"] == 0.95


def test_list_runs(tmp_path):
    tracker = ExperimentTracker(tracking_dir=str(tmp_path))
    for i in range(3):
        tracker.start_run("exp")
        tracker.log_metrics({"score": i * 0.1})
        tracker.end_run()
    runs = tracker.list_runs("exp")
    assert len(runs) == 3
