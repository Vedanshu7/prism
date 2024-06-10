from __future__ import annotations
import click

from prism.utils.logger import setup_logging


@click.command()
@click.option("--model", "-m", required=True, help="Path to saved model (.pkl)")
@click.option("--data",  "-d", required=True, help="Path to test data (CSV)")
@click.option("--target", "-t", default="target", help="Target column name")
@click.option("--task", default="classification", type=click.Choice(["classification", "regression"]))
def evaluate(model: str, data: str, target: str, task: str):
    """Evaluate a saved model on a dataset."""
    setup_logging("INFO")

    from prism.models.base import BaseModel
    from prism.data.loader import DataLoader
    from prism.evaluation.evaluator import Evaluator

    m    = BaseModel.load(model)
    loader = DataLoader()
    df   = loader.load(data)
    X    = df.drop(columns=[target])
    y    = df[target]

    evaluator = Evaluator(task=task)
    results   = evaluator.evaluate(m, X, y)

    click.echo(f"\nEvaluation results ({m.__class__.__name__}):")
    for metric, value in results["metrics"].items():
        click.echo(f"  {metric}: {value:.4f}")
