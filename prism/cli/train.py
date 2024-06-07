from __future__ import annotations
import click
import yaml

from prism.utils.logger import setup_logging


@click.command()
@click.option("--config", "-c", required=True, help="Path to YAML config file")
@click.option("--output", "-o", default="outputs/model.pkl", help="Output model path")
@click.option("--verbose", "-v", is_flag=True)
def train(config: str, output: str, verbose: bool):
    """Train a model from a config file."""
    setup_logging("DEBUG" if verbose else "INFO")

    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg["model_path"] = output

    from prism.experiments.runner import ExperimentRunner
    runner = ExperimentRunner()
    results = runner.run_experiment(cfg)

    click.echo(f"Training complete.")
    for metric, value in results["metrics"].items():
        click.echo(f"  {metric}: {value:.4f}")
    click.echo(f"Model saved to {output}")
