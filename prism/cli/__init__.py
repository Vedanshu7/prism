import click
from prism.cli.train import train
from prism.cli.evaluate import evaluate
from prism.cli.serve import serve

@click.group()
def cli():
    """Prism ML pipeline CLI."""

cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(serve)
