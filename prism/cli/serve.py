from __future__ import annotations
import click


@click.command()
@click.option("--model", "-m", required=True, help="Path to saved model (.pkl)")
@click.option("--port",  "-p", default=8000,  help="Port to run on")
@click.option("--host",        default="0.0.0.0")
def serve(model: str, port: int, host: str):
    """Serve a trained model as a FastAPI prediction endpoint."""
    import os
    os.environ["MODEL_PATH"] = model

    import uvicorn
    uvicorn.run("prism.api.app:app", host=host, port=port, reload=False)
