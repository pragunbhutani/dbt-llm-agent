import click
import logging
import uvicorn

# Use the logger configured by cli.py
logger = logging.getLogger(__name__)


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the server to.")
@click.option("--port", default=8000, type=int, help="Port to bind the server to.")
@click.option(
    "--reload", is_flag=True, default=False, help="Enable auto-reload for development."
)
def serve(host: str, port: int, reload: bool):
    """Run the RAGstar FastAPI API server (for Slack integration)."""
    logger.info(f"Starting API server on {host}:{port} (Reload: {reload})...")

    # Assume necessary env vars are loaded by the main cli entry point or dotenv in slack_handler
    uvicorn.run(
        "ragstar.api.slack_handler:app",  # Path to the FastAPI app instance
        host=host,
        port=port,
        reload=reload,
        log_level="info",  # Set a default log level for the server
    )
