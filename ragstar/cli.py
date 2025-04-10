"""
CLI interface for ragstar
"""

import click
import logging
import os

# Set up logging
from ragstar.utils.logging import setup_logging, get_logger
from ragstar.utils.cli_utils import load_dotenv_once, get_config_value

# Initialize logging with default settings
setup_logging()
logger = get_logger(__name__)

# Import all commands
from ragstar.commands import (
    parse,
    embed,
    ask,
    interpret,
    list,
    model_details,
    questions,
    feedback,
    migrate,
    init_db,
    reset_db,
)
from ragstar.commands.version_cmd import version
from ragstar.commands.init_cmd import init

# Import the new serve command
from ragstar.commands.serve_cmd import serve as serve_command


@click.group()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to custom config file."
)
@click.pass_context
def cli(ctx, config):
    """RAGstar CLI Tool"""
    ctx.ensure_object(dict)
    # Load config centrally for other commands if needed
    config_source = "Default environment variables/.env"
    if config:
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=config, override=True)
            config_source = config
            logger.debug(f"Loaded environment variables from {config}")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Cannot load custom config file."
            )
        except FileNotFoundError:
            logger.error(f"Custom config file not found: {config}")
            # Decide if you want to exit or continue with default env vars
            # For now, continue and log the source used below

    logger.info(f"Using config source: {config_source}")


# Register all commands
cli.add_command(version)
cli.add_command(parse)
cli.add_command(embed)
cli.add_command(ask)
cli.add_command(interpret)
cli.add_command(list, name="list")
cli.add_command(model_details)
cli.add_command(questions)
cli.add_command(feedback)
cli.add_command(migrate)
cli.add_command(init_db)
cli.add_command(reset_db)
cli.add_command(init)
# Register the serve command
cli.add_command(serve_command, "serve")


# --- New API Server Command ---
# Remove the serve command definition from here
# @cli.command()
# @click.option("--host", default="0.0.0.0", help="Host to bind the server to.")
# @click.option("--port", default=8000, type=int, help="Port to bind the server to.")
# @click.option(
#     "--reload", is_flag=True, default=False, help="Enable auto-reload for development."
# )
# def serve(host: str, port: int, reload: bool):
#     """Run the RAGstar FastAPI API server (for Slack integration)."""
#     logger.info(f"Starting API server on {host}:{port} (Reload: {reload})...")
#
#     # Ensure necessary env vars are available for the server startup within slack_handler
#     # Although loaded by dotenv there, double-check critical ones if needed?
#     # For now, assume slack_handler's startup handles env vars correctly.
#
#     uvicorn.run(
#         "ragstar.api.slack_handler:app",  # Path to the FastAPI app instance
#         host=host,
#         port=port,
#         reload=reload,
#         # Optionally add log_level configuration based on CLI verbosity later
#         # log_level="info",
#     )


# --- End New API Server Command ---


def main():
    # Set up colored logging - only configure once
    if not logging.root.handlers:
        setup_logging()

    # Load environment variables from .env file
    load_dotenv_once()

    cli()


if __name__ == "__main__":
    main()
