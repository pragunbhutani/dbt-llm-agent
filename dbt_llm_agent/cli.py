"""
CLI interface for dbt-llm-agent
"""

import click
import os
import subprocess
import sys
import logging
from typing import Optional, Any
import pathlib
import json

# Set up logging
from dbt_llm_agent.utils.logging import setup_logging, get_logger, COLORS

# Initialize logging with default settings
setup_logging()
logger = get_logger(__name__)

# Import all commands
from dbt_llm_agent.commands import (
    parse,
    embed,
    ask,
    interpret,
    list_models,
    model_details,
    questions,
    feedback,
    migrate,
    init_db,
)
from dbt_llm_agent.commands.version_cmd import version


def colored_echo(text, color=None, bold=False):
    """Echo text with color and styling.

    Args:
        text: The text to echo
        color: Color name from COLORS dict or color code
        bold: Whether to make the text bold
    """
    # Define a mapping of capitalized color names to the COLORS keys
    color_mapping = {
        "BLUE": "INFO",  # Use INFO (green) for BLUE
        "GREEN": "INFO",  # Use INFO (green) for GREEN
        "CYAN": "DEBUG",  # Use DEBUG (cyan) for CYAN
        "RED": "ERROR",  # Use ERROR (red) for RED
        "YELLOW": "WARNING",  # Use WARNING (yellow) for YELLOW
    }

    prefix = ""
    suffix = COLORS["RESET"]

    # Map color name to a color in the COLORS dict if needed
    if color and color in color_mapping and color_mapping[color] in COLORS:
        prefix += COLORS[color_mapping[color]]
    elif color and color in COLORS:
        prefix += COLORS[color]
    elif color:
        prefix += color

    if bold:
        prefix += "\033[1m"

    click.echo(f"{prefix}{text}{suffix}")


def get_env_var(var_name: str, default: Any = None) -> Any:
    """
    Get an environment variable with a default value.

    Args:
        var_name: The name of the environment variable
        default: The default value if the environment variable is not set

    Returns:
        The value of the environment variable or the default value
    """
    return os.environ.get(var_name, default)


def set_logging_level(verbose: bool):
    """Set the logging level based on the verbose flag.

    Args:
        verbose: Whether to enable verbose logging
    """
    # Just set the level of our logger, don't reconfigure logging
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value from environment variables.

    Args:
        key: The configuration key
        default: The default value if the key is not found

    Returns:
        The configuration value or the default value
    """
    env_var = f"{key.upper()}"
    return get_env_var(env_var, default)


@click.group()
def cli():
    """dbt LLM Agent CLI"""
    pass


# Register all commands
cli.add_command(version)
cli.add_command(parse)
cli.add_command(embed)
cli.add_command(ask)
cli.add_command(interpret)
cli.add_command(list_models, name="list")
cli.add_command(model_details)
cli.add_command(questions)
cli.add_command(feedback)
cli.add_command(migrate)
cli.add_command(init_db)


def main():
    # Set up colored logging - only configure once
    if not logging.root.handlers:
        setup_logging()

    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
        logger.debug("Loaded environment variables from .env file")
    except ImportError:
        logger.warning(
            "python-dotenv not installed. Environment variables may not be properly loaded."
        )

    cli()


if __name__ == "__main__":
    main()
