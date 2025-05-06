"""
Common utility functions for CLI commands
"""

import os
import logging
import click
from typing import Any

from ragstar_archive.utils.logging import COLORS, get_logger

# Get logger
logger = get_logger(__name__)


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
    # Set the level for the root logger to affect all handlers
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    # Optional: You might want to adjust the level of specific handlers
    # if they have their own level settings, but setting the root logger level
    # is usually sufficient.
    logger.info(
        f"Logging level set to {logging.getLevelName(log_level)}"
    )  # Log the change


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value from environment variables.

    Args:
        key: The configuration key
        default: The default value if the key is not found

    Returns:
        The configuration value or the default value
    """
    return get_env_var(key.upper(), default)


def format_model_reference(model_name: str) -> str:
    """
    Format a model name as a model reference.

    Args:
        model_name: The model name to format

    Returns:
        A formatted model reference string
    """
    return f"ref('{model_name}')"
