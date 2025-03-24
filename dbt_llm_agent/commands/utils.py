"""
Utility functions used by CLI commands.
"""

import os
import logging
import click
from typing import Any

from dbt_llm_agent.utils.logging import COLORS, get_logger

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
    # Set the level of our logger
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


def format_model_as_yaml(model):
    """Format a DBT model as a dbt YAML document.

    Args:
        model: The DBT model to format

    Returns:
        A string containing the YAML document
    """
    # Start the YAML document
    yaml_lines = ["version: 2", "", "models:", f"  - name: {model.name}"]

    # Add description
    description = (
        model.interpreted_description
        if model.interpreted_description
        else model.description
    )
    if description:
        # Indent multiline descriptions correctly
        formatted_desc = description.replace("\n", "\n      ")
        yaml_lines.append(f"    description: >\n      {formatted_desc}")

    # Add model metadata
    if model.schema:
        yaml_lines.append(f"    schema: {model.schema}")

    if model.database:
        yaml_lines.append(f"    database: {model.database}")

    if model.materialization:
        yaml_lines.append(f"    config:")
        yaml_lines.append(f"      materialized: {model.materialization}")

    if model.tags:
        yaml_lines.append(f"    tags: [{', '.join(model.tags)}]")

    # Add column specifications
    if model.columns or model.interpreted_columns:
        yaml_lines.append("    columns:")

        # First try to use YML columns with their full specs
        if model.columns:
            for col_name, col in model.columns.items():
                yaml_lines.append(f"      - name: {col_name}")
                if col.description:
                    # Indent multiline descriptions correctly
                    col_desc = col.description.replace("\n", "\n          ")
                    yaml_lines.append(f"        description: >\n          {col_desc}")
                if col.data_type:
                    yaml_lines.append(f"        data_type: {col.data_type}")
        # Fall back to interpreted columns if available
        elif model.interpreted_columns:
            for col_name, description in model.interpreted_columns.items():
                yaml_lines.append(f"      - name: {col_name}")
                # Indent multiline descriptions correctly
                col_desc = description.replace("\n", "\n          ")
                yaml_lines.append(f"        description: >\n          {col_desc}")

    # Add tests if available
    if model.tests:
        test_lines = []
        for test in model.tests:
            if test.column_name:
                # This is a column-level test, which should go under the column definition
                continue
            else:
                # This is a model-level test
                test_name = test.test_type or test.name
                if test_name:
                    test_lines.append(f"      - {test_name}")

        if test_lines:
            yaml_lines.append("    tests:")
            yaml_lines.extend(test_lines)

    return "\n".join(yaml_lines)
