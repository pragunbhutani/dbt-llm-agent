"""
List command for dbt-llm-agent CLI.
"""

import click
import sys

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import (
    get_env_var,
    set_logging_level,
    colored_echo,
)

# Initialize logger
logger = get_logger(__name__)


@click.command(name="list")
@click.option(
    "--select",
    required=True,
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def list_models(select, postgres_uri, verbose):
    """List selected models from the database."""
    # Import required modules
    from dbt_llm_agent.storage.postgres_storage import PostgresStorage
    from dbt_llm_agent.utils.model_selector import ModelSelector

    set_logging_level(verbose)

    if not postgres_uri:
        postgres_uri = get_env_var("POSTGRES_URI")

    if not postgres_uri:
        colored_echo(
            "PostgreSQL URI not provided and not found in config",
            color="RED",
            bold=True,
        )
        sys.exit(1)

    try:
        # Initialize PostgreSQL storage
        postgres_storage = PostgresStorage(postgres_uri)

        # Fetch all models from the database
        all_models = postgres_storage.get_all_models()

        if not all_models:
            colored_echo("No models found in the database", color="YELLOW")
            return

        # Convert the list of models to a dictionary for the selector
        models_dict = {model.name: model for model in all_models}

        # Select models based on the provided selection
        selector = ModelSelector(models_dict)
        selected_model_names = selector.select(select)

        # Get the actual model objects for the selected names
        selected_models = [
            models_dict[name] for name in selected_model_names if name in models_dict
        ]

        if not selected_models:
            colored_echo(
                f"No models selected using '{select}'", color="YELLOW", bold=True
            )
            return

        colored_echo(
            f"Selected {len(selected_models)} model(s) using '{select}':",
            color="GREEN",
            bold=True,
        )

        for idx, model in enumerate(selected_models, 1):
            # Only show parentheses if either materialization or schema has a value
            model_info = f"{idx}. {model.name}"
            if model.materialization or model.schema:
                mat = model.materialization if model.materialization else ""
                schema = model.schema if model.schema else ""
                if mat and schema:
                    model_info += f" ({mat}, {schema})"
                elif mat:
                    model_info += f" ({mat})"
                elif schema:
                    model_info += f" ({schema})"

            colored_echo(model_info)
            if verbose:
                if model.description:
                    colored_echo(f"   Description: {model.description}", color="CYAN")
                colored_echo(f"   Path: {model.path}", color="CYAN")
                if model.columns:
                    colored_echo(f"   Columns:", color="CYAN")
                    for col_name, col in model.columns.items():
                        desc = f" - {col.description}" if col.description else ""
                        colored_echo(f"     - {col_name}{desc}", color="CYAN")
                colored_echo("")

    except Exception as e:
        colored_echo(f"Error listing models: {str(e)}", color="RED", bold=True)
        if verbose:
            import traceback

            colored_echo(traceback.format_exc(), color="RED")
        sys.exit(1)
