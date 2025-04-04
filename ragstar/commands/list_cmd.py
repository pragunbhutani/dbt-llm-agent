"""
Command to list available dbt models.
"""

import click
import sys
import json
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table

from ragstar.utils.logging import get_logger
from ragstar.utils.cli_utils import get_config_value, set_logging_level
from ragstar.utils.model_selector import ModelSelector

# Initialize logger
logger = get_logger(__name__)

# Initialize console for rich output
console = Console()


@click.command()
@click.option(
    "--select",
    "-s",
    help="Select models to list using dbt-style selectors (e.g., '+tag:marts')",
)
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def list_models(select, output_json, verbose):
    """List available dbt models.

    This command displays all models in the database, with optional filtering
    using dbt-style selectors.

    Examples:
        dbt-llm list
        dbt-llm list --select "tag:marts"
        dbt-llm list --select "customers" --json
        dbt-llm list --verbose
    """
    set_logging_level(verbose)

    # Load configuration from environment
    postgres_uri = get_config_value("postgres_uri")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        # Import necessary modules
        from ragstar.storage.model_storage import ModelStorage

        # Initialize storage
        model_storage = ModelStorage(postgres_uri)

        # Get all models
        all_models = model_storage.get_all_models()

        if not all_models:
            console.print("No models found in the database")
            sys.exit(0)

        # Apply selector if provided
        if select:
            models_dict = {model.name: model for model in all_models}
            selector = ModelSelector(models_dict)
            selected_model_names = selector.select(select)

            if not selected_model_names:
                console.print(f"No models matched the selector: {select}")
                sys.exit(0)

            # Filter models based on selection
            models = [
                model for model in all_models if model.name in selected_model_names
            ]
        else:
            models = all_models

        # Sort models by name for consistent output
        models.sort(key=lambda m: m.name)

        # Output as JSON if requested
        if output_json:
            models_json = []
            for model in models:
                model_dict = {
                    "name": model.name,
                    "schema": model.schema,
                    "materialization": model.materialization,
                }
                # Add selected fields for non-verbose output
                if verbose:
                    model_dict.update(
                        {
                            "description": model.description,
                            "tags": model.tags,
                            "column_count": len(model.columns) if model.columns else 0,
                        }
                    )
                models_json.append(model_dict)

            print(json.dumps(models_json, indent=2))
            return

        # Create table display
        table = Table(title=f"dbt Models (showing {len(models)} models)")
        table.add_column("Name", style="cyan")
        table.add_column("Schema", style="green")
        table.add_column("Type", style="blue")

        if verbose:
            table.add_column("Columns", justify="right")
            table.add_column("Description")

        for model in models:
            row = [model.name, model.schema or "", model.materialization or "view"]

            if verbose:
                # Add column count
                column_count = len(model.columns) if model.columns else 0
                row.append(str(column_count))

                # Add description (truncated)
                description = model.description or ""
                if len(description) > 60:
                    description = description[:57] + "..."
                row.append(description)

            table.add_row(*row)

        console.print(table)
        console.print(f"\nShowing {len(models)} models.")
        console.print(
            "Use 'dbt-llm model-details [model_name]' to see details of a specific model."
        )

    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
