"""
Command to display details of a dbt model.
"""

import click
import sys
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import get_config_value, set_logging_level

# Initialize logger
logger = get_logger(__name__)

# Initialize console for rich output
console = Console()


@click.command()
@click.argument("model_name", required=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
def model_details(model_name, verbose, output_json):
    """Display details of a dbt model.

    This command shows comprehensive information about a specific dbt model,
    including its description, columns, and relationship to other models.

    Examples:
        dbt-llm model-details customers
        dbt-llm model-details orders --json
    """
    set_logging_level(verbose)

    # Load configuration from environment
    postgres_uri = get_config_value("postgres_uri")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        # Import necessary modules
        from dbt_llm_agent.storage.model_storage import ModelStorage

        # Initialize storage
        model_storage = ModelStorage(postgres_uri)

        # Get model
        model = model_storage.get_model(model_name)
        if not model:
            logger.error(f"Model '{model_name}' not found in database")
            sys.exit(1)

        # Output as JSON if requested
        if output_json:
            model_dict = model.to_dict()
            print(json.dumps(model_dict, indent=2))
            return

        # Display model details
        console.print(f"\n[bold blue]Model:[/bold blue] {model.name}")

        if model.description:
            console.print("\n[bold]Description:[/bold]")
            console.print(model.description)

        if model.interpreted_description:
            console.print("\n[bold]Interpreted Description:[/bold]")
            console.print(model.interpreted_description)

        # Display materialiation and schema
        console.print(f"\n[bold]Materialization:[/bold] {model.materialization}")
        console.print(f"[bold]Schema:[/bold] {model.schema}")

        # Display columns
        if model.columns:
            console.print("\n[bold]Columns:[/bold]")
            columns_table = Table(show_header=True)
            columns_table.add_column("Name")
            columns_table.add_column("Description")

            for name, column in model.columns.items():
                columns_table.add_row(
                    name, column.description if column.description else ""
                )

            console.print(columns_table)

        # Display dependencies
        if model.depends_on:
            console.print("\n[bold]Depends on:[/bold]")
            for dep in model.depends_on:
                console.print(f"- {dep}")

        # Display upstream models
        if model.all_upstream_models:
            console.print("\n[bold]All upstream models:[/bold]")
            for upstream in model.all_upstream_models:
                console.print(f"- {upstream}")

        # Display tests
        if model.tests:
            console.print("\n[bold]Tests:[/bold]")
            tests_table = Table(show_header=True)
            tests_table.add_column("Type")
            tests_table.add_column("Column")

            for test in model.tests:
                tests_table.add_row(test.test_type or "", test.column_name or "")

            console.print(tests_table)

        # Display raw SQL if verbose
        if verbose and model.raw_sql:
            console.print("\n[bold]SQL:[/bold]")
            console.print(f"```sql\n{model.raw_sql}\n```")

    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
