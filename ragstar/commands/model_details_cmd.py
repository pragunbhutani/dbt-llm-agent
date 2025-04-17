"""
Command to display details of a dbt model.
"""

import click
import sys
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from ragstar.utils.logging import get_logger
from ragstar.utils.cli_utils import get_config_value, set_logging_level
from dotenv import load_dotenv_once

# Initialize logger
logger = get_logger(__name__)

# Initialize console for rich output
console = Console()


@click.command()
@click.argument("model_name", required=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.option("--yaml", "output_yaml", is_flag=True, help="Output in YAML format")
@click.option(
    "--use-interpretation",
    is_flag=True,
    help="Use LLM-interpreted descriptions when generating YAML (use with --yaml)",
)
def model_details(model_name, verbose, output_json, output_yaml, use_interpretation):
    """Display details of a dbt model.

    This command shows comprehensive information about a specific dbt model,
    including its description, columns, and relationship to other models.

    Examples:
        dbt-llm model-details customers
        dbt-llm model-details orders --json
        dbt-llm model-details orders --yaml
        dbt-llm model-details orders --yaml --use-interpretation
    """
    set_logging_level(verbose)

    load_dotenv_once()

    # Load configuration from environment
    postgres_uri = get_config_value("database_url")
    openai_api_key = get_config_value("openai_api_key")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        # Import necessary modules
        from ragstar.storage.model_storage import ModelStorage
        from ragstar.core.models import DBTModel, Column

        # Initialize storage
        model_storage = ModelStorage(postgres_uri)

        # Get model
        model = model_storage.get_model(model_name)
        if not model:
            logger.error(f"Model '{model_name}' not found in database")
            sys.exit(1)

        # Output as JSON if requested
        if output_json:
            # Use the model's own method to convert to embedding-compatible JSON
            result = model.to_dict()

            # Output as JSON
            print(json.dumps(result, indent=2))
            return

        # Output as YAML if requested
        if output_yaml:
            if use_interpretation and model.interpreted_columns:
                # Create a copy of the model with interpreted columns converted to the format
                # expected by format_as_yaml
                model_copy = DBTModel(
                    name=model.name,
                    description=(
                        model.interpreted_description
                        if model.interpreted_description
                        else model.description
                    ),
                    schema=model.schema,
                    database=model.database,
                    materialization=model.materialization,
                    tags=model.tags,
                    depends_on=model.depends_on,
                    all_upstream_models=model.all_upstream_models,
                    path=model.path,
                    unique_id=model.unique_id,
                    meta=model.meta,
                    raw_sql=model.raw_sql,
                    compiled_sql=model.compiled_sql,
                    tests=model.tests,
                    interpretation_details=model.interpretation_details,
                )

                # Convert interpreted columns to Column objects
                if model.interpreted_columns:
                    model_copy.columns = {
                        col_name: Column(
                            name=col_name,
                            description=col_desc,
                            data_type=(
                                model.columns.get(col_name).data_type
                                if col_name in model.columns
                                else ""
                            ),
                        )
                        for col_name, col_desc in model.interpreted_columns.items()
                    }

                # Use the modified model for YAML generation
                yaml_representation = model_copy.format_as_yaml()
            else:
                # Use the model's format_as_yaml method to generate YAML
                yaml_representation = model.format_as_yaml()

            print(yaml_representation)
            return

        # Display model details using the same format as embeddings
        text_representation = model.get_text_representation()
        console.print(text_representation)

        # Display raw SQL if verbose (not included in embedding text)
        if verbose and model.raw_sql:
            console.print("\n[bold]SQL:[/bold]")
            console.print(f"```sql\n{model.raw_sql}\n```")

    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
