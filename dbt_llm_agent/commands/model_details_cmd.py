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
            # Use the model's own method to convert to embedding-compatible JSON
            result = model.to_embedding_json()

            # Output as JSON
            print(json.dumps(result, indent=2))
            return

        # Display model details using the same format as embeddings
        embedding_text = model.to_embedding_text()
        console.print(embedding_text)

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
