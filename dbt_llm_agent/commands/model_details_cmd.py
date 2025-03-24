"""
Model details command for dbt-llm-agent CLI.
"""

import click
import sys

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.commands.utils import (
    get_env_var,
    set_logging_level,
    colored_echo,
    format_model_as_yaml,
)

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.argument("model_name", required=True)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--yaml", is_flag=True, help="Output as dbt YAML document")
@click.option("--sql", is_flag=True, help="Output the raw SQL code")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def model_details(model_name, postgres_uri, yaml, sql, verbose):
    """
    Get details for a specific dbt model.

    MODEL_NAME is the name of the model to view details for.
    """
    try:
        # Set logging level based on verbosity
        set_logging_level(verbose)

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage

        # Get PostgreSQL URI from args or env var
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Initialize storage
        logger.info(f"Connecting to PostgreSQL database: {postgres_uri}")
        postgres = PostgresStorage(postgres_uri)

        # Get model
        model = postgres.get_model(model_name)
        if not model:
            logger.error(f"Model '{model_name}' not found in the database")
            sys.exit(1)

        if sql:
            # Show the raw SQL code
            if model.raw_sql:
                colored_echo(f"-- SQL for model: {model_name}", color="INFO", bold=True)
                colored_echo(model.raw_sql, color="INFO")
            else:
                colored_echo(
                    f"No SQL code found for model: {model_name}", color="WARNING"
                )
        elif yaml:
            # Format model as dbt YAML document
            yaml_output = format_model_as_yaml(model)
            colored_echo(yaml_output, color="INFO")
        else:
            # Show readable representation
            colored_echo(model.get_readable_representation(), color="INFO")

    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        sys.exit(1)
