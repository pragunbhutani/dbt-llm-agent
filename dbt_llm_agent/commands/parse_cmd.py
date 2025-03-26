"""
Parse command for dbt-llm-agent CLI.
"""

import click
import logging
import pathlib
import sys

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import (
    get_env_var,
    colored_echo,
    set_logging_level,
    load_dotenv_once,
)

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.argument(
    "project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--select",
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
    default=None,
)
@click.option("--force", is_flag=True, help="Force re-parsing of all models")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def parse(project_path, select, force, verbose):
    """
    Parse a dbt project and store models in the database.

    PROJECT_PATH is the path to the root of the dbt project.
    """
    try:
        # Set logging level based on verbosity
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Load environment variables from .env file (if not already loaded)
        load_dotenv_once()

        # Normalize and validate project path
        project_path = pathlib.Path(project_path).resolve()
        if not project_path.exists():
            logger.error(f"Project path does not exist: {project_path}")
            sys.exit(1)

        if not (project_path / "dbt_project.yml").exists():
            logger.error(
                f"Not a valid dbt project (no dbt_project.yml found): {project_path}"
            )
            sys.exit(1)

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage
        from dbt_llm_agent.core.dbt_parser import DBTProjectParser
        from dbt_llm_agent.utils.model_selector import ModelSelector

        # Get PostgreSQL URI from environment
        postgres_uri = get_env_var("POSTGRES_URI")
        if not postgres_uri:
            logger.error(
                "PostgreSQL URI not provided in environment variables (.env file)"
            )
            sys.exit(1)

        # Initialize storage
        logger.info(f"Connecting to PostgreSQL database: {postgres_uri}")
        postgres = PostgresStorage(postgres_uri)

        # Initialize parser
        logger.info(f"Parsing dbt project at: {project_path}")
        parser = DBTProjectParser(project_path)

        # Parse project
        project = parser.parse_project()

        # Create model selector if selection is provided
        if select:
            logger.info(f"Filtering models with selector: {select}")
            selector = ModelSelector(project.models)
            selected_models = selector.select(select)
            logger.info(f"Selected {len(selected_models)} models")

            # Filter project.models to only include selected models
            project.models = {
                name: model
                for name, model in project.models.items()
                if name in selected_models
            }

        # Store models in database
        logger.info(f"Found {len(project.models)} models")
        if force:
            logger.info("Force flag enabled - re-parsing all models")

        for model_name, model in project.models.items():
            if verbose:
                logger.debug(f"Processing model: {model_name}")
            postgres.store_model(model, force=force)

        logger.info(f"Successfully parsed and stored {len(project.models)} models")

        # Store sources if available
        if hasattr(project, "sources") and project.sources:
            logger.info(f"Found {len(project.sources)} sources")
            for source_name, source in project.sources.items():
                if verbose:
                    logger.debug(f"Processing source: {source_name}")
                postgres.store_source(source, force=force)

            logger.info(
                f"Successfully parsed and stored {len(project.sources)} sources"
            )

        return 0

    except Exception as e:
        logger.error(f"Error parsing dbt project: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
