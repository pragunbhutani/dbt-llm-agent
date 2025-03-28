"""
Database management commands for dbt-llm-agent CLI.
"""

import click
import sys

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import get_config_value, set_logging_level

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--revision", help="Target revision (default: head)", default="head")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def migrate(postgres_uri, revision, verbose):
    """Update the database schema to the latest version.

    This command applies Alembic migrations to update the database schema.

    You can specify a specific revision with --revision (default: head).
    """
    set_logging_level(verbose)

    # Load configuration if not provided
    if not postgres_uri:
        postgres_uri = get_config_value("postgres_uri")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided and not found in config")
        sys.exit(1)

    try:
        logger.info("Running database migrations...")

        # Initialize PostgresStorage and apply migrations explicitly
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage

        postgres_storage = PostgresStorage(postgres_uri)

        # Apply migrations using the storage class method
        success = postgres_storage.apply_migrations()

        if success:
            logger.info("Migrations completed successfully")
        else:
            logger.error("Migration failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)


@click.command()
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def init_db(postgres_uri, verbose):
    """Initialize the database schema.

    This command creates all tables and initializes the database with the latest schema.
    """
    set_logging_level(verbose)

    # Import necessary modules
    import sqlalchemy as sa
    from dbt_llm_agent.storage.models import Base
    from dbt_llm_agent.storage.postgres_storage import PostgresStorage

    # Load configuration if not provided
    if not postgres_uri:
        postgres_uri = get_config_value("postgres_uri")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided and not found in config")
        sys.exit(1)

    try:
        logger.info("Initializing database schema...")

        # Create the storage instance
        postgres_storage = PostgresStorage(postgres_uri)

        # Apply migrations explicitly
        success = postgres_storage.apply_migrations()

        if success:
            logger.info("Database initialization completed successfully")
        else:
            logger.error("Database initialization failed during migration step")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
