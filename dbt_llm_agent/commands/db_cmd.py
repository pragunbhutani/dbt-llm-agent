"""
Database management commands for dbt-llm-agent CLI.
"""

import click
import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError, ProgrammingError
from alembic.config import Config
from alembic import command
from rich.console import Console
from rich.prompt import Confirm

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import get_config_value, set_logging_level
from dbt_llm_agent.core.models import Base

# Initialize logger and console
logger = get_logger(__name__)
console = Console()


@click.command()
@click.option("--revision", help="Target revision (default: head)", default="head")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def migrate(revision, verbose):
    """Update the database schema to the latest version.

    This command applies Alembic migrations to update the database schema.

    You can specify a specific revision with --revision (default: head).
    """
    set_logging_level(verbose)

    # Load configuration from environment
    postgres_uri = get_config_value("postgres_uri")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        logger.info("Running database migrations...")

        # Initialize ModelStorage and apply migrations explicitly
        from dbt_llm_agent.storage.model_storage import ModelStorage

        model_storage = ModelStorage(postgres_uri)

        # Apply migrations using the storage class method
        success = model_storage.apply_migrations()

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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def init_db(verbose):
    """Initialize the database schema and Alembic tracking."""
    set_logging_level(verbose)
    logger.info("Initializing database schema...")
    postgres_uri = get_config_value("postgres_uri")
    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        engine = create_engine(postgres_uri)
        with engine.connect() as connection:
            # Ensure pgvector extension is enabled
            try:
                logger.info("Ensuring pgvector extension is enabled...")
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                connection.commit()
                logger.info("pgvector extension is enabled.")
            except ProgrammingError as e:
                logger.error(f"Error enabling pgvector extension: {e}")
                console.print(
                    "[red]Failed to enable pgvector extension. Check database user permissions.[/red]"
                )
                connection.rollback()
                sys.exit(1)
            except Exception as e:
                logger.error(f"Unexpected error enabling pgvector extension: {e}")
                console.print(
                    "[red]An unexpected error occurred while enabling pgvector.[/red]"
                )
                connection.rollback()
                sys.exit(1)

            # Create all tables defined in models.py
            logger.info("Creating database tables...")
            Base.metadata.create_all(engine)
            logger.info("Database tables created successfully.")

            # Stamp the database with the latest Alembic revision
            logger.info("Stamping database with latest Alembic revision...")
            alembic_cfg = Config("alembic.ini")
            command.stamp(alembic_cfg, "head")
            logger.info("Database stamped successfully.")

        console.print("[green]Database initialized successfully.[/green]")

    except OperationalError as e:
        logger.error(f"Could not connect to the database: {e}")
        console.print("[red]Error: Could not connect to the database.[/red]")
        console.print(f"Check your POSTGRES_URI in the .env file: {postgres_uri}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)


@click.command()
@click.option("--force", is_flag=True, help="Force reset without confirmation")
@click.option(
    "--drop-tables",
    is_flag=True,
    help="Drop tables completely instead of just truncating them",
)
@click.option(
    "--cascade",
    is_flag=True,
    help="Use CASCADE option when dropping tables to automatically drop dependent objects",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def reset_db(force, drop_tables, cascade, verbose):
    """Reset the database by clearing all data.

    This command truncates all tables, removing all data while preserving the schema.
    Use --drop-tables to completely drop the tables instead of just truncating them.
    Use --cascade with --drop-tables to automatically drop dependent tables.
    Use with caution as this operation cannot be undone.
    """
    set_logging_level(verbose)

    # Load configuration from environment
    postgres_uri = get_config_value("postgres_uri")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    # Confirm action unless --force is used
    drop_message = "DROP TABLES" if drop_tables else "DELETE"
    cascade_warning = " AND ALL DEPENDENT OBJECTS" if cascade and drop_tables else ""
    if not force:
        confirmation = input(
            f"WARNING: This will {'DROP ALL TABLES' + cascade_warning + ' in' if drop_tables else 'delete ALL data from'} the database. This action cannot be undone.\n"
            f"Type '{drop_message}' to confirm: "
        )
        if confirmation != drop_message:
            logger.info("Reset cancelled")
            sys.exit(0)

    try:
        action = "Dropping tables" if drop_tables else "Resetting database"
        logger.info(f"{action}...")

        # Import necessary modules
        import sqlalchemy as sa
        from sqlalchemy import inspect, text
        from dbt_llm_agent.core.models import (
            Base,
            ModelTable,
            QuestionTable,
            QuestionModelTable,
            ModelEmbeddingTable,
            ColumnTable,
            TestTable,
            DependencyTable,
        )

        # Create engine
        engine = sa.create_engine(postgres_uri)
        inspector = inspect(engine)

        # Create a session
        Session = sa.orm.sessionmaker(bind=engine)
        session = Session()

        # Complete list of all known tables in dependency order (children/dependent tables first)
        tables_to_process = [
            (ColumnTable, "columns"),
            (TestTable, "tests"),
            (DependencyTable, "dependencies"),
            (QuestionModelTable, "question_models"),
            (ModelEmbeddingTable, "model_embeddings"),
            (QuestionTable, "questions"),
            (ModelTable, "models"),
        ]

        try:
            # Process tables in dependency order
            logger.info(f"{'Dropping' if drop_tables else 'Truncating'} tables...")

            existing_tables = inspector.get_table_names()
            logger.info(f"Found tables in database: {', '.join(existing_tables)}")

            success_count = 0
            if drop_tables and cascade:
                # With CASCADE option, we can drop parent tables directly
                # and PostgreSQL will handle dropping dependent tables
                for model_class, table_name in reversed(tables_to_process):
                    if table_name in existing_tables:
                        try:
                            # Use raw SQL to drop table with CASCADE
                            connection = engine.connect()
                            connection.execute(text(f"DROP TABLE {table_name} CASCADE"))
                            connection.commit()
                            connection.close()
                            logger.info(f"Dropped table {table_name} CASCADE")
                            success_count += 1
                        except Exception as e:
                            logger.warning(f"Could not drop {table_name}: {str(e)}")
                    else:
                        logger.warning(f"Table {table_name} does not exist, skipping")
            else:
                # Process each table in dependency order
                for model_class, table_name in tables_to_process:
                    if table_name in existing_tables:
                        try:
                            if drop_tables:
                                # Drop the table
                                model_class.__table__.drop(engine)
                                logger.info(f"Dropped table {table_name}")
                            else:
                                # Delete data from the table
                                session.query(model_class).delete()
                                logger.info(f"Deleted data from {table_name}")
                            success_count += 1
                        except Exception as e:
                            action = "drop" if drop_tables else "truncate"
                            logger.warning(f"Could not {action} {table_name}: {str(e)}")
                    else:
                        logger.warning(f"Table {table_name} does not exist, skipping")

            # Commit the transaction if not dropping tables
            if not drop_tables:
                session.commit()

            action = "Dropped" if drop_tables else "Reset"
            logger.info(
                f"{action} {success_count} of {len(tables_to_process)} tables successfully"
            )

        except Exception as e:
            if not drop_tables:
                session.rollback()
            logger.error(f"Error during table processing: {str(e)}")
            raise
        finally:
            session.close()

        logger.info(
            f"Database {'reset' if not drop_tables else 'cleanup'} completed successfully"
        )

    except Exception as e:
        logger.error(
            f"Error {'dropping tables' if drop_tables else 'resetting database'}: {str(e)}"
        )
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
