"""
CLI interface for dbt-llm-agent
"""

import click
import os
import subprocess
import sys
import logging
from typing import Optional, Any
import pathlib

# Set up logging
from dbt_llm_agent.utils.logging import setup_logging, get_logger, COLORS

# Initialize logging with default settings
setup_logging()
logger = get_logger(__name__)


def colored_echo(text, color=None, bold=False):
    """Echo text with color and styling.

    Args:
        text: The text to echo
        color: Color name from COLORS dict or color code
        bold: Whether to make the text bold
    """
    prefix = ""
    suffix = COLORS["RESET"]

    if color and color in COLORS:
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


@click.group()
def cli():
    """dbt LLM Agent CLI"""
    pass


@cli.command()
def version():
    """Get the version of dbt-llm-agent"""
    colored_echo("dbt-llm-agent version 0.1.0", color="INFO", bold=True)


@cli.command()
@click.argument(
    "project_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option(
    "--select",
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
    default=None,
)
@click.option("--force", is_flag=True, help="Force re-parsing of all models")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def parse(project_path, postgres_uri, select, force, verbose):
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
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            logger.info("Loaded environment variables from .env file")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

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


@cli.command()
@click.option(
    "--select",
    required=True,
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option(
    "--postgres-connection-string",
    help="PostgreSQL connection string for vector database",
    envvar="POSTGRES_CONNECTION_STRING",
)
@click.option(
    "--embedding-model", help="Embedding model to use", default="text-embedding-ada-002"
)
@click.option("--force", is_flag=True, help="Force re-embedding of models")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def embed(
    select, postgres_uri, postgres_connection_string, embedding_model, force, verbose
):
    """
    Embed selected models in the vector database.
    """
    try:
        # Set logging level based on verbosity
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            logger.info("Loaded environment variables from .env file")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage
        from dbt_llm_agent.storage.vector_store import PostgresVectorStore
        from dbt_llm_agent.utils.model_selector import ModelSelector

        # Get PostgreSQL URI
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Get PostgreSQL connection string for vector database
        if not postgres_connection_string:
            postgres_connection_string = get_env_var("POSTGRES_CONNECTION_STRING")
            if not postgres_connection_string:
                postgres_connection_string = postgres_uri
                logger.info("Using POSTGRES_URI for vector database connection string")

        # Initialize storage
        logger.info(f"Connecting to PostgreSQL database: {postgres_uri}")
        postgres = PostgresStorage(postgres_uri)

        # Initialize vector store
        logger.info(f"Connecting to vector database: {postgres_connection_string}")
        vector_store = PostgresVectorStore(
            connection_string=postgres_connection_string,
            embedding_model=embedding_model,
        )

        # Get all models from the database
        all_models = postgres.get_all_models()
        logger.info(f"Found {len(all_models)} models in the database")

        # Create model selector
        logger.info(f"Selecting models with selector: {select}")
        models_dict = {model.name: model for model in all_models}
        selector = ModelSelector(models_dict)
        selected_model_names = selector.select(select)

        if not selected_model_names:
            logger.warning(f"No models matched the selector: {select}")
            return 0

        logger.info(f"Selected {len(selected_model_names)} models for embedding")

        # Filter to only selected models
        selected_models = [
            model for model in all_models if model.name in selected_model_names
        ]

        # Embed each model
        models_dict = {}
        metadata_dict = {}

        for model in selected_models:
            model_text = model.get_readable_representation()
            models_dict[model.name] = model_text

            # Create metadata
            metadata = {
                "schema": model.schema,
                "materialization": model.materialization,
            }
            if hasattr(model, "tags") and model.tags:
                metadata["tags"] = model.tags

            metadata_dict[model.name] = metadata

            if verbose:
                logger.debug(f"Prepared model {model.name} for embedding")

        # Store models in vector database
        logger.info(f"Storing {len(models_dict)} models in vector database")
        vector_store.store_models(models_dict, metadata_dict)

        logger.info(f"Successfully embedded {len(models_dict)} models")
        return 0

    except Exception as e:
        logger.error(f"Error embedding models: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option(
    "--postgres-connection-string",
    help="PostgreSQL connection string for vector database",
    envvar="POSTGRES_CONNECTION_STRING",
)
@click.option("--openai-api-key", help="OpenAI API key", envvar="OPENAI_API_KEY")
@click.option("--openai-model", help="OpenAI model to use", envvar="OPENAI_MODEL")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def ask(
    question,
    postgres_uri,
    postgres_connection_string,
    openai_api_key,
    openai_model,
    verbose,
):
    """
    Ask a question about your dbt project.
    """
    try:
        # Set logging level based on verbosity
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            logger.info("Loaded environment variables from .env file")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage
        from dbt_llm_agent.storage.vector_store import PostgresVectorStore
        from dbt_llm_agent.storage.question_service import QuestionTrackingService
        from dbt_llm_agent.core.agent import DBTAgent

        # Get PostgreSQL URI
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Get PostgreSQL connection string for vector database
        if not postgres_connection_string:
            postgres_connection_string = get_env_var("POSTGRES_CONNECTION_STRING")
            if not postgres_connection_string:
                postgres_connection_string = postgres_uri
                logger.info("Using POSTGRES_URI for vector database connection string")

        # Get OpenAI API key
        if not openai_api_key:
            openai_api_key = get_env_var("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error(
                    "OpenAI API key not provided. Please either:\n"
                    "1. Add OPENAI_API_KEY to your .env file\n"
                    "2. Pass it as --openai-api-key argument"
                )
                sys.exit(1)

        # Get OpenAI model
        if not openai_model:
            openai_model = get_env_var("OPENAI_MODEL", "gpt-4-turbo")

        # Initialize storage
        logger.info(f"Connecting to PostgreSQL database: {postgres_uri}")
        postgres = PostgresStorage(postgres_uri)

        # Initialize vector store
        logger.info(f"Connecting to vector database: {postgres_connection_string}")
        vector_store = PostgresVectorStore(connection_string=postgres_connection_string)

        # Initialize question tracking
        question_tracking = QuestionTrackingService(postgres_uri)

        # Initialize agent
        logger.info(f"Initializing DBT agent with {openai_model} model")
        agent = DBTAgent(
            postgres_storage=postgres,
            vector_store=vector_store,
            openai_api_key=openai_api_key,
            model_name=openai_model,
        )

        # Ask the question
        logger.info(f"Asking: {question}")
        result = agent.answer_question(question)

        # Output the answer
        colored_echo("\nAnswer:", color="INFO", bold=True)
        colored_echo(result["answer"])

        # List the models used
        if result["relevant_models"]:
            colored_echo("\nModels used:", color="INFO", bold=True)
            for model_data in result["relevant_models"]:
                colored_echo(f"- {model_data['name']}", color="DEBUG")

        # Record the question and answer
        model_names = [model["name"] for model in result["relevant_models"]]
        question_id = question_tracking.record_question(
            question_text=question,
            answer_text=result["answer"],
            model_names=model_names,
        )

        colored_echo(f"\nQuestion ID: {question_id}", color="INFO", bold=True)
        colored_echo("You can provide feedback on this answer with:", color="INFO")
        colored_echo(
            f"  dbt-llm-agent feedback {question_id} --useful=true", color="DEBUG"
        )

        return 0

    except Exception as e:
        logger.error(f"Error asking question: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("question_id", type=int)
@click.option("--useful", type=bool, required=True, help="Was the answer useful?")
@click.option("--feedback", help="Additional feedback")
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
def feedback(question_id, useful, feedback, postgres_uri):
    """
    Provide feedback on an answer.
    """
    try:
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.question_service import QuestionTrackingService

        # Get PostgreSQL URI
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Initialize question tracking
        question_tracking = QuestionTrackingService(postgres_uri)

        # Get the question to make sure it exists
        question = question_tracking.get_question(question_id)
        if not question:
            logger.error(f"Question with ID {question_id} not found")
            sys.exit(1)

        # Update feedback
        success = question_tracking.update_feedback(
            question_id=question_id, was_useful=useful, feedback=feedback
        )

        if success:
            colored_echo(
                f"Feedback recorded for question {question_id}", color="INFO", bold=True
            )
        else:
            colored_echo(
                f"Failed to record feedback for question {question_id}", color="ERROR"
            )

        return 0

    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        sys.exit(1)


@cli.command()
@click.option("--limit", type=int, default=10, help="Number of questions to show")
@click.option("--offset", type=int, default=0, help="Offset for pagination")
@click.option("--useful", type=bool, help="Filter by usefulness")
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
def questions(limit, offset, useful, postgres_uri):
    """
    List questions and answers.
    """
    try:
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.question_service import QuestionTrackingService

        # Get PostgreSQL URI
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Initialize question tracking
        question_tracking = QuestionTrackingService(postgres_uri)

        # Get questions
        questions = question_tracking.get_all_questions(
            limit=limit, offset=offset, was_useful=useful
        )

        if not questions:
            colored_echo("No questions found", color="WARNING")
            return 0

        colored_echo(f"Found {len(questions)} questions:", color="INFO", bold=True)
        for q in questions:
            colored_echo(f"\nID: {q['id']}", color="INFO", bold=True)
            colored_echo(f"Question: {q['question_text']}", color="INFO")
            colored_echo(f"Answer: {q['answer_text'][:100]}...", color="DEBUG")
            # Use different colors based on usefulness
            usefulness_color = "INFO" if q["was_useful"] else "WARNING"
            colored_echo(f"Was useful: {q['was_useful']}", color=usefulness_color)
            colored_echo(f"Models: {', '.join(q['models'])}", color="DEBUG")
            colored_echo(f"Created at: {q['created_at']}", color="DEBUG")

        return 0

    except Exception as e:
        logger.error(f"Error listing questions: {e}")
        sys.exit(1)


@cli.command()
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option(
    "--postgres-connection-string",
    help="PostgreSQL connection string for vector database",
    envvar="POSTGRES_CONNECTION_STRING",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--confirm", is_flag=True, help="Confirm database reset without prompting"
)
def cleanup(postgres_uri, postgres_connection_string, verbose, confirm):
    """
    Clean up database tables (for development or after schema changes).
    WARNING: This will drop all tables and recreate them.
    """
    try:
        # Set logging level based on verbosity
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            logger.info("Loaded environment variables from .env file")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage
        from dbt_llm_agent.storage.vector_store import PostgresVectorStore
        from dbt_llm_agent.storage.models import Base

        # Get PostgreSQL URI
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Get PostgreSQL connection string for vector database
        if not postgres_connection_string:
            postgres_connection_string = get_env_var("POSTGRES_CONNECTION_STRING")
            if not postgres_connection_string:
                postgres_connection_string = postgres_uri
                logger.info("Using POSTGRES_URI for vector database connection string")

        # Confirm reset if not already confirmed
        if not confirm:
            click.confirm(
                "WARNING: This will drop all tables and recreate them. Are you sure?",
                abort=True,
            )

        # Initialize database connections
        logger.info("Initializing database connections")
        from sqlalchemy import create_engine
        from sqlalchemy.sql import text

        metadata_engine = create_engine(postgres_uri)
        vector_engine = create_engine(postgres_connection_string)

        # Drop and recreate tables
        logger.info("Dropping and recreating metadata tables")
        Base.metadata.drop_all(metadata_engine)
        Base.metadata.create_all(metadata_engine)

        logger.info("Dropping and recreating vector tables")
        # Make sure pgvector extension exists
        with vector_engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        Base.metadata.drop_all(vector_engine)
        Base.metadata.create_all(vector_engine)

        logger.info("Database cleanup complete")
        return 0

    except Exception as e:
        logger.error(f"Error cleaning up database: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    "--select",
    required=True,
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def list(select, postgres_uri, verbose):
    """
    List all models matching the selection criteria.
    """
    try:
        # Set logging level based on verbosity
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
            logger.info("Loaded environment variables from .env file")
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage
        from dbt_llm_agent.utils.model_selector import ModelSelector

        # Get PostgreSQL URI
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

        # Get all models from the database
        all_models = postgres.get_all_models()
        logger.info(f"Found {len(all_models)} models in the database")

        # Create model selector
        logger.info(f"Selecting models with selector: {select}")
        models_dict = {model.name: model for model in all_models}
        selector = ModelSelector(models_dict)
        selected_model_names = selector.select(select)

        if not selected_model_names:
            logger.warning(f"No models matched the selector: {select}")
            return 0

        # Display selected models in alphabetical order
        colored_echo(
            f"Selected models ({len(selected_model_names)}):", color="INFO", bold=True
        )
        for model_name in sorted(selected_model_names):
            model = models_dict[model_name]
            # If verbose, show more details about each model
            if verbose:
                tags = getattr(model, "tags", [])
                materialization = getattr(model, "materialization", "unknown")
                tags_str = ", ".join(tags) if tags else "none"
                colored_echo(
                    f"  - {model_name} [materialization: {materialization}, tags: {tags_str}]",
                    color="DEBUG",
                )
            else:
                colored_echo(f"  - {model_name}", color="DEBUG")

        return 0

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def main():
    # Set up colored logging
    setup_logging()

    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
        logger.debug("Loaded environment variables from .env file")
    except ImportError:
        logger.warning(
            "python-dotenv not installed. Environment variables may not be properly loaded."
        )

    cli()


if __name__ == "__main__":
    main()
