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
import json

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
    # Define a mapping of capitalized color names to the COLORS keys
    color_mapping = {
        "BLUE": "INFO",  # Use INFO (green) for BLUE
        "GREEN": "INFO",  # Use INFO (green) for GREEN
        "CYAN": "DEBUG",  # Use DEBUG (cyan) for CYAN
        "RED": "ERROR",  # Use ERROR (red) for RED
        "YELLOW": "WARNING",  # Use WARNING (yellow) for YELLOW
    }

    prefix = ""
    suffix = COLORS["RESET"]

    # Map color name to a color in the COLORS dict if needed
    if color and color in color_mapping and color_mapping[color] in COLORS:
        prefix += COLORS[color_mapping[color]]
    elif color and color in COLORS:
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


def set_logging_level(verbose: bool):
    """Set the logging level based on the verbose flag.

    Args:
        verbose: Whether to enable verbose logging
    """
    # Just set the level of our logger, don't reconfigure logging
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value from environment variables.

    Args:
        key: The configuration key
        default: The default value if the key is not found

    Returns:
        The configuration value or the default value
    """
    env_var = f"{key.upper()}"
    return get_env_var(env_var, default)


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
    "--embedding-model", help="Embedding model to use", default="text-embedding-ada-002"
)
@click.option("--force", is_flag=True, help="Force re-embedding of models")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--documentation-only",
    is_flag=True,
    help="Only embed documentation (not interpretation)",
)
@click.option(
    "--interpret",
    is_flag=True,
    help="Interpret models if their interpretation is missing and include interpretations in the embedding. Will create embeddings even for models without documentation.",
)
def embed(
    select,
    postgres_uri,
    embedding_model,
    force,
    verbose,
    documentation_only,
    interpret,
):
    """Embed model documentation in the vector database.

    This command creates vector embeddings for model documentation
    to enable semantic search capabilities.

    Models can be selected using dbt selector syntax with the --select option.

    Use --interpret to generate interpretations for models that don't have one
    and include them in the embedding for improved semantic search results.
    This uses the OpenAI API to generate interpretations. When a model has no
    documentation but has an interpretation (either existing or newly generated),
    a placeholder documentation will be created from the interpretation.
    """
    set_logging_level(verbose)

    # Import modules
    from dbt_llm_agent.storage.postgres_storage import PostgresStorage
    from dbt_llm_agent.storage.vector_store import PostgresVectorStore
    from dbt_llm_agent.utils.model_selector import ModelSelector
    from dbt_llm_agent.core.agent import DBTAgent

    # Load configuration
    if not postgres_uri:
        postgres_uri = get_env_var("POSTGRES_URI")

    # Validate configuration
    if not postgres_uri:
        logger.error(
            "PostgreSQL URI not provided. Use --postgres-uri or set POSTGRES_URI env var."
        )
        sys.exit(1)

    try:
        # Initialize storage and model selector
        logger.info(f"Connecting to PostgreSQL database: {postgres_uri}")
        postgres = PostgresStorage(postgres_uri)

        # Initialize vector store using the same postgres URI
        logger.info(f"Connecting to vector database: {postgres_uri}")
        vector_store = PostgresVectorStore(
            connection_string=postgres_uri,
            embedding_model=embedding_model,
        )

        # Get all models
        logger.info("Retrieving models from database")
        all_models = postgres.get_all_models()
        if not all_models:
            logger.error("No models found in database")
            sys.exit(1)

        # Create model dictionary for selector
        model_dict = {model.name: model for model in all_models}

        # Select models to embed
        logger.info(f"Selecting models with selector: {select}")
        selector = ModelSelector(model_dict)
        selected_model_names = selector.select(select)

        if not selected_model_names:
            logger.error(f"No models matched the selector: {select}")
            sys.exit(1)

        logger.info(f"Selected {len(selected_model_names)} models for embedding")

        # Initialize DBTAgent if interpret flag is set
        agent = None
        if interpret:
            openai_api_key = get_env_var("OPENAI_API_KEY")
            openai_model = get_env_var("OPENAI_MODEL", "gpt-4-turbo")

            if not openai_api_key:
                logger.error("OpenAI API key not provided. Set OPENAI_API_KEY env var.")
                sys.exit(1)

            logger.info("Initializing DBT Agent for model interpretation")
            agent = DBTAgent(
                postgres_storage=postgres,
                vector_store=vector_store,
                openai_api_key=openai_api_key,
                model_name=openai_model,
            )

        # Create embeddings for each selected model
        success_count = 0
        interpret_count = 0

        for model_name in selected_model_names:
            if verbose:
                logger.debug(f"Processing model: {model_name}")

            model = model_dict.get(model_name)
            if not model:
                logger.warning(f"Model {model_name} not found, skipping")
                continue

            # If interpret flag is set and model doesn't have interpretation, interpret it
            if interpret and agent and not model.interpreted_description:
                logger.info(
                    f"Model {model_name} has no interpretation, interpreting now"
                )

                # Interpret the model
                result = agent.interpret_model(model_name)

                if "success" in result and result["success"]:
                    # Save the interpretation
                    save_result = agent.save_interpreted_documentation(
                        model_name,
                        result["yaml_documentation"],
                        embed=False,  # Don't embed yet, we'll do it below
                    )

                    if save_result["success"]:
                        interpret_count += 1
                        logger.info(f"Successfully interpreted model {model_name}")

                        # Refresh model from database since it was updated
                        model = postgres.get_model(model_name)
                    else:
                        logger.warning(
                            f"Failed to save interpretation for model {model_name}"
                        )
                else:
                    logger.warning(f"Failed to interpret model {model_name}")

            # Store documentation embedding if available
            if model.documentation:
                logger.info(f"Storing documentation embedding for model {model.name}")
                vector_store.store_model_documentation(
                    model.name, model.documentation, force=force
                )
                success_count += 1
            # If no documentation but we have interpretation and interpret flag was set, use interpretation for embedding
            elif model.interpreted_description:
                logger.info(
                    f"No documentation available for model {model.name}, using interpretation for embedding"
                )
                # Use vector_store.store_model directly to utilize the homogeneous document structure
                vector_store.store_model(
                    model.name, model.get_readable_representation()
                )
                success_count += 1
            else:
                logger.warning(
                    f"Skipping documentation embedding for model {model.name} - no documentation or interpretation available"
                )

        logger.info(f"Successfully embedded {success_count} models")
        if interpret:
            logger.info(f"Successfully interpreted {interpret_count} models")

    except Exception as e:
        logger.error(f"Error embedding models: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--openai-api-key", help="OpenAI API key", envvar="OPENAI_API_KEY")
@click.option("--openai-model", help="OpenAI model to use", envvar="OPENAI_MODEL")
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Temperature for the OpenAI model",
    envvar="TEMPERATURE",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--no-color", is_flag=True, help="Disable colored output", default=False)
@click.option("--json", is_flag=True, help="Output results as JSON", default=False)
def ask(
    question,
    postgres_uri,
    openai_api_key,
    openai_model,
    temperature,
    verbose,
    no_color,
    json,
):
    """Ask a question about the dbt project.

    This command allows you to ask questions about your dbt project and get answers
    based on the information in the model documentation and schema.
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

        # Import necessary modules
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage
        from dbt_llm_agent.storage.vector_store import PostgresVectorStore
        from dbt_llm_agent.core.agent import DBTAgent
        from dbt_llm_agent.utils.model_selector import ModelSelector

        # Load configuration
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")

        if not openai_api_key:
            openai_api_key = get_env_var("OPENAI_API_KEY")

        if not openai_model:
            openai_model = get_env_var("OPENAI_MODEL")
            if not openai_model:
                openai_model = "gpt-4-turbo"

        # Validate configuration
        if not postgres_uri:
            logger.error("PostgreSQL URI not provided and not found in config")
            sys.exit(1)

        if not openai_api_key:
            logger.error("OpenAI API key not provided and not found in config")
            sys.exit(1)

        # Initialize storage and agent
        postgres_storage = PostgresStorage(postgres_uri)
        vector_store = PostgresVectorStore(postgres_uri)
        agent = DBTAgent(
            postgres_storage=postgres_storage,
            vector_store=vector_store,
            openai_api_key=openai_api_key,
            model_name=openai_model,
        )

        # Ask the question
        result = agent.answer_question(question)

        if "error" in result:
            logger.error(f"Error: {result['error']}")
            sys.exit(1)

        # Print the answer
        if not json:
            try:
                from rich.markdown import Markdown
                from rich.console import Console

                markdown = True
                console = Console()
            except ImportError:
                markdown = False
                console = None

            if markdown and console:
                md_answer = result["answer"]
                md = Markdown(md_answer)
                console.print(md)
            else:
                print(result["answer"])

            # Print models if available
            if "relevant_models" in result and result["relevant_models"]:
                print("\nBased on models:")
                for model_data in result["relevant_models"]:
                    model_info = f"- {model_data['name']}"
                    print(model_info)

        # Store the question and answer if there's a result
        if "answer" in result and "relevant_models" in result:
            try:
                from dbt_llm_agent.storage.question_service import (
                    QuestionTrackingService,
                )

                # Initialize question tracking service
                question_tracking = QuestionTrackingService(postgres_uri)

                question_id = question_tracking.record_question(
                    question_text=question,
                    answer_text=result["answer"],
                    model_names=[m["name"] for m in result["relevant_models"]],
                )

                logger.info(
                    f"Question and answer stored with ID: {question_id}. "
                    f"Use 'dbt-llm-agent feedback {question_id} --useful=True/False' to provide feedback."
                )
            except Exception as e:
                logger.error(f"Error storing question: {e}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
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


@cli.command()
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


@cli.command()
@click.option(
    "--select",
    required=True,
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def list(select, postgres_uri, verbose):
    """List selected models from the database."""
    # Import required modules
    from dbt_llm_agent.storage.postgres_storage import PostgresStorage
    from dbt_llm_agent.utils.model_selector import ModelSelector

    set_logging_level(verbose)

    if not postgres_uri:
        postgres_uri = get_config_value("postgres_uri")

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


@cli.command()
@click.argument("model_name", required=False)
@click.option(
    "--select",
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
    default=None,
)
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
@click.option("--openai-api-key", help="OpenAI API key", envvar="OPENAI_API_KEY")
@click.option("--openai-model", help="OpenAI model to use", envvar="OPENAI_MODEL")
@click.option(
    "--no-save", is_flag=True, help="Don't save the documentation to the database"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--embed", is_flag=True, help="Embed the interpretation in the vector store"
)
def interpret(
    model_name,
    select,
    postgres_uri,
    openai_api_key,
    openai_model,
    no_save,
    verbose,
    embed,
):
    """Interpret a model and generate documentation for it.

    This command analyzes the SQL code of a model along with its upstream dependencies
    to generate documentation in YAML format.

    By default, the documentation is saved to the database. Use --no-save to disable this.

    You can interpret a single model by providing MODEL_NAME as an argument
    or interpret multiple models using the --select option with dbt selector syntax.
    """
    set_logging_level(verbose)

    # Import necessary modules
    from dbt_llm_agent.storage.postgres_storage import PostgresStorage
    from dbt_llm_agent.storage.vector_store import PostgresVectorStore
    from dbt_llm_agent.core.agent import DBTAgent
    from dbt_llm_agent.utils.model_selector import ModelSelector

    # Validate that either model_name or select is provided, but not both
    if model_name and select:
        logger.error(
            "Cannot provide both MODEL_NAME and --select. Please use only one."
        )
        sys.exit(1)

    if not model_name and not select:
        logger.error("Must provide either MODEL_NAME or --select option.")
        sys.exit(1)

    # Load configuration and override with command line arguments
    if not postgres_uri:
        postgres_uri = get_config_value("postgres_uri")

    if not openai_api_key:
        openai_api_key = get_config_value("openai_api_key")

    if not openai_model:
        openai_model = get_config_value("openai_model")
        if not openai_model:
            openai_model = "gpt-4-turbo"

    # Validate configuration
    if not postgres_uri:
        logger.error("PostgreSQL URI not provided and not found in config")
        sys.exit(1)

    if not openai_api_key:
        logger.error("OpenAI API key not provided and not found in config")
        sys.exit(1)

    try:
        # Initialize storage and agent
        postgres_storage = PostgresStorage(postgres_uri)
        # Use postgres_uri also for vector store connection
        vector_store = PostgresVectorStore(postgres_uri)
        agent = DBTAgent(
            postgres_storage=postgres_storage,
            vector_store=vector_store,
            openai_api_key=openai_api_key,
            model_name=openai_model,
        )

        # Process model selection if --select is used
        models_to_interpret = []
        if select:
            logger.info(f"Selecting models with selector: {select}")
            # Get all models from the database
            all_models = postgres_storage.get_all_models()
            if not all_models:
                logger.error(
                    "No models found in the database. Run 'parse' command first."
                )
                sys.exit(1)

            # Convert models to dictionary format for selector
            models_dict = {model.name: model for model in all_models}

            # Use ModelSelector to select models
            selector = ModelSelector(models_dict)
            selected_model_names = selector.select(select)

            if not selected_model_names:
                logger.error(f"No models found matching selector: {select}")
                sys.exit(1)

            logger.info(f"Selected {len(selected_model_names)} models to interpret")
            models_to_interpret = selected_model_names
        else:
            # Check if the single model exists
            model = postgres_storage.get_model(model_name)
            if not model:
                logger.error(f"Model '{model_name}' not found in the database")
                sys.exit(1)
            models_to_interpret = [model_name]

        # Process each model
        for current_model_name in models_to_interpret:
            logger.info(f"Interpreting model: {current_model_name}")

            # Interpret the model
            result = agent.interpret_model(current_model_name)

            if "error" in result:
                logger.error(f"Error interpreting model: {result['error']}")
                continue

            # If verbose, print the prompt that was used
            if verbose and "prompt" in result:
                logger.info("Prompt used for interpretation:")
                print(result["prompt"])

            # Print the resulting YAML documentation
            if verbose:
                logger.info(f"Generated YAML Documentation for {current_model_name}:")
            print(result["yaml_documentation"])

            # Save the documentation by default unless --no-save is provided
            if not no_save:
                logger.info("Saving documentation to database...")
                save_result = agent.save_interpreted_documentation(
                    current_model_name, result["yaml_documentation"], embed=embed
                )

                if save_result["success"]:
                    logger.info(
                        f"Documentation saved successfully for model: {current_model_name}"
                    )

                    # Embed the interpretation if requested
                    if embed:
                        logger.info("Embedding interpretation in vector store...")
                        # Get the updated model with the interpretation
                        updated_model = postgres_storage.get_model(current_model_name)
                        if updated_model and updated_model.interpreted_description:
                            vector_store.store_model_interpretation(
                                current_model_name,
                                updated_model.interpreted_description,
                            )
                            logger.info(
                                f"Interpretation embedded successfully for model: {current_model_name}"
                            )
                        else:
                            logger.error("Could not find interpretation to embed")
                else:
                    error_msg = save_result.get("error", "Unknown error")
                    logger.error(f"Failed to save documentation: {error_msg}")

    except Exception as e:
        logger.error(f"Error interpreting model: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)


@cli.command()
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


def format_model_as_yaml(model):
    """Format a DBT model as a dbt YAML document.

    Args:
        model: The DBT model to format

    Returns:
        A string containing the YAML document
    """
    # Start the YAML document
    yaml_lines = ["version: 2", "", "models:", f"  - name: {model.name}"]

    # Add description
    description = (
        model.interpreted_description
        if model.interpreted_description
        else model.description
    )
    if description:
        # Indent multiline descriptions correctly
        formatted_desc = description.replace("\n", "\n      ")
        yaml_lines.append(f"    description: >\n      {formatted_desc}")

    # Add model metadata
    if model.schema:
        yaml_lines.append(f"    schema: {model.schema}")

    if model.database:
        yaml_lines.append(f"    database: {model.database}")

    if model.materialization:
        yaml_lines.append(f"    config:")
        yaml_lines.append(f"      materialized: {model.materialization}")

    if model.tags:
        yaml_lines.append(f"    tags: [{', '.join(model.tags)}]")

    # Add column specifications
    if model.columns or model.interpreted_columns:
        yaml_lines.append("    columns:")

        # First try to use YML columns with their full specs
        if model.columns:
            for col_name, col in model.columns.items():
                yaml_lines.append(f"      - name: {col_name}")
                if col.description:
                    # Indent multiline descriptions correctly
                    col_desc = col.description.replace("\n", "\n          ")
                    yaml_lines.append(f"        description: >\n          {col_desc}")
                if col.data_type:
                    yaml_lines.append(f"        data_type: {col.data_type}")
        # Fall back to interpreted columns if available
        elif model.interpreted_columns:
            for col_name, description in model.interpreted_columns.items():
                yaml_lines.append(f"      - name: {col_name}")
                # Indent multiline descriptions correctly
                col_desc = description.replace("\n", "\n          ")
                yaml_lines.append(f"        description: >\n          {col_desc}")

    # Add tests if available
    if model.tests:
        test_lines = []
        for test in model.tests:
            if test.column_name:
                # This is a column-level test, which should go under the column definition
                continue
            else:
                # This is a model-level test
                test_name = test.test_type or test.name
                if test_name:
                    test_lines.append(f"      - {test_name}")

        if test_lines:
            yaml_lines.append("    tests:")
            yaml_lines.extend(test_lines)

    return "\n".join(yaml_lines)


def main():
    # Set up colored logging - only configure once
    if not logging.root.handlers:
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
