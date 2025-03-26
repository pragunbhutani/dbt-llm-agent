"""
Embed command for dbt-llm-agent CLI.
"""

import click
import logging
import sys

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import (
    get_env_var,
    set_logging_level,
    colored_echo,
)

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.option(
    "--select",
    required=True,
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
)
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

    # Load configuration from environment
    postgres_uri = get_env_var("POSTGRES_URI")

    # Validate configuration
    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
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
