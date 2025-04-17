"""
Embedding command for ragstar CLI.

This command allows embedding models into vector storage for semantic search.
"""

import click
import sys
import json
import logging
from typing import Dict, List, Optional

from ragstar.utils.logging import get_logger
from ragstar.utils.cli_utils import get_config_value, set_logging_level
from ragstar.utils.model_selector import ModelSelector
from dotenv import load_dotenv_once

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.option(
    "--select",
    "-s",
    help="Select models to embed (e.g., 'customers' or '+tag:marts')",
    required=True,
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force regeneration of embeddings for existing models",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def embed(select, force, verbose):
    """Embed models in the vector store.

    This command embeds selected models in the vector store for semantic search.
    You can select models using the same syntax as dbt's select option.

    Examples:
        dbt-llm embed --select "customers"
        dbt-llm embed --select "+tag:marts"
        dbt-llm embed --select "*" --force
    """
    set_logging_level(verbose)

    load_dotenv_once()

    # Load configuration from environment
    postgres_uri = get_config_value("database_url")
    openai_api_key = get_config_value("openai_api_key")
    openai_embedding_model = get_config_value("openai_embedding_model")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        # Import necessary modules
        from ragstar.storage.model_storage import ModelStorage
        from ragstar.storage.model_embedding_storage import ModelEmbeddingStorage

        logger.info("Connecting to database...")
        model_storage = ModelStorage(postgres_uri)

        # First, get all models from database
        logger.info("Retrieving models from database...")
        models = model_storage.get_all_models()
        logger.info(f"Found {len(models)} models in total")

        # Create model selector
        models_dict = {model.name: model for model in models}
        selector = ModelSelector(models_dict)
        selected_model_names = selector.select(select)

        if not selected_model_names:
            logger.error(f"No models matched the selector: {select}")
            sys.exit(1)

        logger.info(f"Selected {len(selected_model_names)} models for embedding")

        # Filter to only selected models
        selected_models = [
            model for model in models if model.name in selected_model_names
        ]

        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = ModelEmbeddingStorage(connection_string=postgres_uri)

        # Embed each model individually to track progress
        logger.info("Embedding models...")
        total_models = len(selected_models)

        for i, model in enumerate(selected_models):
            model_name = model.name
            logger.info(f"Embedding model: {model_name} ({i+1}/{total_models})")

            model_text = model.get_text_representation()

            # Create metadata
            metadata = {
                "schema": model.schema,
                "materialization": model.materialization,
            }
            if hasattr(model, "tags") and model.tags:
                metadata["tags"] = model.tags

            # Store model with progress info
            try:
                vector_store.store_model_embedding(
                    model_name=model_name, model_text=model_text, metadata=metadata
                )
            except Exception as e:
                logger.error(f"Error embedding model {model_name}: {str(e)}")
                if verbose:
                    import traceback

                    logger.debug(traceback.format_exc())

        logger.info(f"Successfully embedded {total_models} models")

    except Exception as e:
        logger.error(f"Error embedding models: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
