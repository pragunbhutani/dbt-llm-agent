"""
Command to interpret dbt models using LLM.

This command uses an LLM to analyze SQL and generate documentation for dbt models.
"""

import click
import sys
import logging
import yaml
import json
from typing import Dict, List, Optional, Any

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import get_config_value, set_logging_level
from dbt_llm_agent.utils.model_selector import ModelSelector

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.option(
    "--select",
    "-s",
    help="Select models to interpret (e.g., 'customers' or '+tag:marts')",
    required=True,
)
@click.option(
    "--save",
    is_flag=True,
    help="Save the interpretations to the database",
)
@click.option("--embed", is_flag=True, help="Embed interpreted models in vector store")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--only-when-missing",
    is_flag=True,
    help="Only interpret models that don't already have an interpretation",
)
@click.option(
    "--recursive",
    is_flag=True,
    help="Recursively interpret upstream models if they lack interpretation",
)
@click.option(
    "--force-recursive",
    is_flag=True,
    help="Recursively interpret all upstream models, even if they have existing interpretations",
)
def interpret(
    select, save, embed, verbose, only_when_missing, recursive, force_recursive
):
    """Interpret dbt models and generate documentation.

    This command analyzes selected dbt models using LLM and generates documentation,
    including model descriptions and column definitions.

    Examples:
        dbt-llm interpret --select "customers"
        dbt-llm interpret --select "+tag:marts" --save
        dbt-llm interpret --select "orders" --save --embed
        dbt-llm interpret --select "customers" --save --only-when-missing
        dbt-llm interpret --select "+fct_orders" --recursive --save
        dbt-llm interpret --select "+fct_orders" --force-recursive --save
    """
    set_logging_level(verbose)

    # Add warning if recursive flags are used without save
    if (recursive or force_recursive) and not save:
        logger.warning(
            "Using --recursive or --force-recursive without --save means upstream interpretations "
            "will only be used for context in this run and not persisted."
        )

    # Load configuration from environment
    postgres_uri = get_config_value("postgres_uri")
    openai_api_key = get_config_value("openai_api_key")
    openai_model = get_config_value("openai_model", "gpt-4o")
    temperature = float(get_config_value("temperature", "0.0"))

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    if not openai_api_key:
        logger.error("OpenAI API key not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        # Import necessary modules
        from dbt_llm_agent.storage.model_storage import ModelStorage
        from dbt_llm_agent.storage.model_embedding_storage import ModelEmbeddingStorage
        from dbt_llm_agent.integrations.llm.client import LLMClient
        from dbt_llm_agent.core.agent import Agent

        # Initialize storage
        model_storage = ModelStorage(postgres_uri)
        vector_store = ModelEmbeddingStorage(postgres_uri)

        # Initialize LLM client
        llm = LLMClient(
            api_key=openai_api_key, model=openai_model, temperature=temperature
        )

        # Initialize agent
        agent = Agent(
            llm_client=llm,
            model_storage=model_storage,
            vector_store=vector_store,
        )

        # Get all models from database
        all_models = model_storage.get_all_models()
        if not all_models:
            logger.error("No models found in database")
            sys.exit(1)

        # Create model dictionary for selector
        models_dict = {model.name: model for model in all_models}

        # Select models to interpret
        selector = ModelSelector(models_dict)
        selected_model_names = selector.select(select)

        if not selected_model_names:
            logger.error(f"No models matched the selector: {select}")
            sys.exit(1)

        logger.info(f"Selected {len(selected_model_names)} models for interpretation")

        # Filter models based on existing interpretations if --only-when-missing flag is set
        if only_when_missing:
            original_count = len(selected_model_names)
            filtered_model_names = []

            for model_name in selected_model_names:
                model = models_dict.get(model_name)
                if model and not model.interpreted_description:
                    filtered_model_names.append(model_name)
                elif model:
                    logger.info(
                        f"Skipping model {model_name} as it already has an interpretation"
                    )

            selected_model_names = filtered_model_names
            logger.info(
                f"Filtered to {len(selected_model_names)} models that need interpretation (skipped {original_count - len(selected_model_names)})"
            )

            if not selected_model_names:
                logger.info("No models need interpretation, exiting")
                sys.exit(0)

        # Interpret each selected model
        total_models = len(selected_model_names)
        for i, model_name in enumerate(selected_model_names):
            model = models_dict.get(model_name)
            if not model:
                logger.warning(f"Model {model_name} not found in database, skipping")
                continue

            logger.info(f"Interpreting model: {model_name} ({i+1}/{total_models})")

            # Get interpretations from the agent
            result = agent.interpret_model(
                model_name, recursive=recursive, force_recursive=force_recursive
            )

            if not result.get("success", False):
                logger.error(
                    f"Failed to interpret model {model_name}: {result.get('error', 'Unknown error')}"
                )
                continue

            # Display interpretation
            print(
                f"\n=== Interpretation for model: {model_name} ({i+1}/{total_models}) ===\n"
            )
            # If verbose, print the prompt used
            if verbose and "prompt" in result:
                print(f"\n--- Prompt used for {model_name} ---")
                print(result["prompt"])
                print("--- End Prompt ---\n")

            print(result["yaml_documentation"])

            # Save interpretation if requested
            if save:
                logger.info(f"Saving interpretation for model {model_name}")
                save_result = agent.save_interpreted_documentation(
                    model_name, result["yaml_documentation"], embed=embed
                )

                if save_result.get("success", False):
                    logger.info(
                        f"Successfully saved interpretation for model {model_name}"
                    )
                else:
                    logger.error(
                        f"Failed to save interpretation for model {model_name}: {save_result.get('error', 'Unknown error')}"
                    )

    except Exception as e:
        logger.error(f"Error interpreting models: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
