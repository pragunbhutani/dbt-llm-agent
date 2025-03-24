"""
Interpret command for dbt-llm-agent CLI.
"""

import click
import sys

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import (
    get_env_var,
    set_logging_level,
    get_config_value,
)

# Initialize logger
logger = get_logger(__name__)


@click.command()
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
