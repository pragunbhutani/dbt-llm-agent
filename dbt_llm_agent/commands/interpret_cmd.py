"""
Interpret command for dbt-llm-agent CLI.
"""

import click
import sys
import os

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import (
    get_env_var,
    set_logging_level,
    get_config_value,
)

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.option(
    "--select",
    required=True,
    help="Model selection using dbt syntax (e.g. 'tag:marketing,+downstream_model')",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--force", is_flag=True, help="Force reinterpretation of models")
@click.option(
    "--embed", is_flag=True, help="Embed the interpretations after generating"
)
@click.option("--save-yml", is_flag=True, help="Save interpretation to YAML files")
@click.option(
    "--output-dir",
    help="Directory for saving YAML files",
    type=click.Path(file_okay=False),
)
def interpret(select, verbose, force, embed, save_yml, output_dir):
    """Interpret dbt models and generate enhanced documentation.

    This command uses LLMs to generate interpretations for each model,
    extracting high-level descriptions and contextual information about
    metrics, dimensions, and relationships.

    Models can be selected using dbt selector syntax (--select),
    such as 'tag:metrics' or '+model_name'.

    By default, interpretations are saved to the database. Use --save-yml
    to also save them to YAML files in your project directory.
    """
    # Set logging level
    set_logging_level(verbose)

    # Import dependencies here to avoid circular imports
    from dbt_llm_agent.storage.postgres_storage import PostgresStorage
    from dbt_llm_agent.storage.vector_store import PostgresVectorStore
    from dbt_llm_agent.utils.model_selector import ModelSelector
    from dbt_llm_agent.core.agent import DBTAgent

    # Load configuration from environment
    openai_api_key = get_env_var("OPENAI_API_KEY")
    openai_model = get_env_var("OPENAI_MODEL", "gpt-4-turbo")
    postgres_uri = get_env_var("POSTGRES_URI")

    # Validate configuration
    if not openai_api_key:
        logger.error("OpenAI API key not provided in environment variables (.env file)")
        sys.exit(1)

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
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

            # Save the documentation to the database
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

                # Save to YAML file if requested
                if save_yml:
                    if output_dir:
                        # Create output directory if it doesn't exist
                        os.makedirs(output_dir, exist_ok=True)
                        file_path = os.path.join(
                            output_dir, f"{current_model_name}.yml"
                        )
                    else:
                        file_path = f"{current_model_name}.yml"

                    logger.info(f"Saving YAML to file: {file_path}")
                    with open(file_path, "w") as f:
                        f.write(result["yaml_documentation"])
                    logger.info(f"YAML file saved: {file_path}")
            else:
                error_msg = save_result.get("error", "Unknown error")
                logger.error(f"Failed to save documentation: {error_msg}")

    except Exception as e:
        logger.error(f"Error interpreting model: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
