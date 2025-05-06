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

from ragstar_archive.utils.logging import get_logger
from ragstar_archive.utils.cli_utils import get_config_value, set_logging_level
from ragstar_archive.utils.model_selector import ModelSelector
from rich.console import Console
from rich.progress import Progress

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
@click.option(
    "--workflow",
    is_flag=True,
    help="Use simpler workflow mode for interpretation",
)
@click.option(
    "--iterations",
    "-i",
    type=click.IntRange(0, 5),
    default=1,
    help="Number of verification iterations to run (default: 1, min: 0, max: 5)",
)
def interpret(
    select,
    save,
    embed,
    verbose,
    only_when_missing,
    recursive,
    force_recursive,
    workflow,
    iterations,
):
    """Interpret dbt models and generate documentation.

    This command analyzes selected dbt models using LLM and generates documentation,
    including model descriptions and column definitions.

    The command uses an agentic workflow that:
    1. Reads the model source code to identify upstream dependencies
    2. Fetches details of upstream models for context
    3. Creates a draft interpretation
    4. Iteratively verifies the interpretation against upstream models:
       - Directly analyzes source SQL of upstream models to identify all columns
       - Provides structured recommendations for columns to add, remove, or modify
       - Refines the interpretation until verification passes or iterations complete
    5. Saves the final interpretation if requested

    Use --iterations to control the number of verification iterations (default: 1).
    Use --verbose to see detailed verification results and column recommendations for each iteration.

    Examples:
        dbt-llm interpret --select "customers"
        dbt-llm interpret --select "+tag:marts" --save
        dbt-llm interpret --select "orders" --save --embed
        dbt-llm interpret --select "customers" --save --only-when-missing
        dbt-llm interpret --select "+fct_orders" --recursive --save
        dbt-llm interpret --select "+fct_orders" --force-recursive --save
        dbt-llm interpret --select "customers" --iterations 3 --verbose
    """
    set_logging_level(verbose)

    # Validate iterations parameter
    if iterations > 5:
        logger.warning("Number of iterations is capped at 5")
        iterations = 5

    logger.info(
        f"Using {iterations} verification iteration(s) for model interpretation"
    )

    # Warn if workflow mode is enabled with recursive flags
    if workflow and (recursive or force_recursive):
        logger.warning(
            "Ignoring --recursive or --force-recursive flags in workflow mode."
        )

    # Add warning if recursive flags are used without save
    if (recursive or force_recursive) and not save:
        logger.warning(
            "Using --recursive or --force-recursive without --save means upstream interpretations "
            "will only be used for context in this run and not persisted."
        )

    # Load configuration from environment
    postgres_uri = get_config_value("database_url")
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
        from ragstar_archive.storage.model_storage import ModelStorage
        from ragstar_archive.storage.model_embedding_storage import (
            ModelEmbeddingStorage,
        )
        from ragstar_archive.core.llm.client import LLMClient
        from ragstar_archive.core.agents import ModelInterpreter

        # Initialize storage
        model_storage = ModelStorage(postgres_uri)
        vector_store = ModelEmbeddingStorage(postgres_uri)

        # Initialize LLM client
        llm = LLMClient(
            api_key=openai_api_key, model=openai_model, temperature=temperature
        )

        # Initialize agent - now ModelInterpreter
        interpreter = ModelInterpreter(
            llm_client=llm,
            model_storage=model_storage,
            vector_store=vector_store,
            verbose=verbose,
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
            # Choose mode
            mode = "workflow" if workflow else "agentic"
            logger.info(f"Using {mode} workflow for {model_name}")
            # Get interpretations based on mode
            if workflow:
                result = interpreter.interpret_model_workflow(model_name)
            else:
                result = interpreter.run_interpretation_workflow(model_name)

            if not result.get("success", False):
                logger.error(
                    f"Failed to interpret model {model_name}: {result.get('error', 'Unknown error')}"
                )
                continue

            # Display interpretation
            print(
                f"\n=== Interpretation for model: {model_name} ({i+1}/{total_models}) ===\n"
            )

            # === New Code: Pretty-print YAML ===
            documentation_dict = result.get("documentation")
            if documentation_dict and isinstance(documentation_dict, dict):
                # Structure for dbt YAML
                dbt_yaml_structure = {
                    "models": [
                        {
                            "name": documentation_dict.get("name", model_name),
                            "description": documentation_dict.get("description", ""),
                            "columns": documentation_dict.get("columns", []),
                        }
                    ]
                }
                # Convert to YAML string
                try:
                    yaml_output = yaml.dump(
                        dbt_yaml_structure,
                        sort_keys=False,
                        indent=2,
                        allow_unicode=True,
                    )
                    print("--- Generated dbt YAML ---")
                    print(yaml_output)
                    print("--- End Generated dbt YAML ---")
                except Exception as yaml_err:
                    logger.error(f"Failed to format documentation as YAML: {yaml_err}")
                    # Optionally print the raw dict if YAML fails
                    print(
                        "--- Raw Documentation Dictionary (YAML formatting failed) ---"
                    )
                    print(json.dumps(documentation_dict, indent=2))
                    print("--- End Raw Documentation ---")
            else:
                logger.warning(
                    f"No valid documentation dictionary found in the result for {model_name} to print as YAML."
                )
            # === End New Code ===

            # Display iteration configuration
            if "verification_iterations" in result:
                total_iterations = result.get("verification_iterations", 1)
                print(f"Verification iterations: {total_iterations}/{iterations} used")

            # If verbose, print the prompt used
            if verbose and "prompt" in result:
                print(f"\n--- Prompt used for {model_name} ---")
                print(result["prompt"])
                print("--- End Prompt ---\n")

            # If verbose, print verification details
            if verbose and "verification_result" in result:
                print(f"\n--- Verification for {model_name} ---")
                print(result["verification_result"])

                # We now display this information at the top level
                if (
                    "verification_iterations" in result
                    and total_iterations < iterations
                ):
                    print(
                        f"\nVerification completed early after {total_iterations}/{iterations} iteration(s)"
                    )

                print("--- End Verification ---\n")

                # Display structured column recommendations if available
                if verbose and "column_recommendations" in result:
                    recommendations = result["column_recommendations"]
                    has_recommendations = (
                        len(recommendations.get("columns_to_add", [])) > 0
                        or len(recommendations.get("columns_to_remove", [])) > 0
                        or len(recommendations.get("columns_to_modify", [])) > 0
                    )

                    if has_recommendations:
                        print(
                            f"\n--- Column Recommendations for {model_name} (Final Iteration) ---"
                        )

                        # Columns to add
                        columns_to_add = recommendations.get("columns_to_add", [])
                        if columns_to_add:
                            print("\nColumns To Add:")
                            for column in columns_to_add:
                                print(f"  - {column.get('name', 'Unknown')}")
                                if "description" in column:
                                    print(f"    Description: {column['description']}")
                                if "reason" in column:
                                    print(f"    Reason: {column['reason']}")

                        # Columns to remove
                        columns_to_remove = recommendations.get("columns_to_remove", [])
                        if columns_to_remove:
                            print("\nColumns To Remove:")
                            for column in columns_to_remove:
                                print(f"  - {column.get('name', 'Unknown')}")
                                if "reason" in column:
                                    print(f"    Reason: {column['reason']}")

                        # Columns to modify
                        columns_to_modify = recommendations.get("columns_to_modify", [])
                        if columns_to_modify:
                            print("\nColumns To Modify:")
                            for column in columns_to_modify:
                                print(f"  - {column.get('name', 'Unknown')}")
                                if "current_description" in column:
                                    print(
                                        f"    Current: {column['current_description']}"
                                    )
                                if "suggested_description" in column:
                                    print(
                                        f"    Suggested: {column['suggested_description']}"
                                    )
                                if "reason" in column:
                                    print(f"    Reason: {column['reason']}")

                        print("--- End Column Recommendations ---\n")

                if (
                    "draft_yaml" in result
                    and result["draft_yaml"] != result["yaml_documentation"]
                ):
                    print(f"\n--- Draft vs Final ---")
                    print(
                        "Draft interpretation was refined based on verification feedback."
                    )
                    print("--- End Draft vs Final ---\n")

            # Save interpretation if requested
            if save:
                logger.info(f"Saving interpretation for model {model_name}")
                save_result = interpreter.save_interpreted_documentation(
                    model_name, result["documentation"], embed=embed
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
