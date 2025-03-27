"""
Command line interface for asking questions to the dbt-llm-agent.
"""

import click
import sys
import os
import json
import textwrap
from rich.console import Console
from rich.markdown import Markdown

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import (
    get_config_value,
    set_logging_level,
    format_model_reference,
)

# Initialize logger
logger = get_logger(__name__)

# Initialize console for rich output
console = Console()


@click.command()
@click.option(
    "--no-history", is_flag=True, help="Don't store the question and answer in history"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.argument("question")
def ask(question, no_history, verbose):
    """Ask a question about your dbt models.

    This command lets you ask questions about your dbt models in natural language.
    dbt-llm-agent will use its knowledge of your models to provide an answer.

    Examples:
        dbt-llm ask "What models do we have related to customers?"
        dbt-llm ask "How is the orders model related to customers?"
        dbt-llm ask "What are the primary key columns in orders model?"
    """
    set_logging_level(verbose)

    # Check for OpenAI API key
    openai_api_key = get_config_value("openai_api_key")
    if not openai_api_key:
        logger.error("OpenAI API key not provided in environment variables (.env file)")
        sys.exit(1)

    # Load configuration
    postgres_uri = get_config_value("postgres_uri")
    vector_db_path = get_config_value("vector_db_path", "~/.dbt-llm-agent/vector_db")
    temperature = float(get_config_value("temperature", "0.0"))
    openai_model = get_config_value("openai_model", "gpt-4o")

    try:
        # Initialize storage
        from dbt_llm_agent.storage.model_storage import ModelStorage
        from dbt_llm_agent.storage.model_embedding_storage import ModelEmbeddingStorage

        model_storage = ModelStorage(postgres_uri)
        vector_store = ModelEmbeddingStorage(postgres_uri)

        # First check if we have stored models
        models = model_storage.get_all_models()
        if not models:
            logger.error(
                "No dbt models found in the database. "
                "Please run 'dbt-llm parse' first to import your models."
            )
            sys.exit(1)

        # Initialize LLM client
        from dbt_llm_agent.llm.client import LLMClient

        llm = LLMClient(api_key=openai_api_key, model=openai_model)

        # Initialize question tracking if we are storing history
        question_id = None
        if not no_history:
            from dbt_llm_agent.storage.question_storage import QuestionStorage

            question_tracking = QuestionStorage(postgres_uri)

        # Search for relevant models
        console.print("[bold]🔍 Searching for relevant models...[/bold]")
        relevant_results = vector_store.search_models(question, n_results=5)
        relevant_models = []

        if relevant_results:
            for result in relevant_results:
                model_name = result["model_name"]
                similarity = result["similarity_score"]
                if similarity > 0.5:  # Only include if reasonably similar
                    # Get full model details
                    model = model_storage.get_model(model_name)
                    if model:
                        relevant_models.append(
                            {
                                "name": model_name,
                                "description": model.description or "No description",
                                "similarity": similarity,
                            }
                        )

        # Display relevant models if in verbose mode
        if verbose and relevant_models:
            console.print("\n[bold]Relevant Models:[/bold]")
            for model in relevant_models:
                console.print(
                    f"- {model['name']} "
                    f"({model['similarity']:.2f}): {model['description']}"
                )

        # Generate the answer
        console.print("[bold]🤔 Generating answer...[/bold]")

        from dbt_llm_agent.core.agent import Agent

        agent = Agent(
            llm_client=llm,
            model_storage=model_storage,
            vector_store=vector_store,
            temperature=temperature,
        )

        answer = agent.answer_question(question, relevant_models)

        # Display the answer
        console.print("\n[bold]Question:[/bold]")
        console.print(question)
        console.print("\n[bold]Answer:[/bold]")

        # Format answer as markdown
        md = Markdown(answer)
        console.print(md)

        # Store the question and answer
        if not no_history:
            # Extract model names
            model_names = [model["name"] for model in relevant_models]

            question_id = question_tracking.record_question(
                question_text=question,
                answer_text=answer,
                model_names=model_names,
            )

            console.print(
                f"\n[italic]This conversation has been saved with ID {question_id}.[/italic]"
            )
            console.print(
                "[italic]You can provide feedback with "
                f"'dbt-llm feedback {question_id} --useful/--not-useful'[/italic]"
            )

        return 0

    except Exception as e:
        logger.error(f"Error asking question: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1


if __name__ == "__main__":
    ask()
