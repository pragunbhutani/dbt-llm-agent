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
from dbt_llm_agent.storage.question_storage import QuestionStorage

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
    """Ask a question about your dbt models using an agentic workflow."""
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
        from dbt_llm_agent.integrations.llm.client import LLMClient

        llm = LLMClient(api_key=openai_api_key, model=openai_model)

        # Initialize question tracking if we are storing history
        question_tracking = None
        if not no_history:
            question_tracking = QuestionStorage(
                connection_string=postgres_uri, openai_api_key=openai_api_key
            )

        # Initialize the Agent
        from dbt_llm_agent.core.agent import Agent

        agent = Agent(
            llm_client=llm,
            model_storage=model_storage,
            vector_store=vector_store,
            question_storage=question_tracking,
            temperature=temperature,
            console=console,
            verbose=verbose,
            openai_api_key=openai_api_key,
        )

        # *** Start of New Agentic Workflow ***
        console.print("[bold]🚀 Starting agentic workflow...[/bold]")

        # Run the agentic workflow
        # This method will handle understanding, searching, fetching,
        # feedback checking, synthesis, validation, and refinement.
        workflow_result = agent.run_agentic_workflow(question)

        if not workflow_result or not workflow_result.get("final_answer"):
            console.print(
                "[bold red]Agent could not generate a satisfactory answer.[/bold red]"
            )
            return 1

        final_answer = workflow_result["final_answer"]
        used_model_names = workflow_result.get("used_model_names", [])
        conversation_id = workflow_result.get("conversation_id")

        # Display the final answer
        console.print("\n[bold]Question:[/bold]")
        console.print(question)
        console.print("\n[bold]Final Answer:[/bold]")
        md = Markdown(final_answer)
        console.print(md)

        # Handle history saving confirmation (if managed by workflow)
        if conversation_id and not no_history:
            console.print(
                f"\n[italic]This conversation has been saved with ID {conversation_id}.[/italic]"
            )
            console.print(
                "[italic]You can provide feedback with "
                f"'dbt-llm feedback {conversation_id} --useful/--not-useful'[/italic]"
            )
        elif not no_history and not conversation_id and question_tracking:
            # Fallback if workflow didn't save history
            # Only try if question_tracking was initialized
            try:
                new_id = question_tracking.record_question(
                    question_text=question,
                    answer_text=final_answer,
                    model_names=used_model_names,
                )
                console.print(
                    f"\n[italic]Saved conversation with ID {new_id} (fallback).[/italic]"
                )
            except Exception as hist_err:
                logger.warning(
                    f"Could not save question history via fallback: {hist_err}"
                )

        return 0
        # *** End of New Agentic Workflow ***

    except Exception as e:
        logger.error(f"Error asking question: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return 1


if __name__ == "__main__":
    ask()
