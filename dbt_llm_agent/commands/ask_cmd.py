"""
Ask command for dbt-llm-agent CLI.
"""

import click
import logging
import sys

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.commands.utils import (
    get_env_var,
    set_logging_level,
    get_config_value,
)

# Initialize logger
logger = get_logger(__name__)


@click.command()
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
