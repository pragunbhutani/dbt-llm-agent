"""
Questions command for dbt-llm-agent CLI.
"""

import click
import sys
import json as json_lib
import logging

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import (
    get_env_var,
    colored_echo,
    set_logging_level,
    load_dotenv_once,
)

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--limit", type=int, default=10, help="Number of questions to show")
@click.option("--json", is_flag=True, help="Output as JSON", default=False)
def questions(verbose, limit, json):
    """List recent questions and answers.

    This command shows the history of questions asked and their answers.
    """
    try:
        if verbose:
            set_logging_level(True)

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage

        # Get PostgreSQL URI from environment
        postgres_uri = get_env_var("POSTGRES_URI")
        if not postgres_uri:
            logger.error(
                "PostgreSQL URI not provided in environment variables (.env file)"
            )
            sys.exit(1)

        # Initialize storage
        logger.info(f"Connecting to PostgreSQL database")
        postgres = PostgresStorage(postgres_uri)

        # Load environment variables from .env file
        load_dotenv_once()

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.question_service import QuestionTrackingService

        # Initialize question tracking
        question_tracking = QuestionTrackingService(postgres_uri)

        # Get questions
        questions = question_tracking.get_all_questions(
            limit=limit, offset=0, was_useful=None
        )

        print("\n")

        if not questions:
            colored_echo("No questions found", color="WARNING")
            return 0

        colored_echo(f"Found {len(questions)} questions:\n", color="INFO", bold=True)
        for q in questions:
            colored_echo(f"\nID: {q['id']}", color="INFO", bold=True)
            colored_echo(f"Question: {q['question_text']}", color="INFO")
            colored_echo(f"Answer: {q['answer_text']}", color="DEBUG")
            # Use different colors based on usefulness
            usefulness_color = "INFO" if q["was_useful"] else "WARNING"
            colored_echo(f"Was useful: {q['was_useful']}", color=usefulness_color)
            # Get model names from the dictionary
            model_names = [m["name"] for m in q["models"]]
            colored_echo(f"Models: {', '.join(model_names)}", color="DEBUG")
            colored_echo(f"Created at: {q['created_at']}", color="DEBUG")

        return 0

    except Exception as e:
        logger.error(f"Error listing questions: {e}")
        sys.exit(1)
