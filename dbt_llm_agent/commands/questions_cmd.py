"""
Questions command for dbt-llm-agent CLI.
"""

import click
import sys

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import get_env_var, colored_echo

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.option("--limit", type=int, default=10, help="Number of questions to show")
@click.option("--offset", type=int, default=0, help="Offset for pagination")
@click.option("--useful", type=bool, help="Filter by usefulness")
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
def questions(limit, offset, useful, postgres_uri):
    """
    List questions and answers.
    """
    try:
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
        except ImportError:
            logger.warning(
                "python-dotenv not installed. Environment variables may not be properly loaded."
            )

        # Import here to avoid circular imports
        from dbt_llm_agent.storage.question_service import QuestionTrackingService

        # Get PostgreSQL URI
        if not postgres_uri:
            postgres_uri = get_env_var("POSTGRES_URI")
            if not postgres_uri:
                logger.error(
                    "PostgreSQL URI not provided. Please either:\n"
                    "1. Add POSTGRES_URI to your .env file\n"
                    "2. Pass it as --postgres-uri argument"
                )
                sys.exit(1)

        # Initialize question tracking
        question_tracking = QuestionTrackingService(postgres_uri)

        # Get questions
        questions = question_tracking.get_all_questions(
            limit=limit, offset=offset, was_useful=useful
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
