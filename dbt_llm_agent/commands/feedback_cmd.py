"""
Feedback command for dbt-llm-agent CLI.
"""

import click
import sys

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import get_env_var, colored_echo

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.argument("question_id", type=int)
@click.option("--useful", type=bool, required=True, help="Was the answer useful?")
@click.option("--feedback", help="Additional feedback")
@click.option("--postgres-uri", help="PostgreSQL connection URI", envvar="POSTGRES_URI")
def feedback(question_id, useful, feedback, postgres_uri):
    """
    Provide feedback on an answer.
    """
    try:
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

        # Get the question to make sure it exists
        question = question_tracking.get_question(question_id)
        if not question:
            logger.error(f"Question with ID {question_id} not found")
            sys.exit(1)

        # Update feedback
        success = question_tracking.update_feedback(
            question_id=question_id, was_useful=useful, feedback=feedback
        )

        if success:
            colored_echo(
                f"Feedback recorded for question {question_id}", color="INFO", bold=True
            )
        else:
            colored_echo(
                f"Failed to record feedback for question {question_id}", color="ERROR"
            )

        return 0

    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        sys.exit(1)
