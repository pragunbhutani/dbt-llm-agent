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
@click.option("--feedback", help="Feedback on the response", required=True)
@click.option("--rating", help="Rating (1-5)", type=int, default=None)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def feedback(question_id, feedback, rating, verbose):
    """Provide feedback on a specific answer.

    QUESTION_ID is the ID of the question to provide feedback for.
    """
    set_logging_level(verbose)

    # Load configuration from environment
    postgres_uri = get_env_var("POSTGRES_URI")

    # Validate configuration
    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        # Import necessary modules
        from dbt_llm_agent.storage.postgres_storage import PostgresStorage

        # Initialize storage and agent
        logger.info(f"Connecting to PostgreSQL database")
        postgres_storage = PostgresStorage(postgres_uri)

        # Store feedback
        logger.info(f"Storing feedback for question {question_id}")
        postgres_storage.store_feedback(question_id, feedback, rating)

        logger.info("Feedback stored successfully")

    except Exception as e:
        logger.error(f"Error storing feedback: {str(e)}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
