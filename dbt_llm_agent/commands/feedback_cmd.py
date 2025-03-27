"""
Feedback command for dbt-llm-agent CLI.
"""

import click
import sys
import logging

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import get_config_value, set_logging_level

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.argument("question_id", type=int)
@click.option("--useful", is_flag=True, help="Mark the answer as useful")
@click.option("--not-useful", is_flag=True, help="Mark the answer as not useful")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def feedback(question_id, useful, not_useful, verbose):
    """Provide feedback on a previous question.

    This command allows you to mark a previous question and answer as useful or not useful.
    You need to provide the question ID, which is displayed when you ask a question.

    Examples:
        dbt-llm feedback 1 --useful
        dbt-llm feedback 2 --not-useful
    """
    set_logging_level(verbose)

    # Validate arguments
    if not (useful or not_useful) or (useful and not_useful):
        logger.error("Please specify either --useful or --not-useful")
        sys.exit(1)

    # Load configuration from environment
    postgres_uri = get_config_value("postgres_uri")

    # Import necessary modules
    from dbt_llm_agent.storage.model_storage import ModelStorage
    from dbt_llm_agent.storage.question_storage import QuestionStorage

    # Initialize storage
    model_storage = ModelStorage(postgres_uri)
    question_storage = QuestionStorage(postgres_uri)

    # Get question
    question = question_storage.get_question(question_id)
    if not question:
        logger.error(f"Question with ID {question_id} not found")
        sys.exit(1)

    # Update feedback
    was_useful = useful
    success = question_storage.update_feedback(question_id, was_useful=was_useful)

    if success:
        logger.info(f"Feedback recorded for question {question_id}")
        if verbose:
            logger.info(f"Question: {question.question_text}")
            logger.info(f"Answer: {question.answer_text}")
    else:
        logger.error(f"Failed to record feedback for question {question_id}")
        sys.exit(1)
