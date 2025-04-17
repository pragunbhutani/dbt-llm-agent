"""
Feedback command for ragstar CLI.
"""

import click
import sys
import logging
from dotenv import load_dotenv
from rich import console

from ragstar.utils.logging import get_logger
from ragstar.utils.cli_utils import get_config_value, set_logging_level

# Initialize logger
logger = get_logger(__name__)


@click.command()
@click.argument("question_id", type=int)
@click.option("--useful", is_flag=True, help="Mark the answer as useful")
@click.option("--not-useful", is_flag=True, help="Mark the answer as not useful")
@click.option(
    "--text",
    "feedback_text",
    type=str,
    help="Provide detailed text feedback or correction.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def feedback(question_id, useful, not_useful, feedback_text, verbose):
    """Provide feedback on a previous question.

    This command allows you to mark a previous question and answer as useful or not useful,
    and optionally provide detailed text feedback or corrections.

    You need to provide the question ID, which is displayed when you ask a question.

    Examples:
        dbt-llm feedback 1 --useful
        dbt-llm feedback 2 --not-useful --text "The join key should be user_id, not email."
        dbt-llm feedback 3 --text "This answer is correct but too verbose."
    """
    set_logging_level(verbose)

    # Load environment variables
    load_dotenv()

    # Validate arguments - Allow providing text without useful/not-useful flags
    if not (useful or not_useful or feedback_text):
        logger.error(
            "Please specify feedback: either --useful/--not-useful or provide --text, or both."
        )
        sys.exit(1)
    if useful and not_useful:
        logger.error("Cannot specify both --useful and --not-useful")
        sys.exit(1)

    console.print("[bold green]Interpreting feedback...[/bold green]")

    openai_api_key = get_config_value("openai_api_key")
    postgres_uri = get_config_value("database_url")

    if not openai_api_key:
        console.print("[bold red]OPENAI_API_KEY not found[/bold red]")
        sys.exit(1)

    # Import necessary modules
    from ragstar.storage.model_storage import ModelStorage
    from ragstar.storage.question_storage import QuestionStorage

    # Initialize storage
    model_storage = ModelStorage(postgres_uri)
    question_storage = QuestionStorage(
        connection_string=postgres_uri, openai_api_key=openai_api_key
    )

    # Get question
    question = question_storage.get_question(question_id)
    if not question:
        logger.error(f"Question with ID {question_id} not found")
        sys.exit(1)

    # Determine usefulness - default to None if only text is provided
    was_useful = None
    if useful:
        was_useful = True
    elif not_useful:
        was_useful = False

    # Update feedback, passing the text
    success = question_storage.update_feedback(
        question_id, was_useful=was_useful, feedback=feedback_text
    )

    if success:
        logger.info(f"Feedback recorded for question {question_id}")
        if verbose:
            logger.info(f"Question: {question.question_text}")
            logger.info(f"Answer: {question.answer_text}")
            logger.info(f"Useful: {was_useful}")
            logger.info(f"Feedback Text: {feedback_text}")
    else:
        logger.error(f"Failed to record feedback for question {question_id}")
        sys.exit(1)
