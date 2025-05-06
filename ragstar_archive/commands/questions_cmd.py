"""
Command to list and manage previously asked questions.
"""

import click
import sys
import json
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from typing import Dict, List, Optional

from ragstar.utils.logging import get_logger
from ragstar.utils.cli_utils import get_config_value, set_logging_level
from dotenv import load_dotenv

# Initialize logger
logger = get_logger(__name__)

# Initialize console for rich output
console = Console()


@click.command()
@click.option(
    "--limit", type=int, default=10, help="Maximum number of questions to show"
)
@click.option("--offset", type=int, default=0, help="Offset for pagination")
@click.option("--useful", is_flag=True, help="Only show questions marked as useful")
@click.option(
    "--not-useful", is_flag=True, help="Only show questions marked as not useful"
)
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def questions(limit, offset, useful, not_useful, output_json, verbose):
    """List previously asked questions and their answers.

    This command allows you to view your question history with optional filtering.

    Examples:
        dbt-llm questions
        dbt-llm questions --limit 5 --offset 10
        dbt-llm questions --useful
        dbt-llm questions --not-useful
    """
    set_logging_level(verbose)

    load_dotenv()

    # Load configuration
    postgres_uri = get_config_value("database_url")
    openai_api_key = get_config_value("openai_api_key")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        # Import necessary modules
        from ragstar.storage.model_storage import ModelStorage
        from ragstar.storage.question_storage import QuestionStorage

        # Initialize storage
        model_storage = ModelStorage(postgres_uri)
        question_storage = QuestionStorage(postgres_uri)

        # Determine filter
        was_useful = None
        if useful:
            was_useful = True
        elif not_useful:
            was_useful = False

        # Get questions
        results = question_storage.get_all_questions(
            limit=limit, offset=offset, was_useful=was_useful
        )

        if not results:
            console.print("No questions found")
            return

        # Display results
        if output_json:
            # Convert to JSON serializable format
            questions_json = []
            for q in results:
                question_dict = {
                    "id": q.id,
                    "question": q.question_text,
                    "answer": q.answer_text,
                    "was_useful": q.was_useful,
                    "feedback": q.feedback,
                    "created_at": q.created_at.isoformat() if q.created_at else None,
                }
                questions_json.append(question_dict)

            print(json.dumps(questions_json, indent=2))
            return

        # Create table display
        table = Table(title=f"Questions (showing {len(results)} results)")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Question", style="green")
        table.add_column("Useful", justify="center")
        table.add_column("Created", style="dim")

        for question in results:
            useful_str = (
                "✅"
                if question.was_useful == True
                else "❌"
                if question.was_useful == False
                else "?"
            )
            created_str = (
                question.created_at.strftime("%Y-%m-%d %H:%M")
                if question.created_at
                else "unknown"
            )

            table.add_row(
                str(question.id),
                (
                    question.question_text[:50] + "..."
                    if len(question.question_text) > 50
                    else question.question_text
                ),
                useful_str,
                created_str,
            )

        console.print(table)
        console.print(
            f"\nShowing {len(results)} of {limit} results. Use --limit and --offset for pagination."
        )
        console.print(
            "To see details of a specific question, use: dbt-llm question [ID]"
        )

    except Exception as e:
        logger.error(f"Error listing questions: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
