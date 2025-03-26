"""
CLI interface for dbt-llm-agent
"""

import click
import logging

# Set up logging
from dbt_llm_agent.utils.logging import setup_logging, get_logger
from dbt_llm_agent.utils.cli_utils import load_dotenv_once

# Initialize logging with default settings
setup_logging()
logger = get_logger(__name__)

# Import all commands
from dbt_llm_agent.commands import (
    parse,
    embed,
    ask,
    interpret,
    list,
    model_details,
    questions,
    feedback,
    migrate,
    init_db,
)
from dbt_llm_agent.commands.version_cmd import version


@click.group()
def cli():
    """dbt LLM Agent CLI"""
    pass


# Register all commands
cli.add_command(version)
cli.add_command(parse)
cli.add_command(embed)
cli.add_command(ask)
cli.add_command(interpret)
cli.add_command(list)
cli.add_command(model_details)
cli.add_command(questions)
cli.add_command(feedback)
cli.add_command(migrate)
cli.add_command(init_db)


def main():
    # Set up colored logging - only configure once
    if not logging.root.handlers:
        setup_logging()

    # Load environment variables from .env file
    load_dotenv_once()

    cli()


if __name__ == "__main__":
    main()
