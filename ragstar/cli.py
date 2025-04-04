"""
CLI interface for ragstar
"""

import click
import logging

# Set up logging
from ragstar.utils.logging import setup_logging, get_logger
from ragstar.utils.cli_utils import load_dotenv_once

# Initialize logging with default settings
setup_logging()
logger = get_logger(__name__)

# Import all commands
from ragstar.commands import (
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
    reset_db,
)
from ragstar.commands.version_cmd import version
from ragstar.commands.init_cmd import init


@click.group()
def cli():
    """Ragstar CLI"""
    pass


# Register all commands
cli.add_command(version)
cli.add_command(parse)
cli.add_command(embed)
cli.add_command(ask)
cli.add_command(interpret)
cli.add_command(list, name="list")
cli.add_command(model_details)
cli.add_command(questions)
cli.add_command(feedback)
cli.add_command(migrate)
cli.add_command(init_db)
cli.add_command(reset_db)
cli.add_command(init)


def main():
    # Set up colored logging - only configure once
    if not logging.root.handlers:
        setup_logging()

    # Load environment variables from .env file
    load_dotenv_once()

    cli()


if __name__ == "__main__":
    main()
