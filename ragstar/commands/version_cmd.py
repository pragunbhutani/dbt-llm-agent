"""
Version command for ragstar CLI.
"""

import click
from ragstar.utils.cli_utils import colored_echo


@click.command()
def version():
    """Get the version of ragstar"""
    colored_echo("ragstar version 0.5.0", color="INFO", bold=True)
