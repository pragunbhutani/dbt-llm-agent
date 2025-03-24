"""
Version command for dbt-llm-agent CLI.
"""

import click
from dbt_llm_agent.utils.cli_utils import colored_echo


@click.command()
def version():
    """Get the version of dbt-llm-agent"""
    colored_echo("dbt-llm-agent version 0.1.0", color="INFO", bold=True)
