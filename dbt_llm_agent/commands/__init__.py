"""
This package contains all the CLI commands for dbt-llm-agent.
Each command is defined in its own module.
"""

from dbt_llm_agent.commands.parse_cmd import parse
from dbt_llm_agent.commands.embed_cmd import embed
from dbt_llm_agent.commands.ask_cmd import ask
from dbt_llm_agent.commands.interpret_cmd import interpret
from dbt_llm_agent.commands.list_cmd import list
from dbt_llm_agent.commands.model_details_cmd import model_details
from dbt_llm_agent.commands.questions_cmd import questions
from dbt_llm_agent.commands.feedback_cmd import feedback
from dbt_llm_agent.commands.db_cmd import migrate, init_db
from dbt_llm_agent.commands.version_cmd import version

__all__ = [
    "parse",
    "embed",
    "ask",
    "interpret",
    "list",
    "model_details",
    "questions",
    "feedback",
    "migrate",
    "init_db",
    "version",
]
