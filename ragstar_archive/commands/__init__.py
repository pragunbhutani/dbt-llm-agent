"""
This package contains all the CLI commands for ragstar.
Each command is defined in its own module.
"""

from ragstar.commands.parse_cmd import parse
from ragstar.commands.embed_cmd import embed
from ragstar.commands.ask_cmd import ask
from ragstar.commands.interpret_cmd import interpret
from ragstar.commands.list_cmd import list_models as list
from ragstar.commands.model_details_cmd import model_details
from ragstar.commands.questions_cmd import questions
from ragstar.commands.feedback_cmd import feedback
from ragstar.commands.db_cmd import migrate, init_db, reset_db
from ragstar.commands.version_cmd import version

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
    "reset_db",
    "version",
]
