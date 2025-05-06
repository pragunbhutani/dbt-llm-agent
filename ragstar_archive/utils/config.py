"""Configuration utilities for ragstar."""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path

# Define constants
RAGSTAR_ENV_FILE = ".env"

logger = logging.getLogger(__name__)

# Check ~/.ragstar/.env first
env_path = Path.home() / ".ragstar" / RAGSTAR_ENV_FILE
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    logger.debug(f"Loaded environment variables from global {env_path}")

# Then check .env in the current working directory
# This is now handled by load_dotenv() called in cli.py and commands
cwd_env_path = Path(".env")
if cwd_env_path.exists():
    load_dotenv(dotenv_path=cwd_env_path, override=True)
    logger.debug(f"Loaded environment variables from local .env")


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {
        "database_url": os.environ.get("DATABASE_URL", None),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", None),
        "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
        "openai_embedding_model": os.environ.get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        "temperature": float(os.environ.get("TEMPERATURE", 0.0)),
        "vector_db_path": os.environ.get("VECTOR_DB_PATH", "data/vector_db"),
        "dbt_project_path": os.environ.get("DBT_PROJECT_PATH", None),
        "slack_bot_token": os.environ.get("SLACK_BOT_TOKEN", ""),
        "slack_signing_secret": os.environ.get("SLACK_SIGNING_SECRET", ""),
    }
    return config


def get_config_value(key: str, default: Optional[Any] = None) -> Any:
    """Get a configuration value from the environment.

    Args:
        key: Configuration key
        default: Default value if the key is not found

    Returns:
        Configuration value
    """
    config = load_config()
    return config.get(key, default)
