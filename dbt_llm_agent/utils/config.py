"""Configuration utilities for dbt-llm-agent."""

import os
import logging
import warnings
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {
        "postgres_uri": os.environ.get("POSTGRES_URI", None),
        # Keep postgres_connection_string for backward compatibility but
        # default to postgres_uri if available
        "postgres_connection_string": os.environ.get(
            "POSTGRES_CONNECTION_STRING",
            os.environ.get("POSTGRES_URI", None),
        ),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", None),
        "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
        "openai_embedding_model": os.environ.get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        "temperature": float(os.environ.get("TEMPERATURE", 0.0)),
        "vector_db_path": os.environ.get("VECTOR_DB_PATH", "data/vector_db"),
        "dbt_project_path": os.environ.get("DBT_PROJECT_PATH", None),
        "slack_bot_token": os.environ.get("SLACK_BOT_TOKEN", ""),
        "slack_app_token": os.environ.get("SLACK_APP_TOKEN", ""),
        "slack_signing_secret": os.environ.get("SLACK_SIGNING_SECRET", ""),
    }
    return config


def save_config(config: Dict[str, Any]) -> None:
    """This function is deprecated and no longer saves to a config file.

    Update your .env file instead.

    Args:
        config: Configuration dictionary
    """
    warnings.warn(
        "save_config is deprecated. Please update your .env file directly. "
        "Changes made through this function will not persist.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "Configuration changes won't be saved. Please update your .env file directly."
    )


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


# Deprecated function to maintain backward compatibility
def get_config_path() -> str:
    """DEPRECATED. Configuration is now read directly from environment variables.

    Returns:
        Empty string
    """
    warnings.warn(
        "get_config_path is deprecated. Configuration is now read directly from environment variables.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ""
