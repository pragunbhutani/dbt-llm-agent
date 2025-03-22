"""
Configuration management for dbt-llm-agent.
DEPRECATED: Use dbt_llm_agent.utils.config instead.
"""

import os
import json
import logging
import warnings
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Default config file path
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.dbt-llm-agent/config.json")


def load_config(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load configuration from environment variables.
    DEPRECATED: Use dbt_llm_agent.utils.config.load_config() instead.

    Args:
        config_path: Path to the config file. This parameter is ignored and
                    only kept for backward compatibility.

    Returns:
        Dictionary containing configuration.
    """
    # Show deprecation warning
    warnings.warn(
        "dbt_llm_agent.core.config is deprecated; use dbt_llm_agent.utils.config instead",
        DeprecationWarning,
        stacklevel=2,
    )

    # Create configuration dict from environment variables
    config = {
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4"),
        "temperature": float(os.environ.get("TEMPERATURE", "0.0")),
        "postgres_uri": os.environ.get(
            "POSTGRES_URI",
            "postgresql://postgres:postgres@localhost:5432/dbt_llm_agent",
        ),
        "postgres_connection_string": os.environ.get(
            "POSTGRES_CONNECTION_STRING",
            os.environ.get(
                "POSTGRES_URI",
                "postgresql://postgres:postgres@localhost:5432/dbt_llm_agent",
            ),
        ),
        "vector_db_path": os.environ.get("VECTOR_DB_PATH", "./data/vector_db"),
        "dbt_project_path": os.environ.get("DBT_PROJECT_PATH", ""),
        "slack_bot_token": os.environ.get("SLACK_BOT_TOKEN", ""),
        "slack_app_token": os.environ.get("SLACK_APP_TOKEN", ""),
        "slack_signing_secret": os.environ.get("SLACK_SIGNING_SECRET", ""),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
    }

    return config


def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> bool:
    """
    DEPRECATED: Use dbt_llm_agent.utils.config.save_config() instead.
    This function is a no-op since we're moving away from file-based configuration.

    Args:
        config: Dictionary containing configuration.
        config_path: Path to the config file. If None, uses the default path.

    Returns:
        True if successful, False otherwise.
    """
    warnings.warn(
        "dbt_llm_agent.core.config.save_config is deprecated and is a no-op; use dbt_llm_agent.utils.config.save_config instead",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "save_config is deprecated; configuration is now stored in environment variables"
    )
    return True


def get_default_config() -> Dict[str, Any]:
    """
    Returns a dictionary with default configuration values.
    DEPRECATED: Use dbt_llm_agent.utils.config.load_config() instead.
    """
    warnings.warn(
        "dbt_llm_agent.core.config.get_default_config is deprecated; use dbt_llm_agent.utils.config.load_config instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_config()
