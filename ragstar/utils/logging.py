"""Logging utilities for ragstar."""

import os
import logging
import logging.config
from typing import Dict, Any

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = "~/.ragstar/logs/ragstar.log"


# Define ANSI color codes for log levels
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",  # Reset to default color
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels and more whitespace."""

    def format(self, record):
        levelname = record.levelname
        # Add color to level name
        if levelname in COLORS:
            colored_levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
            record.levelname = colored_levelname

        # Format the record
        result = super().format(record)

        # Restore the original levelname
        record.levelname = levelname

        return result


def setup_logging(log_level: str = None, log_file: str = None) -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level
        log_file: Path to the log file
    """
    # Check if logging has already been configured
    if hasattr(setup_logging, "configured") and setup_logging.configured:
        return

    # Get log level from environment or use default
    log_level = log_level or os.environ.get("RAGSTAR_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    log_level = getattr(logging, log_level.upper())

    # Get log file from environment or use default
    log_file = log_file or os.environ.get("RAGSTAR_LOG_FILE", DEFAULT_LOG_FILE)
    log_file = os.path.expanduser(log_file)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Format with more whitespace between identifier and message
    colored_format = "%(asctime)s - %(levelname)s   -   %(name)s   -   %(message)s"
    standard_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    # Configure logging
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "colored": {
                    "()": ColoredFormatter,
                    "format": colored_format,
                },
                "standard": {"format": standard_format},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "colored",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "filename": log_file,
                    "mode": "a",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["console", "file"],
                    "level": log_level,
                    "propagate": True,
                },
                "ragstar": {
                    "handlers": ["console", "file"],
                    "level": log_level,
                    "propagate": False,
                },
            },
        }
    )

    # Mark as configured
    setup_logging.configured = True

    # Log configuration
    logging.info(f"Logging configured with level {log_level}")
    logging.info(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
