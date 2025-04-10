"""Slack utilities."""

from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError
import logging

logger = logging.getLogger(__name__)


def get_async_slack_client(token: str) -> AsyncWebClient | None:
    """
    Initializes and returns an async Slack WebClient.

    Args:
        token: The Slack bot token.

    Returns:
        An initialized AsyncWebClient instance, or None if initialization fails.
    """
    if not token:
        logger.error("SLACK_BOT_TOKEN is missing.")
        return None
    try:
        client = AsyncWebClient(token=token)
        # Optional: Perform a quick test to ensure the token is valid
        # asyncio.run(client.auth_test()) # This might be better done elsewhere or made async
        logger.debug("Async Slack client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Async Slack client: {e}", exc_info=True)
        return None
