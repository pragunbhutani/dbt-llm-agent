import logging
from typing import Optional

# Import models and services
from apps.knowledge_base.models import Model
from apps.accounts.models import OrganisationSettings

# Import workflow
from .model_interpreter.workflow import ModelInterpreterWorkflow

# Optional Rich console integration
try:
    from rich.console import Console as RichConsole

    console = RichConsole()
except ImportError:
    console = None
    RichConsole = None

# Attempt to import AsyncWebClient at the module level for type hinting
try:
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError:
    AsyncWebClient = None  # Define as None if not available, for type hints

logger = logging.getLogger(__name__)

# --- Slack Client Service ---
_slack_client_instance: Optional[AsyncWebClient] = None


def get_slack_web_client() -> Optional[AsyncWebClient]:
    """
    Provides a singleton instance of the Slack AsyncWebClient.
    Initializes the client on first call using INTEGRATIONS_SLACK_BOT_TOKEN from Django settings.
    Returns None if slack_sdk is not installed or token is not configured.
    """
    global _slack_client_instance

    if AsyncWebClient is None:  # SDK was not imported successfully at module level
        logger.warning(
            "slack_sdk is not installed or AsyncWebClient could not be imported. Slack functionalities will be unavailable."
        )
        return None

    if _slack_client_instance is None:
        slack_bot_token = getattr(settings, "INTEGRATIONS_SLACK_BOT_TOKEN", None)
        if not slack_bot_token:
            logger.error(
                "INTEGRATIONS_SLACK_BOT_TOKEN is not configured in Django settings. Slack client cannot be initialized."
            )
            return None

        # AsyncWebClient is already imported at module level if available
        _slack_client_instance = AsyncWebClient(token=slack_bot_token)
        logger.info("Slack AsyncWebClient initialized singleton.")

    return _slack_client_instance


def trigger_model_interpretation(
    model: Model, org_settings: OrganisationSettings, verbosity: int = 0
) -> bool:
    """Triggers model interpretation using the simplified workflow.

    Args:
        model: The knowledge_base.Model instance to interpret.
        org_settings: The OrganisationSettings instance for the model.
        verbosity: Verbosity level for logging.

    Returns:
        True if interpretation was successful and saved, False otherwise.
    """
    try:
        # Initialize workflow
        workflow = ModelInterpreterWorkflow(
            org_settings=org_settings, verbosity=verbosity
        )

        # Run interpretation
        return workflow.interpret_model(model)

    except Exception as init_err:
        logger.error(
            f"Failed to initialize ModelInterpreterWorkflow for {model.name}: {init_err}",
            exc_info=True,
        )
        return False
