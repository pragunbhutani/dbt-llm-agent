import logging
from typing import Dict, Any, Optional
from django.db import transaction
from django.conf import settings

# Import models and services/agents
from apps.knowledge_base.models import Model
from apps.llm_providers.services import default_chat_service
from .model_interpreter import ModelInterpreterAgent, ModelDocumentation

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
# _slack_sdk_available_check is implicitly handled by whether AsyncWebClient is None or not after import attempt
# We can remove _slack_sdk_available_check global variable as its role is covered by AsyncWebClient itself.


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


# --- Model Interpretation Service ---
def trigger_model_interpretation(model: Model, verbosity: int = 0) -> bool:
    """Triggers the ModelInterpreterAgent workflow for a single model.

    Args:
        model: The knowledge_base.Model instance to interpret.
        verbosity: Verbosity level for agent execution.

    Returns:
        True if interpretation was successful and saved, False otherwise.
    """
    logger.info(f"Starting interpretation workflow for model: {model.name}")

    if not model.raw_sql:
        logger.warning(f"Skipping interpretation for {model.name}: Raw SQL is missing.")
        return False

    try:
        # Initialize Agent
        interpreter_agent = ModelInterpreterAgent(
            chat_service=default_chat_service,
            verbosity=verbosity,
            console=(console if verbosity > 0 else None),
        )
    except ValueError as e:
        logger.error(f"Failed to initialize ModelInterpreterAgent: {e}")
        return False

    try:
        # Run Workflow
        agent_result = interpreter_agent.run_interpretation_workflow(
            model_name=model.name, raw_sql=model.raw_sql
        )

        # Process Result
        if agent_result["success"] and agent_result["documentation"]:
            documentation_dict = agent_result["documentation"]
            try:
                validated_doc = ModelDocumentation(**documentation_dict)
            except Exception as pydantic_err:
                logger.error(
                    f"Agent returned invalid documentation structure for {model.name}: {pydantic_err}. Data: {documentation_dict}",
                    exc_info=True,
                )
                return False  # Failed validation

            # Save results
            try:
                with transaction.atomic():
                    model.interpreted_description = validated_doc.description
                    model.interpreted_columns = {
                        col.name: col.description for col in validated_doc.columns
                    }
                    # Note: updated_at is auto-updated by Django
                    model.save(
                        update_fields=["interpreted_description", "interpreted_columns"]
                    )
                logger.info(f"Successfully saved interpretation for model {model.name}")
                return True  # Success!
            except Exception as save_err:
                logger.error(
                    f"Error saving interpretation for model {model.name}: {save_err}",
                    exc_info=True,
                )
                return False  # Failed save
        else:
            # Agent workflow failed
            error_msg = agent_result.get("error", "Unknown agent error")
            logger.error(f"Agent failed to interpret {model.name}: {error_msg}")
            return False  # Agent failure

    except Exception as workflow_err:
        logger.error(
            f"Unexpected error during interpretation workflow for {model.name}: {workflow_err}",
            exc_info=True,
        )
        return False  # Unexpected workflow error
