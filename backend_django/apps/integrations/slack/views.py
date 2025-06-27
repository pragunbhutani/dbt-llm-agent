# apps/integrations/slack/views.py
import json
import logging
import re  # For parsing Slack message link
from typing import Optional
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import (
    settings as django_settings,
)  # For INTEGRATIONS_SLACK_SIGNING_SECRET

# For Slack request verification (using slack_sdk utility)
from slack_sdk.signature import SignatureVerifier
from slack_sdk.errors import SlackApiError  # For client calls
from slack_sdk import WebClient
from slack_sdk.web.async_client import AsyncWebClient

from apps.accounts.models import OrganisationSettings
from apps.integrations.models import OrganisationIntegration

logger = logging.getLogger(__name__)

# Import the Metabase handler logic and callback ID
try:
    from apps.integrations.metabase.client import (
        MetabaseClient,
    )  # To check if available

    # Note: process_metabase_creation is not available in handlers, removing it
    METABASE_SHORTCUT_CALLBACK_ID = "create_metabase_query"
except ImportError:
    METABASE_SHORTCUT_CALLBACK_ID = "create_metabase_query"  # Fallback
    MetabaseClient = None
    logger.warning("Could not import Metabase related handlers or client for views.py.")


def get_signing_secret_for_request(request: HttpRequest) -> Optional[str]:
    """
    Get the Slack signing secret for the request based on the team ID.
    This function extracts the team ID from the request and looks up the
    appropriate signing secret from the organization's integration credentials.
    """
    try:
        # Try to extract team_id from different sources in the request
        team_id = None

        # Check POST data for team_id (common in Slack payloads)
        if request.method == "POST":
            # If it's a JSON payload
            if request.content_type == "application/json":
                try:
                    body = json.loads(request.body.decode("utf-8"))
                    team_id = body.get("team", {}).get("id") or body.get("team_id")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            # If it's form data or payload parameter
            elif hasattr(request, "POST"):
                team_id = request.POST.get("team_id")
                # Check if there's a payload parameter (common in interactive components)
                payload_str = request.POST.get("payload")
                if payload_str and not team_id:
                    try:
                        payload = json.loads(payload_str)
                        team_id = payload.get("team", {}).get("id") or payload.get(
                            "team_id"
                        )
                    except json.JSONDecodeError:
                        pass

        if not team_id:
            logger.warning(
                "Could not extract team_id from Slack request for signature verification"
            )
            return None

        # Find the organization integration for this team using the new system
        from apps.integrations.models import OrganisationIntegration

        org_integration = OrganisationIntegration.objects.filter(
            integration_key="slack",
            is_enabled=True,
            configuration__team_id=team_id,
        ).first()

        if not org_integration:
            logger.warning(f"No enabled Slack integration found for team: {team_id}")
            return None

        # Get signing secret from credentials
        signing_secret = org_integration.credentials.get("signing_secret")
        if not signing_secret:
            logger.warning(
                f"No signing secret configured for Slack integration in organization: {org_integration.organisation.name}"
            )
            return None

        return signing_secret

    except Exception as e:
        logger.error(f"Error getting signing secret for Slack request: {e}")
        return None


def verify_slack_request(request: HttpRequest) -> bool:
    """Verifies the request signature from Slack using the organization's signing secret."""
    try:
        # Get the signing secret for this specific request/organization
        signing_secret = get_signing_secret_for_request(request)

        if not signing_secret:
            logger.error("Could not get signing secret for Slack request verification")
            # In development, allow requests without verification
            if django_settings.DEBUG:
                logger.warning("Slack request verification skipped in DEBUG mode")
                return True
            return False

        # Create signature verifier with the organization's signing secret
        signature_verifier = SignatureVerifier(signing_secret)

        if not signature_verifier.is_valid_request(
            body=request.body.decode("utf-8"), headers=dict(request.headers)
        ):
            logger.warning("Slack request verification failed")
            return False
        return True

    except Exception as e:
        logger.error(f"Error during Slack request verification: {e}")
        return False


def get_slack_web_client_for_team(team_id: str) -> Optional[AsyncWebClient]:
    """
    Get a Slack web client for a specific team using the organization's bot token.
    """
    try:
        # Find the organization integration for this team using the new system
        from apps.integrations.models import OrganisationIntegration

        org_integration = OrganisationIntegration.objects.filter(
            integration_key="slack",
            is_enabled=True,
            configuration__team_id=team_id,
        ).first()

        if not org_integration:
            logger.error(f"No enabled Slack integration found for team: {team_id}")
            return None

        bot_token = org_integration.credentials.get("bot_token")
        if not bot_token:
            logger.error(
                f"No bot token found for organization: {org_integration.organisation.name}"
            )
            return None

        return AsyncWebClient(token=bot_token)

    except Exception as e:
        logger.error(f"Error creating Slack web client for team {team_id}: {e}")
        return None


def parse_slack_message_link(link: str) -> tuple[Optional[str], Optional[str]]:
    """Parses a Slack message link to extract channel ID and message TS."""
    # Example link: https://your-workspace.slack.com/archives/C12345ABC/p1620000000000100?thread_ts=123&cid=C123
    # Simpler regex for basic /archives/CHANNEL/pTIMESTAMP format
    match = re.search(r"/archives/([^/]+)/p(\d{10})(\d{6})", link)
    if match:
        channel_id = match.group(1)
        ts_integer_part = match.group(2)
        ts_decimal_part = match.group(3)
        message_ts = f"{ts_integer_part}.{ts_decimal_part}"
        return channel_id, message_ts
    logger.warning(f"Could not parse Slack message link: {link}")
    return None, None


@csrf_exempt
def slack_events_view(request: HttpRequest):
    """Placeholder for event handling - events are handled by slack_bolt in handlers.py"""
    return HttpResponse("OK", status=200)


@csrf_exempt
def slack_interactive_view(request: HttpRequest):
    """Placeholder for interactive components - handled by slack_bolt in handlers.py"""
    return HttpResponse("OK", status=200)


@csrf_exempt
def slack_shortcut_view(request: HttpRequest):
    """
    Handles interactive shortcut payloads from Slack.

    This endpoint must respond within 3 seconds to acknowledge receipt of the shortcut.
    All processing is done asynchronously via Celery tasks.
    """
    if not verify_slack_request(request):
        return HttpResponse("Slack request verification failed.", status=403)

    try:
        payload_str = request.POST.get("payload")
        if not payload_str:
            logger.warning("slack_shortcut_view: No payload found in request.")
            return HttpResponse("Missing payload.", status=400)

        payload = json.loads(payload_str)
        logger.info(f"Received shortcut payload: {json.dumps(payload, indent=2)}")

        callback_id = payload.get("callback_id")
        user_id = payload.get("user", {}).get("id")
        channel_id = payload.get("channel", {}).get("id")
        message_data = payload.get("message", {})
        trigger_message_ts = message_data.get("ts")
        # Determine the actual thread_ts for the workflow
        # If 'thread_ts' is present in the message, it's a reply in a thread, so use that.
        # Otherwise, the message itself is the start of the thread (or a standalone message).
        actual_thread_ts = message_data.get("thread_ts", trigger_message_ts)
        team_id = payload.get("team", {}).get("id")

        if not all(
            [user_id, channel_id, trigger_message_ts, actual_thread_ts, team_id]
        ):
            logger.error(
                f"Missing critical information in shortcut payload for callback_id '{callback_id}': user_id={user_id}, channel_id={channel_id}, trigger_ts={trigger_message_ts}, actual_thread_ts={actual_thread_ts}, team_id={team_id}"
            )
            return HttpResponse(status=200)  # Acknowledge receipt

        # Get organization settings for this team using the new integration system
        from apps.integrations.slack.handlers import get_org_settings_for_team

        org_settings = get_org_settings_for_team(team_id)
        if not org_settings:
            logger.error(f"No organization settings found for Slack team: {team_id}")
            return HttpResponse("Organization not found", status=400)

        # Get the Slack integration and bot token
        from apps.integrations.models import OrganisationIntegration

        org_integration = OrganisationIntegration.objects.filter(
            organisation=org_settings.organisation,
            integration_key="slack",
            is_enabled=True,
        ).first()

        if not org_integration:
            logger.error(
                f"Slack integration not enabled for organization: {org_settings.organisation.name}"
            )
            return HttpResponse("Slack integration not configured", status=400)

        bot_token = org_integration.credentials.get("bot_token")
        if not bot_token:
            logger.error(
                f"No bot token configured for Slack integration in organization: {org_settings.organisation.name}"
            )
            return HttpResponse("Bot token not configured", status=400)

        # Import Celery tasks
        from apps.workflows.tasks import (
            run_query_executor_workflow,
            run_knowledge_extractor_workflow,
        )

        # Check if it's a message action and the correct callback_id
        if payload.get("type") == "message_action":
            if callback_id == "execute_query":
                logger.info(
                    f"Processing 'execute_query' shortcut: user_id={user_id}, channel_id={channel_id}, trigger_message_ts={trigger_message_ts}, actual_thread_ts={actual_thread_ts}"
                )

                # Queue the QueryExecutor workflow task
                task_result = run_query_executor_workflow.delay(
                    channel_id=channel_id,
                    thread_ts=actual_thread_ts,
                    user_id=user_id,
                    trigger_message_ts=trigger_message_ts,
                    org_settings_id=org_settings.organisation.id,
                    slack_bot_token=bot_token,
                )

                logger.info(
                    f"QueryExecutor workflow task {task_result.id} queued for {channel_id}/{actual_thread_ts}"
                )
                return HttpResponse(status=200)  # Acknowledge the shortcut immediately

            elif callback_id == "learn_from_thread":
                logger.info(
                    f"Processing 'learn_from_thread' shortcut: user_id={user_id}, channel_id={channel_id}, trigger_message_ts={trigger_message_ts}, actual_thread_ts={actual_thread_ts}"
                )

                # Queue the KnowledgeExtractor workflow task
                task_result = run_knowledge_extractor_workflow.delay(
                    channel_id=channel_id,
                    thread_ts=actual_thread_ts,
                    user_id=user_id,
                    org_settings_id=org_settings.organisation.id,
                    slack_bot_token=bot_token,
                )

                logger.info(
                    f"KnowledgeExtractor workflow task {task_result.id} queued for {channel_id}/{actual_thread_ts}"
                )
                return HttpResponse(status=200)  # Acknowledge the shortcut immediately

            elif callback_id == METABASE_SHORTCUT_CALLBACK_ID:
                logger.info(
                    f"Processing 'create_metabase_query' shortcut: user_id={user_id}, channel_id={channel_id}, trigger_message_ts={trigger_message_ts}, actual_thread_ts={actual_thread_ts}"
                )
                # Handle Metabase shortcut - this can remain as it was since it's likely simpler
                # For now, just acknowledge it
                logger.warning(
                    "Metabase shortcut handling not yet implemented in Celery task"
                )
                return HttpResponse(status=200)

        # If we get here, it's an unhandled shortcut type
        logger.warning(f"Unhandled shortcut callback_id: {callback_id}")
        return HttpResponse(status=200)

    except json.JSONDecodeError:
        logger.error("slack_shortcut_view: Failed to parse JSON payload.")
        return HttpResponse("Invalid JSON payload.", status=400)
    except Exception as e:
        logger.error(f"slack_shortcut_view: Unexpected error: {e}", exc_info=True)
        return HttpResponse("Internal server error.", status=500)


# Slack integration now uses OrganisationIntegration model for configuration instead of Django settings
