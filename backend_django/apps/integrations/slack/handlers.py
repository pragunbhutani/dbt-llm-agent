"""
Slack handlers for Ragstar AI data analyst.

This module provides the main Slack integration using:
- Celery for background processing to avoid timeouts
- Multi-agent workflow with SlackResponder orchestrator for robust user experience
- Comprehensive conversation logging and analytics
"""

import logging
import os

from django.conf import settings
from slack_bolt import App  # Changed from AsyncApp
from slack_sdk.web import WebClient  # Changed from AsyncWebClient

from apps.workflows.tasks import run_conversation_workflow
from apps.workflows.models import (
    Conversation,
    ConversationTrigger,
    ConversationStatus,
    Question,
)
from apps.accounts.models import OrganisationSettings

logger = logging.getLogger(__name__)

# --- Helper Functions ---


def get_team_info_from_token(bot_token: str) -> dict:
    """
    Helper function to get team information from a Slack bot token.
    This can be used during setup to auto-populate the team ID.
    """
    try:
        client = WebClient(token=bot_token)
        response = client.team_info()

        if response.get("ok"):
            team = response.get("team", {})
            return {
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "team_domain": team.get("domain"),
            }
        else:
            logger.error(f"Failed to get team info: {response.get('error')}")
            return None

    except Exception as e:
        logger.error(f"Error getting team info from token: {e}")
        return None


# --- Slack App Setup ---
slack_bot_token = getattr(
    settings,
    "INTEGRATIONS_SLACK_BOT_TOKEN",
    os.environ.get("INTEGRATIONS_SLACK_BOT_TOKEN"),
)
slack_signing_secret = getattr(
    settings,
    "INTEGRATIONS_SLACK_SIGNING_SECRET",
    os.environ.get("INTEGRATIONS_SLACK_SIGNING_SECRET"),
)

app = App(  # Changed from AsyncApp
    token=slack_bot_token,
    signing_secret=slack_signing_secret,
)


# Add a catch-all event handler for debugging
@app.event({"type": ".*"})
def handle_all_events(event, ack, body):
    """Catch-all handler to log all events for debugging."""
    ack()
    logger.info(
        f"Received Slack event: type={event.get('type')}, subtype={event.get('subtype')}"
    )
    logger.info(f"Full event data: {event}")


@app.event("app_mention")
def handle_app_mention(
    event, say, ack, body, client: WebClient
):  # Removed async, changed WebClient
    """
    Main app mention handler using Celery for background processing
    and comprehensive conversation logging.
    """

    # 1. Acknowledge immediately
    ack()  # Removed await

    try:
        logger.info(f"Received app_mention event: {event}")
        logger.info(f"Event body: {body}")

        # 2. Extract event details
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts", event.get("ts"))
        user_question = event.get("text", "").strip()
        user_id = event.get("user")
        team_id = body.get("team_id") or event.get("team")

        # Log team ID for setup purposes
        logger.info(f"Slack event from team: {team_id}")
        logger.info(
            f"Event details - channel: {channel_id}, thread_ts: {thread_ts}, user: {user_id}, question: {user_question}"
        )

        if not all([channel_id, thread_ts, user_id, team_id]):
            logger.warning(
                f"Missing event data - channel: {channel_id}, thread_ts: {thread_ts}, user: {user_id}, team: {team_id}"
            )
            return

        # 3. Clean up question text
        bot_user_id = None
        if body.get("authorizations"):
            bot_user_id = body["authorizations"][0].get("user_id")

        if bot_user_id:
            user_question = user_question.replace(f"<@{bot_user_id}>", "").strip()

        if not user_question:
            say(  # Removed await
                text="Hi! I'm here to help you analyze your dbt models and data. What would you like to know?",
                thread_ts=thread_ts,
            )
            return

        # 4. Get organization settings
        logger.info(f"Looking up organization settings for team: {team_id}")
        org_settings = get_org_settings_for_team(team_id)  # Removed await
        if not org_settings:
            logger.error(f"No organization settings found for team: {team_id}")
            say(  # Removed await
                text="I'm sorry, I couldn't access your organization settings. Please contact support.",
                thread_ts=thread_ts,
            )
            return
        else:
            logger.info(f"Found organization settings: {org_settings.organisation.id}")

        # 5. Create conversation record
        conversation = create_conversation_record(  # Removed await
            org_settings=org_settings,
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            question=user_question,
        )

        # 6. Launch Celery task for background processing (no immediate acknowledgment)
        # Check if there's already an active task for this conversation to prevent duplicates
        existing_task_id = conversation.conversation_context.get("celery_task_id")
        if existing_task_id:
            # Check if the task is still running
            from celery import current_app

            try:
                task_result = current_app.AsyncResult(existing_task_id)
                task_state = task_result.state

                # If task is still pending or running, skip to prevent duplicates
                if task_state in ["PENDING", "STARTED", "RETRY"]:
                    logger.info(
                        f"Task {existing_task_id} is still {task_state} for {channel_id}/{thread_ts}, skipping duplicate"
                    )
                    return
                else:
                    # Task is completed, failed, or revoked - allow new task
                    logger.info(
                        f"Previous task {existing_task_id} has state {task_state}, allowing new task for {channel_id}/{thread_ts}"
                    )
            except Exception as e:
                logger.warning(
                    f"Could not check task state for {existing_task_id}: {e}, allowing new task"
                )
                # If we can't check the task state, err on the side of allowing the new task

        task_result = run_conversation_workflow.delay(
            question=user_question,
            channel_id=channel_id,
            thread_ts=thread_ts,
            user_id=user_id,
            org_settings_id=org_settings.organisation.id,
            slack_bot_token=slack_bot_token,
            conversation_metadata={
                "conversation_id": str(conversation.id),
                "trigger_event": event.get("event_ts"),
                "user_name": event.get("user_name", "Unknown"),
            },
        )

        logger.info(
            f"Conversation task {task_result.id} queued for {channel_id}/{thread_ts}"
        )
        logger.info(f"Task details: {task_result}")

        # 7. Store task ID for potential monitoring
        # Preserve existing context but update task-related fields
        updated_context = (
            conversation.conversation_context.copy()
            if conversation.conversation_context
            else {}
        )
        updated_context.update(
            {
                "celery_task_id": task_result.id,
                "slack_event_ts": event.get("event_ts"),
                "last_task_started_at": event.get("event_ts"),
            }
        )
        conversation.conversation_context = updated_context
        conversation.save()  # Removed await and sync_to_async

    except Exception as e:
        logger.exception(f"Error in app mention handler: {e}")
        try:
            say(  # Removed await
                text="I'm sorry, I encountered an issue while processing your request. Please try again.",
                thread_ts=event.get("thread_ts", event.get("ts")),
            )
        except Exception as say_err:
            logger.error(f"Failed to send error message: {say_err}")


@app.event("reaction_added")
def handle_reaction_added(event, client, ack):  # Removed async
    """Handle user feedback via emoji reactions."""
    ack()  # Removed await

    try:
        user_id = event.get("user")
        reaction = event.get("reaction")
        message_ts = event.get("item", {}).get("ts")

        if not all([user_id, reaction, message_ts]):
            return

        # Map reactions to feedback
        feedback_mapping = {
            "thumbsup": True,
            "+1": True,
            "heavy_check_mark": True,
            "white_check_mark": True,
            "thumbsdown": False,
            "-1": False,
            "x": False,
            "heavy_multiplication_x": False,
        }

        if reaction in feedback_mapping:
            was_useful = feedback_mapping[reaction]
            update_feedback_in_db(message_ts, user_id, was_useful)  # Removed await

    except Exception as e:
        logger.exception(f"Error handling reaction: {e}")


@app.event("reaction_removed")
def handle_reaction_removed(event, client, ack):  # Removed async
    """Handle removal of feedback reactions."""
    ack()  # Removed await

    try:
        user_id = event.get("user")
        reaction = event.get("reaction")
        message_ts = event.get("item", {}).get("ts")

        if not all([user_id, reaction, message_ts]):
            return

        # Map reactions to feedback
        feedback_mapping = {
            "thumbsup": True,
            "+1": True,
            "heavy_check_mark": True,
            "white_check_mark": True,
            "thumbsdown": False,
            "-1": False,
            "x": False,
            "heavy_multiplication_x": False,
        }

        if reaction in feedback_mapping:
            # Remove feedback (set to None)
            update_feedback_in_db(message_ts, user_id, None)  # Removed await

    except Exception as e:
        logger.exception(f"Error handling reaction removal: {e}")


# Handle direct messages to the bot
@app.event("message")
def handle_message(event, say, ack, body, client: WebClient):
    """Handle direct messages to the bot."""
    ack()

    # Only handle direct messages (DMs)
    channel_type = event.get("channel_type")
    subtype = event.get("subtype")
    bot_id = event.get("bot_id")
    user_id = event.get("user")

    logger.info(
        f"Received message event: channel_type={channel_type}, subtype={subtype}, bot_id={bot_id}, user={user_id}"
    )

    # Skip bot messages, message edits, and other subtypes
    if bot_id or subtype or not user_id:
        logger.info("Skipping message: bot message, has subtype, or no user")
        return

    # Only handle direct messages (channel_type 'im')
    if channel_type == "im":
        logger.info("Processing direct message")
        # Process similar to app_mention
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts", event.get("ts"))
        user_question = event.get("text", "").strip()
        team_id = body.get("team_id") or event.get("team")

        logger.info(
            f"DM details - channel: {channel_id}, thread_ts: {thread_ts}, user: {user_id}, question: {user_question}"
        )

        if not all([channel_id, thread_ts, user_id, team_id, user_question]):
            logger.warning(
                f"Missing DM data - channel: {channel_id}, thread_ts: {thread_ts}, user: {user_id}, team: {team_id}, question: {user_question}"
            )
            return

        # Get organization settings
        logger.info(f"Looking up organization settings for team: {team_id}")
        org_settings = get_org_settings_for_team(team_id)
        if not org_settings:
            logger.error(f"No organization settings found for team: {team_id}")
            say(
                text="I'm sorry, I couldn't access your organization settings. Please contact support.",
                channel=channel_id,
            )
            return

        logger.info(f"Found organization settings: {org_settings.organisation.id}")

        # Create conversation record
        conversation = create_conversation_record(
            org_settings=org_settings,
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            question=user_question,
        )

        # Launch Celery task
        task_result = run_conversation_workflow.delay(
            question=user_question,
            channel_id=channel_id,
            thread_ts=thread_ts,
            user_id=user_id,
            org_settings_id=org_settings.organisation.id,
            slack_bot_token=slack_bot_token,
            conversation_metadata={
                "conversation_id": str(conversation.id),
                "trigger_event": event.get("event_ts"),
                "user_name": event.get("user_name", "Unknown"),
            },
        )

        logger.info(f"DM task {task_result.id} queued for {channel_id}/{thread_ts}")

        # Store task ID
        updated_context = (
            conversation.conversation_context.copy()
            if conversation.conversation_context
            else {}
        )
        updated_context.update(
            {
                "celery_task_id": task_result.id,
                "slack_event_ts": event.get("event_ts"),
                "last_task_started_at": event.get("event_ts"),
            }
        )
        conversation.conversation_context = updated_context
        conversation.save()


def get_org_settings_for_team(team_id: str):  # Removed @sync_to_async and async
    """Get organization settings for a Slack team/workspace."""
    try:
        return OrganisationSettings.objects.filter(slack_team_id=team_id).first()
    except Exception as e:
        logger.error(f"Error getting org settings for team {team_id}: {e}")
        return None


def create_conversation_record(  # Removed @sync_to_async and async
    org_settings, user_id: str, channel_id: str, thread_ts: str, question: str
) -> Conversation:
    """Create or get conversation record for this Slack thread."""
    try:
        # Try to get existing conversation
        conversation = Conversation.objects.get(
            external_id=thread_ts,
            channel_id=channel_id,
            organisation=org_settings.organisation,
        )
        logger.info(
            f"Using existing conversation {conversation.id} for C{channel_id}/{thread_ts}"
        )
        return conversation

    except Conversation.DoesNotExist:
        # Create new conversation
        conversation = Conversation.objects.create(
            organisation=org_settings.organisation,
            external_id=thread_ts,
            channel="slack",
            channel_type="slack",
            channel_id=channel_id,
            user_id=user_id,
            user_external_id=user_id,
            status=ConversationStatus.ACTIVE,
            trigger=ConversationTrigger.SLACK_MENTION,
            initial_question=question,
            llm_provider=getattr(org_settings, "llm_anthropic_api_key", None)
            and "anthropic"
            or "openai",
            enabled_integrations=org_settings.enabled_integrations or [],
        )
        logger.info(
            f"Created new conversation {conversation.id} for {channel_id}/{thread_ts}"
        )
        return conversation


def update_feedback_in_db(
    message_ts: str, user_id: str, was_useful: bool = None
):  # Removed @sync_to_async and async
    """Update feedback in the database based on reaction."""
    try:
        # This is a simplified implementation - you might want to link this to
        # your conversation parts based on message timestamp
        logger.info(
            f"User {user_id} {'gave positive feedback' if was_useful else 'gave negative feedback' if was_useful is False else 'removed feedback'} on message {message_ts}"
        )
        # TODO: Implement actual feedback storage logic here
    except Exception as e:
        logger.exception(f"Error updating feedback: {e}")
