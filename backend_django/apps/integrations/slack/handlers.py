"""
Slack handlers for Ragstar AI data analyst.

This module provides the main Slack integration using:
- Celery for background processing to avoid timeouts
- Multi-agent workflow with SlackResponder orchestrator for robust user experience
- Comprehensive conversation logging and analytics
"""

import logging
import os
import re
import traceback
from typing import Dict, Any, Optional

from celery import shared_task
from django.conf import settings
from slack_bolt import App
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from apps.workflows.tasks import run_conversation_workflow
from apps.workflows.models import (
    Conversation,
    ConversationTrigger,
    ConversationStatus,
    Question,
    ConversationPart,
)
from apps.accounts.models import OrganisationSettings, Organisation

logger = logging.getLogger(__name__)

# Module-level app instances cache - keyed by team_id
_slack_apps_cache: Dict[str, App] = {}


def get_team_info_from_token(bot_token: str) -> dict:
    """
    Get team information from a Slack bot token.
    Used for testing and validation purposes.
    """
    try:
        client = WebClient(token=bot_token)
        response = client.auth_test()

        if response["ok"]:
            return {
                "team_id": response.get("team_id"),
                "team_name": response.get("team"),
                "bot_user_id": response.get("user_id"),
                "user": response.get("user"),
            }
        else:
            logger.error(f"Slack auth test failed: {response}")
            return None
    except Exception as e:
        logger.error(f"Error getting team info from token: {e}")
        return None


def get_slack_app_for_team(team_id: str) -> Optional[App]:
    """
    Get or create a Slack App instance for a specific team.
    Uses database configuration instead of environment variables.
    """
    # Check cache first
    if team_id in _slack_apps_cache:
        return _slack_apps_cache[team_id]

    try:
        # Get organization integration for this Slack team ID
        from apps.integrations.models import OrganisationIntegration

        # Find the Slack integration with this team ID in configuration
        org_integration = OrganisationIntegration.objects.filter(
            integration_key="slack",
            is_enabled=True,
            configuration__team_id=team_id,
        ).first()

        if not org_integration:
            logger.error(f"No enabled Slack integration found for team: {team_id}")
            return None

        # Get bot token and signing secret from credentials
        credentials = org_integration.credentials
        bot_token = credentials.get("bot_token")
        signing_secret = credentials.get("signing_secret")

        if not bot_token:
            logger.error(
                f"No bot token configured for Slack integration in organization: {org_integration.organisation.name}"
            )
            return None

        if not signing_secret:
            logger.error(
                f"No signing secret configured for Slack integration in organization: {org_integration.organisation.name}"
            )
            return None

        # Validate bot token format
        if not bot_token.startswith("xoxb-"):
            logger.error(f"Invalid bot token format for team {team_id}")
            return None

        # Create and configure the Slack App
        app = App(
            token=bot_token,
            signing_secret=signing_secret,
        )

        # Register event handlers for this app instance
        _register_event_handlers(app)

        # Cache the app instance
        _slack_apps_cache[team_id] = app

        logger.info(f"Created Slack app instance for team: {team_id}")
        return app

    except Exception as e:
        logger.error(f"Error creating Slack app for team {team_id}: {e}")
        return None


def _register_event_handlers(app: App):
    """
    Register all event handlers for a Slack App instance.
    """

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
    def handle_app_mention(event, say, ack, body, client: WebClient):
        """
        Main app mention handler using Celery for background processing
        and comprehensive conversation logging.
        """
        # 1. Acknowledge immediately
        ack()

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
                say(
                    text="Hi! I'm here to help you analyze your dbt models and data. What would you like to know?",
                    thread_ts=thread_ts,
                )
                return

            # 4. Get organization settings
            logger.info(f"Looking up organization settings for team: {team_id}")
            org_settings = get_org_settings_for_team(team_id)
            if not org_settings:
                logger.error(f"No organization settings found for team: {team_id}")
                say(
                    text="I'm sorry, I couldn't access your organization settings. Please contact support.",
                    thread_ts=thread_ts,
                )
                return
            else:
                logger.info(
                    f"Found organization settings: {org_settings.organisation.id}"
                )

            # ----------------------------------------------------------
            # NEW: Fetch user display name for better UX
            # ----------------------------------------------------------
            user_display_name: Optional[str] = None
            try:
                if client and user_id:
                    user_info_resp = client.users_info(user=user_id)
                    if user_info_resp.get("ok"):
                        profile = user_info_resp["user"].get("profile", {})
                        user_display_name = (
                            profile.get("real_name")
                            or profile.get("display_name")
                            or user_info_resp["user"].get("name")
                        )
            except SlackApiError as e:
                logger.warning(
                    f"Could not fetch Slack user info for {user_id}: {e.response['error']}"
                )
            except Exception as e:
                logger.warning(
                    f"Unexpected error fetching Slack user info for {user_id}: {e}"
                )

            # 5. Create conversation record
            conversation = get_or_create_conversation_record(
                org_settings=org_settings,
                user_id=user_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
                question=user_question,
                user_display_name=user_display_name,
            )

            # 6. Get bot token for the Celery task
            from apps.integrations.models import OrganisationIntegration

            org_integration = OrganisationIntegration.objects.filter(
                organisation=org_settings.organisation,
                integration_key="slack",
                is_enabled=True,
            ).first()

            if not org_integration:
                logger.error("Slack integration not found for organization")
                say(
                    text="I'm sorry, Slack integration is not properly configured. Please contact support.",
                    thread_ts=thread_ts,
                )
                return

            bot_token = org_integration.credentials.get("bot_token")
            if not bot_token:
                logger.error("Bot token not found in Slack integration")
                say(
                    text="I'm sorry, Slack bot token is not configured. Please contact support.",
                    thread_ts=thread_ts,
                )
                return

            # 7. Launch Celery task for background processing (no immediate acknowledgment)
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
                slack_bot_token=bot_token,
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

            # 8. Store task ID for potential monitoring
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
            conversation.save()

        except Exception as e:
            logger.exception(f"Error in app mention handler: {e}")
            try:
                say(
                    text="I'm sorry, I encountered an issue while processing your request. Please try again.",
                    thread_ts=event.get("thread_ts", event.get("ts")),
                )
            except Exception as say_err:
                logger.error(f"Failed to send error message: {say_err}")

    @app.event("reaction_added")
    def handle_reaction_added(event, client, ack):
        """Handle user feedback via emoji reactions."""
        ack()

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
                update_feedback_in_db(message_ts, user_id, was_useful)

        except Exception as e:
            logger.exception(f"Error in reaction_added handler: {e}")

    @app.event("reaction_removed")
    def handle_reaction_removed(event, client, ack):
        """Handle removal of user feedback via emoji reactions."""
        ack()

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
                # When reaction is removed, we set feedback to None (neutral)
                update_feedback_in_db(message_ts, user_id, None)

        except Exception as e:
            logger.exception(f"Error in reaction_removed handler: {e}")

    @app.event("message")
    def handle_message(event, say, ack, body, client: WebClient):
        """
        Handle direct messages to the bot.
        This provides a similar experience to app mentions but in DMs.
        """
        ack()

        try:
            # Only handle direct messages, not messages in channels
            channel_type = event.get("channel_type")
            if channel_type != "im":
                return

            # Skip messages from bots (including our own responses)
            if event.get("bot_id") or event.get("subtype") == "bot_message":
                return

            logger.info(f"Received DM event: {event}")

            # Extract event details
            channel_id = event.get("channel")
            thread_ts = event.get("thread_ts", event.get("ts"))
            user_question = event.get("text", "").strip()
            user_id = event.get("user")
            team_id = body.get("team_id") or event.get("team")

            logger.info(
                f"DM details - channel: {channel_id}, thread_ts: {thread_ts}, user: {user_id}, question: {user_question}"
            )

            if not all([channel_id, thread_ts, user_id, team_id]):
                logger.warning(
                    f"Missing DM data - channel: {channel_id}, thread_ts: {thread_ts}, user: {user_id}, team: {team_id}"
                )
                return

            if not user_question:
                say(
                    text="Hi! I'm here to help you analyze your dbt models and data. What would you like to know?",
                    thread_ts=thread_ts,
                )
                return

            # ----------------------------------------------------------
            # NEW: Fetch user display name for DM conversations
            # ----------------------------------------------------------
            user_display_name: Optional[str] = None
            try:
                if client and user_id:
                    user_info_resp = client.users_info(user=user_id)
                    if user_info_resp.get("ok"):
                        profile = user_info_resp["user"].get("profile", {})
                        user_display_name = (
                            profile.get("real_name")
                            or profile.get("display_name")
                            or user_info_resp["user"].get("name")
                        )
            except SlackApiError as e:
                logger.warning(
                    f"Could not fetch Slack user info for {user_id}: {e.response['error']}"
                )
            except Exception as e:
                logger.warning(
                    f"Unexpected error fetching Slack user info for {user_id}: {e}"
                )

            # ----------------------------------------------------------
            # Fetch organisation settings (restored)
            # ----------------------------------------------------------
            org_settings = get_org_settings_for_team(team_id)
            if not org_settings:
                logger.error(f"No organization settings found for team: {team_id}")
                say(
                    text="I'm sorry, I couldn't access your organization settings. Please contact support.",
                    thread_ts=thread_ts,
                )
                return

            # Get or create conversation record
            conversation = get_or_create_conversation_record(
                org_settings=org_settings,
                user_id=user_id,
                channel_id=channel_id,
                thread_ts=thread_ts,
                question=user_question,
                user_display_name=user_display_name,
            )

            # Get bot token for the Celery task
            from apps.integrations.models import OrganisationIntegration

            org_integration = OrganisationIntegration.objects.filter(
                organisation=org_settings.organisation,
                integration_key="slack",
                is_enabled=True,
            ).first()

            if not org_integration:
                logger.error("Slack integration not found for organization")
                say(
                    text="I'm sorry, Slack integration is not properly configured. Please contact support.",
                    thread_ts=thread_ts,
                )
                return

            bot_token = org_integration.credentials.get("bot_token")
            if not bot_token:
                logger.error("Bot token not found in Slack integration")
                say(
                    text="I'm sorry, Slack bot token is not configured. Please contact support.",
                    thread_ts=thread_ts,
                )
                return

            # Launch Celery task for background processing
            task_result = run_conversation_workflow.delay(
                question=user_question,
                channel_id=channel_id,
                thread_ts=thread_ts,
                user_id=user_id,
                org_settings_id=org_settings.organisation.id,
                slack_bot_token=bot_token,
                conversation_metadata={
                    "conversation_id": str(conversation.id),
                    "trigger_event": event.get("event_ts"),
                    "user_name": event.get("user_name", "Unknown"),
                    "is_dm": True,
                },
            )

            logger.info(
                f"DM conversation task {task_result.id} queued for {channel_id}/{thread_ts}"
            )

            # Store task ID for monitoring
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
                    "is_dm": True,
                }
            )
            conversation.conversation_context = updated_context
            conversation.save()

        except Exception as e:
            logger.exception(f"Error in DM message handler: {e}")
            try:
                say(
                    text="I'm sorry, I encountered an issue while processing your request. Please try again.",
                    thread_ts=event.get("thread_ts", event.get("ts")),
                )
            except Exception as say_err:
                logger.error(f"Failed to send error message: {say_err}")


def get_org_settings_for_team(team_id: str):
    """Get organization settings for a Slack team ID."""
    try:
        # Find the Slack integration with this team ID in configuration
        from apps.integrations.models import OrganisationIntegration

        org_integration = OrganisationIntegration.objects.filter(
            integration_key="slack",
            is_enabled=True,
            configuration__team_id=team_id,
        ).first()

        if not org_integration:
            logger.error(f"No Slack integration found for team: {team_id}")
            return None

        # Return the organization settings
        return org_integration.organisation.settings
    except Exception as e:
        logger.error(f"Error getting org settings for team {team_id}: {e}")
        return None


def get_or_create_conversation_record(
    org_settings,
    user_id: str,
    channel_id: str,
    thread_ts: str,
    question: str,
    user_display_name: Optional[str] = None,
) -> Conversation:
    """Get or create a conversation record in the database for a Slack thread."""
    try:
        preferred_display = user_display_name or user_id

        # Try to get existing conversation first
        conversation, created = Conversation.objects.get_or_create(
            organisation=org_settings.organisation,
            external_id=thread_ts,  # Use thread_ts as external_id for proper threading
            defaults={
                "initial_question": question,
                "channel": "slack",
                "user_id": user_id,
                "user_external_id": preferred_display,
                "channel_id": channel_id,
                "trigger": ConversationTrigger.SLACK_MENTION,
                # Store the LLM provider & model at creation time for analytics
                "llm_provider": org_settings.llm_chat_provider or "unknown",
                "llm_chat_model": org_settings.llm_chat_model,
                "conversation_context": {
                    "slack_user_id": user_id,
                    "slack_channel_id": channel_id,
                    "slack_thread_ts": thread_ts,
                    "platform": "slack",
                },
            },
        )

        # If the conversation already existed, we may need to backfill missing fields.
        fields_to_update: list[str] = []

        # Update display name if we now have a better one
        if user_display_name and (
            not conversation.user_external_id
            or conversation.user_external_id == conversation.user_id
        ):
            conversation.user_external_id = user_display_name
            fields_to_update.append("user_external_id")

        # Backfill provider / model if they were previously missing
        if not conversation.llm_provider and org_settings.llm_chat_provider:
            conversation.llm_provider = org_settings.llm_chat_provider
            fields_to_update.append("llm_provider")

        if not conversation.llm_chat_model and org_settings.llm_chat_model:
            conversation.llm_chat_model = org_settings.llm_chat_model
            fields_to_update.append("llm_chat_model")

        if fields_to_update:
            conversation.save(update_fields=fields_to_update)

        if created:
            logger.info(
                f"Created new conversation record: {conversation.id} for thread {thread_ts}"
            )
        else:
            logger.info(
                f"Found existing conversation record: {conversation.id} for thread {thread_ts}"
            )

        return conversation

    except Exception as e:
        logger.error(f"Error getting/creating conversation record: {e}")
        # Create a minimal conversation as fallback
        return Conversation.objects.create(
            organisation=org_settings.organisation,
            initial_question=question,
            channel="slack",
            llm_provider=org_settings.llm_chat_provider or "unknown",
            llm_chat_model=org_settings.llm_chat_model,
            conversation_context={"error": "Failed to create full record"},
        )


def update_feedback_in_db(message_ts: str, user_id: str, was_useful: bool = None):
    """Update feedback in the database based on Slack reactions."""
    try:
        # Find the conversation part that corresponds to this message
        # We'll search by the Slack thread timestamp in metadata
        conversation_parts = ConversationPart.objects.filter(
            metadata__slack_thread_ts=message_ts, actor="assistant"
        )

        if conversation_parts.exists():
            # Update the first matching conversation part
            part = conversation_parts.first()
            if not part.metadata:
                part.metadata = {}

            part.metadata.update(
                {
                    "feedback_user_id": user_id,
                    "was_useful": was_useful,
                    "feedback_updated_at": message_ts,
                }
            )
            part.save()

            logger.info(
                f"Updated feedback for conversation part {part.id}: {was_useful}"
            )
        else:
            logger.warning(f"No conversation part found for message_ts: {message_ts}")

    except Exception as e:
        logger.error(f"Error updating feedback in database: {e}")
