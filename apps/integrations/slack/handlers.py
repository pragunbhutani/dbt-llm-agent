# apps/integrations/slack/handlers.py
import logging
import os
import asyncio
from threading import Thread
from asgiref.sync import sync_to_async
from typing import Optional

from django.conf import settings
from django.db import models
from django.utils import timezone

# Revert to sync App and SlackRequestHandler for Django -> Replace with Async versions
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

# Update imports for workflows and models
from apps.workflows.slack_responder import SlackResponderAgent
from apps.workflows.models import Question

logger = logging.getLogger(__name__)

# --- Slack Bolt App Initialization ---
try:
    slack_bot_token = settings.SLACK_BOT_TOKEN
    slack_signing_secret = settings.SLACK_SIGNING_SECRET
except AttributeError:
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    slack_signing_secret = os.environ.get("SLACK_SIGNING_SECRET")

if not slack_bot_token:
    logger.error("SLACK_BOT_TOKEN not configured.")
    raise ValueError("SLACK_BOT_TOKEN is not set.")
if not slack_signing_secret:
    logger.error("SLACK_SIGNING_SECRET not configured.")
    raise ValueError("SLACK_SIGNING_SECRET is not set.")

# Use sync App -> Use AsyncApp -> Rename to app
app = AsyncApp(
    token=slack_bot_token,
    signing_secret=slack_signing_secret,
)


# --- Event Handlers ---
# Listener remains async - Bolt should handle this internally
@app.event("app_mention")
async def handle_app_mention(event, say, ack, body, client: AsyncWebClient):
    """Handles app mentions: acknowledges, extracts info, and launches workflow task."""
    # 1. Acknowledge the event immediately
    logger.info(f"handle_app_mention: Acknowledging event {event.get('event_ts')}")
    try:
        await ack()
        logger.info(f"handle_app_mention: Acknowledged event {event.get('event_ts')}")
    except Exception as e:
        logger.error(
            f"handle_app_mention: Failed to acknowledge event {event.get('event_ts')}: {e}"
        )
        # Don't proceed if ack fails?
        return

    # 2. Extract necessary details from the event
    try:
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts", event.get("ts"))
        user_question = event.get("text", "").strip()
        user_id = event.get("user")

        if not all([channel_id, thread_ts, user_id]):
            logger.warning(
                "Missing crucial event data in app_mention, skipping workflow."
            )
            return

        # 3. Remove bot mention from question text (using the provided client)
        # Bot user ID is typically available in body["authorizations"][0]["user_id"]
        # or fallback to auth_test if needed (though less efficient)
        bot_user_id = None
        if body.get("authorizations") and len(body["authorizations"]) > 0:
            bot_user_id = body["authorizations"][0].get("user_id")
        else:
            # Fallback: This requires an extra API call
            try:
                auth_test_res = await client.auth_test()
                bot_user_id = auth_test_res.get("user_id")
            except Exception as auth_err:
                logger.error(f"Error calling auth.test: {auth_err}")

        if bot_user_id:
            user_question = user_question.replace(f"<@{bot_user_id}>", "").strip()

        if not user_question:
            logger.info(
                f"App mention in {channel_id}/{thread_ts} has no question text after mention removal."
            )
            # Optionally send a message back to the user asking for a question
            # await say(text="Please ask a question after mentioning me!", thread_ts=thread_ts)
            return

        logger.info(
            f"handle_app_mention: Launching workflow for {channel_id}/{thread_ts}"
        )

        # 4. Instantiate Agent (passing the client provided by Bolt)
        # Use Django settings for verbosity
        verbose = getattr(settings, "AGENT_DEFAULT_VERBOSITY", 0) > 1
        responder_agent = SlackResponderAgent(
            slack_client=client,  # Use the client from the handler args
            verbose=verbose,
            # If memory/checkpointer is needed, it must be instantiated here or passed
            # memory=...
        )

        # 5. Launch workflow as a background task
        asyncio.create_task(
            responder_agent.run_slack_workflow(
                question=user_question,
                channel_id=channel_id,
                thread_ts=thread_ts,
                user_id=user_id,
            )
        )
        logger.info(
            f"handle_app_mention: Background task created for {channel_id}/{thread_ts}"
        )

    except Exception as e:
        logger.exception(
            f"Error processing app_mention event {event.get('event_ts')} in handler: {e}"
        )
        # Attempt to notify user of failure if possible
        try:
            await say(
                text=f"Sorry, I encountered an error trying to process your request.",
                thread_ts=event.get(
                    "thread_ts", event.get("ts")
                ),  # Use thread_ts if available
            )
        except Exception as say_err:
            logger.error(f"Failed to send error message to user: {say_err}")


# --- Feedback Handlers ---
@sync_to_async
def _update_feedback_in_db(message_ts: str, user_id: str, was_useful: Optional[bool]):
    """Helper sync function to update feedback in DB."""
    try:
        # Try finding by response_message_ts first
        q_filter = models.Q(response_message_ts=message_ts)
        # If that's null (e.g., only file was posted), try response_file_message_ts
        q_filter |= models.Q(
            response_file_message_ts=message_ts, response_message_ts__isnull=True
        )

        updated_count = Question.objects.filter(q_filter).update(
            was_useful=was_useful,
            feedback_provided_at=timezone.now() if was_useful is not None else None,
            feedback_provider_user_id=user_id,
        )

        if updated_count > 0:
            logger.info(
                f"Updated feedback (was_useful={was_useful}) for question related to message {message_ts} by user {user_id}"
            )
            return True
        else:
            # Added more specific logging
            logger.warning(
                f"Could not find Question with response_message_ts OR response_file_message_ts matching {message_ts} to update feedback."
            )
            return False
    except Exception as e:
        logger.error(
            f"Error updating feedback in DB for message {message_ts}: {e}",
            exc_info=True,
        )
        return False


@app.event("reaction_added")
async def handle_reaction_added(event, client, ack):
    await ack()
    reaction = event.get("reaction")
    user_id = event.get("user")
    item_user_id = event.get("item_user")
    item_details = event.get("item", {})
    item_type = item_details.get("type")

    bot_user_id = None
    try:
        auth_test_res = await client.auth_test()
        bot_user_id = auth_test_res.get("user_id")
    except Exception as e:
        logger.error(f"Error getting bot user ID for reaction check: {e}")
        return

    if item_user_id != bot_user_id or item_type != "message":
        logger.debug(
            f"Ignoring reaction (not on bot message: item_user={item_user_id}, item_type={item_type})"
        )
        return

    was_useful: Optional[bool] = None
    if reaction in ["+1", "thumbsup"]:
        was_useful = True
    elif reaction in ["-1", "thumbsdown"]:
        was_useful = False
    else:
        logger.debug(f"Ignoring non-feedback reaction: {reaction}")
        return

    message_ts = item_details.get("ts")
    if not message_ts:
        logger.warning(f"Missing 'ts' in item details for message reaction: {event}")
        return

    logger.info(
        f"Processing feedback reaction '{reaction}' on message {message_ts} by user {user_id}"
    )
    await _update_feedback_in_db(
        message_ts=message_ts, user_id=user_id, was_useful=was_useful
    )


@app.event("reaction_removed")
async def handle_reaction_removed(event, client, ack):
    await ack()
    reaction = event.get("reaction")
    user_id = event.get("user")
    item_user_id = event.get("item_user")
    item_details = event.get("item", {})
    item_type = item_details.get("type")

    bot_user_id = None
    try:
        auth_test_res = await client.auth_test()
        bot_user_id = auth_test_res.get("user_id")
    except Exception as e:
        logger.error(f"Error getting bot user ID for reaction removal check: {e}")
        return

    if item_user_id != bot_user_id or item_type != "message":
        logger.debug(
            f"Ignoring reaction removal (not on bot message: item_user={item_user_id}, item_type={item_type})"
        )
        return

    if reaction not in ["+1", "thumbsup", "-1", "thumbsdown"]:
        logger.debug(f"Ignoring removal of non-feedback reaction: {reaction}")
        return

    message_ts = item_details.get("ts")
    if not message_ts:
        logger.warning(
            f"Missing 'ts' in item details for message reaction removal: {event}"
        )
        return

    logger.info(
        f"Processing feedback removal (reaction: '{reaction}') on message {message_ts} by user {user_id}"
    )
    # Remove feedback by setting was_useful to None
    await _update_feedback_in_db(
        message_ts=message_ts, user_id=user_id, was_useful=None
    )
