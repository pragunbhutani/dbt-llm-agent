# apps/integrations/slack/handlers.py
import logging
import os
import asyncio
import aiohttp  # Added for file download
from datetime import datetime  # Added for timestamp formatting
import pytz  # Added for timezone conversion
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

try:
    from apps.integrations.metabase.client import (
        MetabaseClient,
        DEFAULT_INTEGRATIONS_METABASE_DATABASE_ID,
    )
except ImportError:
    MetabaseClient = (
        None  # Allow app to run if metabase integration is not fully set up
    )
    DEFAULT_INTEGRATIONS_METABASE_DATABASE_ID = 1  # Placeholder
    logger.warning(
        "MetabaseClient could not be imported. Metabase-related shortcuts will not function."
    )

logger = logging.getLogger(__name__)

# --- Slack Bolt App Initialization ---
try:
    slack_bot_token = settings.INTEGRATIONS_SLACK_BOT_TOKEN
    slack_signing_secret = settings.INTEGRATIONS_SLACK_SIGNING_SECRET
except AttributeError:
    slack_bot_token = os.environ.get("INTEGRATIONS_SLACK_BOT_TOKEN")
    slack_signing_secret = os.environ.get("INTEGRATIONS_SLACK_SIGNING_SECRET")

if not slack_bot_token:
    logger.error("INTEGRATIONS_SLACK_BOT_TOKEN not configured.")
    raise ValueError("INTEGRATIONS_SLACK_BOT_TOKEN is not set.")
if not slack_signing_secret:
    logger.error("INTEGRATIONS_SLACK_SIGNING_SECRET not configured.")
    raise ValueError("INTEGRATIONS_SLACK_SIGNING_SECRET is not set.")

# Check for Metabase env vars for awareness, though client handles enforcement
if MetabaseClient:  # Only check if client was imported
    metabase_url = os.environ.get("INTEGRATIONS_METABASE_URL")
    metabase_api_key = os.environ.get("INTEGRATIONS_METABASE_API_KEY")
    metabase_db_id = os.environ.get("INTEGRATIONS_METABASE_DATABASE_ID")
    if not metabase_url:
        logger.warning(
            "INTEGRATIONS_METABASE_URL environment variable is not set. Metabase integration may not work."
        )
    if not metabase_api_key:
        logger.warning(
            "INTEGRATIONS_METABASE_API_KEY environment variable is not set. Metabase integration may not work."
        )
    if not metabase_db_id:
        logger.warning(
            f"INTEGRATIONS_METABASE_DATABASE_ID environment variable is not set. Using default {DEFAULT_INTEGRATIONS_METABASE_DATABASE_ID}. Metabase integration may not work as expected."
        )

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
        # Verbosity is now handled internally by the agent based on Django settings
        responder_agent = SlackResponderAgent(
            slack_client=client,  # Use the client from the handler args
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


# --- Metabase Shortcut Handler ---
METABASE_SHORTCUT_CALLBACK_ID = "create_metabase_query"


@app.shortcut(METABASE_SHORTCUT_CALLBACK_ID)
async def handle_create_metabase_query_shortcut(
    shortcut, ack, client: AsyncWebClient, body, logger: logging.Logger, say
):
    """Handles the 'Create Metabase Query' message shortcut."""
    await ack()
    logger.info(f"Received Metabase shortcut: {shortcut.get('callback_id')}")

    if not MetabaseClient:
        logger.error(
            "MetabaseClient is not available. Cannot process 'create_metabase_query' shortcut."
        )
        try:
            await client.chat_postEphemeral(
                channel=shortcut["channel"]["id"],
                user=shortcut["user"]["id"],
                thread_ts=shortcut["message"]["ts"],
                text="Sorry, the Metabase integration is not configured correctly. Please contact an administrator.",
            )
        except Exception as e_say:
            logger.error(
                f"Error sending ephemeral message about misconfiguration: {e_say}"
            )
        return

    try:
        user_id = shortcut["user"]["id"]
        channel_id = shortcut["channel"]["id"]
        original_message_ts = shortcut["message"]["ts"]
        message_data = shortcut["message"]

        sql_query = None

        # Attempt to get SQL from an attached file first
        if message_data.get("files"):
            for file_info in message_data["files"]:
                is_sql_filetype = file_info.get("filetype") == "sql"
                is_plain_text_sql_file = file_info.get(
                    "mimetype"
                ) == "text/plain" and file_info.get("name", "").lower().endswith(".sql")
                if is_sql_filetype or is_plain_text_sql_file:
                    download_url = file_info.get("url_private_download")
                    if download_url:
                        logger.info(
                            f"Found SQL file: {file_info.get('name')}. Attempting to download from {download_url}"
                        )
                        try:
                            logger.info("Creating aiohttp session for file download")
                            async with aiohttp.ClientSession() as session:
                                logger.info("Making HTTP GET request to download file")
                                try:
                                    resp = await asyncio.wait_for(
                                        session.get(
                                            download_url,
                                            headers={
                                                "Authorization": f"Bearer {slack_bot_token}"
                                            },
                                        ),
                                        timeout=10.0,  # 10 second timeout for file download
                                    )
                                    async with resp:
                                        logger.info(
                                            f"HTTP response status: {resp.status}"
                                        )
                                        if resp.status == 200:
                                            logger.info("Reading response text...")
                                            sql_query = await resp.text()
                                            logger.info(
                                                f"Successfully downloaded SQL query from file: {file_info.get('name')}. Length: {len(sql_query)}"
                                            )
                                            break  # Use the first SQL file found
                                        else:
                                            logger.error(
                                                f"Failed to download SQL file {file_info.get('name')}. Status: {resp.status}. Response: {await resp.text()}"
                                            )
                                except asyncio.TimeoutError:
                                    logger.error(
                                        f"Timeout downloading file {file_info.get('name')} after 10 seconds"
                                    )
                        except Exception as e_download:
                            logger.error(
                                f"Exception during file download: {e_download}",
                                exc_info=True,
                            )

        # If no SQL query from file, try message text
        if not sql_query:
            raw_text = message_data.get("text", "").strip()
            # Remove bot mention if it's there (though less likely in a shortcut on a user message)
            bot_user_id = body.get("authorizations", [{}])[0].get(
                "user_id"
            )  # Safer access
            if bot_user_id:
                mention_pattern = f"<@{bot_user_id}>"
                if raw_text.startswith(mention_pattern):
                    raw_text = raw_text[len(mention_pattern) :].strip()
            sql_query = raw_text
            if sql_query:
                logger.info(
                    f"Using SQL query from message text. Length: {len(sql_query)}"
                )

        if not sql_query:
            logger.warning(
                f"User {user_id} triggered Metabase shortcut on message {original_message_ts} in {channel_id} with no usable SQL query in text or .sql attachment."
            )
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                thread_ts=original_message_ts,
                text="The message you selected is empty. Please use this shortcut on a message containing a SQL query.",
            )
            return

        # Get user's display name for the collection name
        user_info = await client.users_info(user=user_id)
        user_info_data = user_info.get("user", {})
        profile_data = user_info_data.get("profile", {})

        user_name = profile_data.get("display_name")
        if not user_name:
            user_name = user_info_data.get("real_name")
        if not user_name:
            user_name = user_info_data.get("name", "UnknownUser")

        # Sanitize user_name for collection name (optional, Metabase might handle it)
        # For simplicity, we'll use it as is, but consider replacing non-alphanumeric chars

        # Post an initial "processing" message
        processing_message = await client.chat_postMessage(
            channel=channel_id,
            thread_ts=original_message_ts,
            text=f"🤖 Understood, {user_name}! Creating a Metabase question with your query...",
        )

        # Background task for Metabase interaction
        asyncio.create_task(
            process_metabase_creation(
                sql_query=sql_query,
                user_name=user_name,
                user_timezone_str=None,  # Assuming UTC for now
                channel_id=channel_id,
                thread_ts=original_message_ts,  # Thread to reply under
                # Pass the ts of our "processing..." message to update it later
                status_update_message_ts=processing_message.get("ts"),
                slack_client=client,
                logger_instance=logger,
            )
        )

    except SlackApiError as e:
        logger.error(f"Slack API error in Metabase shortcut: {e}")
        # Try to send an ephemeral message if possible
        try:
            await client.chat_postEphemeral(
                channel=shortcut.get("channel", {}).get("id"),
                user=shortcut.get("user", {}).get("id"),
                thread_ts=shortcut.get("message", {}).get("ts"),
                text=f"A Slack API error occurred: {e.strerror}. Please try again or contact support.",
            )
        except Exception as e_say:
            logger.error(f"Error sending ephemeral error message: {e_say}")

    except Exception as e:
        logger.exception(f"Generic error in Metabase shortcut handler: {e}")
        try:
            await client.chat_postEphemeral(
                channel=shortcut.get("channel", {}).get("id"),
                user=shortcut.get("user", {}).get("id"),
                thread_ts=shortcut.get("message", {}).get("ts"),
                text="An unexpected error occurred while processing your request for Metabase.",
            )
        except Exception as e_say:
            logger.error(f"Error sending ephemeral generic error message: {e_say}")


async def process_metabase_creation(
    sql_query: str,
    user_name: str,
    user_timezone_str: Optional[str],
    channel_id: str,
    thread_ts: str,
    status_update_message_ts: Optional[str],
    slack_client: AsyncWebClient,
    logger_instance: logging.Logger,
):
    """Asynchronous task to interact with Metabase and update Slack."""
    # ADDED ENTRY LOG
    logger_instance.info(
        f"process_metabase_creation: Task started for user '{user_name}', thread_ts '{thread_ts}', query: '{sql_query[:50]}...'"
    )
    try:
        metabase_client = MetabaseClient()  # Initializes with env vars
        collection_path = ["Ragstar", user_name]  # As per requirement

        # Convert thread_ts to human-readable format in user's timezone
        readable_timestamp = thread_ts  # Fallback
        try:
            utc_dt = datetime.fromtimestamp(float(thread_ts), tz=pytz.utc)
            if user_timezone_str:
                try:
                    user_tz = pytz.timezone(user_timezone_str)
                    local_dt = utc_dt.astimezone(user_tz)
                    readable_timestamp = local_dt.strftime(
                        "%Y-%m-%d %H:%M:%S %Z"
                    )  # e.g., 2024-05-27 15:30:45 PDT
                except pytz.UnknownTimeZoneError:
                    logger_instance.warning(
                        f"Unknown timezone string: '{user_timezone_str}'. Using UTC for question name."
                    )
                    readable_timestamp = utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                logger_instance.info(
                    "User timezone not available. Using UTC for question name."
                )
                readable_timestamp = utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except ValueError:
            logger_instance.warning(
                f"Could not parse thread_ts '{thread_ts}' into a datetime. Using raw ts for question name."
            )
            # readable_timestamp remains raw thread_ts here

        question_name = f"Slack Query - {readable_timestamp}"
        description = f"SQL query submitted via Slack by {user_name} from channel {channel_id}, thread {thread_ts} (local time: {readable_timestamp}). Query: \n{sql_query[:1000]}{'...' if len(sql_query) > 1000 else ''}"

        logger_instance.info(
            f"Attempting to get/create Metabase collection: {collection_path}"
        )
        target_collection_id = await sync_to_async(
            metabase_client.get_or_create_collection_by_path
        )(collection_path)

        if not target_collection_id:
            error_msg = "Failed to create or find the necessary Metabase collection."
            logger_instance.error(error_msg)
            if status_update_message_ts:
                await slack_client.chat_update(
                    channel=channel_id,
                    ts=status_update_message_ts,
                    text=f"⚠️ Error: {error_msg} Could not save your Metabase question.",
                )
            else:  # Fallback if no initial message to update
                await slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=f"⚠️ Error: {error_msg}",
                )
            return

        logger_instance.info(
            f"Metabase collection ID {target_collection_id} obtained. Creating question: {question_name}"
        )
        new_question = await sync_to_async(
            metabase_client.create_native_query_question
        )(
            name=question_name,
            sql_query=sql_query,
            collection_id=target_collection_id,
            description=description,
        )

        question_id = new_question.get("id")
        question_url = (
            f"{metabase_client.metabase_url.rstrip('/')}/question/{question_id}"
        )
        success_message = f"✅ {user_name}, your Metabase question has been created!\n<{question_url}|View it here>"
        logger_instance.info(
            f"Successfully created Metabase question {question_id} for user {user_name}. Message: {success_message}"  # Log success message
        )

        # ADDED DETAILED LOGGING FOR FINAL SLACK UPDATE
        final_update_blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": success_message},
            }
        ]
        if status_update_message_ts:
            logger_instance.info(
                f"Attempting to update Slack message {channel_id}/{status_update_message_ts} with success."
            )
            await slack_client.chat_update(
                channel=channel_id,
                ts=status_update_message_ts,
                text=success_message,  # Fallback text
                blocks=final_update_blocks,
            )
            logger_instance.info(
                f"Successfully updated Slack message {channel_id}/{status_update_message_ts}."
            )
        else:
            logger_instance.info(
                f"Attempting to post new Slack message to {channel_id}/{thread_ts} with success."
            )
            await slack_client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=success_message,  # Fallback text
                blocks=final_update_blocks,
            )
            logger_instance.info(
                f"Successfully posted new Slack message to {channel_id}/{thread_ts}."
            )

    except ValueError as ve:  # Config errors from MetabaseClient init
        logger_instance.error(f"Metabase configuration error: {ve}")
        err_text = f"⚠️ Metabase Configuration Error: {ve}. Please check server logs and environment variables."
        if status_update_message_ts:
            logger_instance.error(
                f"Attempting to update Slack message {channel_id}/{status_update_message_ts} with error: {err_text}"
            )
            await slack_client.chat_update(
                channel=channel_id, ts=status_update_message_ts, text=err_text
            )
            logger_instance.info(
                f"Updated Slack message {channel_id}/{status_update_message_ts} with error notification."
            )
        else:
            logger_instance.error(
                f"Attempting to post new Slack message to {channel_id}/{thread_ts} with error: {err_text}"
            )
            await slack_client.chat_postMessage(
                channel=channel_id, thread_ts=thread_ts, text=err_text
            )
            logger_instance.info(
                f"Posted new Slack message to {channel_id}/{thread_ts} with error notification."
            )

    except Exception as e:
        logger_instance.exception(f"Error during Metabase question creation task: {e}")
        err_text = (
            f"⚠️ An error occurred while creating your Metabase question: {str(e)[:200]}"
        )
        # ADDED DETAILED LOGGING FOR FINAL SLACK UPDATE (ERROR CASE)
        if status_update_message_ts:
            logger_instance.error(
                f"Attempting to update Slack message {channel_id}/{status_update_message_ts} with error: {err_text}"
            )
            await slack_client.chat_update(
                channel=channel_id, ts=status_update_message_ts, text=err_text
            )
            logger_instance.info(
                f"Updated Slack message {channel_id}/{status_update_message_ts} with error notification."
            )
        else:
            logger_instance.error(
                f"Attempting to post new Slack message to {channel_id}/{thread_ts} with error: {err_text}"
            )
            await slack_client.chat_postMessage(
                channel=channel_id, thread_ts=thread_ts, text=err_text
            )
            logger_instance.info(
                f"Posted new Slack message to {channel_id}/{thread_ts} with error notification."
            )


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
