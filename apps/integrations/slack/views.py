# apps/integrations/slack/views.py
import asyncio
import json
import logging
import re  # For parsing Slack message link
import aiohttp  # Added for file download
from typing import Optional
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import (
    settings as django_settings,
)  # For INTEGRATIONS_SLACK_SIGNING_SECRET

# For Slack request verification (using slack_sdk utility)
from slack_sdk.signature import SignatureVerifier
from slack_sdk.errors import SlackApiError  # For client calls

# Import the workflow
from apps.workflows.query_executor.workflow import QueryExecutorWorkflow
from apps.workflows.services import get_slack_web_client  # For the workflow
from apps.workflows.knowledge_extractor.workflow import KnowledgeExtractorWorkflow
from apps.llm_providers.services import (
    default_chat_service,
)  # Import default_chat_service

# Import the Metabase handler logic and callback ID
try:
    from apps.integrations.slack.handlers import (
        process_metabase_creation,
        METABASE_SHORTCUT_CALLBACK_ID,
        slack_bot_token,  # For file download, if needed by process_metabase_creation's context
    )
    from apps.integrations.metabase.client import (
        MetabaseClient,
    )  # To check if available
except ImportError:
    process_metabase_creation = None
    METABASE_SHORTCUT_CALLBACK_ID = "create_metabase_query"  # Fallback
    MetabaseClient = None
    slack_bot_token = None
    logger.warning("Could not import Metabase related handlers or client for views.py.")

logger = logging.getLogger(__name__)


# Initialize the signature verifier
# IMPORTANT: Store your Slack Signing Secret securely, e.g., in Django settings or env var
INTEGRATIONS_SLACK_SIGNING_SECRET = getattr(
    django_settings, "INTEGRATIONS_SLACK_SIGNING_SECRET", None
)
if not INTEGRATIONS_SLACK_SIGNING_SECRET:
    logger.critical(
        "INTEGRATIONS_SLACK_SIGNING_SECRET is not configured in Django settings. Slack request verification will fail!"
    )
    # In a real app, you might want to prevent startup or raise an ImproperlyConfigured exception
    # For now, verifier will be None if secret is missing, and verification will skip/fail.
    signature_verifier = None
else:
    signature_verifier = SignatureVerifier(INTEGRATIONS_SLACK_SIGNING_SECRET)


def verify_slack_request(request: HttpRequest) -> bool:
    """Verifies the request signature from Slack."""
    if not signature_verifier:
        logger.error(
            "Slack request verification skipped: SignatureVerifier not initialized (INTEGRATIONS_SLACK_SIGNING_SECRET missing)."
        )
        # Depending on security policy, you might return False here to block unverified requests.
        # For development, you might allow if DEBUG=True, but be cautious.
        return (
            not django_settings.DEBUG
        )  # Only allow if not in DEBUG mode when secret is missing

    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    # The request body must be a raw string
    if not signature_verifier.is_valid_request(
        body=request.body.decode("utf-8"), headers=dict(request.headers)
    ):
        logger.warning("Slack request verification failed.")
        return False
    return True


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
async def slack_shortcut_view(request: HttpRequest):
    """Handles interactive shortcut payloads from Slack."""
    if request.method != "POST":
        logger.warning(
            f"Received non-POST request to slack_shortcut_view: {request.method}"
        )
        return HttpResponse("Invalid request method.", status=405)

    if not verify_slack_request(request):
        return HttpResponse("Slack request verification failed.", status=403)

    try:
        # For interactive components (like shortcuts), payload is in request.POST['payload'] as a JSON string
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

        if not all([user_id, channel_id, trigger_message_ts, actual_thread_ts]):
            logger.error(
                f"Missing critical information in shortcut payload for callback_id '{callback_id}': user_id={user_id}, channel_id={channel_id}, trigger_ts={trigger_message_ts}, actual_thread_ts={actual_thread_ts}"
            )
            return HttpResponse(status=200)  # Acknowledge receipt

        slack_client = get_slack_web_client()
        if not slack_client:
            logger.error(
                f"slack_shortcut_view ({callback_id}): Slack client not available."
            )
            return HttpResponse(status=200)

        # Get the bot's user ID to filter its own messages later
        bot_user_id_for_filtering = None
        try:
            logger.info(
                f"slack_shortcut_view ({callback_id}): Calling auth.test to get bot user ID..."
            )
            auth_test_response = await asyncio.wait_for(
                slack_client.auth_test(), timeout=5.0  # 5 second timeout for auth.test
            )
            if auth_test_response.get("ok"):
                bot_user_id_for_filtering = auth_test_response.get("user_id")
                logger.info(
                    f"slack_shortcut_view ({callback_id}): Successfully fetched bot user ID: {bot_user_id_for_filtering}"
                )
            else:
                logger.warning(
                    f"slack_shortcut_view ({callback_id}): auth.test call was not ok: {auth_test_response.get('error')}. Cannot filter bot messages by ID."
                )
        except asyncio.TimeoutError:
            logger.error(
                f"slack_shortcut_view ({callback_id}): Timeout calling auth.test after 5 seconds"
            )
        except Exception as e:
            logger.error(
                f"slack_shortcut_view ({callback_id}): Failed to call auth.test to get bot user ID: {e}",
                exc_info=True,
            )
            # Proceed without bot_user_id_for_filtering, filtering will be less effective

        # Check if it's a message action and the correct callback_id
        if payload.get("type") == "message_action":
            if callback_id == "execute_query":
                logger.info(
                    f"Processing 'execute_query' shortcut: user_id={user_id}, channel_id={channel_id}, trigger_message_ts={trigger_message_ts}, actual_thread_ts={actual_thread_ts}"
                )

                async def run_execute_query_workflow_task():
                    workflow = QueryExecutorWorkflow(slack_client=slack_client)
                    try:
                        logger.info(
                            f"Triggering QueryExecutorWorkflow from shortcut for target thread: {channel_id}/{actual_thread_ts}"
                        )
                        await workflow.run_workflow(
                            channel_id=channel_id,
                            thread_ts=actual_thread_ts,
                            user_id=user_id,
                            trigger_message_ts=trigger_message_ts,  # The message the shortcut was invoked on
                        )
                    except Exception as e:
                        logger.error(
                            f"Error running QueryExecutorWorkflow from shortcut: {e}",
                            exc_info=True,
                        )
                        try:
                            await slack_client.chat_postMessage(
                                channel=channel_id,
                                thread_ts=actual_thread_ts,
                                text=f"<@{user_id}> Sorry, an unexpected error occurred while processing the shortcut: {e}",
                            )
                        except Exception as post_error:
                            logger.error(
                                f"Failed to post critical workflow start error from shortcut to Slack: {post_error}"
                            )

                asyncio.create_task(run_execute_query_workflow_task())
                return HttpResponse(status=200)  # Acknowledge the shortcut immediately

            elif callback_id == "learn_from_thread":
                logger.info(
                    f"Processing 'learn_from_thread' shortcut: user_id={user_id}, channel_id={channel_id}, trigger_message_ts={trigger_message_ts}, actual_thread_ts={actual_thread_ts}"
                )
                # Acknowledge immediately
                # Logic to process this will be in a background task.

                async def process_thread_and_log_learnings_task():
                    try:
                        llm_client = default_chat_service.get_client()
                        if not llm_client:
                            logger.error(
                                "learn_from_thread shortcut: LLM client not available from default_chat_service. Check LLM provider settings."
                            )
                            await slack_client.chat_postMessage(
                                channel=user_id,  # DM the user who invoked
                                text=f"Sorry <@{user_id}>, I can't learn from the thread right now because the Language Model service isn't configured correctly. Please contact an administrator.",
                            )
                            return

                        thread_replies_response = (
                            await slack_client.conversations_replies(
                                channel=channel_id,
                                ts=actual_thread_ts,  # Use actual_thread_ts
                            )
                        )
                        if thread_replies_response.get(
                            "ok"
                        ) and thread_replies_response.get("messages"):
                            thread_messages = thread_replies_response["messages"]
                            logger.info(
                                f"learn_from_thread shortcut: Fetched {len(thread_messages)} messages from thread {channel_id}/{actual_thread_ts}"
                            )

                            knowledge_workflow = KnowledgeExtractorWorkflow(
                                llm_client=llm_client,
                                bot_user_id_to_ignore=bot_user_id_for_filtering,  # Pass the fetched bot_user_id
                            )
                            extracted_data = (
                                await knowledge_workflow.extract_from_thread(
                                    thread_messages,
                                    source_slack_channel_id=channel_id,
                                    source_slack_thread_ts=actual_thread_ts,
                                )
                            )

                            # Check for errors from the LLM call within extracted_data
                            if isinstance(extracted_data, dict) and extracted_data.get(
                                "error"
                            ):
                                logger.error(
                                    f"learn_from_thread shortcut: Error during knowledge extraction: {extracted_data.get('error')}"
                                )
                                raw_response_snippet = extracted_data.get(
                                    "raw_response", ""
                                )[:200]
                                await slack_client.chat_postMessage(
                                    channel=user_id,  # DM the user
                                    text=f"Sorry <@{user_id}>, I encountered an issue trying to understand the thread: {extracted_data.get('error')}. Details: {raw_response_snippet}...",
                                )
                                return

                            logger.info(
                                f"learn_from_thread shortcut: Successfully processed thread {channel_id}/{actual_thread_ts}. Learnings: {json.dumps(extracted_data, indent=2)}"
                            )

                            # Use user_id for DM confirmation, or channel_id/actual_thread_ts for public confirmation
                            public_confirmation_channel = channel_id
                            public_confirmation_thread_ts = actual_thread_ts

                            confirmation_text = f"<@{user_id}> I've finished learning from the thread: {payload.get('message',{}).get('permalink','this thread')}. Summary: {extracted_data.get('summary','No summary generated.')}"
                            if extracted_data.get("questions_and_answers"):
                                confirmation_text += f"\nIdentified {len(extracted_data['questions_and_answers'])} Q&A pairs."

                            try:
                                await slack_client.chat_postMessage(
                                    channel=public_confirmation_channel,
                                    thread_ts=public_confirmation_thread_ts,
                                    text=confirmation_text,
                                )
                            except SlackApiError as e_slack:
                                logger.error(
                                    f"learn_from_thread shortcut: Failed to send public confirmation: {e_slack}"
                                )
                                # Fallback to DM if public post fails
                                await slack_client.chat_postMessage(
                                    channel=user_id,  # DM the user
                                    text=f"I've finished learning from the thread. Summary: {extracted_data.get('summary','No summary generated.')}",
                                )

                        else:
                            logger.warning(
                                f"learn_from_thread shortcut: Could not fetch replies for thread {channel_id}/{actual_thread_ts}. Response: {thread_replies_response.get('error', 'Unknown error')}"
                            )
                            await slack_client.chat_postMessage(
                                channel=user_id,  # DM the user
                                text=f"Sorry <@{user_id}>, I couldn't fetch the messages from the thread to learn from them.",
                            )
                    except Exception as e:
                        logger.error(
                            f"learn_from_thread shortcut: Unexpected error in background task: {e}",
                            exc_info=True,
                        )
                        try:
                            await slack_client.chat_postMessage(
                                channel=user_id,  # DM the user
                                text=f"Sorry <@{user_id}>, an unexpected error occurred while trying to learn from the thread.",
                            )
                        except Exception as e_slack_crit:
                            logger.error(
                                f"learn_from_thread shortcut: Failed to send critical error DM: {e_slack_crit}"
                            )

                asyncio.create_task(process_thread_and_log_learnings_task())
                return HttpResponse(status=200)  # Acknowledge immediately

            elif (
                METABASE_SHORTCUT_CALLBACK_ID
                and callback_id == METABASE_SHORTCUT_CALLBACK_ID
            ):
                logger.info(
                    f"Processing '{METABASE_SHORTCUT_CALLBACK_ID}' shortcut: user_id={user_id}, channel_id={channel_id}, trigger_message_ts={trigger_message_ts}"
                )

                if (
                    not MetabaseClient
                    or not process_metabase_creation
                    or not slack_bot_token
                ):
                    logger.error(
                        f"Metabase integration is not properly configured or imported for callback_id {METABASE_SHORTCUT_CALLBACK_ID}. Cannot process."
                    )
                    # Try to send an ephemeral message back to the user
                    try:
                        await slack_client.chat_postEphemeral(
                            channel=channel_id,
                            user=user_id,
                            thread_ts=actual_thread_ts,  # Use actual_thread_ts or trigger_message_ts
                            text="Sorry, the Metabase integration is not configured correctly on the server. Please contact an administrator.",
                        )
                    except Exception as e_say:
                        logger.error(
                            f"Error sending ephemeral message about Metabase misconfiguration: {e_say}"
                        )
                    return HttpResponse(status=200)  # Acknowledge anyway

                # Get user_name and timezone
                user_info_response = None  # Initialize
                user_data = {}  # Initialize
                user_profile = {}  # Initialize
                user_timezone_str = None  # Initialize
                user_name_for_collection = "UnknownUser"  # Initialize with a default

                try:
                    logger.info(
                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': Calling users.info for user {user_id}"
                    )
                    user_info_response = await asyncio.wait_for(
                        slack_client.users_info(user=user_id),
                        timeout=5.0,  # 5-second timeout
                    )
                    user_data = user_info_response.get("user", {})
                    user_profile = user_data.get("profile", {})
                    user_timezone_str = user_data.get("tz")
                    logger.info(
                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': Successfully fetched users.info for {user_id}. Timezone: {user_timezone_str}"
                    )

                    user_name_for_collection = user_profile.get("display_name")
                    if not user_name_for_collection:
                        user_name_for_collection = user_profile.get("real_name")
                    if not user_name_for_collection:
                        user_name_for_collection = user_data.get("name", "UnknownUser")
                    logger.info(
                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': User name for collection: {user_name_for_collection}"
                    )

                except asyncio.TimeoutError:
                    logger.error(
                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': Timeout calling users.info for user {user_id} after 5 seconds."
                    )
                    # Decide how to proceed: maybe use default user_name and no timezone, or send error to user
                    # For now, we'll proceed with defaults, which means user_name_for_collection is "UnknownUser"
                    # and user_timezone_str is None.
                    # Optionally, send an ephemeral message to the user about this specific failure.
                    await slack_client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        thread_ts=actual_thread_ts,
                        text="Sorry, I couldn't fetch your user details from Slack in time. Using default information for now.",
                    )
                except Exception as e_user_info:
                    logger.error(
                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': Error calling users.info for user {user_id}: {e_user_info}",
                        exc_info=True,
                    )
                    # Same fallback as timeout
                    await slack_client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        thread_ts=actual_thread_ts,
                        text="Sorry, an error occurred while fetching your user details from Slack. Using default information for now.",
                    )

                # Initialize SQL query content variable
                sql_query_content = None

                # Attempt to get SQL from an attached file first
                if message_data.get("files"):
                    for file_info in message_data["files"]:
                        is_sql_filetype = file_info.get("filetype") == "sql"
                        is_plain_text_sql_file = file_info.get(
                            "mimetype"
                        ) == "text/plain" and file_info.get(
                            "name", ""
                        ).lower().endswith(
                            ".sql"
                        )
                        if is_sql_filetype or is_plain_text_sql_file:
                            download_url = file_info.get("url_private_download")
                            if download_url:
                                logger.info(
                                    f"'{METABASE_SHORTCUT_CALLBACK_ID}': Found SQL file: {file_info.get('name')}. Starting download from {download_url}"
                                )
                                try:
                                    logger.info(
                                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': Preparing aiohttp.ClientSession for file download."
                                    )
                                    async with aiohttp.ClientSession() as http_session:
                                        logger.info(
                                            f"'{METABASE_SHORTCUT_CALLBACK_ID}': aiohttp.ClientSession created. Preparing to make GET request."
                                        )
                                        try:
                                            logger.info(
                                                f"'{METABASE_SHORTCUT_CALLBACK_ID}': Calling http_session.get() within asyncio.wait_for for URL: {download_url}"
                                            )
                                            resp = await asyncio.wait_for(
                                                http_session.get(
                                                    download_url,
                                                    headers={
                                                        "Authorization": f"Bearer {slack_bot_token}"
                                                    },
                                                ),
                                                timeout=10.0,  # 10 second timeout for file download
                                            )
                                            logger.info(
                                                f"'{METABASE_SHORTCUT_CALLBACK_ID}': http_session.get() completed. Response status: {resp.status if resp else 'No response'}"
                                            )
                                            async with (
                                                resp
                                            ):  # This line will fail if resp is None
                                                logger.info(
                                                    f"'{METABASE_SHORTCUT_CALLBACK_ID}': HTTP response status: {resp.status}"
                                                )
                                                if resp.status == 200:
                                                    logger.info(
                                                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': Reading response text..."
                                                    )
                                                    sql_query_content = (
                                                        await resp.text()
                                                    )
                                                    logger.info(
                                                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': Downloaded SQL from file {file_info.get('name')}"
                                                    )
                                                    logger.info(
                                                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': SQL content from file '{file_info.get('name')}' has length {len(sql_query_content if sql_query_content else '')}. Breaking from file loop."
                                                    )
                                                    break
                                                else:
                                                    logger.error(
                                                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': Failed to download SQL file {file_info.get('name')}. Status: {resp.status}"
                                                    )
                                        except asyncio.TimeoutError:
                                            logger.error(
                                                f"'{METABASE_SHORTCUT_CALLBACK_ID}': Timeout downloading file {file_info.get('name')} after 10 seconds"
                                            )
                                except Exception as e_download:
                                    logger.error(
                                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': Exception during file download: {e_download}",
                                        exc_info=True,
                                    )

                if not sql_query_content:
                    sql_query_content = message_data.get("text", "").strip()
                    # Basic bot mention removal (if any)
                    if bot_user_id_for_filtering and sql_query_content.startswith(
                        f"<@{bot_user_id_for_filtering}>"
                    ):
                        sql_query_content = sql_query_content[
                            len(f"<@{bot_user_id_for_filtering}>") :
                        ].strip()
                    if sql_query_content:
                        logger.info(
                            f"'{METABASE_SHORTCUT_CALLBACK_ID}': Using SQL from message text."
                        )

                if not sql_query_content:
                    logger.warning(
                        f"'{METABASE_SHORTCUT_CALLBACK_ID}': No SQL query found in message text or .sql attachment for user {user_id} on {trigger_message_ts}."
                    )
                    await slack_client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        thread_ts=actual_thread_ts,
                        text="The message selected doesn't contain a SQL query in its text or as a .sql file attachment.",
                    )
                    return HttpResponse(status=200)

                # Skip the initial "processing" message to avoid hanging issues
                # Just proceed directly to the Metabase task
                logger.info(
                    f"'{METABASE_SHORTCUT_CALLBACK_ID}': Proceeding directly to Metabase creation task for user {user_id}."
                )
                asyncio.create_task(
                    process_metabase_creation(
                        sql_query=sql_query_content,
                        user_name=user_name_for_collection,
                        user_timezone_str=user_timezone_str,  # Pass timezone
                        channel_id=channel_id,
                        thread_ts=actual_thread_ts,
                        status_update_message_ts=None,  # No initial message to update
                        slack_client=slack_client,
                        logger_instance=logger,  # Use the logger from views.py
                    )
                )
                return HttpResponse(status=200)  # Acknowledge immediately

            else:
                logger.warning(
                    f"slack_shortcut_view: Received unhandled message_action callback_id: {callback_id}"
                )
                return HttpResponse(status=200)

        logger.info(
            f"Received unhandled shortcut/interactive payload type or callback_id: type='{payload.get('type')}', callback_id='{callback_id}'"
        )
        return HttpResponse(status=200)

    except json.JSONDecodeError:
        logger.error(
            "slack_shortcut_view: Failed to decode JSON payload.", exc_info=True
        )
        return HttpResponse("Invalid JSON payload.", status=400)
    except Exception as e:
        logger.error(f"Error processing shortcut: {e}", exc_info=True)
        # Acknowledge to prevent Slack retries, even for unexpected errors
        return HttpResponse(status=200)


# You might also need to add INTEGRATIONS_SLACK_SIGNING_SECRET to your ragstar/settings.py
# e.g., INTEGRATIONS_SLACK_SIGNING_SECRET = os.environ.get("INTEGRATIONS_SLACK_SIGNING_SECRET")
