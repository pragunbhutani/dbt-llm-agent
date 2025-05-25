# apps/integrations/slack/views.py
import asyncio
import json
import logging
import re  # For parsing Slack message link
from typing import Optional
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings as django_settings  # For SLACK_SIGNING_SECRET

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

logger = logging.getLogger(__name__)


# Initialize the signature verifier
# IMPORTANT: Store your Slack Signing Secret securely, e.g., in Django settings or env var
SLACK_SIGNING_SECRET = getattr(django_settings, "SLACK_SIGNING_SECRET", None)
if not SLACK_SIGNING_SECRET:
    logger.critical(
        "SLACK_SIGNING_SECRET is not configured in Django settings. Slack request verification will fail!"
    )
    # In a real app, you might want to prevent startup or raise an ImproperlyConfigured exception
    # For now, verifier will be None if secret is missing, and verification will skip/fail.
    signature_verifier = None
else:
    signature_verifier = SignatureVerifier(SLACK_SIGNING_SECRET)


def verify_slack_request(request: HttpRequest) -> bool:
    """Verifies the request signature from Slack."""
    if not signature_verifier:
        logger.error(
            "Slack request verification skipped: SignatureVerifier not initialized (SLACK_SIGNING_SECRET missing)."
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
            auth_test_response = await slack_client.auth_test()
            if auth_test_response.get("ok"):
                bot_user_id_for_filtering = auth_test_response.get("user_id")
                logger.info(
                    f"slack_shortcut_view ({callback_id}): Successfully fetched bot user ID: {bot_user_id_for_filtering}"
                )
            else:
                logger.warning(
                    f"slack_shortcut_view ({callback_id}): auth.test call was not ok: {auth_test_response.get('error')}. Cannot filter bot messages by ID."
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
                                channel=user_id,
                                text=f"Sorry <@{user_id}>, I can't learn from the thread right now because the Language Model service isn't configured correctly. Please contact an administrator.",
                            )
                            return

                        thread_replies_response = (
                            await slack_client.conversations_replies(
                                channel=channel_id, ts=actual_thread_ts
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
                                bot_user_id_to_ignore=bot_user_id_for_filtering,
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
                                    channel=user_id,
                                    text=f"Sorry <@{user_id}>, I encountered an issue trying to understand the thread: {extracted_data.get('error')}. Details: {raw_response_snippet}...",
                                )
                                return

                            logger.info(
                                f"learn_from_thread shortcut: Successfully processed thread {channel_id}/{actual_thread_ts}. Learnings: {json.dumps(extracted_data, indent=2)}"
                            )

                            await slack_client.chat_postMessage(
                                channel=user_id,  # DM to the user
                                text=f"I've finished learning from the thread in <#{channel_id}>. The extracted insights have been logged.",
                            )

                        else:
                            logger.error(
                                f"learn_from_thread shortcut: Failed to fetch thread replies for {channel_id}/{actual_thread_ts}: {thread_replies_response.get('error')}"
                            )
                            await slack_client.chat_postMessage(
                                channel=user_id,  # DM to the user
                                text=f"Sorry <@{user_id}>, I couldn't fetch the messages from the thread in <#{channel_id}>: {thread_replies_response.get('error')}. Please check my logs.",
                            )
                    except SlackApiError as e:
                        logger.error(
                            f"learn_from_thread shortcut: Slack API error: {e.response['error']}",
                            exc_info=True,
                        )
                        try:
                            await slack_client.chat_postMessage(
                                channel=user_id,
                                text=f"Sorry <@{user_id}>, a Slack API error occurred while processing the thread: {e.response['error']}.",
                            )
                        except Exception as e_msg:
                            logger.error(f"Failed to send error DM to user: {e_msg}")
                    except Exception as e:
                        logger.error(
                            f"learn_from_thread shortcut: Unexpected error: {e}",
                            exc_info=True,
                        )
                        try:
                            await slack_client.chat_postMessage(
                                channel=user_id,
                                text=f"Sorry <@{user_id}>, an unexpected error occurred while I was processing the thread. Please check application logs.",
                            )
                        except Exception as e_msg:
                            logger.error(f"Failed to send error DM to user: {e_msg}")

                asyncio.create_task(process_thread_and_log_learnings_task())
                return HttpResponse(status=200)  # Acknowledge shortcut

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


# You might also need to add SLACK_SIGNING_SECRET to your ragstar/settings.py
# e.g., SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
