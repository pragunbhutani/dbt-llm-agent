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


@csrf_exempt  # Slack POST requests won't have a CSRF token
async def execute_sql_command_view(request: HttpRequest):
    """Handles the /execute-sql slash command from Slack."""
    if request.method != "POST":
        logger.warning(
            f"Received non-POST request to execute_sql_command_view: {request.method}"
        )
        return HttpResponse("Invalid request method.", status=405)

    if not verify_slack_request(request):
        return HttpResponse("Slack request verification failed.", status=403)

    try:
        # Slack sends command data as application/x-www-form-urlencoded
        payload = request.POST
        logger.info(f"Received /execute-sql command payload: {payload}")

        command_text = payload.get("text", "").strip()
        user_id = payload.get("user_id")
        # channel_id_command_invoked_in = payload.get("channel_id") # Where user typed /execute-sql

        if not command_text:
            return JsonResponse(
                {
                    "response_type": "ephemeral",
                    "text": "Please provide a link to a message in the thread containing the SQL query. Usage: `/execute-sql <link_to_message>`",
                }
            )

        target_channel_id, target_message_ts = parse_slack_message_link(command_text)

        if not target_channel_id or not target_message_ts:
            return JsonResponse(
                {
                    "response_type": "ephemeral",
                    "text": "Invalid Slack message link provided. Please ensure it's a valid link to a message.",
                }
            )

        # Now, we need to determine the actual thread_ts for the QueryExecutorWorkflow.
        # The target_message_ts *could* be the thread_ts if it's the parent message,
        # or it could be a reply within the thread. We should fetch the message info
        # to get its `thread_ts` property if it exists.

        slack_client = get_slack_web_client()
        if not slack_client:
            logger.error("execute_sql_command_view: Slack client not available.")
            return JsonResponse(
                {
                    "response_type": "ephemeral",
                    "text": "Error: Could not connect to Slack to process the request.",
                },
                status=500,
            )

        actual_thread_ts = None
        try:
            # Fetch history for the single message to get its thread_ts if it's part of a thread
            # Using conversations.replies with limit 1 is also an option if we know it's a parent.
            # conversations.history is more robust for getting info about a specific message TS.
            history_response = await slack_client.conversations_history(
                channel=target_channel_id,
                latest=target_message_ts,
                inclusive=True,
                limit=1,
            )
            if history_response.get("ok") and history_response.get("messages"):
                message_info = history_response["messages"][0]
                actual_thread_ts = message_info.get(
                    "thread_ts", target_message_ts
                )  # Use thread_ts if present, else the message itself is the root
                logger.info(
                    f"Determined thread_ts: {actual_thread_ts} for linked message {target_message_ts} in channel {target_channel_id}"
                )
            else:
                logger.error(
                    f"Failed to fetch message info for {target_channel_id}/{target_message_ts}: {history_response.get('error')}"
                )
                return JsonResponse(
                    {
                        "response_type": "ephemeral",
                        "text": "Error: Could not fetch information for the linked message.",
                    },
                    status=200,
                )
        except SlackApiError as e:
            logger.error(
                f"Slack API error fetching message info: {e.response['error']}",
                exc_info=True,
            )
            return JsonResponse(
                {
                    "response_type": "ephemeral",
                    "text": f"Slack API Error: {e.response['error']}",
                },
                status=200,
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching message info: {e}", exc_info=True)
            return JsonResponse(
                {
                    "response_type": "ephemeral",
                    "text": "An unexpected error occurred while fetching message details.",
                },
                status=500,
            )

        if not actual_thread_ts:  # Should be set if message was found
            logger.error(
                "Could not determine actual_thread_ts after fetching message info."
            )
            return JsonResponse(
                {
                    "response_type": "ephemeral",
                    "text": "Error processing the linked message's thread information.",
                },
                status=200,
            )

        ack_text = f"Got it, <@{user_id}>! I'll process the thread containing the linked message (`{actual_thread_ts}`) and look for an SQL query to execute. Results will be posted in that thread."

        async def run_workflow_task():
            # Slack client is already fetched above
            workflow = QueryExecutorWorkflow(
                slack_client=slack_client
            )  # Pass the same client
            try:
                logger.info(
                    f"Triggering QueryExecutorWorkflow for target thread: {target_channel_id}/{actual_thread_ts}"
                )
                await workflow.run_workflow(
                    channel_id=target_channel_id,  # Use channel from link
                    thread_ts=actual_thread_ts,  # Use determined thread_ts
                    user_id=user_id,
                    trigger_message_ts=actual_thread_ts,  # The workflow expects original_trigger_message_ts to be the relevant thread
                )
            except Exception as e:
                logger.error(
                    f"Error running QueryExecutorWorkflow from slash command: {e}",
                    exc_info=True,
                )
                try:
                    await slack_client.chat_postMessage(
                        channel=target_channel_id,
                        thread_ts=actual_thread_ts,
                        text=f"<@{user_id}> Sorry, an unexpected error occurred trying to start the SQL execution for the linked message: {e}",
                    )
                except Exception as post_error:
                    logger.error(
                        f"Failed to post critical workflow start error to Slack: {post_error}"
                    )

        asyncio.create_task(run_workflow_task())

        return JsonResponse({"response_type": "ephemeral", "text": ack_text})

    except Exception as e:
        logger.error(f"Error processing /execute-sql command: {e}", exc_info=True)
        return JsonResponse(
            {
                "response_type": "ephemeral",
                "text": f"An general unexpected error occurred: {e}",
            },
            status=500,
        )


# You might also need to add SLACK_SIGNING_SECRET to your ragstar/settings.py
# e.g., SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")


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

        # Check if it's a message action and the correct callback_id
        if (
            payload.get("type") == "message_action"
            and payload.get("callback_id") == "execute_query"
        ):
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
                    f"Missing critical information in shortcut payload: user_id={user_id}, channel_id={channel_id}, trigger_ts={trigger_message_ts}, actual_thread_ts={actual_thread_ts}"
                )
                # Slack expects a 200 OK even for errors if we can't process,
                # but we shouldn't proceed. We can log the error.
                # Sending an ephemeral message here is tricky as we might not have response_url for all errors.
                return HttpResponse(status=200)  # Acknowledge receipt

            logger.info(
                f"Processing 'execute_query_csv' shortcut: user_id={user_id}, channel_id={channel_id}, trigger_message_ts={trigger_message_ts}, actual_thread_ts={actual_thread_ts}"
            )

            slack_client = get_slack_web_client()
            if not slack_client:
                logger.error("slack_shortcut_view: Slack client not available.")
                # Acknowledge to prevent Slack retries, but log the issue.
                return HttpResponse(status=200)

            async def run_workflow_task():
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
                    # Attempt to notify the user in the thread if possible
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

            asyncio.create_task(run_workflow_task())

            # Acknowledge the shortcut interaction immediately
            return HttpResponse(status=200)

        else:
            logger.info(
                f"Received unhandled shortcut/interactive payload type or callback_id: type='{payload.get('type')}', callback_id='{payload.get('callback_id')}'"
            )
            # Acknowledge other types of interactions too, but do nothing.
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
