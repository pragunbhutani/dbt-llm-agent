"""
Celery tasks for running workflows in the background.
This allows for better scalability and user experience by handling
long-running operations asynchronously.
"""

import logging
import json
from typing import Dict, Any, Optional
from celery import shared_task
from django.utils import timezone

from apps.workflows.slack_responder import SlackResponderAgent
from apps.accounts.models import OrganisationSettings
from slack_sdk.web.async_client import AsyncWebClient
import asyncio

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def run_conversation_workflow(
    self,
    question: str,
    channel_id: str,
    thread_ts: str,
    user_id: str,
    org_settings_id: int,
    slack_bot_token: str,
    conversation_metadata: Optional[Dict[str, Any]] = None,
):
    """
    Celery task to run multi-agent slack workflow in the background.

    Args:
        question: User's question
        channel_id: Slack channel ID
        thread_ts: Slack thread timestamp
        user_id: Slack user ID
        org_settings_id: Organization settings ID
        slack_bot_token: Slack bot token for API calls
        conversation_metadata: Additional metadata about the conversation
    """

    try:
        logger.info(f"Starting multi-agent workflow task for {channel_id}/{thread_ts}")

        # Check if there's already an active task for this conversation
        from apps.workflows.models import Conversation, ConversationStatus

        # Look for existing active conversations for this thread
        active_conversation = Conversation.objects.filter(
            external_id=thread_ts,
            channel_id=channel_id,
            status=ConversationStatus.ACTIVE,
        ).first()

        if active_conversation:
            # Check if there's already a task running for this conversation
            task_id_in_context = active_conversation.conversation_context.get(
                "celery_task_id"
            )
            if task_id_in_context and task_id_in_context != self.request.id:
                logger.warning(
                    f"Another task {task_id_in_context} is already processing conversation {channel_id}/{thread_ts}, skipping"
                )
                return {
                    "success": False,
                    "error": "Duplicate task prevented",
                    "existing_task_id": task_id_in_context,
                }

        # Get organization settings
        try:
            org_settings = OrganisationSettings.objects.get(
                organisation_id=org_settings_id
            )
        except OrganisationSettings.DoesNotExist:
            logger.error(f"Organization settings not found: {org_settings_id}")
            return {"success": False, "error": "Organization settings not found"}

        # Create Slack client
        slack_client = AsyncWebClient(token=slack_bot_token)

        # Create slack responder agent (multi-agent orchestrator)
        slack_responder = SlackResponderAgent(
            org_settings=org_settings,
            slack_client=slack_client,
            memory=None,  # Could add Redis-based checkpointing here
        )

        # Run the multi-agent workflow asynchronously
        async def run_workflow():
            return await slack_responder.run_slack_workflow(
                question=question,
                channel_id=channel_id,
                thread_ts=thread_ts,
                user_id=user_id,
            )

        # Execute the async workflow
        result = asyncio.run(run_workflow())

        # Clear the task ID from conversation context on successful completion
        if active_conversation:
            try:
                context = active_conversation.conversation_context or {}
                if context.get("celery_task_id") == self.request.id:
                    context.pop("celery_task_id", None)
                    context["last_completed_task_id"] = self.request.id
                    context["last_completed_at"] = timezone.now().isoformat()
                    active_conversation.conversation_context = context
                    active_conversation.save()
            except Exception as cleanup_exc:
                logger.warning(
                    f"Failed to clear task ID from conversation context: {cleanup_exc}"
                )

        logger.info(
            f"Multi-agent workflow completed successfully for {channel_id}/{thread_ts}"
        )

        # Extract only serializable data from result to avoid serialization errors
        serializable_result = {}
        if isinstance(result, dict):
            for key, value in result.items():
                if key == "final_state" and isinstance(value, dict):
                    # Handle final_state specially since it contains langchain objects
                    final_state_serializable = {}
                    for fs_key, fs_value in value.items():
                        if fs_key == "messages":
                            # Convert messages to serializable format
                            final_state_serializable[fs_key] = (
                                len(fs_value) if fs_value else 0
                            )
                        elif isinstance(
                            fs_value, (str, int, float, bool, list, dict, type(None))
                        ):
                            final_state_serializable[fs_key] = fs_value
                        else:
                            # Convert complex objects to strings, truncated
                            str_value = str(fs_value)
                            final_state_serializable[fs_key] = (
                                str_value[:200] + "..."
                                if len(str_value) > 200
                                else str_value
                            )
                    serializable_result[key] = final_state_serializable
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable_result[key] = value
                elif hasattr(value, "__dict__"):
                    # For complex objects, store a summary
                    str_value = str(value)
                    serializable_result[key] = (
                        str_value[:200] + "..." if len(str_value) > 200 else str_value
                    )
                else:
                    str_value = str(value)
                    serializable_result[key] = (
                        str_value[:200] + "..." if len(str_value) > 200 else str_value
                    )

        return {
            "success": True,
            "result": serializable_result,
            "completed_at": timezone.now().isoformat(),
        }

    except Exception as exc:
        logger.exception(f"Error in multi-agent workflow task: {exc}")

        # Retry on transient errors (but not on duplicate key errors which indicate data issues)
        if (
            self.request.retries < self.max_retries
            and "duplicate key value violates unique constraint" not in str(exc)
        ):
            logger.info(
                f"Retrying multi-agent workflow task (attempt {self.request.retries + 1})"
            )
            raise self.retry(exc=exc)

        # Send error message to user after all retries exhausted
        try:
            asyncio.run(
                send_error_message_to_slack(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    user_id=user_id,
                    slack_bot_token=slack_bot_token,
                    error_message="I'm sorry, I encountered a persistent issue while processing your request. Please try again later or contact support.",
                )
            )
        except Exception as slack_error:
            logger.error(f"Failed to send error message to Slack: {slack_error}")

        # Clear the task ID from conversation context on failure after all retries
        if active_conversation:
            try:
                context = active_conversation.conversation_context or {}
                if context.get("celery_task_id") == self.request.id:
                    context.pop("celery_task_id", None)
                    context["last_failed_task_id"] = self.request.id
                    context["last_failed_at"] = timezone.now().isoformat()
                    context["last_failure_reason"] = str(exc)
                    active_conversation.conversation_context = context
                    active_conversation.save()
            except Exception as cleanup_exc:
                logger.warning(
                    f"Failed to clear task ID from conversation context on failure: {cleanup_exc}"
                )

        # Ensure error message is serializable
        error_msg = str(exc)[:500] + "..." if len(str(exc)) > 500 else str(exc)
        return {"success": False, "error": error_msg, "retries_exhausted": True}


async def send_error_message_to_slack(
    channel_id: str,
    thread_ts: str,
    user_id: str,
    slack_bot_token: str,
    error_message: str,
):
    """Helper function to send error messages to Slack."""

    slack_client = AsyncWebClient(token=slack_bot_token)

    try:
        await slack_client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f"<@{user_id}> {error_message}",
        )
    except Exception as e:
        logger.error(f"Failed to send error message to Slack: {e}")


@shared_task(bind=True)
def cleanup_old_conversations(self, days_old: int = 30):
    """
    Celery task to clean up old conversation records.

    Args:
        days_old: Number of days after which to clean up conversations
    """

    try:
        from apps.workflows.models import Conversation
        from django.utils import timezone
        from datetime import timedelta

        cutoff_date = timezone.now() - timedelta(days=days_old)

        # Delete old conversations
        deleted_count, _ = Conversation.objects.filter(
            created_at__lt=cutoff_date
        ).delete()

        logger.info(
            f"Cleaned up {deleted_count} old conversations older than {days_old} days"
        )

        return {
            "success": True,
            "deleted_count": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
        }

    except Exception as exc:
        logger.exception(f"Error in cleanup task: {exc}")
        return {"success": False, "error": str(exc)}


@shared_task(bind=True)
def generate_conversation_analytics(self, org_id: int, days_back: int = 7):
    """
    Generate analytics for conversations over the specified period.

    Args:
        org_id: Organization ID
        days_back: Number of days to analyze
    """

    try:
        from apps.workflows.models import Conversation
        from django.utils import timezone
        from datetime import timedelta

        start_date = timezone.now() - timedelta(days=days_back)

        conversations = Conversation.objects.filter(
            org_settings__organisation_id=org_id, created_at__gte=start_date
        )

        analytics = {
            "total_conversations": conversations.count(),
            "successful_conversations": conversations.filter(
                status="completed"
            ).count(),
            "failed_conversations": conversations.filter(status="failed").count(),
            "average_duration": 0,  # Could calculate based on conversation parts
            "most_common_intents": {},  # Could analyze based on conversation metadata
            "user_satisfaction": {},  # Based on feedback
        }

        logger.info(f"Generated analytics for org {org_id}: {analytics}")

        return {
            "success": True,
            "analytics": analytics,
            "period": f"{days_back} days",
            "generated_at": timezone.now().isoformat(),
        }

    except Exception as exc:
        logger.exception(f"Error generating analytics: {exc}")
        return {"success": False, "error": str(exc)}


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def run_query_executor_workflow(
    self,
    channel_id: str,
    thread_ts: str,
    user_id: str,
    trigger_message_ts: str,
    org_settings_id: int,
    slack_bot_token: str,
):
    """
    Celery task to run QueryExecutor workflow in the background.

    Args:
        channel_id: Slack channel ID
        thread_ts: Slack thread timestamp
        user_id: Slack user ID who triggered the shortcut
        trigger_message_ts: Timestamp of the message the shortcut was triggered on
        org_settings_id: Organization settings ID
        slack_bot_token: Slack bot token for API calls
    """

    try:
        logger.info(
            f"Starting QueryExecutor workflow task for {channel_id}/{thread_ts}"
        )

        # Get organization settings
        try:
            org_settings = OrganisationSettings.objects.get(
                organisation_id=org_settings_id
            )
        except OrganisationSettings.DoesNotExist:
            logger.error(f"Organization settings not found: {org_settings_id}")
            return {"success": False, "error": "Organization settings not found"}

        # Create Slack client
        slack_client = AsyncWebClient(token=slack_bot_token)

        # Import and create QueryExecutor workflow
        from apps.workflows.query_executor.workflow import QueryExecutorWorkflow

        async def run_workflow():
            try:
                workflow = QueryExecutorWorkflow(
                    org_settings=org_settings, slack_client=slack_client
                )

                await workflow.run_workflow(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    user_id=user_id,
                    trigger_message_ts=trigger_message_ts,
                )

                return {"success": True}

            except Exception as e:
                logger.error(f"Error in QueryExecutor workflow: {e}", exc_info=True)

                # Send error message to user
                try:
                    await slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=f"<@{user_id}> Sorry, an unexpected error occurred while processing the query executor: {e}",
                    )
                except Exception as post_error:
                    logger.error(f"Failed to post error message to Slack: {post_error}")

                raise e

        # Execute the async workflow
        result = asyncio.run(run_workflow())

        logger.info(
            f"QueryExecutor workflow completed successfully for {channel_id}/{thread_ts}"
        )

        return {
            "success": True,
            "result": result,
            "completed_at": timezone.now().isoformat(),
        }

    except Exception as exc:
        logger.exception(f"Error in QueryExecutor workflow task: {exc}")

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying QueryExecutor workflow task (attempt {self.request.retries + 1})"
            )
            raise self.retry(exc=exc)

        # Send error message to user after all retries exhausted
        try:
            asyncio.run(
                send_error_message_to_slack(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    user_id=user_id,
                    slack_bot_token=slack_bot_token,
                    error_message="I'm sorry, I encountered a persistent issue while processing the query executor. Please try again later or contact support.",
                )
            )
        except Exception as slack_error:
            logger.error(f"Failed to send error message to Slack: {slack_error}")

        return {"success": False, "error": str(exc), "retries_exhausted": True}


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def run_knowledge_extractor_workflow(
    self,
    channel_id: str,
    thread_ts: str,
    user_id: str,
    org_settings_id: int,
    slack_bot_token: str,
):
    """
    Celery task to run KnowledgeExtractor workflow in the background.

    Args:
        channel_id: Slack channel ID
        thread_ts: Slack thread timestamp
        user_id: Slack user ID who triggered the shortcut
        org_settings_id: Organization settings ID
        slack_bot_token: Slack bot token for API calls
    """

    try:
        logger.info(
            f"Starting KnowledgeExtractor workflow task for {channel_id}/{thread_ts}"
        )

        # Get organization settings
        try:
            org_settings = OrganisationSettings.objects.get(
                organisation_id=org_settings_id
            )
        except OrganisationSettings.DoesNotExist:
            logger.error(f"Organization settings not found: {org_settings_id}")
            return {"success": False, "error": "Organization settings not found"}

        # Create Slack client
        slack_client = AsyncWebClient(token=slack_bot_token)

        # Import required classes
        from apps.workflows.knowledge_extractor.workflow import (
            KnowledgeExtractorWorkflow,
        )
        from apps.llm_providers.services import ChatService

        async def run_workflow():
            try:
                # Initialize LLM service
                llm_service = ChatService(org_settings=org_settings)
                llm_client = llm_service.get_client()

                if not llm_client:
                    error_msg = "LLM client not available. Check LLM provider settings."
                    logger.error(f"KnowledgeExtractor workflow: {error_msg}")
                    await slack_client.chat_postMessage(
                        channel=user_id,  # DM the user who invoked
                        text=f"Sorry <@{user_id}>, I can't learn from the thread right now because the Language Model service isn't configured correctly. Please contact an administrator.",
                    )
                    return {"success": False, "error": error_msg}

                # Get bot user ID for filtering
                bot_user_id_for_filtering = None
                try:
                    auth_test_response = await asyncio.wait_for(
                        slack_client.auth_test(), timeout=5.0
                    )
                    if auth_test_response.get("ok"):
                        bot_user_id_for_filtering = auth_test_response.get("user_id")
                        logger.info(
                            f"Fetched bot user ID for filtering: {bot_user_id_for_filtering}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to get bot user ID for filtering: {e}")

                # Fetch thread messages
                thread_replies_response = await slack_client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                )

                if not (
                    thread_replies_response.get("ok")
                    and thread_replies_response.get("messages")
                ):
                    error_msg = f"Failed to fetch thread messages: {thread_replies_response.get('error')}"
                    logger.error(error_msg)
                    await slack_client.chat_postMessage(
                        channel=user_id,  # DM the user
                        text=f"Sorry <@{user_id}>, I couldn't fetch the thread messages to learn from.",
                    )
                    return {"success": False, "error": error_msg}

                thread_messages = thread_replies_response["messages"]
                logger.info(
                    f"Fetched {len(thread_messages)} messages from thread {channel_id}/{thread_ts}"
                )

                # Run knowledge extraction workflow
                knowledge_workflow = KnowledgeExtractorWorkflow(
                    llm_client=llm_client,
                    bot_user_id_to_ignore=bot_user_id_for_filtering,
                )

                extracted_data = await knowledge_workflow.extract_learnings_from_thread(
                    thread_messages=thread_messages,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                )

                # Send success message to user
                await slack_client.chat_postMessage(
                    channel=user_id,  # DM the user
                    text=f"<@{user_id}> I've successfully analyzed the thread and extracted valuable insights! The learnings have been added to the knowledge base.",
                )

                return {"success": True, "extracted_data": extracted_data}

            except Exception as e:
                logger.error(
                    f"Error in KnowledgeExtractor workflow: {e}", exc_info=True
                )

                # Send error message to user
                try:
                    await slack_client.chat_postMessage(
                        channel=user_id,  # DM the user
                        text=f"Sorry <@{user_id}>, an unexpected error occurred while learning from the thread: {e}",
                    )
                except Exception as post_error:
                    logger.error(f"Failed to post error message to Slack: {post_error}")

                raise e

        # Execute the async workflow
        result = asyncio.run(run_workflow())

        logger.info(
            f"KnowledgeExtractor workflow completed successfully for {channel_id}/{thread_ts}"
        )

        return {
            "success": True,
            "result": result,
            "completed_at": timezone.now().isoformat(),
        }

    except Exception as exc:
        logger.exception(f"Error in KnowledgeExtractor workflow task: {exc}")

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying KnowledgeExtractor workflow task (attempt {self.request.retries + 1})"
            )
            raise self.retry(exc=exc)

        # Send error message to user after all retries exhausted
        try:
            asyncio.run(
                send_error_message_to_slack(
                    channel_id=user_id,  # Send as DM since this is what the original did
                    thread_ts=None,  # DM doesn't need thread_ts
                    user_id=user_id,
                    slack_bot_token=slack_bot_token,
                    error_message="I'm sorry, I encountered a persistent issue while learning from the thread. Please try again later or contact support.",
                )
            )
        except Exception as slack_error:
            logger.error(f"Failed to send error message to Slack: {slack_error}")

        return {"success": False, "error": str(exc), "retries_exhausted": True}
