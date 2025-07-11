"""
Services for workflow-related functionality.

This module contains utility classes and services that support
the workflow models and business logic.
"""

import logging
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime
from django.utils import timezone
from django.db.models import Max
from django.db.utils import IntegrityError

# Import models and services
from .models import Conversation, ConversationPart
from apps.knowledge_base.models import Model
from apps.accounts.models import OrganisationSettings

# Import workflow
from .model_interpreter.workflow import ModelInterpreterWorkflow

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


def get_slack_web_client() -> Optional[AsyncWebClient]:
    """
    Provides a singleton instance of the Slack AsyncWebClient.
    Uses the first available organization's Slack integration credentials.
    Returns None if slack_sdk is not installed or no Slack integration is configured.
    """
    global _slack_client_instance

    if AsyncWebClient is None:  # SDK was not imported successfully at module level
        logger.warning(
            "slack_sdk is not installed or AsyncWebClient could not be imported. Slack functionalities will be unavailable."
        )
        return None

    if _slack_client_instance is None:
        try:
            from apps.integrations.models import OrganisationIntegration

            # Get the first available Slack integration
            org_integration = OrganisationIntegration.objects.filter(
                integration_key="slack",
                is_enabled=True,
            ).first()

            if not org_integration:
                logger.error(
                    "No enabled Slack integration found. Slack client cannot be initialized."
                )
                return None

            slack_bot_token = org_integration.credentials.get("bot_token")
            if not slack_bot_token:
                logger.error(
                    "No bot token found in Slack integration. Slack client cannot be initialized."
                )
                return None

            # AsyncWebClient is already imported at module level if available
            _slack_client_instance = AsyncWebClient(token=slack_bot_token)
            logger.info(
                "Slack AsyncWebClient initialized singleton using integration credentials."
            )

        except Exception as e:
            logger.error(
                f"Error creating Slack client from integration credentials: {e}"
            )
            return None

    return _slack_client_instance


def trigger_model_interpretation(
    model: Model, org_settings: OrganisationSettings, verbosity: int = 0
) -> bool:
    """Triggers model interpretation using the simplified workflow.

    Args:
        model: The knowledge_base.Model instance to interpret.
        org_settings: The OrganisationSettings instance for the model.
        verbosity: Verbosity level for logging.

    Returns:
        True if interpretation was successful and saved, False otherwise.
    """
    try:
        # Initialize workflow
        workflow = ModelInterpreterWorkflow(
            org_settings=org_settings, verbosity=verbosity
        )

        # Run interpretation
        return workflow.interpret_model(model)

    except Exception as init_err:
        logger.error(
            f"Failed to initialize ModelInterpreterWorkflow for {model.name}: {init_err}",
            exc_info=True,
        )
        return False


class ConversationLogger:
    """
    Service for logging conversation parts and managing conversation state.
    Handles proper sequence numbering and deduplication of conversation parts.
    """

    def __init__(self, conversation: Conversation):
        self.conversation = conversation
        # Remove the in-memory sequence counter to prevent race conditions
        self.logger = logging.getLogger(f"{__name__}.{conversation.id}")

    def _get_next_sequence_number(self) -> int:
        """Get the next sequence number for this conversation from the database."""
        # Use database aggregation to get the next sequence number atomically
        max_sequence = ConversationPart.objects.filter(
            conversation=self.conversation
        ).aggregate(max_seq=Max("sequence_number"))["max_seq"]

        return (max_sequence or 0) + 1

    def log_user_message(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationPart:
        """Log a user message."""
        return self._create_part(
            actor="user",
            message_type="message",
            part_type="user_message",  # Legacy field
            content=content,
            metadata=metadata or {},
        )

    def log_agent_response(
        self,
        content: str,
        tokens_used: int = 0,
        cost: Decimal = Decimal("0"),
        duration_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationPart:
        """Log an agent response with performance metrics."""
        # Determine actor and message_type from metadata
        actor = metadata.get("type", "agent") if metadata else "agent"
        message_type = metadata.get("type", "message") if metadata else "message"

        # Map specific types to proper actor/message_type
        if message_type == "llm_input":
            actor = "agent"
        elif message_type == "llm_output":
            actor = "llm"
        elif message_type in ["slack_output", "slack_file_output"]:
            actor = "agent"
        elif message_type in ["tool_execution", "tool_error"]:
            actor = "tool"
        elif message_type == "intent_classification":
            actor = "agent"
        elif message_type == "workflow_completion":
            actor = "system"

        return self._create_part(
            actor=actor,
            message_type=message_type,
            part_type="agent_response",  # Legacy field
            content=content,
            tokens_used=tokens_used,
            cost=cost,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationPart:
        """Log a tool call."""
        content = content or f"Calling tool: {tool_name}"
        return self._create_part(
            actor="tool",
            message_type="tool_call",
            part_type="tool_call",  # Legacy field
            content=content,
            tool_name=tool_name,
            tool_input=tool_input,
            metadata=metadata or {},
        )

    def log_tool_result(
        self,
        tool_name: str,
        tool_output: Dict[str, Any],
        content: Optional[str] = None,
        duration_ms: int = 0,
        result_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationPart:
        """Log a tool execution result."""
        content = content or f"Tool {tool_name} completed"
        return self._create_part(
            actor="tool",
            message_type="tool_execution",
            part_type="tool_result",  # Legacy field
            content=content,
            tool_name=tool_name,
            tool_output=tool_output,
            duration_ms=duration_ms,
            metadata=metadata or {},
            result_summary=result_summary,
        )

    def log_agent_thinking(
        self,
        content: str,
        tokens_used: int = 0,
        cost: Decimal = Decimal("0"),
        duration_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationPart:
        """Log agent thinking/reasoning steps."""
        return self._create_part(
            actor="agent",
            message_type="thinking",
            part_type="agent_thinking",  # Legacy field
            content=content,
            tokens_used=tokens_used,
            cost=cost,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

    def log_error(
        self,
        content: str,
        error_details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationPart:
        """Log an error that occurred during the conversation."""
        combined_metadata = metadata or {}
        if error_details:
            combined_metadata["error_details"] = error_details

        return self._create_part(
            actor="system",
            message_type="error",
            part_type="error",  # Legacy field
            content=content,
            metadata=combined_metadata,
        )

    def log_system_message(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationPart:
        """Log a system message."""
        return self._create_part(
            actor="system",
            message_type="message",
            part_type="system",  # Legacy field
            content=content,
            metadata=metadata or {},
        )

    def _create_part(
        self,
        actor: str,
        message_type: str,
        content: str,
        part_type: Optional[str] = None,  # Legacy field
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        tool_output: Optional[Dict[str, Any]] = None,
        result_summary: Optional[str] = None,
        tokens_used: int = 0,
        cost: Decimal = Decimal("0"),
        duration_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationPart:
        """Create a conversation part with the given parameters."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                part = ConversationPart.objects.create(
                    conversation=self.conversation,
                    sequence_number=self._get_next_sequence_number(),
                    actor=actor,
                    message_type=message_type,
                    part_type=part_type,  # Legacy field for backward compatibility
                    content=content,
                    tool_name=tool_name,
                    tool_input=tool_input or {},
                    tool_output=tool_output or {},
                    result_summary=result_summary,
                    tokens_used=tokens_used,
                    cost=cost,
                    duration_ms=duration_ms,
                    metadata=metadata or {},
                )

                # Update conversation totals
                self._update_conversation_totals(tokens_used, cost)

                return part

            except IntegrityError as e:
                if (
                    "conversation_parts_conversation_id_sequence" in str(e)
                    and attempt < max_retries - 1
                ):
                    # Handle duplicate sequence number - retry with new sequence number
                    self.logger.warning(
                        f"Sequence number conflict, retrying... (attempt {attempt + 1})"
                    )
                    continue
                else:
                    self.logger.error(
                        f"Failed to create conversation part after {attempt + 1} attempts: {e}"
                    )
                    raise
            except Exception as e:
                self.logger.error(f"Failed to create conversation part: {e}")
                raise

        # This should never be reached due to the raise statements above
        raise Exception("Failed to create conversation part after maximum retries")

    def _update_conversation_totals(self, tokens_used: int, cost: Decimal):
        """Update conversation-level totals."""
        try:
            self.conversation.total_parts += 1
            self.conversation.total_tokens_used += tokens_used
            self.conversation.total_cost += cost
            self.conversation.save(
                update_fields=["total_parts", "total_tokens_used", "total_cost"]
            )
        except Exception as e:
            self.logger.error(f"Failed to update conversation totals: {e}")

    def set_conversation_status(
        self, status: str, completed_at: Optional[datetime] = None
    ):
        """Update the conversation status."""
        try:
            self.conversation.status = status
            if completed_at:
                self.conversation.completed_at = completed_at
            elif status in ["completed", "error", "timeout"]:
                self.conversation.completed_at = timezone.now()

            self.conversation.save(update_fields=["status", "completed_at"])

        except Exception as e:
            self.logger.error(f"Failed to update conversation status: {e}")

    def add_user_feedback(
        self, rating: Optional[int] = None, feedback: Optional[str] = None
    ):
        """Add user feedback to the conversation."""
        try:
            if rating is not None:
                self.conversation.user_rating = rating
            if feedback is not None:
                self.conversation.user_feedback = feedback

            self.conversation.save(update_fields=["user_rating", "user_feedback"])

        except Exception as e:
            self.logger.error(f"Failed to add user feedback: {e}")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        parts = self.conversation.parts.all()

        summary = {
            "id": self.conversation.id,
            "status": self.conversation.status,
            "started_at": self.conversation.started_at,
            "completed_at": self.conversation.completed_at,
            "total_parts": len(parts),
            "total_tokens_used": self.conversation.total_tokens_used,
            "total_cost": float(self.conversation.total_cost),
            "user_rating": self.conversation.user_rating,
            "parts_by_type": {},
        }

        # Count parts by type
        for part in parts:
            part_type = part.part_type
            if part_type not in summary["parts_by_type"]:
                summary["parts_by_type"][part_type] = 0
            summary["parts_by_type"][part_type] += 1

        return summary

    def get_conversation_timeline(self) -> List[Dict[str, Any]]:
        """Get a timeline view of the conversation."""
        parts = self.conversation.parts.all().order_by("sequence_number")

        timeline = []
        for part in parts:
            timeline_item = {
                "sequence_number": part.sequence_number,
                "part_type": part.part_type,
                "content": part.content[:200]
                + ("..." if len(part.content) > 200 else ""),
                "created_at": part.created_at,
                "tokens_used": part.tokens_used,
                "cost": float(part.cost),
                "duration_ms": part.duration_ms,
            }

            if part.tool_name:
                timeline_item["tool_name"] = part.tool_name

            timeline.append(timeline_item)

        return timeline


class ConversationAnalytics:
    """Service for analyzing conversation patterns and performance."""

    @staticmethod
    def get_conversation_metrics(
        organisation_id: int, days: int = 30
    ) -> Dict[str, Any]:
        """Get conversation metrics for an organization."""
        from django.utils import timezone
        from datetime import timedelta

        cutoff_date = timezone.now() - timedelta(days=days)

        conversations = Conversation.objects.filter(
            organisation_id=organisation_id, started_at__gte=cutoff_date
        )

        total_conversations = conversations.count()
        completed_conversations = conversations.filter(status="completed").count()

        total_tokens = sum(c.total_tokens_used for c in conversations)
        total_cost = sum(c.total_cost for c in conversations)

        avg_rating = None
        rated_conversations = conversations.filter(user_rating__isnull=False)
        if rated_conversations.exists():
            avg_rating = (
                sum(c.user_rating for c in rated_conversations)
                / rated_conversations.count()
            )

        return {
            "total_conversations": total_conversations,
            "completed_conversations": completed_conversations,
            "completion_rate": (
                (completed_conversations / total_conversations * 100)
                if total_conversations > 0
                else 0
            ),
            "total_tokens_used": total_tokens,
            "total_cost": float(total_cost),
            "average_tokens_per_conversation": (
                total_tokens / total_conversations if total_conversations > 0 else 0
            ),
            "average_cost_per_conversation": (
                float(total_cost / total_conversations)
                if total_conversations > 0
                else 0
            ),
            "average_user_rating": avg_rating,
            "period_days": days,
        }
