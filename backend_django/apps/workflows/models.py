from django.db import models
from pgvector.django import VectorField
from apps.knowledge_base.models import Model
from apps.accounts.models import OrganisationScopedModelMixin


class Question(OrganisationScopedModelMixin, models.Model):
    question_text = models.TextField(null=False)
    answer_text = models.TextField(null=True, blank=True)
    question_embedding = VectorField(
        dimensions=3072,
        null=True,
        blank=True,
        help_text="Embedding vector for the question text",
    )
    was_useful = models.BooleanField(null=True, blank=True)
    feedback = models.TextField(null=True, blank=True)
    feedback_embedding = VectorField(
        dimensions=3072,
        null=True,
        blank=True,
        help_text="Embedding vector for the feedback text",
    )
    question_metadata = models.JSONField(null=True, blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    original_message_text = models.TextField(null=True, blank=True)
    original_message_ts = models.CharField(max_length=50, null=True, blank=True)
    response_message_ts = models.CharField(max_length=50, null=True, blank=True)
    original_message_embedding = VectorField(
        dimensions=3072,
        null=True,
        blank=True,
        help_text="Embedding vector for the original message text",
    )
    response_file_message_ts = models.CharField(
        max_length=50, null=True, blank=True, db_index=True
    )

    models_used = models.ManyToManyField(
        Model,
        through="QuestionModel",
        related_name="questions_used_in",
    )

    def __str__(self):
        return f"Q{self.id}: {self.question_text[:50]}..."

    class Meta:
        db_table = "questions"  # Match existing table name
        verbose_name = "Question"
        verbose_name_plural = "Questions"


class QuestionModel(models.Model):
    """Association table tracking which Models were used for each Question."""

    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    model = models.ForeignKey(Model, on_delete=models.CASCADE)
    relevance_score = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "question_models"  # Match existing table name
        unique_together = (("question", "model"),)
        verbose_name = "Question Model Usage"
        verbose_name_plural = "Question Model Usages"


# Conversation-related choices
class ConversationStatus(models.TextChoices):
    ACTIVE = "active", "Active"
    COMPLETED = "completed", "Completed"
    ERROR = "error", "Error"
    TIMEOUT = "timeout", "Timeout"


class ConversationTrigger(models.TextChoices):
    SLACK_MENTION = "slack_mention", "Slack Mention"
    WEB_INTERFACE = "web_interface", "Web Interface"
    MCP_SERVER = "mcp_server", "MCP Server"
    API_CALL = "api_call", "API Call"


class Conversation(OrganisationScopedModelMixin, models.Model):
    """
    Represents a complete conversation session with a user.
    Could be from Slack, web interface, MCP server, etc.
    """

    # Basic conversation info
    external_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="External ID (e.g., Slack thread ID, web session ID)",
    )
    channel = models.CharField(
        max_length=50,
        choices=[
            ("slack", "Slack"),
            ("web", "Web Interface"),
            ("mcp", "MCP Server"),
            ("api", "Direct API"),
        ],
        help_text="Channel where conversation originated",
    )
    user_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="User identifier from the external system",
    )

    # Status tracking
    status = models.CharField(
        max_length=20,
        choices=ConversationStatus.choices,
        default=ConversationStatus.ACTIVE,
    )

    # Trigger information
    trigger = models.CharField(
        max_length=50,
        choices=ConversationTrigger.choices,
        default=ConversationTrigger.SLACK_MENTION,
        help_text="What triggered this conversation",
    )

    # Conversation metadata
    title = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Auto-generated or user-provided conversation title",
    )
    summary = models.TextField(
        null=True, blank=True, help_text="AI-generated summary of the conversation"
    )

    # Initial question and response
    initial_question = models.TextField(
        help_text="The first question that started this conversation"
    )

    # Channel-specific fields
    channel_type = models.CharField(
        max_length=50, default="slack", help_text="Type of channel (slack, web, etc.)"
    )
    channel_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Channel ID where conversation took place",
    )
    user_external_id = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="External user ID from the source system",
    )

    # LLM and integration settings
    llm_provider = models.CharField(
        max_length=50,
        default="anthropic",
        help_text="LLM provider used for this conversation",
    )
    llm_chat_model = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        help_text="Specific LLM model used for this conversation",
    )
    enabled_integrations = models.JSONField(
        default=list, help_text="List of integrations enabled for this conversation"
    )

    # Performance metrics
    total_parts = models.PositiveIntegerField(default=0)
    total_tokens_used = models.PositiveIntegerField(default=0)
    total_cost = models.DecimalField(
        max_digits=10, decimal_places=4, default=0, help_text="Total cost in USD"
    )

    # Timing
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Feedback
    user_rating = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        choices=[(i, i) for i in range(1, 6)],
        help_text="User rating 1-5",
    )
    user_feedback = models.TextField(null=True, blank=True)

    # Additional context
    conversation_context = models.JSONField(
        default=dict, help_text="Additional conversation context and metadata"
    )

    class Meta:
        db_table = "conversations"
        indexes = [
            models.Index(fields=["external_id", "channel"]),
            models.Index(fields=["organisation", "started_at"]),
            models.Index(fields=["status"]),
            models.Index(fields=["user_id"]),
            models.Index(fields=["organisation", "external_id"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["organisation", "external_id"],
                condition=models.Q(external_id__isnull=False),
                name="unique_conversation_per_external_id",
            ),
        ]
        verbose_name = "Conversation"
        verbose_name_plural = "Conversations"

    def __str__(self):
        title = self.title or f"Conv {self.id}"
        return f"{title} ({self.channel})"

    @property
    def calculated_total_parts(self):
        """Calculate the actual number of conversation parts."""
        return self.parts.count()

    @property
    def calculated_visible_parts(self):
        """Calculate the number of user-facing conversation parts (excludes internal system messages)."""
        hidden_types = ["thinking", "llm_input", "tool_execution"]
        return self.parts.exclude(message_type__in=hidden_types).count()

    @property
    def calculated_total_tokens(self):
        """Calculate the total tokens used across all parts."""
        return self.parts.aggregate(total=models.Sum("tokens_used"))["total"] or 0

    @property
    def calculated_total_cost(self):
        """Calculate the total cost across all parts."""
        return self.parts.aggregate(total=models.Sum("cost"))["total"] or 0

    # --------------------------------------------------------------
    # Token breakdown helpers
    # --------------------------------------------------------------

    @property
    def calculated_input_tokens(self):
        """Total tokens used for messages sent *to* the LLM."""
        return (
            self.parts.filter(message_type="llm_input").aggregate(
                total=models.Sum("tokens_used")
            )["total"]
            or 0
        )

    @property
    def calculated_output_tokens(self):
        """Total tokens produced *by* the LLM."""
        return (
            self.parts.filter(message_type="llm_output").aggregate(
                total=models.Sum("tokens_used")
            )["total"]
            or 0
        )

    @property
    def calculated_thinking_tokens(self):
        """Tokens used for internal agent reasoning / thinking steps."""
        return (
            self.parts.filter(message_type="thinking").aggregate(
                total=models.Sum("tokens_used")
            )["total"]
            or 0
        )


class ConversationPart(models.Model):
    """
    Represents individual parts of a conversation:
    - User messages
    - Agent responses
    - Tool calls
    - Agent thinking/reasoning
    - Errors
    """

    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="parts"
    )

    # Ordering within conversation
    sequence_number = models.PositiveIntegerField(
        help_text="Order of this part within the conversation"
    )

    # Actor and message type
    actor = models.CharField(
        max_length=20,
        choices=[
            ("user", "User"),
            ("agent", "Agent"),
            ("system", "System"),
            ("llm", "LLM"),
            ("tool", "Tool"),
        ],
        help_text="Who/what generated this part",
    )

    message_type = models.CharField(
        max_length=30,
        choices=[
            ("message", "Message"),
            ("intent_classification", "Intent Classification"),
            ("llm_input", "LLM Input"),
            ("llm_output", "LLM Output"),
            ("tool_call", "Tool Call"),
            ("tool_execution", "Tool Execution"),
            ("tool_error", "Tool Error"),
            ("slack_output", "Slack Output"),
            ("slack_file_output", "Slack File Output"),
            ("workflow_completion", "Workflow Completion"),
            ("error", "Error"),
            ("thinking", "Agent Thinking"),
        ],
        help_text="Type of message/interaction",
    )

    # Legacy field for backward compatibility
    part_type = models.CharField(
        max_length=20,
        choices=[
            ("user_message", "User Message"),
            ("agent_response", "Agent Response"),
            ("tool_call", "Tool Call"),
            ("tool_result", "Tool Result"),
            ("agent_thinking", "Agent Thinking"),
            ("error", "Error"),
            ("system", "System Message"),
        ],
        null=True,
        blank=True,
        help_text="Legacy field - use actor and message_type instead",
    )

    # Content
    content = models.TextField(help_text="The actual content of this conversation part")

    # Tool-specific fields
    tool_name = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        help_text="Name of tool used (if part_type is tool_call/tool_result)",
    )
    tool_input = models.JSONField(
        null=True, blank=True, help_text="Input parameters for tool call"
    )
    tool_output = models.JSONField(
        null=True, blank=True, help_text="Output from tool execution"
    )

    # Optional short summary of tool result (helps avoid storing large payloads)
    result_summary = models.TextField(
        null=True,
        blank=True,
        help_text="Truncated or human-friendly summary of the tool output",
    )

    # Performance tracking
    tokens_used = models.PositiveIntegerField(
        default=0, help_text="Tokens used for this part"
    )
    cost = models.DecimalField(
        max_digits=8, decimal_places=4, default=0, help_text="Cost for this part in USD"
    )
    duration_ms = models.PositiveIntegerField(
        default=0, help_text="Duration in milliseconds"
    )

    # Metadata
    metadata = models.JSONField(
        default=dict, help_text="Additional metadata for this part"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "conversation_parts"
        indexes = [
            models.Index(fields=["conversation", "sequence_number"]),
            models.Index(fields=["actor"]),
            models.Index(fields=["message_type"]),
            models.Index(fields=["actor", "message_type"]),
            models.Index(fields=["part_type"]),  # Keep for backward compatibility
            models.Index(fields=["tool_name"]),
            models.Index(fields=["created_at"]),
        ]
        unique_together = [("conversation", "sequence_number")]
        ordering = ["sequence_number"]
        verbose_name = "Conversation Part"
        verbose_name_plural = "Conversation Parts"

    def __str__(self):
        return f"Conv {self.conversation.id} - Part {self.sequence_number} ({self.part_type})"


# Note: Removed Model and ModelEmbedding classes.
