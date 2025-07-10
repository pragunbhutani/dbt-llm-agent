from rest_framework import serializers

# Import from local models
from .models import Question, QuestionModel, Conversation, ConversationPart
from apps.accounts.serializers import OrganisationSerializer
from apps.knowledge_base.serializers import ModelSerializer


class QuestionModelSerializer(serializers.ModelSerializer):
    # Optionally, to show model details instead of just ID in QuestionModel through table
    model = ModelSerializer(read_only=True)

    class Meta:
        model = QuestionModel
        fields = ["model", "relevance_score"]


class QuestionSerializer(serializers.ModelSerializer):
    organisation = OrganisationSerializer(read_only=True)
    # models_used = serializers.PrimaryKeyRelatedField(many=True, read_only=True) # Original
    # To provide more detail for models_used through QuestionModel:
    models_used_details = QuestionModelSerializer(
        source="questionmodel_set", many=True, read_only=True
    )

    class Meta:
        model = Question
        fields = [
            "id",
            "question_text",
            "answer_text",
            "question_embedding",
            "was_useful",
            "feedback",
            "feedback_embedding",
            "question_metadata",
            "created_at",
            "updated_at",
            "original_message_text",
            "original_message_ts",
            "response_message_ts",
            "original_message_embedding",
            "response_file_message_ts",
            "models_used",  # Keep this for writing if you use it, or remove if models_used_details is primary way
            "models_used_details",
            "organisation",
        ]
        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "organisation",
            "question_embedding",
            "feedback_embedding",
            "original_message_embedding",
        ]
        # models_used is often handled via the through model or specific endpoints


class ConversationPartSerializer(serializers.ModelSerializer):
    """Serializer for individual conversation parts."""

    class Meta:
        model = ConversationPart
        fields = [
            "id",
            "sequence_number",
            "actor",
            "message_type",
            "content",
            "tool_name",
            "tool_input",
            "tool_output",
            "tokens_used",
            "cost",
            "duration_ms",
            "metadata",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class ConversationSerializer(serializers.ModelSerializer):
    """Serializer for conversations."""

    organisation = OrganisationSerializer(read_only=True)
    parts = ConversationPartSerializer(many=True, read_only=True)
    total_parts = serializers.ReadOnlyField(source="calculated_total_parts")
    total_tokens_used = serializers.ReadOnlyField(source="calculated_total_tokens")
    total_cost = serializers.ReadOnlyField(source="calculated_total_cost")

    # Detailed token metrics
    input_tokens = serializers.ReadOnlyField(source="calculated_input_tokens")
    output_tokens = serializers.ReadOnlyField(source="calculated_output_tokens")
    thinking_tokens = serializers.ReadOnlyField(source="calculated_thinking_tokens")

    class Meta:
        model = Conversation
        fields = [
            "id",
            "external_id",
            "channel",
            "user_id",
            "status",
            "trigger",
            "title",
            "summary",
            "initial_question",
            "channel_type",
            "channel_id",
            "user_external_id",
            "llm_provider",
            "llm_chat_model",
            "enabled_integrations",
            "total_parts",
            "total_tokens_used",
            "total_cost",
            "started_at",
            "completed_at",
            "user_rating",
            "user_feedback",
            "conversation_context",
            "organisation",
            "parts",
            # Added detailed token fields for analytics
            "input_tokens",
            "output_tokens",
            "thinking_tokens",
        ]
        read_only_fields = [
            "id",
            "started_at",
            "organisation",
            "total_parts",
            "total_tokens_used",
            "total_cost",
            "input_tokens",
            "output_tokens",
            "thinking_tokens",
        ]


class ConversationListSerializer(serializers.ModelSerializer):
    """Lighter serializer for conversation listings without parts."""

    organisation = OrganisationSerializer(read_only=True)
    total_parts = serializers.ReadOnlyField(source="calculated_visible_parts")
    total_tokens_used = serializers.ReadOnlyField(source="calculated_total_tokens")
    total_cost = serializers.ReadOnlyField(source="calculated_total_cost")

    class Meta:
        model = Conversation
        fields = [
            "id",
            "channel",
            "user_id",
            "user_external_id",
            "status",
            "trigger",
            "title",
            "initial_question",
            "total_parts",
            "total_tokens_used",
            "total_cost",
            "started_at",
            "completed_at",
            "user_rating",
            "organisation",
        ]
        read_only_fields = [
            "id",
            "started_at",
            "organisation",
            "total_parts",
            "total_tokens_used",
            "total_cost",
            "user_external_id",
        ]
