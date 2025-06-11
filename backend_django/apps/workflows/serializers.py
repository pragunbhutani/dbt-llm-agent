from rest_framework import serializers

# Import from local models
from .models import Question, QuestionModel
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
