from rest_framework import serializers

# Import from local models
from .models import ModelEmbedding
from apps.accounts.serializers import OrganisationSerializer  # For nested Organisation
from apps.knowledge_base.serializers import ModelSerializer


class ModelEmbeddingSerializer(serializers.ModelSerializer):
    organisation = OrganisationSerializer(read_only=True)
    model = ModelSerializer(read_only=True)
    model_name = serializers.SerializerMethodField()
    # Embedding field is generated, so it should be read-only
    embedding = serializers.ListField(child=serializers.FloatField(), read_only=True)

    def get_model_name(self, obj):
        """Get the model name from the related model."""
        return obj.model.name if obj.model else None

    class Meta:
        model = ModelEmbedding
        fields = [
            "id",
            "model",
            "model_name",
            "document",
            "embedding",
            "model_metadata",
            "can_be_used_for_answers",
            "created_at",
            "updated_at",
            "organisation",
        ]
        read_only_fields = [
            "id",
            "model_name",
            "created_at",
            "updated_at",
            "organisation",
            "embedding",
        ]
