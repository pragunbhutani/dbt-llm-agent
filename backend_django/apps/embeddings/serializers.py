from rest_framework import serializers

# Import from local models
from .models import ModelEmbedding
from apps.accounts.serializers import OrganisationSerializer  # For nested Organisation


class ModelEmbeddingSerializer(serializers.ModelSerializer):
    organisation = OrganisationSerializer(read_only=True)
    # Embedding field is generated, so it should be read-only
    embedding = serializers.ListField(child=serializers.FloatField(), read_only=True)

    class Meta:
        model = ModelEmbedding
        fields = [
            "id",
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
            "created_at",
            "updated_at",
            "organisation",
            "embedding",
        ]
