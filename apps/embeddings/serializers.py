from rest_framework import serializers

# Import from local models
from .models import ModelEmbedding


class ModelEmbeddingSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelEmbedding
        fields = "__all__"
