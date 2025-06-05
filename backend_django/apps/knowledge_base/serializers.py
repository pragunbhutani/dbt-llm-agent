from rest_framework import serializers

# Import from local models
from .models import Model


class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Model
        fields = "__all__"  # Start by including all fields
