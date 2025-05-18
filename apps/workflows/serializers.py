from rest_framework import serializers

# Import from local models
from .models import Question


class QuestionSerializer(serializers.ModelSerializer):
    # Make models_used read-only for now, as it's managed via QuestionModel
    # or could be handled by nested serializers later if needed.
    models_used = serializers.PrimaryKeyRelatedField(many=True, read_only=True)

    class Meta:
        model = Question
        fields = "__all__"
