from rest_framework import serializers
from .models import DbtProject


class DbtProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = DbtProject
        fields = "__all__"
        read_only_fields = ("organisation",)
