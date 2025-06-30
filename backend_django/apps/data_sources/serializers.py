from rest_framework import serializers
from .models import DbtProject


class DbtProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = DbtProject
        exclude = ("credentials_path",)


class DbtCloudProjectCreateSerializer(serializers.Serializer):
    dbt_cloud_url = serializers.URLField()
    dbt_cloud_account_id = serializers.IntegerField()
    dbt_cloud_api_key = serializers.CharField()
    name = serializers.CharField(required=False, allow_blank=True)

    def create(self, validated_data):
        # This serializer is for input validation, not for creating an object directly.
        # The service will handle the logic.
        return validated_data
