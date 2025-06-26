from rest_framework import serializers
from .models import OrganisationIntegration
from .constants import get_integration_definition, get_active_integration_definitions


class IntegrationSerializer(serializers.Serializer):
    """Serializer for available integrations (now from constants)."""

    key = serializers.CharField()
    name = serializers.CharField()
    description = serializers.CharField()
    integration_type = serializers.CharField()
    icon_url = serializers.URLField(allow_null=True)
    documentation_url = serializers.URLField(allow_null=True)
    configuration_schema = serializers.JSONField()
    is_active = serializers.BooleanField()


class OrganisationIntegrationSerializer(serializers.ModelSerializer):
    """Serializer for organization-specific integration configurations."""

    integration = serializers.SerializerMethodField(read_only=True)
    connection_status = serializers.CharField(read_only=True)
    is_configured = serializers.BooleanField(read_only=True)
    integration_name = serializers.CharField(read_only=True)
    integration_type = serializers.CharField(read_only=True)

    class Meta:
        model = OrganisationIntegration
        fields = [
            "id",
            "integration",
            "integration_key",
            "integration_name",
            "integration_type",
            "is_enabled",
            "configuration",
            "credentials",
            "connection_status",
            "is_configured",
            "last_test_result",
            "last_tested_at",
            "created_at",
            "updated_at",
        ]
        extra_kwargs = {
            "credentials": {
                "write_only": True
            },  # Don't expose credentials in responses
        }

    def get_integration(self, obj):
        """Get the integration definition from constants."""
        definition = obj.get_integration_definition()
        if definition:
            return IntegrationSerializer(definition.to_dict()).data
        return None

    def validate_integration_key(self, value):
        """Validate that the integration key exists in constants."""
        try:
            get_integration_definition(value)
            return value
        except ValueError:
            raise serializers.ValidationError(f"Unknown integration key: {value}")


class IntegrationStatusSerializer(serializers.Serializer):
    """Serializer for integration status information."""

    id = serializers.IntegerField(allow_null=True)
    key = serializers.CharField()
    name = serializers.CharField()
    integration_type = serializers.CharField()
    is_enabled = serializers.BooleanField()
    is_configured = serializers.BooleanField()
    connection_status = serializers.CharField()
    last_tested_at = serializers.DateTimeField(allow_null=True)
    tools_count = serializers.IntegerField()


class ConnectionTestSerializer(serializers.Serializer):
    """Serializer for connection test results."""

    success = serializers.BooleanField()
    message = serializers.CharField()
    tested_at = serializers.DateTimeField()


class IntegrationToolSerializer(serializers.Serializer):
    """Serializer for integration tools."""

    name = serializers.CharField()
    description = serializers.CharField()
    parameters = serializers.JSONField()
    integration_key = serializers.CharField()
    integration_name = serializers.CharField()
