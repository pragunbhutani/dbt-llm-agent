from django.db import models
from apps.accounts.models import Organisation
from .constants import get_integration_definition, IntegrationType


class OrganisationIntegration(models.Model):
    """
    Integration configuration for a specific organization.

    Instead of using a foreign key to an Integration model, we now reference
    integrations by their key which is defined in constants.py.
    """

    organisation = models.ForeignKey(
        Organisation, on_delete=models.CASCADE, related_name="integrations"
    )
    integration_key = models.CharField(
        max_length=50,
        default="unknown",  # Temporary default for migration
        help_text="Key of the integration as defined in constants.py",
    )
    is_enabled = models.BooleanField(default=False)
    configuration = models.JSONField(
        default=dict, help_text="Integration-specific configuration (non-sensitive)"
    )
    credentials = models.JSONField(
        default=dict, help_text="Encrypted credentials and sensitive data"
    )

    # Status tracking
    last_test_result = models.JSONField(
        default=dict, help_text="Last connection test result"
    )
    last_tested_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("organisation", "integration_key")
        ordering = ["integration_key"]

    def __str__(self):
        status = "enabled" if self.is_enabled else "disabled"
        integration_name = self.get_integration_definition().name
        return f"{self.organisation.name} - {integration_name} ({status})"

    def get_integration_definition(self):
        """Get the integration definition from constants."""
        try:
            return get_integration_definition(self.integration_key)
        except ValueError:
            # Handle case where integration key is no longer valid
            # This could happen if an integration is removed from constants
            return None

    @property
    def integration_name(self):
        """Get the name of the integration."""
        definition = self.get_integration_definition()
        return definition.name if definition else f"Unknown ({self.integration_key})"

    @property
    def integration_type(self):
        """Get the type of the integration."""
        definition = self.get_integration_definition()
        return definition.integration_type if definition else "unknown"

    @property
    def is_configured(self):
        """Check if integration has required configuration."""
        # This will be implemented by each integration class
        return bool(self.configuration or self.credentials)

    @property
    def connection_status(self):
        """Get connection status from last test."""
        if not self.last_test_result:
            return "unknown"
        return "connected" if self.last_test_result.get("success") else "error"
