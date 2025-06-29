from django.db import models
from apps.accounts.models import Organisation
from .constants import get_integration_definition, IntegrationType
from typing import Dict, Any, Optional
import json
from django.conf import settings


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

    # Single path to the credentials JSON object in Parameter Store
    credentials_path = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Path to credentials JSON object in Parameter Store, e.g., '/ragstar/{environment}/org-{org_id}/integrations/{integration_key}/credentials'",
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
        return bool(self.configuration or self.credentials_path)

    @property
    def connection_status(self):
        """Get connection status from last test."""
        if not self.last_test_result:
            return "unknown"
        return "connected" if self.last_test_result.get("success") else "error"

    def _get_secret_manager(self):
        """Get the secret manager instance."""
        from apps.accounts.services import secret_manager

        return secret_manager

    def _get_credentials_parameter_path(self) -> str:
        """Generate parameter path for credentials JSON."""
        return f"/ragstar/{settings.ENVIRONMENT}/org-{self.organisation.id}/integrations/{self.integration_key}/credentials"

    @property
    def credentials(self) -> Dict[str, Any]:
        """Get all credentials for this integration from Parameter Store."""
        if not self.credentials_path:
            return {}

        secret_manager = self._get_secret_manager()
        credentials_json = secret_manager.get_secret(self.credentials_path)

        if not credentials_json:
            return {}

        try:
            return json.loads(credentials_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    def set_credentials(self, credentials_dict: Dict[str, Any]) -> bool:
        """Store credentials dictionary as JSON in Parameter Store."""
        if not credentials_dict:
            return False

        parameter_path = self._get_credentials_parameter_path()
        secret_manager = self._get_secret_manager()

        description = (
            f"{self.integration_key.title()} credentials for {self.organisation.name}"
        )
        credentials_json = json.dumps(credentials_dict)

        if secret_manager.put_secret(
            parameter_path, credentials_json, description=description
        ):
            self.credentials_path = parameter_path
            self.save(update_fields=["credentials_path"])
            return True
        return False

    def update_credentials(self, updates: Dict[str, Any]) -> bool:
        """Update specific credentials without replacing the entire object."""
        current_credentials = self.credentials
        current_credentials.update(updates)
        return self.set_credentials(current_credentials)

    def get_credential(self, credential_name: str) -> Optional[Any]:
        """Get a specific credential value."""
        return self.credentials.get(credential_name)

    def delete_credentials(self) -> bool:
        """Delete credentials from Parameter Store."""
        if not self.credentials_path:
            return True

        secret_manager = self._get_secret_manager()
        success = secret_manager.delete_secret(self.credentials_path)

        if success:
            self.credentials_path = None
            self.save(update_fields=["credentials_path"])

        return success
