from django.db import models
from apps.accounts.models import Organisation
from .constants import get_integration_definition, IntegrationType
from typing import Dict, Any, Optional
import json
from django.conf import settings
import secrets
from datetime import datetime, timedelta, timezone


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


class MCPOAuthClient(models.Model):
    """
    OAuth client registrations for MCP server.
    Replaces the in-memory client_storage from the working FastMCP implementation.
    """

    client_id = models.CharField(max_length=100, unique=True, primary_key=True)
    client_secret = models.CharField(max_length=100)
    client_name = models.CharField(max_length=200)
    redirect_uris = models.JSONField(
        default=list, help_text="List of allowed redirect URIs"
    )
    grant_types = models.JSONField(
        default=list, help_text="List of allowed grant types"
    )
    response_types = models.JSONField(
        default=list, help_text="List of allowed response types"
    )
    scope = models.CharField(
        max_length=500,
        default="mcp:tools mcp:resources mcp:prompts",
        help_text="OAuth scopes",
    )
    auto_registered = models.BooleanField(
        default=False, help_text="Whether this client was auto-registered"
    )

    # Link to organization for auto-registered clients
    organisation = models.ForeignKey(
        Organisation,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        help_text="Organization that this client is linked to (for auto-registered clients)",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "MCP OAuth Client"
        verbose_name_plural = "MCP OAuth Clients"

    def __str__(self):
        return f"{self.client_name} ({self.client_id})"

    @classmethod
    def generate_client_credentials(cls) -> Dict[str, str]:
        """Generate client ID and secret for dynamic client registration."""
        client_id = f"mcp_client_{secrets.token_urlsafe(16)}"
        client_secret = secrets.token_urlsafe(32)
        return {"client_id": client_id, "client_secret": client_secret}

    @classmethod
    def create_client(
        cls,
        client_name: str,
        redirect_uris: list,
        grant_types: list = None,
        response_types: list = None,
        scope: str = None,
        auto_registered: bool = False,
        organisation: Organisation = None,
    ) -> "MCPOAuthClient":
        """Create a new OAuth client with generated credentials."""
        credentials = cls.generate_client_credentials()

        return cls.objects.create(
            client_id=credentials["client_id"],
            client_secret=credentials["client_secret"],
            client_name=client_name,
            redirect_uris=redirect_uris,
            grant_types=grant_types or ["authorization_code"],
            response_types=response_types or ["code"],
            scope=scope or "mcp:tools mcp:resources mcp:prompts",
            auto_registered=auto_registered,
            organisation=organisation,
        )

    def is_redirect_uri_valid(self, redirect_uri: str) -> bool:
        """Check if the redirect URI is valid for this client."""
        return redirect_uri in self.redirect_uris


class MCPOAuthAuthorizationCode(models.Model):
    """
    Authorization codes for OAuth flow.
    Replaces the in-memory auth_code_storage from the working FastMCP implementation.
    """

    code = models.CharField(max_length=100, unique=True, primary_key=True)
    client = models.ForeignKey(
        MCPOAuthClient, on_delete=models.CASCADE, related_name="authorization_codes"
    )
    user_data = models.JSONField(help_text="User data for token generation")
    scopes = models.JSONField(default=list, help_text="Requested scopes")
    code_challenge = models.CharField(max_length=200, help_text="PKCE code challenge")
    redirect_uri = models.CharField(max_length=500, help_text="Redirect URI")
    expires_at = models.DateTimeField(help_text="When this code expires")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "MCP OAuth Authorization Code"
        verbose_name_plural = "MCP OAuth Authorization Codes"

    def __str__(self):
        return f"Auth Code for {self.client.client_name} ({self.code[:10]}...)"

    @classmethod
    def create_code(
        cls,
        client: MCPOAuthClient,
        user_data: Dict[str, Any],
        scopes: list,
        code_challenge: str,
        redirect_uri: str,
    ) -> "MCPOAuthAuthorizationCode":
        """Create a new authorization code."""
        code = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

        return cls.objects.create(
            code=code,
            client=client,
            user_data=user_data,
            scopes=scopes,
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
            expires_at=expires_at,
        )

    @property
    def is_expired(self) -> bool:
        """Check if this authorization code has expired."""
        return self.expires_at < datetime.now(timezone.utc)

    def delete_expired(self):
        """Delete this code if it's expired."""
        if self.is_expired:
            self.delete()


class MCPOAuthAuthorizationRequest(models.Model):
    """
    Authorization requests during OAuth flow.
    Replaces the in-memory auth_request_storage from the working FastMCP implementation.
    """

    request_id = models.CharField(max_length=100, unique=True, primary_key=True)
    client_id = models.CharField(max_length=100, help_text="OAuth client ID")
    redirect_uri = models.CharField(max_length=500, help_text="Redirect URI")
    response_type = models.CharField(max_length=20, help_text="OAuth response type")
    scope = models.CharField(max_length=500, help_text="Requested scopes")
    code_challenge = models.CharField(max_length=200, help_text="PKCE code challenge")
    code_challenge_method = models.CharField(
        max_length=10, default="S256", help_text="PKCE code challenge method"
    )
    state = models.CharField(
        max_length=500, null=True, blank=True, help_text="OAuth state parameter"
    )
    expires_at = models.DateTimeField(help_text="When this request expires")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "MCP OAuth Authorization Request"
        verbose_name_plural = "MCP OAuth Authorization Requests"

    def __str__(self):
        return f"Auth Request {self.request_id[:10]}... for {self.client_id}"

    @classmethod
    def create_request(
        cls,
        client_id: str,
        redirect_uri: str,
        response_type: str,
        scope: str,
        code_challenge: str,
        code_challenge_method: str = "S256",
        state: str = None,
    ) -> "MCPOAuthAuthorizationRequest":
        """Create a new authorization request."""
        request_id = secrets.token_urlsafe(16)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)

        return cls.objects.create(
            request_id=request_id,
            client_id=client_id,
            redirect_uri=redirect_uri,
            response_type=response_type,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            state=state,
            expires_at=expires_at,
        )

    @property
    def is_expired(self) -> bool:
        """Check if this authorization request has expired."""
        return self.expires_at < datetime.now(timezone.utc)

    def delete_expired(self):
        """Delete this request if it's expired."""
        if self.is_expired:
            self.delete()
