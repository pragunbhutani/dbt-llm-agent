from django.db import models
from django.utils.translation import gettext_lazy as _
from django.conf import settings

# TODO: Replace TextField with a secure EncryptedTextField once a compatible
# library for Django 5.2 is found.
# from django_cryptography.fields import EncryptedTextField

from apps.accounts.models import OrganisationScopedModelMixin

# Create your models here.


class DbtProject(OrganisationScopedModelMixin, models.Model):
    class ConnectionType(models.TextChoices):
        DBT_CLOUD = "DBT_CLOUD", _("dbt Cloud")
        GITHUB = "GITHUB", _("GitHub")

    name = models.CharField(max_length=255)
    connection_type = models.CharField(
        max_length=20, choices=ConnectionType.choices, default=ConnectionType.DBT_CLOUD
    )
    dbt_cloud_url = models.URLField(blank=True, null=True)
    dbt_cloud_account_id = models.BigIntegerField(blank=True, null=True)

    # GitHub connection settings
    github_repository_url = models.URLField(blank=True, null=True)
    github_branch = models.CharField(max_length=255, blank=True, null=True)
    github_project_folder = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Path to the folder containing dbt_project.yml within the repository.",
    )

    # Path in AWS SSM Parameter Store where a JSON blob containing
    # sensitive credentials (e.g. {"dbt_cloud_api_key": "..."}) is stored.
    credentials_path = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Path to credentials JSON in Parameter Store, e.g. '/ragstar/{env}/org-{org_id}/dbt-projects/{project_id}/credentials'",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    # --- Secrets handling helpers ---

    def _get_secret_manager(self):
        from apps.accounts.services import secret_manager

        return secret_manager

    def _get_credentials_parameter_path(self) -> str:
        """Generate parameter path for this project's credentials."""
        from django.conf import settings

        return f"/ragstar/{settings.ENVIRONMENT}/org-{self.organisation.id}/dbt-projects/{self.id}/credentials"

    @property
    def credentials(self) -> dict:
        """Return credentials dict from Parameter Store (or empty)."""
        if not self.credentials_path:
            return {}

        secret_manager = self._get_secret_manager()
        import json

        raw = secret_manager.get_secret(self.credentials_path)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def set_credentials(self, credentials: dict) -> bool:
        """Store credentials in Parameter Store and persist path."""
        if not credentials:
            return False

        import json

        description = f"dbt Cloud credentials for project '{self.name}' (ID {self.id})"
        parameter_path = self._get_credentials_parameter_path()
        secret_manager = self._get_secret_manager()

        ok = secret_manager.put_secret(
            parameter_path, json.dumps(credentials), description=description
        )
        if ok:
            self.credentials_path = parameter_path
            # Avoid recursion by updating direct fields only
            self.save(update_fields=["credentials_path"])
        return ok

    def get_credential(self, key: str, default=None):
        return self.credentials.get(key, default)

    # Backwards-compatibility shim so existing code can still access project.dbt_cloud_api_key
    @property
    def dbt_cloud_api_key(self):
        return self.get_credential("dbt_cloud_api_key")
