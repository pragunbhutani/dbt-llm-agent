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
    dbt_cloud_api_key = models.TextField(blank=True, null=True)  # TODO: Encrypt this

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name
