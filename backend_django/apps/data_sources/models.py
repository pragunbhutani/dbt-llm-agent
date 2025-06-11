from django.db import models

# TODO: Replace TextField with a secure EncryptedTextField once a compatible
# library for Django 5.2 is found.
# from django_cryptography.fields import EncryptedTextField

from apps.accounts.models import OrganisationScopedModelMixin

# Create your models here.


class DbtProject(OrganisationScopedModelMixin, models.Model):
    """
    Represents a dbt project, which can be either a dbt Cloud project or a
    dbt Core project connected via a Git repository.
    """

    class ProjectType(models.TextChoices):
        DBT_CLOUD = "DBT_CLOUD", "dbt Cloud"
        DBT_CORE = "DBT_CORE", "dbt Core"

    name = models.CharField(max_length=255)
    project_type = models.CharField(
        max_length=10,
        choices=ProjectType.choices,
        default=ProjectType.DBT_CLOUD,
    )

    # dbt Cloud specific fields
    # TODO: Encrypt this field
    dbt_cloud_api_key = models.TextField(blank=True, null=True)
    dbt_cloud_project_id = models.CharField(max_length=255, blank=True, null=True)
    dbt_cloud_job_id = models.CharField(max_length=255, blank=True, null=True)

    # dbt Core specific fields
    git_repository_url = models.CharField(max_length=255, blank=True, null=True)
    # TODO: Encrypt this field
    git_ssh_key = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name

    def clean(self):
        super().clean()
        if self.project_type == self.ProjectType.DBT_CLOUD:
            if not all([self.dbt_cloud_api_key, self.dbt_cloud_project_id]):
                raise models.ValidationError(
                    "For dbt Cloud projects, API key and project ID are required."
                )
            self.git_repository_url = None
            self.git_ssh_key = None
        elif self.project_type == self.ProjectType.DBT_CORE:
            if not self.git_repository_url:
                raise models.ValidationError(
                    "For dbt Core projects, a Git repository URL is required."
                )
            self.dbt_cloud_api_key = None
            self.dbt_cloud_project_id = None
            self.dbt_cloud_job_id = None
