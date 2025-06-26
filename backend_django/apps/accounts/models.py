from django.db import models
import uuid
from django.contrib.auth.models import AbstractUser
from django.conf import settings

# Create your models here.
# All models previously in this file have been moved to their respective apps:
# - Model -> apps.knowledge_base.models
# - Question, QuestionModel -> apps.agents.models
# - ModelEmbedding -> apps.embeddings.models

# If core app needs its own models later (e.g., UserProfile, Settings),
# define them here.


class Organisation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # Use settings.AUTH_USER_MODEL for ForeignKey to User
        related_name="owned_organisations",
        on_delete=models.SET_NULL,  # Or models.PROTECT, depending on desired behavior
        null=True,  # Owner can be set later or might not exist for system orgs
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class User(AbstractUser):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    organisation = models.ForeignKey(
        Organisation,
        related_name="users",
        on_delete=models.PROTECT,  # Prevent deleting organisation if users are still part of it
        null=True,  # Allow users (like superadmin) not to belong to an organisation
        blank=True,
    )

    # Use email as the primary identifier
    email = models.EmailField(unique=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = [
        "username"
    ]  # username is still required for superuser creation etc. but not for login.

    def __str__(self):
        return self.email


# --- Multi-tenancy Scoping Utilities ---


class OrganisationScopedManager(models.Manager):
    def get_queryset(self):
        # This manager is intended to be used in conjunction with view-level filtering
        # or by explicitly calling for_organisation.
        return super().get_queryset()

    def for_organisation(self, organisation):
        if not organisation:
            # Depending on policy, could raise an error or return none().
            # Returning none() is safer for general use to prevent accidental data exposure.
            return self.none()
        return self.get_queryset().filter(organisation=organisation)


class OrganisationScopedModelMixin(models.Model):
    organisation = models.ForeignKey(
        Organisation,
        on_delete=models.CASCADE,
        related_name="%(app_label)s_%(class)s_org_related",
        null=True,  # Keeping it nullable for now
        blank=True,  # Keeping it nullable for now
    )

    objects = OrganisationScopedManager()  # Default manager instance

    class Meta:
        abstract = True


class OrganisationSettings(models.Model):
    """
    Holds all configuration settings for a specific organisation.
    """

    organisation = models.OneToOneField(
        Organisation,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="settings",
    )

    # LLM Provider Settings
    llm_chat_provider = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="e.g., 'openai', 'google', 'anthropic'",
    )
    llm_chat_model = models.CharField(
        max_length=100, blank=True, null=True, help_text="e.g., 'gpt-4', 'gemini-pro'"
    )
    llm_embeddings_provider = models.CharField(
        max_length=50, blank=True, null=True, help_text="e.g., 'openai', 'google'"
    )
    llm_embeddings_model = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="e.g., 'text-embedding-ada-002'",
    )

    # LLM API Keys
    # TODO: These fields should be encrypted.
    llm_openai_api_key = models.TextField(blank=True, null=True)
    llm_google_api_key = models.TextField(blank=True, null=True)
    llm_anthropic_api_key = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Settings for {self.organisation.name}"


# --- End of file --- Ensure everything below this line is removed ---
