from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Organisation, OrganisationSettings


@receiver(post_save, sender=Organisation)
def create_default_organisation_settings(
    sender, instance: Organisation, created: bool, **kwargs
):
    """Ensure every new Organisation has a corresponding OrganisationSettings row.

    When an Organisation is first created we initialise the settings with
    sensible defaults so that users can start experimenting immediately
    without manually configuring LLM providers.

    Defaults chosen:
    - Chat provider / model: Google Gemini 2.5 Flash
    - Embeddings provider / model: OpenAI text-embedding-3-small
    """
    if not created:
        return

    # Guard against tests or custom factories that may have pre-created a
    # settings object for the organisation.
    if OrganisationSettings.objects.filter(organisation=instance).exists():
        return

    OrganisationSettings.objects.create(
        organisation=instance,
        llm_chat_provider="google",
        llm_chat_model="models/gemini-2.5-flash-preview-05-20",
        llm_embeddings_provider="openai",
        llm_embeddings_model="text-embedding-3-small",
    )
