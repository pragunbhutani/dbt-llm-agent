import logging
from typing import Dict, Any
from django.utils import timezone

# Import models and services from other apps
from apps.knowledge_base.models import Model
from apps.llm_providers.services import EmbeddingService
from .models import ModelEmbedding  # Local import

logger = logging.getLogger(__name__)


def embed_knowledge_model(model: Model, include_docs: bool = True) -> bool:
    """Generates and saves embedding for a single knowledge_base Model.

    Args:
        model: The Model instance to embed.
        include_docs: Whether to include YML documentation in the embedded text.

    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"Starting embedding process for model: {model.name}")
    try:
        # 0. Get OrganisationSettings from the model
        if not model.dbt_project or not model.dbt_project.organisation:
            logger.error(
                f"Cannot embed model '{model.name}': It is not linked to an organisation."
            )
            return False
        org_settings = model.dbt_project.organisation.settings
        if not org_settings:
            logger.error(
                f"Cannot embed model '{model.name}': Organisation settings not found."
            )
            return False

        embedding_service = EmbeddingService(org_settings)

        # 1. Generate text representation
        document_text = model.get_text_representation(
            include_documentation=include_docs
        )
        if not document_text:
            logger.warning(
                f"Skipping {model.name}, generated empty text representation."
            )
            return False

        # 2. Generate embedding
        embedding_vector = embedding_service.get_embedding(document_text)
        if not embedding_vector:
            logger.error(f"Failed to generate embedding vector for {model.name}")
            return False

        # 3. Prepare metadata (simplified to avoid UUID serialization issues)
        model_metadata_dict = {
            "name": model.name,
            "path": model.path,
            "schema_name": model.schema_name,
            "database": model.database,
            "materialization": model.materialization,
            "tags": model.tags,
            "depends_on": model.depends_on,
            "all_upstream_models": model.all_upstream_models,
            "unique_id": model.unique_id,
            "organisation_id": (
                str(model.organisation.id) if model.organisation else None
            ),
            "dbt_project_id": model.dbt_project.id if model.dbt_project else None,
        }

        # 4. Save embedding - do NOT override is_processing here
        # The caller (celery task) will handle the is_processing flag appropriately
        embedding_instance, created = ModelEmbedding.objects.update_or_create(
            model=model,
            organisation=model.organisation,
            dbt_project=model.dbt_project,
            defaults={
                "document": document_text,
                "embedding": embedding_vector,
                "model_metadata": model_metadata_dict,
                "updated_at": timezone.now(),
                # Only set can_be_used_for_answers if not already processing
                # This ensures we don't override the processing state
            },
        )

        # Only set can_be_used_for_answers=True if not currently processing
        # This prevents overriding the processing workflow
        if not embedding_instance.is_processing:
            embedding_instance.can_be_used_for_answers = True
            embedding_instance.save()
        action = "Created" if created else "Updated"
        logger.info(f"{action} embedding record for model {model.name}")
        return True

    except Exception as e:
        logger.error(
            f"Error during embedding process for {model.name}: {e}", exc_info=True
        )
        return False
