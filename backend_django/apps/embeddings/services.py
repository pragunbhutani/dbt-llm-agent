import logging
from typing import Dict, Any
from django.utils import timezone

# Import models and services from other apps
from apps.knowledge_base.models import Model
from apps.knowledge_base.serializers import ModelSerializer  # Needed for metadata
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

        # 3. Prepare metadata
        # Use serializer from knowledge_base to get consistent metadata?
        try:
            serializer = ModelSerializer(model)
            model_metadata_dict = serializer.data
            # Clean up potentially large/unwanted fields if necessary
            if model_metadata_dict.get("tests") is not None:
                del model_metadata_dict["tests"]
            # Remove SQL fields?
            # model_metadata_dict.pop("raw_sql", None)
            # model_metadata_dict.pop("compiled_sql", None)
        except Exception as sz_err:
            logger.warning(
                f"Failed to serialize model {model.name} for metadata: {sz_err}"
            )
            model_metadata_dict = {
                "name": model.name,
                "path": model.path,
            }  # Fallback metadata

        # 4. Save embedding
        embedding_instance, created = ModelEmbedding.objects.update_or_create(
            model=model,
            organisation=model.organisation,
            dbt_project=model.dbt_project,
            defaults={
                "document": document_text,
                "embedding": embedding_vector,
                "model_metadata": model_metadata_dict,
                "can_be_used_for_answers": True,
                "updated_at": timezone.now(),
            },
        )
        action = "Created" if created else "Updated"
        logger.info(f"{action} embedding record for model {model.name}")
        return True

    except Exception as e:
        logger.error(
            f"Error during embedding process for {model.name}: {e}", exc_info=True
        )
        return False
