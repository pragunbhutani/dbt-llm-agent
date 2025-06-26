import logging
from celery import shared_task
from .models import Model
from apps.embeddings.models import ModelEmbedding
from apps.workflows.services import trigger_model_interpretation
from apps.embeddings.services import embed_knowledge_model
from apps.accounts.models import OrganisationSettings

logger = logging.getLogger(__name__)


@shared_task
def interpret_and_embed_model_task(model_id: int):
    """
    Celery task to run model interpretation and embedding.
    """
    embedding_record = None
    try:
        model = Model.objects.get(id=model_id)
        logger.info(f"Starting interpretation for model: {model.name}")

        # Get the organisation settings
        try:
            org_settings = OrganisationSettings.objects.get(
                organisation=model.organisation
            )
        except OrganisationSettings.DoesNotExist:
            error_msg = (
                f"OrganisationSettings not found for organisation: {model.organisation}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get or create the ModelEmbedding record
        embedding_record = ModelEmbedding.objects.filter(model=model).first()
        if not embedding_record:
            embedding_record = ModelEmbedding.objects.create(
                model=model,
                organisation=model.organisation,
                dbt_project=model.dbt_project,
                document="",
                embedding=[0] * 3072,  # Placeholder
                is_processing=True,
                can_be_used_for_answers=False,
            )

        # Ensure processing flag is set
        if not embedding_record.is_processing:
            embedding_record.is_processing = True
            embedding_record.save()

        interp_success = trigger_model_interpretation(
            model=model, org_settings=org_settings
        )

        if interp_success:
            logger.info(
                f"Interpretation successful for model: {model.name}. Now embedding."
            )
            embed_success = embed_knowledge_model(model=model, include_docs=True)
            if embed_success:
                logger.info(f"Successfully embedded model: {model.name}")
                # Update the processing status - embedding service handles the record update
                embedding_record.refresh_from_db()  # Get the updated record from embed_knowledge_model
                embedding_record.is_processing = False
                embedding_record.can_be_used_for_answers = True
                embedding_record.save()
            else:
                error_msg = f"Failed to embed model: {model.name}"
                logger.error(error_msg)
                # Delete the placeholder record to avoid tripping up future checks
                embedding_record.delete()
                logger.info(
                    f"Deleted placeholder embedding record for failed model: {model.name}"
                )
                raise RuntimeError(error_msg)
        else:
            error_msg = f"Failed to interpret model: {model.name}"
            logger.error(error_msg)
            # Delete the placeholder record to avoid tripping up future checks
            embedding_record.delete()
            logger.info(
                f"Deleted placeholder embedding record for failed model: {model.name}"
            )
            raise RuntimeError(error_msg)

    except Model.DoesNotExist:
        error_msg = f"Model with id {model_id} not found."
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        logger.error(
            f"Error in interpret_and_embed_model_task for model_id {model_id}: {e}",
            exc_info=True,
        )
        # Try to delete the placeholder record on any error to avoid future issues
        try:
            if embedding_record is None:
                embedding_record = ModelEmbedding.objects.filter(
                    model_id=model_id
                ).first()
            if embedding_record and embedding_record.document == "":
                # Only delete if it's still a placeholder record
                embedding_record.delete()
                logger.info(
                    f"Deleted placeholder embedding record after error for model_id: {model_id}"
                )
        except Exception as cleanup_error:
            logger.error(f"Failed to cleanup placeholder record: {cleanup_error}")

        # Re-raise the original exception so Celery marks the task as failed
        raise
