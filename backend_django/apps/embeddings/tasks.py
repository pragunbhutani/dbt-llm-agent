import logging
from celery import shared_task
from apps.knowledge_base.models import Model
from .models import ModelEmbedding
from .services import embed_knowledge_model

logger = logging.getLogger(__name__)


@shared_task
def embed_model_task(model_id: int):
    """
    Celery task to run embedding only (assumes interpretation already exists).
    """
    embedding_record = None
    try:
        model = Model.objects.get(id=model_id)
        logger.info(f"Starting embedding-only process for model: {model.name}")

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

        # Run embedding
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

    except Model.DoesNotExist:
        error_msg = f"Model with id {model_id} not found."
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        logger.error(
            f"Error in embed_model_task for model_id {model_id}: {e}",
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
