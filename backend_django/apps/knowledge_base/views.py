from django.shortcuts import render
from rest_framework import viewsets, permissions, status
from rest_framework.exceptions import PermissionDenied
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
import logging

# Update imports
from .models import Model
from .serializers import ModelSerializer
from .tasks import interpret_and_embed_model_task
from apps.embeddings.models import ModelEmbedding

# Create your views here.

logger = logging.getLogger(__name__)


# Helper for validating LLM settings before starting interpretation/embedding
def _validate_llm_settings(organisation):
    """Returns (is_valid, message)."""
    from apps.accounts.models import OrganisationSettings

    try:
        org_settings = OrganisationSettings.objects.get(organisation=organisation)
    except OrganisationSettings.DoesNotExist:
        return (
            False,
            "Organisation settings not found. Please configure your LLM provider and API keys before training.",
        )

    missing_items = []

    # Chat provider & model
    if not org_settings.llm_chat_provider or not org_settings.llm_chat_model:
        missing_items.append("chat provider & model")

    # Embeddings provider & model
    if (
        not org_settings.llm_embeddings_provider
        or not org_settings.llm_embeddings_model
    ):
        missing_items.append("embeddings provider & model")

    # API keys (check that a usable key exists for any provider that is selected)
    # Keys can be supplied either via Parameter Store path *or* the corresponding
    # environment variable, so we need to consider both sources.

    provider_to_key_check: dict[str, callable[[], str | None]] = {
        "openai": org_settings.get_llm_openai_api_key,
        "google": org_settings.get_llm_google_api_key,
        "anthropic": org_settings.get_llm_anthropic_api_key,
    }

    selected_providers = {
        org_settings.llm_chat_provider,
        org_settings.llm_embeddings_provider,
    }

    for provider in selected_providers:
        # Skip None/empty providers
        if not provider or provider not in provider_to_key_check:
            continue

        key_value = provider_to_key_check[provider]()
        if not key_value:
            missing_items.append(f"{provider} API key")

    if missing_items:
        pretty = ", ".join(sorted(set(missing_items)))
        return (
            False,
            f"Your LLM configuration is incomplete. Missing: {pretty}. Please complete these settings before training.",
        )

    return (True, "")


class ModelViewSet(viewsets.ModelViewSet):
    """API endpoint that allows Models to be viewed or edited, scoped by organisation."""

    serializer_class = ModelSerializer
    permission_classes = [permissions.IsAuthenticated]  # Ensure user is authenticated

    def get_queryset(self):
        """
        This view should return a list of all the models
        for the currently authenticated user's organisation.
        """
        user = self.request.user
        if hasattr(user, "organisation") and user.organisation:
            # Use the custom manager method if available and preferred,
            # otherwise filter directly.
            # Assumes the default manager on Model is OrganisationScopedManager or similar
            qs = Model.objects.for_organisation(user.organisation)
            project_id = self.request.query_params.get("project")
            if project_id:
                qs = qs.filter(dbt_project_id=project_id)
            return qs.order_by("-updated_at")
        # Return an empty queryset if no organisation is associated with the user
        return Model.objects.none()

    def perform_create(self, serializer):
        user = self.request.user
        if hasattr(user, "organisation") and user.organisation:
            serializer.save(organisation=user.organisation)
        else:
            # This case should ideally not be reached if permissions are set correctly
            # and all authenticated users have an organisation.
            raise PermissionDenied("User is not associated with an organisation.")

    def _is_valid_embedding(self, embedding_record):
        """Return True if the embedding exists and is NOT a placeholder of all-zero values."""
        if not embedding_record:
            return False

        # Convert the stored vector to a plain Python list (VectorField may return a
        # specialised type that doesn't compare cleanly to another list).
        try:
            embedding_values = list(embedding_record.embedding)  # type: ignore
        except Exception:
            # Fallback: treat unknown format as invalid so that we retrain
            return False

        is_zero_vector = len(embedding_values) == 3072 and all(
            v == 0 or v == 0.0 for v in embedding_values
        )

        # A valid embedding must have a non-empty document and a non-zero vector
        return not (embedding_record.document == "" or is_zero_vector)

    @action(detail=True, methods=["post"], url_path="toggle-answering-status")
    def toggle_answering_status(self, request, pk=None):
        """
        Toggles whether a model can be used for answering.
        Always triggers the full interpret and embed workflow for consistency with bulk operations.

        Logic:
        - If enabling: always triggers interpret + embed workflow
        - If disabling: just toggles the can_be_used_for_answers flag
        """
        # Validate LLM configuration before proceeding.
        is_valid, message = _validate_llm_settings(request.user.organisation)
        if not is_valid:
            return Response({"error": message}, status=status.HTTP_400_BAD_REQUEST)

        try:
            model = self.get_object()
            embedding = ModelEmbedding.objects.filter(model=model).first()
            has_interpretation = bool(
                model.interpreted_description or model.interpreted_columns
            )

            # Check if embedding is valid (not a placeholder)
            has_valid_embedding = self._is_valid_embedding(embedding)

            # If currently enabled, disable it
            if embedding and has_valid_embedding and embedding.can_be_used_for_answers:
                if embedding.is_processing:
                    return Response(
                        {"status": "Model is already being processed"},
                        status=status.HTTP_200_OK,
                    )

                # Disable the model
                embedding.can_be_used_for_answers = False
                embedding.save()
                return Response(
                    {"status": "answering status disabled"},
                    status=status.HTTP_200_OK,
                )

            # If currently disabled or no valid embedding, enable with full workflow
            else:
                if embedding and embedding.is_processing:
                    return Response(
                        {"status": "Model is already being processed"},
                        status=status.HTTP_200_OK,
                    )

                # Clean up any existing invalid embedding
                if embedding and not has_valid_embedding:
                    embedding.delete()
                    logger.info(
                        f"Deleted placeholder embedding for model: {model.name}"
                    )

                # Create new embedding record and always run full workflow
                ModelEmbedding.objects.create(
                    model=model,
                    organisation=model.organisation,
                    dbt_project=model.dbt_project,
                    document="",  # Will be populated by the task
                    embedding=[0] * 3072,  # Placeholder, will be updated by task
                    is_processing=True,
                    can_be_used_for_answers=False,
                )

                # Always run the full interpret and embed workflow for consistency
                interpret_and_embed_model_task.delay(model.id)
                return Response(
                    {"status": "interpretation and embedding process started"},
                    status=status.HTTP_202_ACCEPTED,
                )

        except Model.DoesNotExist:
            return Response(
                {"error": "Model not found."}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=["post"], url_path="bulk-toggle-answering")
    def bulk_toggle_answering(self, request):
        """
        Bulk enables or disables models for answering.
        Expects a list of model IDs and an 'action' ('enable' or 'disable').

        For enabling:
        - If model has no interpretation and no embedding: triggers interpretation + embedding
        - If model has interpretation but no embedding: triggers embedding only
        - If model has embedding but is disabled: just enables it
        """
        # Validate LLM configuration before proceeding.
        is_valid, message = _validate_llm_settings(request.user.organisation)
        if not is_valid:
            return Response({"error": message}, status=status.HTTP_400_BAD_REQUEST)

        model_ids = request.data.get("model_ids", [])
        enable = request.data.get("enable", True)

        if not model_ids:
            return Response(
                {"error": "No model IDs provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Get all models belonging to the user's organisation
        models = self.get_queryset().filter(id__in=model_ids)

        started_interpretation_tasks = 0
        started_embedding_tasks = 0
        updated_existing = 0
        already_processing = 0

        for model in models:
            if not enable:
                # Disable answering for model
                ModelEmbedding.objects.filter(model=model).update(
                    can_be_used_for_answers=False
                )
                updated_existing += 1
                continue

            embedding = ModelEmbedding.objects.filter(model=model).first()
            has_interpretation = bool(
                model.interpreted_description or model.interpreted_columns
            )
            has_valid_embedding = self._is_valid_embedding(embedding)

            if not embedding or not has_valid_embedding:
                # No valid embedding exists - clean up any placeholder
                if embedding and not has_valid_embedding:
                    # Delete placeholder embedding
                    embedding.delete()
                    logger.info(
                        f"Deleted placeholder embedding for model: {model.name}"
                    )

                # Create new embedding record and determine what to run
                ModelEmbedding.objects.create(
                    model=model,
                    organisation=model.organisation,
                    dbt_project=model.dbt_project,
                    document="",  # Will be populated by the task
                    embedding=[0] * 3072,  # Placeholder, will be updated by task
                    is_processing=True,
                    can_be_used_for_answers=False,
                )

                if has_interpretation:
                    # Has interpretation, just need embedding
                    from apps.embeddings.tasks import embed_model_task

                    embed_model_task.delay(model.id)
                    started_embedding_tasks += 1
                else:
                    # No interpretation, need both interpretation + embedding
                    interpret_and_embed_model_task.delay(model.id)
                    started_interpretation_tasks += 1

            elif embedding.is_processing:
                # Already processing
                already_processing += 1
            else:
                # Valid embedding exists, check if we need to run interpretation
                if not has_interpretation:
                    # Has embedding but no interpretation - run interpretation only
                    # and update the embedding after
                    embedding.is_processing = True
                    embedding.save()
                    interpret_and_embed_model_task.delay(model.id)
                    started_interpretation_tasks += 1
                else:
                    # Has both interpretation and embedding, just enable it
                    embedding.can_be_used_for_answers = True
                    embedding.save()
                    updated_existing += 1

        response_msg = f"Processed {len(model_ids)} models. "
        if enable:
            total_tasks = started_interpretation_tasks + started_embedding_tasks
            response_msg += f"Started interpretation+embedding: {started_interpretation_tasks}, Started embedding only: {started_embedding_tasks}, Updated existing: {updated_existing}, Already processing: {already_processing}"
        else:
            response_msg += f"Disabled: {updated_existing}"

        return Response({"status": response_msg}, status=status.HTTP_202_ACCEPTED)

    @action(detail=True, methods=["post"], url_path="refresh")
    def refresh_model(self, request, pk=None):
        """
        Forces re-interpretation and re-embedding of a model, even if they already exist.
        """
        try:
            model = self.get_object()
            embedding = ModelEmbedding.objects.filter(model=model).first()

            if embedding and embedding.is_processing:
                return Response(
                    {"status": "Model is already being processed"},
                    status=status.HTTP_200_OK,
                )

            # Delete existing embedding to force a complete refresh
            if embedding:
                embedding.delete()
                logger.info(f"Deleted existing embedding for refresh: {model.name}")

            # Create new embedding record and run full workflow
            ModelEmbedding.objects.create(
                model=model,
                organisation=model.organisation,
                dbt_project=model.dbt_project,
                document="",  # Will be populated by the task
                embedding=[0] * 3072,  # Placeholder, will be updated by task
                is_processing=True,
                can_be_used_for_answers=False,
            )

            # Always run the full interpret and embed workflow
            interpret_and_embed_model_task.delay(model.id)
            return Response(
                {"status": "refresh process started - interpretation and embedding"},
                status=status.HTTP_202_ACCEPTED,
            )

        except Model.DoesNotExist:
            return Response(
                {"error": "Model not found."}, status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=["post"], url_path="bulk-refresh")
    def bulk_refresh_models(self, request):
        """
        Forces re-interpretation and re-embedding of multiple models.
        """
        model_ids = request.data.get("model_ids", [])

        if not model_ids:
            return Response(
                {"error": "No model IDs provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Get all models belonging to the user's organisation
        models = self.get_queryset().filter(id__in=model_ids)

        started_tasks = 0
        already_processing = 0

        for model in models:
            embedding = ModelEmbedding.objects.filter(model=model).first()

            if embedding and embedding.is_processing:
                already_processing += 1
                continue

            # Delete existing embedding to force a complete refresh
            if embedding:
                embedding.delete()
                logger.info(
                    f"Deleted existing embedding for bulk refresh: {model.name}"
                )

            # Create new embedding record and run full workflow
            ModelEmbedding.objects.create(
                model=model,
                organisation=model.organisation,
                dbt_project=model.dbt_project,
                document="",  # Will be populated by the task
                embedding=[0] * 3072,  # Placeholder, will be updated by task
                is_processing=True,
                can_be_used_for_answers=False,
            )

            # Always run the full interpret and embed workflow
            interpret_and_embed_model_task.delay(model.id)
            started_tasks += 1

        response_msg = f"Processed {len(model_ids)} models. Started refresh tasks: {started_tasks}, Already processing: {already_processing}"
        return Response({"status": response_msg}, status=status.HTTP_202_ACCEPTED)
