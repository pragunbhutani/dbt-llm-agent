from django.shortcuts import render
from rest_framework import viewsets, permissions, status
from rest_framework.exceptions import PermissionDenied
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

# Update imports
from .models import Model
from .serializers import ModelSerializer
from .tasks import interpret_and_embed_model_task
from apps.embeddings.models import ModelEmbedding

# Create your views here.


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
            return Model.objects.for_organisation(user.organisation).order_by(
                "-updated_at"
            )
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

    @action(detail=True, methods=["post"], url_path="toggle-answering-status")
    def toggle_answering_status(self, request, pk=None):
        """
        Toggles whether a model can be used for answering.
        If not embedded, it triggers the embedding process.
        If embedded, it flips the 'can_be_used_for_answers' flag.
        """
        try:
            model = self.get_object()
            embedding = ModelEmbedding.objects.filter(model=model).first()

            if embedding:
                if embedding.is_processing:
                    return Response(
                        {"status": "Model is already being processed"},
                        status=status.HTTP_200_OK,
                    )
                # Embedding exists, just flip the boolean
                embedding.can_be_used_for_answers = (
                    not embedding.can_be_used_for_answers
                )
                embedding.save()
                return Response(
                    {"status": "answering status updated"}, status=status.HTTP_200_OK
                )
            else:
                # No embedding exists, create one and start the process
                ModelEmbedding.objects.create(
                    model=model,
                    organisation=model.organisation,
                    dbt_project=model.dbt_project,
                    document="",  # Will be populated by the task
                    embedding=[0] * 1536,  # Placeholder, will be updated by task
                    is_processing=True,
                    can_be_used_for_answers=False,
                )
                interpret_and_embed_model_task.delay(model.id)
                return Response(
                    {"status": "embedding process started"},
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

    @action(detail=False, methods=["post"], url_path="bulk-toggle-answering-status")
    def bulk_toggle_answering_status(self, request):
        """
        Bulk enables or disables models for answering.
        Expects a list of model IDs and an 'action' ('enable' or 'disable').
        """
        model_ids = request.data.get("model_ids", [])
        enable = request.data.get("enable", True)

        if not model_ids:
            return Response(
                {"error": "No model IDs provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Get all models belonging to the user's organisation
        models = self.get_queryset().filter(id__in=model_ids)

        started_tasks = 0
        updated_existing = 0
        already_processing = 0

        for model in models:
            if enable:
                embedding = ModelEmbedding.objects.filter(model=model).first()
                if not embedding:
                    # No embedding exists, create one and start the process
                    ModelEmbedding.objects.create(
                        model=model,
                        organisation=model.organisation,
                        dbt_project=model.dbt_project,
                        document="",  # Will be populated by the task
                        embedding=[0] * 1536,  # Placeholder, will be updated by task
                        is_processing=True,
                        can_be_used_for_answers=False,
                    )
                    interpret_and_embed_model_task.delay(model.id)
                    started_tasks += 1
                elif embedding.is_processing:
                    # Already processing
                    already_processing += 1
                else:
                    # Embedding exists, just enable it
                    embedding.can_be_used_for_answers = True
                    embedding.save()
                    updated_existing += 1
            else:
                # Disable for answering
                ModelEmbedding.objects.filter(model=model).update(
                    can_be_used_for_answers=False
                )
                updated_existing += 1

        response_msg = f"Processed {len(model_ids)} models. "
        if enable:
            response_msg += f"Started tasks: {started_tasks}, Updated existing: {updated_existing}, Already processing: {already_processing}"
        else:
            response_msg += f"Disabled: {updated_existing}"

        return Response({"status": response_msg}, status=status.HTTP_202_ACCEPTED)
