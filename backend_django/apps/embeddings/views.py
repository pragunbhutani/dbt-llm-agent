from django.shortcuts import render
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.exceptions import PermissionDenied, NotFound
from pgvector.django import L2Distance, CosineDistance, MaxInnerProduct

# Update imports
from .models import ModelEmbedding
from .serializers import ModelEmbeddingSerializer
from apps.llm_providers.services import EmbeddingService
from apps.accounts.models import OrganisationSettings


class ModelEmbeddingViewSet(viewsets.ModelViewSet):
    """API endpoint that allows Model Embeddings to be viewed or edited."""

    serializer_class = ModelEmbeddingSerializer
    permission_classes = [permissions.IsAuthenticated]

    def _get_embedding_service(self):
        """Helper to get the embedding service based on the user's organisation."""
        user = self.request.user
        if not (hasattr(user, "organisation") and user.organisation):
            raise PermissionDenied("User is not associated with an organisation.")

        org_settings = OrganisationSettings.objects.filter(
            organisation=user.organisation
        ).first()
        if not org_settings:
            raise NotFound("Organisation settings not found for this user.")

        return EmbeddingService(org_settings)

    def get_queryset(self):
        user = self.request.user
        if hasattr(user, "organisation") and user.organisation:
            return ModelEmbedding.objects.for_organisation(user.organisation).order_by(
                "-created_at"
            )
        return ModelEmbedding.objects.none()

    def perform_create(self, serializer):
        user = self.request.user
        if not (hasattr(user, "organisation") and user.organisation):
            raise PermissionDenied("User is not associated with an organisation.")

        # First, save the instance with the organisation and other serializer-handled fields.
        # The 'embedding' field is read-only in the serializer, so it won't be set here.
        instance = serializer.save(organisation=user.organisation)

        # Now, generate and set the embedding if document text is present.
        document_text = (
            instance.document
        )  # instance.document should be populated by serializer.save()
        if document_text:
            embedding_service = self._get_embedding_service()
            instance.embedding = embedding_service.get_embedding(document_text)
            instance.save(
                update_fields=["embedding"]
            )  # Save again to store the embedding
        # If document_text is empty, embedding remains null (as per model default/previous state)

    def perform_update(self, serializer):
        # get_object ensures the instance is already scoped to the user's organisation.
        # Save changes from serializer (excluding embedding as it's read-only).
        instance = serializer.save()

        # If the document text was part of the update, regenerate the embedding.
        # Check if 'document' was in request.data to see if it was intended to be updated.
        if "document" in serializer.context["request"].data:
            document_text = instance.document
            if document_text:
                embedding_service = self._get_embedding_service()
                instance.embedding = embedding_service.get_embedding(document_text)
            else:
                instance.embedding = None  # Clear embedding if document is cleared
            instance.save(
                update_fields=["embedding"]
            )  # Save again to store the new embedding or None

    @action(detail=False, methods=["post"], url_path="search")
    def search_embeddings(self, request):
        embedding_service = self._get_embedding_service()
        user = (
            self.request.user
        )  # user is safe to use here due to _get_embedding_service check

        query_text = request.data.get("query")
        n_results = int(request.data.get("n_results", 5))
        metric_type = request.data.get("metric", "cosine").lower()
        if not query_text:
            return Response(
                {"error": "Query text is required."}, status=status.HTTP_400_BAD_REQUEST
            )
        query_embedding = embedding_service.get_embedding(query_text)
        if not query_embedding:
            return Response(
                {"error": "Failed to generate query embedding."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Base queryset scoped by organisation
        base_queryset = ModelEmbedding.objects.for_organisation(
            user.organisation
        ).filter(can_be_used_for_answers=True)

        if metric_type == "cosine":
            queryset = base_queryset.annotate(
                distance=CosineDistance("embedding", query_embedding)
            ).order_by("distance")
        elif metric_type == "l2":
            queryset = base_queryset.annotate(
                distance=L2Distance("embedding", query_embedding)
            ).order_by("distance")
        elif metric_type == "inner_product":
            queryset = base_queryset.annotate(
                distance=MaxInnerProduct("embedding", query_embedding)
            ).order_by("-distance")
        else:
            return Response(
                {"error": f"Unsupported metric type: {metric_type}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        results = queryset[:n_results]
        serializer = self.get_serializer(results, many=True)
        data = serializer.data
        for i, item in enumerate(data):
            item["distance"] = results[i].distance
            item["similarity_score"] = (
                1.0 - results[i].distance
                if metric_type != "inner_product"
                else results[i].distance
            )
        return Response(data)
