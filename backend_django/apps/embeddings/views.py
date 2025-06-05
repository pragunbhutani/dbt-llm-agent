from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from pgvector.django import L2Distance, CosineDistance, MaxInnerProduct

# Update imports
from .models import ModelEmbedding
from .serializers import ModelEmbeddingSerializer
from apps.llm_providers.services import default_embedding_service


class ModelEmbeddingViewSet(viewsets.ModelViewSet):
    """API endpoint that allows Model Embeddings to be viewed or edited."""

    queryset = ModelEmbedding.objects.all().order_by("-created_at")
    serializer_class = ModelEmbeddingSerializer

    def perform_create(self, serializer):
        document_text = serializer.validated_data.get("document")
        embedding = None
        if document_text:
            embedding = default_embedding_service.get_embedding(document_text)
        serializer.save(embedding=embedding)

    def perform_update(self, serializer):
        document_text = serializer.validated_data.get("document")
        embedding = None
        if document_text:
            embedding = default_embedding_service.get_embedding(document_text)
            serializer.save(embedding=embedding)
        else:
            serializer.save()

    @action(detail=False, methods=["post"], url_path="search")
    def search_embeddings(self, request):
        query_text = request.data.get("query")
        n_results = int(request.data.get("n_results", 5))
        metric_type = request.data.get("metric", "cosine").lower()
        if not query_text:
            return Response(
                {"error": "Query text is required."}, status=status.HTTP_400_BAD_REQUEST
            )
        query_embedding = default_embedding_service.get_embedding(query_text)
        if not query_embedding:
            return Response(
                {"error": "Failed to generate query embedding."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        queryset = ModelEmbedding.objects.filter(can_be_used_for_answers=True)

        if metric_type == "cosine":
            queryset = queryset.annotate(
                distance=CosineDistance("embedding", query_embedding)
            ).order_by("distance")
        elif metric_type == "l2":
            queryset = queryset.annotate(
                distance=L2Distance("embedding", query_embedding)
            ).order_by("distance")
        elif metric_type == "inner_product":
            queryset = queryset.annotate(
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
