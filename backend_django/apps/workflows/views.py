import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from pgvector.django import L2Distance, CosineDistance, MaxInnerProduct

# Update imports
from .models import Question
from .serializers import QuestionSerializer
from apps.llm_providers.services import default_embedding_service
from .question_answerer import QuestionAnswererAgent  # Relative import within workflows

logger = logging.getLogger(__name__)


class QuestionViewSet(viewsets.ModelViewSet):
    """API endpoint that allows Questions to be viewed or edited."""

    queryset = Question.objects.all().order_by("-created_at")
    serializer_class = QuestionSerializer

    def perform_create(self, serializer):
        instance = serializer.save()
        if instance.question_text:
            instance.question_embedding = default_embedding_service.get_embedding(
                instance.question_text
            )
        if instance.original_message_text:
            instance.original_message_embedding = (
                default_embedding_service.get_embedding(instance.original_message_text)
            )
        if instance.feedback:
            instance.feedback_embedding = default_embedding_service.get_embedding(
                instance.feedback
            )
        instance.save()

    def perform_update(self, serializer):
        instance = serializer.save()
        update_fields = []
        request_data = serializer.context["request"].data
        if "question_text" in request_data and instance.question_text:
            instance.question_embedding = default_embedding_service.get_embedding(
                instance.question_text
            )
            update_fields.append("question_embedding")
        if "original_message_text" in request_data and instance.original_message_text:
            instance.original_message_embedding = (
                default_embedding_service.get_embedding(instance.original_message_text)
            )
            update_fields.append("original_message_embedding")
        if "feedback" in request_data:
            if instance.feedback:
                instance.feedback_embedding = default_embedding_service.get_embedding(
                    instance.feedback
                )
            else:
                instance.feedback_embedding = None
            update_fields.append("feedback_embedding")
        if update_fields:
            instance.save(update_fields=update_fields)

    @action(detail=False, methods=["post"], url_path="search")
    def search_similar(self, request):
        query_text = request.data.get("query")
        search_field = request.data.get("field", "question").lower()
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

        if search_field == "question":
            embedding_db_field = "question_embedding"
            queryset = Question.objects.exclude(question_embedding__isnull=True)
        elif search_field == "feedback":
            embedding_db_field = "feedback_embedding"
            queryset = Question.objects.exclude(feedback_embedding__isnull=True)
        elif search_field == "original":
            embedding_db_field = "original_message_embedding"
            queryset = Question.objects.exclude(original_message_embedding__isnull=True)
        else:
            return Response(
                {"error": f"Unsupported search field: {search_field}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if metric_type == "cosine":
            queryset = queryset.annotate(
                distance=CosineDistance(embedding_db_field, query_embedding)
            ).order_by("distance")
        elif metric_type == "l2":
            queryset = queryset.annotate(
                distance=L2Distance(embedding_db_field, query_embedding)
            ).order_by("distance")
        elif metric_type == "inner_product":
            queryset = queryset.annotate(
                distance=MaxInnerProduct(embedding_db_field, query_embedding)
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


class AskQuestionView(APIView):
    """API endpoint to ask a question and get an answer from the QuestionAnswererAgent."""

    def post(self, request, *args, **kwargs):
        question_text = request.data.get("question")
        thread_context = request.data.get("thread_context")
        conversation_id = request.data.get("conversation_id")
        if not question_text:
            return Response(
                {"error": '"question" field is required.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if thread_context is not None and not isinstance(thread_context, list):
            return Response(
                {"error": '"thread_context" must be a list of objects.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            agent = QuestionAnswererAgent(verbose=True)
            result = agent.run_agentic_workflow(
                question=question_text,
                thread_context=thread_context,
                conversation_id=conversation_id,
            )
            if result.get("error"):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception(f"Unexpected error during /ask processing: {e}")
            return Response(
                {"error": f"An unexpected server error occurred: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
