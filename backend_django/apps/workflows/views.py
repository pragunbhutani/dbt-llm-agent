import logging
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.exceptions import PermissionDenied
from pgvector.django import L2Distance, CosineDistance, MaxInnerProduct

# Update imports
from .models import Question, Conversation, ConversationPart
from .serializers import (
    QuestionSerializer,
    ConversationSerializer,
    ConversationListSerializer,
)
from apps.llm_providers.services import EmbeddingService
from apps.accounts.models import OrganisationSettings
from .question_answerer import QuestionAnswererAgent

logger = logging.getLogger(__name__)


class QuestionViewSet(viewsets.ModelViewSet):
    """API endpoint that allows Questions to be viewed or edited."""

    serializer_class = QuestionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if hasattr(user, "organisation") and user.organisation:
            return Question.objects.for_organisation(user.organisation).order_by(
                "-created_at"
            )
        return Question.objects.none()

    def perform_create(self, serializer):
        user = self.request.user
        if not (hasattr(user, "organisation") and user.organisation):
            raise PermissionDenied("User is not associated with an organisation.")

        # Save with organisation, then handle embeddings
        instance = serializer.save(organisation=user.organisation)
        self._update_embeddings(
            instance, serializer.context["request"].data, is_new=True
        )

    def perform_update(self, serializer):
        # get_object called by the viewset already uses the scoped get_queryset,
        # so we are sure the instance belongs to the user's organisation.
        instance = serializer.save()
        self._update_embeddings(
            instance, serializer.context["request"].data, is_new=False
        )

    def _update_embeddings(self, instance, request_data, is_new=False):
        """Helper to update embeddings for create and update."""
        try:
            org_settings = OrganisationSettings.objects.get(
                organisation=instance.organisation
            )
            embedding_service = EmbeddingService(org_settings=org_settings)
        except OrganisationSettings.DoesNotExist:
            logger.error(
                f"OrganisationSettings not found for {instance.organisation.name}. Cannot generate embeddings."
            )
            return

        update_fields = []
        # For new instances, or if fields are explicitly in request_data for updates
        if is_new or "question_text" in request_data:
            if instance.question_text:
                instance.question_embedding = embedding_service.get_embedding(
                    instance.question_text
                )
                if not is_new:
                    update_fields.append("question_embedding")

        if is_new or "original_message_text" in request_data:
            if instance.original_message_text:
                instance.original_message_embedding = embedding_service.get_embedding(
                    instance.original_message_text
                )
                if not is_new:
                    update_fields.append("original_message_embedding")

        if is_new or "feedback" in request_data:
            if instance.feedback:
                instance.feedback_embedding = embedding_service.get_embedding(
                    instance.feedback
                )
            else:
                instance.feedback_embedding = None  # Clear if feedback is cleared
            if not is_new:
                update_fields.append("feedback_embedding")

        if not is_new and update_fields:
            instance.save(update_fields=update_fields)
        elif (
            is_new
        ):  # For new instances, these are set before initial full save by serializer
            instance.save()  # Save again if any embeddings were set

    @action(detail=False, methods=["post"], url_path="search")
    def search_similar(self, request):
        user = self.request.user
        if not (hasattr(user, "organisation") and user.organisation):
            return Response(
                {"error": "User is not associated with an organisation."},
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            org_settings = OrganisationSettings.objects.get(
                organisation=user.organisation
            )
            embedding_service = EmbeddingService(org_settings=org_settings)
        except OrganisationSettings.DoesNotExist:
            return Response(
                {"error": "Organisation settings not found, cannot perform search."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        query_text = request.data.get("query")
        search_field = request.data.get("field", "question").lower()
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
        base_queryset = Question.objects.for_organisation(user.organisation)

        if search_field == "question":
            embedding_db_field = "question_embedding"
            queryset = base_queryset.exclude(question_embedding__isnull=True)
        elif search_field == "feedback":
            embedding_db_field = "feedback_embedding"
            queryset = base_queryset.exclude(feedback_embedding__isnull=True)
        elif search_field == "original":
            embedding_db_field = "original_message_embedding"
            queryset = base_queryset.exclude(original_message_embedding__isnull=True)
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


class ConversationViewSet(viewsets.ModelViewSet):
    """API endpoint that allows Conversations to be viewed, edited, and deleted."""

    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if hasattr(user, "organisation") and user.organisation:
            return Conversation.objects.for_organisation(user.organisation).order_by(
                "-started_at"
            )
        return Conversation.objects.none()

    def get_serializer_class(self):
        """Use lighter serializer for list view, full serializer for detail view."""
        if self.action == "list":
            return ConversationListSerializer
        return ConversationSerializer

    def perform_create(self, serializer):
        user = self.request.user
        if not (hasattr(user, "organisation") and user.organisation):
            raise PermissionDenied("User is not associated with an organisation.")
        serializer.save(organisation=user.organisation)

    def perform_destroy(self, instance):
        """Delete conversation and all its parts."""
        # The CASCADE relationship in ConversationPart model will handle deletion of parts
        instance.delete()


class AskQuestionView(APIView):
    """API endpoint to ask a question and get an answer from the QuestionAnswererAgent."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        question_text = request.data.get("question")
        thread_context = request.data.get("thread_context")
        conversation_id = request.data.get("conversation_id")

        user = self.request.user
        if not (hasattr(user, "organisation") and user.organisation):
            return Response(
                {"error": "User is not associated with an organisation."},
                status=status.HTTP_403_FORBIDDEN,
            )

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
            org_settings = OrganisationSettings.objects.get(
                organisation=user.organisation
            )
            agent = QuestionAnswererAgent(org_settings=org_settings)
            result = agent.run_agentic_workflow(
                question=question_text,
                thread_context=thread_context,
                conversation_id=conversation_id,
            )
            if result.get("error"):
                return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            return Response(result, status=status.HTTP_200_OK)
        except OrganisationSettings.DoesNotExist:
            return Response(
                {"error": "Organisation settings not found."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        except Exception as e:
            logger.exception(f"Unexpected error during /ask processing: {e}")
            return Response(
                {"error": f"An unexpected server error occurred: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
