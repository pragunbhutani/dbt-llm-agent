from django.shortcuts import render
from django.db import transaction
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import DbtProject
from .serializers import DbtProjectSerializer, DbtCloudProjectCreateSerializer
from rest_framework.permissions import IsAuthenticated
from .services import initialize_dbt_cloud_project
import logging

logger = logging.getLogger(__name__)

# Create your views here.


class DbtProjectViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows dbt projects to be viewed or edited.
    """

    queryset = DbtProject.objects.all()
    serializer_class = DbtProjectSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """
        This view should return a list of all the dbt projects
        for the currently authenticated user's organisation.
        """
        user = self.request.user
        return DbtProject.objects.for_organisation(user.organisation)

    def perform_create(self, serializer):
        """
        Automatically associate the dbt project with the user's organisation.
        """
        serializer.save(organisation=self.request.user.organisation)

    def perform_destroy(self, instance):
        """Delete project and cascade-clean related models & embeddings."""
        from apps.knowledge_base.models import Model
        from apps.embeddings.models import ModelEmbedding

        ModelEmbedding.objects.filter(dbt_project=instance).delete()
        Model.objects.filter(dbt_project=instance).delete()
        super().perform_destroy(instance)

    @action(
        detail=False, methods=["post"], serializer_class=DbtCloudProjectCreateSerializer
    )
    def create_dbt_cloud_project(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated_data = serializer.validated_data

        try:
            with transaction.atomic():
                project = DbtProject.objects.create(
                    name=validated_data.get("name") or "dbt Cloud Project",
                    organisation=request.user.organisation,
                    connection_type=DbtProject.ConnectionType.DBT_CLOUD,
                    dbt_cloud_url=validated_data["dbt_cloud_url"],
                    dbt_cloud_account_id=validated_data["dbt_cloud_account_id"],
                )

                # Store the sensitive API key securely in Parameter Store
                project.set_credentials(
                    {
                        "dbt_cloud_api_key": validated_data["dbt_cloud_api_key"],
                    }
                )

                initialize_dbt_cloud_project(
                    dbt_project=project,
                    organisation=request.user.organisation,
                )
            return Response(
                DbtProjectSerializer(project).data, status=status.HTTP_201_CREATED
            )
        except Exception as e:
            logger.error(f"Failed to create dbt Cloud project: {e}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=["post"])
    def create_github_project(self, request):
        """
        Create a new dbt project from a GitHub repository.
        """
        # TODO: Add a serializer for this action
        name = request.data.get("name")
        github_repository_url = request.data.get("github_repository_url")
        github_branch = request.data.get("github_branch")
        github_project_folder = request.data.get("github_project_folder")

        if not all([name, github_repository_url, github_branch]):
            return Response(
                {"error": "Missing required fields"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            with transaction.atomic():
                project = DbtProject.objects.create(
                    name=name,
                    organisation=request.user.organisation,
                    connection_type=DbtProject.ConnectionType.GITHUB,
                    github_repository_url=github_repository_url,
                    github_branch=github_branch,
                    github_project_folder=github_project_folder,
                )

                from .tasks import parse_github_project_task

                parse_github_project_task.delay(project.id)

            return Response(
                DbtProjectSerializer(project).data, status=status.HTTP_201_CREATED
            )
        except Exception as e:
            logger.error(f"Failed to create GitHub project: {e}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=["post"])
    def refresh(self, request, pk=None):
        """
        Refresh a dbt project by re-fetching and processing its models.
        """
        project = self.get_object()

        try:
            # Re-initialize the project to refresh its models
            results = initialize_dbt_cloud_project(
                dbt_project=project,
                organisation=request.user.organisation,
            )

            # Update the project's updated_at timestamp
            project.save()

            return Response(
                {"message": "Project refreshed successfully", "results": results},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.error(
                f"Failed to refresh dbt project {project.id}: {e}", exc_info=True
            )
            return Response(
                {"error": f"Failed to refresh project: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
