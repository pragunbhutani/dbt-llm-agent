from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from apps.data_sources.models import DbtProject
from apps.knowledge_base.models import Model
from apps.embeddings.models import ModelEmbedding
from apps.workflows.models import Question
from apps.accounts.models import Organisation


class DashboardStatsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        organisation = request.user.organisation
        if not organisation:
            return Response(
                {"detail": "Organisation not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        dbt_projects_count = DbtProject.objects.filter(
            organisation=organisation
        ).count()
        total_models_count = Model.objects.filter(organisation=organisation).count()
        knowledge_base_count = ModelEmbedding.objects.filter(
            organisation=organisation
        ).count()
        questions_count = Question.objects.filter(organisation=organisation).count()

        # Check for slack integration using new integration model
        slack_integrated = False
        try:
            from apps.integrations.models import OrganisationIntegration

            slack_integration = OrganisationIntegration.objects.filter(
                organisation=organisation,
                integration__key="slack",
                is_enabled=True,
                credentials__isnull=False,
            ).first()

            if slack_integration and slack_integration.credentials.get("bot_token"):
                slack_integrated = True
        except Exception:
            slack_integrated = False

        onboarding_steps = {
            "connect_dbt_project": dbt_projects_count > 0,
            "train_knowledge_base": knowledge_base_count > 0,
            "connect_to_slack": slack_integrated,
            "ask_first_question": questions_count > 0,
        }

        stats = {
            "dbt_projects_count": dbt_projects_count,
            "total_models_count": total_models_count,
            "knowledge_base_count": knowledge_base_count,
            "onboarding_steps": onboarding_steps,
        }

        return Response(stats)
