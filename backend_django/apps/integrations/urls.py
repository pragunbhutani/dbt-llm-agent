from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router for API endpoints
router = DefaultRouter(trailing_slash=True)
router.register(r"integrations", views.IntegrationViewSet, basename="integration")
router.register(
    r"organisation-integrations",
    views.OrganisationIntegrationViewSet,
    basename="organisation-integration",
)
router.register(r"slack", views.SlackIntegrationViewSet, basename="slack-integration")
router.register(
    r"snowflake", views.SnowflakeIntegrationViewSet, basename="snowflake-integration"
)
router.register(
    r"metabase", views.MetabaseIntegrationViewSet, basename="metabase-integration"
)

# Define URL patterns for this app
urlpatterns = [
    # Path for Slack events
    path("slack/events/", views.slack_events_handler, name="slack_events"),
    # Include URLs for Slack-specific integrations (like commands)
    # path("slack/", include("apps.integrations.slack.urls")),
    # Add other general integration URLs here if needed
    # Include API endpoints directly (no /api/ prefix since mounted at /api/integrations/)
    path("", include(router.urls)),
]
