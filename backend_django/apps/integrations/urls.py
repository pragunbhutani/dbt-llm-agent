from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import (
    IntegrationViewSet,
    OrganisationIntegrationViewSet,
    SlackIntegrationViewSet,
    SnowflakeIntegrationViewSet,
    MetabaseIntegrationViewSet,
)
from .github.views import GitHubIntegrationViewSet, github_oauth_callback

# Create router for API endpoints
router = DefaultRouter(trailing_slash=True)
router.register(r"definitions", IntegrationViewSet, basename="integration-definitions")
router.register(
    r"organisation-integrations",
    OrganisationIntegrationViewSet,
    basename="organisation-integration",
)
router.register(r"slack", SlackIntegrationViewSet, basename="slack")
router.register(r"snowflake", SnowflakeIntegrationViewSet, basename="snowflake")
router.register(r"metabase", MetabaseIntegrationViewSet, basename="metabase")
router.register(r"github", GitHubIntegrationViewSet, basename="github")


# Define URL patterns for this app
urlpatterns = [
    path("github/callback/", github_oauth_callback, name="github-oauth-callback"),
    # Path for Slack events
    path("slack/events/", views.slack_events_handler, name="slack_events"),
    # Include URLs for Slack-specific integrations (like commands)
    # path("slack/", include("apps.integrations.slack.urls")),
    # Add other general integration URLs here if needed
    # Include API endpoints directly (no /api/ prefix since mounted at /api/integrations/)
    path("", include(router.urls)),
]
