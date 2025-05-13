from django.urls import path, include

# Import the new async view
from .views import slack_events_handler

# Define URL patterns for this app
urlpatterns = [
    # Path for Slack events
    path("slack/events", slack_events_handler, name="slack_events"),
    # Include URLs for Slack-specific integrations (like commands)
    path("slack/", include("apps.integrations.slack.urls")),
    # Add other general integration URLs here if needed
]
