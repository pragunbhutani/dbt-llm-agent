from django.urls import path

# Import the new async view
from .views import slack_events_handler

# Define URL patterns for this app
urlpatterns = [
    # Point the path to the async view function
    path("slack/events", slack_events_handler, name="slack_events"),
    # Add other integration URLs here if needed
]
