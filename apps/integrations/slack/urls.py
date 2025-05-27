from django.urls import path
from . import views  # We will create views.py next or add to existing

urlpatterns = [
    # Add other Slack command URLs here (ensure any command views exist or remove the path)
    path(
        "shortcuts/",
        views.slack_shortcut_view,
        name="slack_shortcut",
    ),
]
