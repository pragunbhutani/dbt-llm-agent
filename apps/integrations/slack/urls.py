from django.urls import path
from . import views  # We will create views.py next or add to existing

urlpatterns = [
    path(
        "commands/execute-sql/",
        views.execute_sql_command_view,
        name="slack_command_execute_sql",
    ),
    # Add other Slack command URLs here
    path(
        "shortcuts/",
        views.slack_shortcut_view,
        name="slack_shortcut",
    ),
]
