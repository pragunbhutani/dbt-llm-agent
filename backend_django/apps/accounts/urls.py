from django.urls import path
from .views import UserRegistrationView, UserView, OrganisationSettingsView

# Core URL patterns can be defined here if needed later.

# URL patterns previously in this file moved to:
# - ModelViewSet -> apps.knowledge_base.urls
# - QuestionViewSet, AskQuestionView -> apps.workflows.urls
# - ModelEmbeddingViewSet -> apps.embeddings.urls
# - slack_events_handler -> apps.integrations.urls

urlpatterns = [
    path("register/", UserRegistrationView.as_view(), name="user-registration"),
    path("me/", UserView.as_view(), name="user-me"),
    path("settings/", OrganisationSettingsView.as_view(), name="organisation-settings"),
    # Add core-specific URLs here if any
]

# --- End of file --- Ensure everything below this line is removed ---
