from django.urls import path, re_path
from .views import (
    UserRegistrationView,
    CustomTokenObtainPairView,
    UserView,
    OrganisationSettingsView,
)
from rest_framework_simplejwt.views import TokenRefreshView

# Core URL patterns can be defined here if needed later.

# URL patterns previously in this file moved to:
# - ModelViewSet -> apps.knowledge_base.urls
# - QuestionViewSet, AskQuestionView -> apps.workflows.urls
# - ModelEmbeddingViewSet -> apps.embeddings.urls
# - slack_events_handler -> apps.integrations.urls

urlpatterns = [
    re_path(r"^register/?$", UserRegistrationView.as_view(), name="user_register"),
    path("token/", CustomTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("me/", UserView.as_view(), name="user_me"),
    path("settings/", OrganisationSettingsView.as_view(), name="organisation_settings"),
    # Add core-specific URLs here if any
]

# --- End of file --- Ensure everything below this line is removed ---
