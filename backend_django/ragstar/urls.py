"""
URL configuration for ragstar_django project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include, re_path
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from django.conf import settings
from django.conf.urls.static import static
from apps.integrations.views import slack_events_handler
from apps.integrations.slack.views import slack_shortcut_view


# Create CSRF-exempt token refresh view
@method_decorator(csrf_exempt, name="dispatch")
class CSRFExemptTokenRefreshView(TokenRefreshView):
    pass


urlpatterns = [
    path("admin/", admin.site.urls),
    # API URLs - All internal APIs under /api/
    path("api/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path(
        "api/token/refresh/", CSRFExemptTokenRefreshView.as_view(), name="token_refresh"
    ),
    path("api/accounts/", include("apps.accounts.urls")),
    path("api/knowledge_base/", include("apps.knowledge_base.urls")),
    path("api/workflows/", include("apps.workflows.urls")),
    path("api/embeddings/", include("apps.embeddings.urls")),
    path("api/data_sources/", include("apps.data_sources.urls")),
    path("api/waitlist/", include("apps.waitlist.urls")),
    # External webhook endpoints (placed BEFORE the generic integrations include so
    # they take precedence). The regex allows both with and without the trailing
    # slash to prevent CSRF issues when Slack calls the URL without it.
    re_path(
        r"^api/integrations/slack/events/?$", slack_events_handler, name="slack_events"
    ),
    re_path(
        r"^api/integrations/slack/shortcuts/?$",
        slack_shortcut_view,
        name="slack_shortcuts",
    ),
    # Generic integrations API (internal, authenticated)
    path("api/integrations/", include("apps.integrations.urls")),
    # MCP Server OAuth endpoints - Available at both root and /api/ for compatibility
    path("api/", include("apps.mcp_server.urls")),  # For FastAPI redirects
    path("", include("apps.mcp_server.urls")),  # For direct access
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
