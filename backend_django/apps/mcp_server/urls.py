"""URL configuration for MCP OAuth endpoints"""

from django.urls import path
from apps.integrations.oauth_server import (
    AuthorizationServerMetadataView,
    DynamicClientRegistrationView,
    AuthorizationView,
    TokenView,
)

app_name = "mcp_server"

urlpatterns = [
    # OAuth 2.0 Authorization Server Metadata (RFC 8414)
    path(
        ".well-known/oauth-authorization-server",
        AuthorizationServerMetadataView.as_view(),
        name="authorization_server_metadata",
    ),
    # OAuth 2.0 Dynamic Client Registration (RFC 7591)
    path(
        "mcp/auth/register",
        DynamicClientRegistrationView.as_view(),
        name="dynamic_client_registration",
    ),
    # OAuth 2.0 Authorization endpoint
    path("mcp/auth/authorize", AuthorizationView.as_view(), name="authorization"),
    # OAuth 2.0 Token endpoint
    path("mcp/auth/token", TokenView.as_view(), name="token"),
]
