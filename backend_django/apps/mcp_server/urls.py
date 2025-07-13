"""URL configuration for MCP OAuth endpoints"""

from django.urls import path
from .oauth_server import (
    AuthorizationServerMetadataView,
    DynamicClientRegistrationView,
    AuthorizationView,
    TokenView,
    OAuthCallbackView,
)
from .views import (
    MCPInfoView,
    MCPAuthorizationServerMetadataView,
    MCPProtectedResourceMetadataView,
    mcp_validate_token,
    mcp_list_models,
    mcp_search_models,
    mcp_get_model_details,
    mcp_get_project_summary,
)

app_name = "mcp_server"

urlpatterns = [
    # Note: The FastAPI MCP app mounted at /mcp now handles both GET and POST
    # requests directly. We therefore do NOT expose a Django route for /mcp.
    # Leaving this path would intercept POST requests and issue a 302 redirect,
    # stripping the request body and breaking the MCP handshake.
    # OAuth 2.0 Authorization Server Metadata (RFC 8414)
    path(
        ".well-known/oauth-authorization-server",
        AuthorizationServerMetadataView.as_view(),
        name="authorization_server_metadata",
    ),
    # OAuth 2.0 Authorization Server Metadata with MCP path
    path(
        ".well-known/oauth-authorization-server/mcp",
        MCPAuthorizationServerMetadataView.as_view(),
        name="mcp_authorization_server_metadata",
    ),
    # OAuth 2.0 Protected Resource Metadata for MCP (RFC 8707)
    path(
        ".well-known/oauth-protected-resource/mcp",
        MCPProtectedResourceMetadataView.as_view(),
        name="mcp_protected_resource_metadata",
    ),
    # OAuth 2.0 Dynamic Client Registration (RFC 7591)
    path(
        "oauth/auth/register",
        DynamicClientRegistrationView.as_view(),
        name="dynamic_client_registration",
    ),
    # OAuth 2.0 Authorization endpoint
    path("oauth/auth/authorize", AuthorizationView.as_view(), name="authorization"),
    # OAuth 2.0 Token endpoint
    path("oauth/auth/token", TokenView.as_view(), name="token"),
    # OAuth 2.0 Callback endpoint (after authentication)
    path("oauth/auth/callback", OAuthCallbackView.as_view(), name="oauth_callback"),
    # MCP API endpoints for standalone MCP server
    path("mcp/auth/validate-token/", mcp_validate_token, name="mcp_validate_token"),
    path("mcp/tools/list-models/", mcp_list_models, name="mcp_list_models"),
    path("mcp/tools/search-models/", mcp_search_models, name="mcp_search_models"),
    path(
        "mcp/tools/model-details/", mcp_get_model_details, name="mcp_get_model_details"
    ),
    path(
        "mcp/tools/project-summary/",
        mcp_get_project_summary,
        name="mcp_get_project_summary",
    ),
]
