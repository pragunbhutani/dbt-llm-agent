"""
OAuth 2.1 Authorization Server for MCP Integration
Implements the MCP authorization specification for remote server authentication.
"""

import json
import secrets
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

from django.conf import settings
from django.http import JsonResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from django.contrib.auth import get_user_model
from django.urls import reverse

from rest_framework_simplejwt.tokens import AccessToken, RefreshToken
from rest_framework_simplejwt.exceptions import InvalidToken

from apps.accounts.models import Organisation, OrganisationSettings
from apps.integrations.models import OrganisationIntegration

User = get_user_model()


class MCPAuthorizationError(Exception):
    """Custom exception for MCP authorization errors"""

    def __init__(
        self, error: str, error_description: str = None, status_code: int = 400
    ):
        self.error = error
        self.error_description = error_description
        self.status_code = status_code
        super().__init__(f"{error}: {error_description}")


class MCPOAuthServer:
    """OAuth 2.1 Authorization Server for MCP"""

    def __init__(self):
        self.authorization_base_url = getattr(
            settings, "MCP_AUTHORIZATION_BASE_URL", "http://localhost:8000"
        )
        self.issuer = f"{self.authorization_base_url}/mcp/auth"

    def get_authorization_server_metadata(self) -> Dict[str, Any]:
        """
        Return OAuth 2.0 Authorization Server Metadata (RFC 8414)
        as required by MCP specification
        """
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.authorization_base_url}/mcp/auth/authorize",
            "token_endpoint": f"{self.authorization_base_url}/mcp/auth/token",
            "registration_endpoint": f"{self.authorization_base_url}/mcp/auth/register",
            "scopes_supported": ["mcp:tools", "mcp:resources", "mcp:prompts"],
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": [
                "client_secret_basic",
                "client_secret_post",
            ],
            "claims_supported": ["sub", "iss", "aud", "exp", "iat", "organization_id"],
        }

    def generate_client_credentials(self) -> Dict[str, str]:
        """Generate client ID and secret for dynamic client registration"""
        client_id = f"mcp_client_{secrets.token_urlsafe(16)}"
        client_secret = secrets.token_urlsafe(32)
        return {"client_id": client_id, "client_secret": client_secret}

    def validate_pkce(
        self,
        code_verifier: str,
        code_challenge: str,
        code_challenge_method: str = "S256",
    ) -> bool:
        """Validate PKCE code challenge"""
        if code_challenge_method == "S256":
            computed_challenge = (
                base64.urlsafe_b64encode(
                    hashlib.sha256(code_verifier.encode()).digest()
                )
                .decode()
                .rstrip("=")
            )
            return computed_challenge == code_challenge
        return False

    def create_authorization_code(
        self,
        client_id: str,
        user: User,
        scopes: list,
        code_challenge: str,
        redirect_uri: str,
    ) -> str:
        """Create authorization code for OAuth flow"""
        code = secrets.token_urlsafe(32)

        # Store authorization code details (in production, use Redis or database)
        # For now, we'll use Django cache
        from django.core.cache import cache

        code_data = {
            "client_id": client_id,
            "user_id": str(user.id),
            "scopes": scopes,
            "code_challenge": code_challenge,
            "redirect_uri": redirect_uri,
            "expires_at": (datetime.utcnow() + timedelta(minutes=10)).isoformat(),
            "organization_id": str(user.organisation.id) if user.organisation else None,
        }

        cache.set(f"mcp_auth_code:{code}", code_data, timeout=600)  # 10 minutes
        return code

    def exchange_code_for_tokens(
        self,
        code: str,
        client_id: str,
        client_secret: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens"""
        from django.core.cache import cache

        # Retrieve and validate authorization code
        code_data = cache.get(f"mcp_auth_code:{code}")
        if not code_data:
            raise MCPAuthorizationError(
                "invalid_grant", "Authorization code expired or invalid"
            )

        # Validate client credentials
        if code_data["client_id"] != client_id:
            raise MCPAuthorizationError("invalid_client", "Client ID mismatch")

        # Validate PKCE
        if not self.validate_pkce(code_verifier, code_data["code_challenge"]):
            raise MCPAuthorizationError("invalid_grant", "PKCE validation failed")

        # Validate redirect URI
        if code_data["redirect_uri"] != redirect_uri:
            raise MCPAuthorizationError("invalid_grant", "Redirect URI mismatch")

        # Get user
        try:
            user = User.objects.get(id=code_data["user_id"])
        except User.DoesNotExist:
            raise MCPAuthorizationError("invalid_grant", "User not found")

        # Create tokens with MCP-specific claims
        access_token = AccessToken()
        access_token["user_id"] = str(user.id)
        access_token["client_id"] = client_id
        access_token["scopes"] = code_data["scopes"]
        access_token["organization_id"] = code_data.get("organization_id")

        refresh_token = RefreshToken()
        refresh_token["user_id"] = str(user.id)
        refresh_token["client_id"] = client_id
        refresh_token["scopes"] = code_data["scopes"]
        refresh_token["organization_id"] = code_data.get("organization_id")

        # Delete used authorization code
        cache.delete(f"mcp_auth_code:{code}")

        # Link Claude.ai account to organization if not already linked
        self.link_claude_account_to_organization(user, client_id)

        return {
            "access_token": str(access_token),
            "refresh_token": str(refresh_token),
            "token_type": "Bearer",
            "expires_in": access_token.token.payload.get("exp")
            - int(datetime.utcnow().timestamp()),
            "scope": " ".join(code_data["scopes"]),
        }

    def link_claude_account_to_organization(self, user: User, client_id: str):
        """Link Claude.ai account to ragstar organization"""
        if not user.organisation:
            return

        # Check if Claude integration already exists for this organization
        integration, created = OrganisationIntegration.objects.get_or_create(
            organisation=user.organisation,
            integration_key="claude_mcp",
            defaults={
                "is_enabled": True,
                "configuration": {
                    "server_url": f"{self.authorization_base_url}/mcp",
                    "auth_provider": "claude",
                    "linked_at": datetime.utcnow().isoformat(),
                },
                "credentials": {"client_id": client_id, "linked_user_id": str(user.id)},
            },
        )

        if not created:
            # Update existing integration
            integration.credentials["linked_user_id"] = str(user.id)
            integration.credentials["last_linked_at"] = datetime.utcnow().isoformat()
            integration.save()


# Views for OAuth endpoints


@method_decorator(csrf_exempt, name="dispatch")
class AuthorizationServerMetadataView(View):
    """OAuth 2.0 Authorization Server Metadata endpoint"""

    def get(self, request):
        oauth_server = MCPOAuthServer()
        return JsonResponse(oauth_server.get_authorization_server_metadata())


@method_decorator(csrf_exempt, name="dispatch")
class DynamicClientRegistrationView(View):
    """OAuth 2.0 Dynamic Client Registration endpoint"""

    def post(self, request):
        try:
            data = json.loads(request.body)
            oauth_server = MCPOAuthServer()

            # Validate required fields
            if not data.get("client_name"):
                raise MCPAuthorizationError(
                    "invalid_request", "client_name is required"
                )

            if not data.get("redirect_uris"):
                raise MCPAuthorizationError(
                    "invalid_request", "redirect_uris is required"
                )

            # Generate client credentials
            credentials = oauth_server.generate_client_credentials()

            # Store client registration (in production, use database)
            from django.core.cache import cache

            client_data = {
                "client_name": data["client_name"],
                "redirect_uris": data["redirect_uris"],
                "grant_types": data.get("grant_types", ["authorization_code"]),
                "response_types": data.get("response_types", ["code"]),
                "scope": data.get("scope", "mcp:tools mcp:resources mcp:prompts"),
                "created_at": datetime.utcnow().isoformat(),
            }

            cache.set(
                f"mcp_client:{credentials['client_id']}",
                client_data,
                timeout=86400 * 30,
            )  # 30 days

            return JsonResponse(
                {
                    "client_id": credentials["client_id"],
                    "client_secret": credentials["client_secret"],
                    "client_name": data["client_name"],
                    "redirect_uris": data["redirect_uris"],
                    "grant_types": client_data["grant_types"],
                    "response_types": client_data["response_types"],
                    "scope": client_data["scope"],
                }
            )

        except json.JSONDecodeError:
            return JsonResponse(
                {"error": "invalid_request", "error_description": "Invalid JSON"},
                status=400,
            )
        except MCPAuthorizationError as e:
            return JsonResponse(
                {"error": e.error, "error_description": e.error_description},
                status=e.status_code,
            )


@method_decorator(csrf_exempt, name="dispatch")
class AuthorizationView(View):
    """OAuth 2.0 Authorization endpoint"""

    def get(self, request):
        # Extract parameters
        client_id = request.GET.get("client_id")
        redirect_uri = request.GET.get("redirect_uri")
        response_type = request.GET.get("response_type")
        scope = request.GET.get("scope", "").split()
        code_challenge = request.GET.get("code_challenge")
        code_challenge_method = request.GET.get("code_challenge_method", "S256")
        state = request.GET.get("state")

        # Validate required parameters
        if not all([client_id, redirect_uri, response_type, code_challenge]):
            return JsonResponse(
                {
                    "error": "invalid_request",
                    "error_description": "Missing required parameters",
                },
                status=400,
            )

        if response_type != "code":
            return JsonResponse(
                {
                    "error": "unsupported_response_type",
                    "error_description": "Only 'code' response type is supported",
                },
                status=400,
            )

        # Validate client exists
        from django.core.cache import cache

        client_data = cache.get(f"mcp_client:{client_id}")
        if not client_data:
            return JsonResponse(
                {"error": "invalid_client", "error_description": "Client not found"},
                status=400,
            )

        # Validate redirect URI
        if redirect_uri not in client_data["redirect_uris"]:
            return JsonResponse(
                {
                    "error": "invalid_request",
                    "error_description": "Invalid redirect_uri",
                },
                status=400,
            )

        # Check if user is authenticated
        if not request.user.is_authenticated:
            # Redirect to login with return URL
            login_url = reverse("signin")
            return_url = request.build_absolute_uri()
            return HttpResponseRedirect(f"{login_url}?next={return_url}")

        # Generate authorization code
        oauth_server = MCPOAuthServer()
        auth_code = oauth_server.create_authorization_code(
            client_id=client_id,
            user=request.user,
            scopes=scope,
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
        )

        # Redirect back to client with authorization code
        redirect_params = f"code={auth_code}"
        if state:
            redirect_params += f"&state={state}"

        separator = "&" if "?" in redirect_uri else "?"
        return HttpResponseRedirect(f"{redirect_uri}{separator}{redirect_params}")


@method_decorator(csrf_exempt, name="dispatch")
class TokenView(View):
    """OAuth 2.0 Token endpoint"""

    def post(self, request):
        try:
            grant_type = request.POST.get("grant_type")

            if grant_type == "authorization_code":
                return self._handle_authorization_code_grant(request)
            elif grant_type == "refresh_token":
                return self._handle_refresh_token_grant(request)
            else:
                raise MCPAuthorizationError(
                    "unsupported_grant_type", f"Grant type '{grant_type}' not supported"
                )

        except MCPAuthorizationError as e:
            return JsonResponse(
                {"error": e.error, "error_description": e.error_description},
                status=e.status_code,
            )

    def _handle_authorization_code_grant(self, request):
        """Handle authorization code grant"""
        code = request.POST.get("code")
        client_id = request.POST.get("client_id")
        client_secret = request.POST.get("client_secret")
        redirect_uri = request.POST.get("redirect_uri")
        code_verifier = request.POST.get("code_verifier")

        if not all([code, client_id, client_secret, redirect_uri, code_verifier]):
            raise MCPAuthorizationError(
                "invalid_request", "Missing required parameters"
            )

        oauth_server = MCPOAuthServer()
        tokens = oauth_server.exchange_code_for_tokens(
            code=code,
            client_id=client_id,
            client_secret=client_secret,
            code_verifier=code_verifier,
            redirect_uri=redirect_uri,
        )

        return JsonResponse(tokens)

    def _handle_refresh_token_grant(self, request):
        """Handle refresh token grant"""
        refresh_token = request.POST.get("refresh_token")
        client_id = request.POST.get("client_id")
        client_secret = request.POST.get("client_secret")

        if not all([refresh_token, client_id, client_secret]):
            raise MCPAuthorizationError(
                "invalid_request", "Missing required parameters"
            )

        try:
            # Validate refresh token
            token = RefreshToken(refresh_token)

            # Validate client
            if token["client_id"] != client_id:
                raise MCPAuthorizationError("invalid_client", "Client ID mismatch")

            # Generate new access token
            access_token = AccessToken()
            access_token["user_id"] = token["user_id"]
            access_token["client_id"] = client_id
            access_token["scopes"] = token["scopes"]
            access_token["organization_id"] = token.get("organization_id")

            return JsonResponse(
                {
                    "access_token": str(access_token),
                    "token_type": "Bearer",
                    "expires_in": access_token.token.payload.get("exp")
                    - int(datetime.utcnow().timestamp()),
                    "scope": " ".join(token["scopes"]),
                }
            )

        except InvalidToken:
            raise MCPAuthorizationError("invalid_grant", "Invalid refresh token")
