"""
Django-based OAuth 2.0 Authorization Server for MCP Integration
Ported from the working FastMCP implementation with database persistence.
"""

import json
import secrets
import base64
import hashlib
from datetime import datetime, timedelta, timezone
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
from django.core.exceptions import ValidationError

from rest_framework_simplejwt.tokens import AccessToken, RefreshToken
from rest_framework_simplejwt.exceptions import InvalidToken
from jose import jwt

from apps.accounts.models import Organisation
from apps.integrations.models import (
    OrganisationIntegration,
    MCPOAuthClient,
    MCPOAuthAuthorizationCode,
    MCPOAuthAuthorizationRequest,
)

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
    """OAuth 2.0 Authorization Server for MCP - Django implementation with database persistence"""

    def __init__(self):
        self.authorization_base_url = getattr(
            settings, "MCP_AUTHORIZATION_BASE_URL", "http://localhost:8000"
        )
        self.issuer = f"{self.authorization_base_url}"

    def get_authorization_server_metadata(self) -> Dict[str, Any]:
        """
        Return OAuth 2.0 Authorization Server Metadata (RFC 8414)
        as required by MCP specification
        """
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.authorization_base_url}/oauth/auth/authorize",
            "token_endpoint": f"{self.authorization_base_url}/oauth/auth/token",
            "registration_endpoint": f"{self.authorization_base_url}/oauth/auth/register",
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
        client: MCPOAuthClient,
        user: User,
        scopes: list,
        code_challenge: str,
        redirect_uri: str,
    ) -> str:
        """Create authorization code for OAuth flow"""
        user_data = {
            "user_id": str(user.id),
            "email": str(user.email),
            "organization_id": str(user.organisation.id) if user.organisation else None,
        }

        auth_code = MCPOAuthAuthorizationCode.create_code(
            client=client,
            user_data=user_data,
            scopes=scopes,
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
        )

        print(f"[OAuth] Created authorization code for client: {client.client_id}")
        return auth_code.code

    def exchange_code_for_tokens(
        self,
        code: str,
        client_id: str,
        client_secret: str,
        code_verifier: str,
        redirect_uri: str,
    ) -> Dict[str, Any]:
        """Exchange authorization code for access and refresh tokens"""
        # Retrieve and validate authorization code
        try:
            code_obj = MCPOAuthAuthorizationCode.objects.get(code=code)
        except MCPOAuthAuthorizationCode.DoesNotExist:
            print(f"[OAuth] Invalid or expired authorization code: {code}")
            raise MCPAuthorizationError(
                "invalid_grant", "Authorization code expired or invalid"
            )

        # Check expiry
        if code_obj.is_expired:
            print(f"[OAuth] Authorization code expired: {code}")
            code_obj.delete()
            raise MCPAuthorizationError("invalid_grant", "Authorization code expired")

        # Validate client credentials
        client = code_obj.client
        if client.client_id != client_id:
            print(f"[OAuth] Client ID mismatch during token exchange: {client_id}")
            raise MCPAuthorizationError("invalid_client", "Client ID mismatch")

        # For auto-registered clients, handle client_secret validation differently
        if client.auto_registered:
            # For auto-registered public clients, client_secret is optional
            # We rely on PKCE for security instead
            print(f"[OAuth] Auto-registered client token exchange: {client_id}")
        else:
            # For manually registered clients, validate client_secret
            if client.client_secret != client_secret:
                print(
                    f"[OAuth] Invalid client secret for manually registered client: {client_id}"
                )
                raise MCPAuthorizationError(
                    "invalid_client", "Invalid client credentials"
                )

        # Validate PKCE (this is the main security mechanism for public clients)
        if not self.validate_pkce(code_verifier, code_obj.code_challenge):
            print(f"[OAuth] PKCE validation failed for client: {client_id}")
            raise MCPAuthorizationError("invalid_grant", "PKCE validation failed")

        # Validate redirect URI
        if code_obj.redirect_uri != redirect_uri:
            print(f"[OAuth] Redirect URI mismatch for client: {client_id}")
            raise MCPAuthorizationError("invalid_grant", "Invalid redirect URI")

        # Create JWT tokens with user claims
        user_data = code_obj.user_data
        now = datetime.now(timezone.utc)

        # Create access token payload
        access_token_payload = {
            "user_id": user_data["user_id"],
            "organization_id": user_data["organization_id"],
            "email": user_data["email"],
            "scopes": code_obj.scopes,
            "client_id": client_id,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
            "iss": self.issuer,
            "aud": client_id,
        }

        # Get JWT secret from Django settings
        jwt_secret = getattr(settings, "SECRET_KEY", "devsecret")
        jwt_algorithm = "HS256"

        access_token = jwt.encode(
            access_token_payload,
            jwt_secret,
            algorithm=jwt_algorithm,
        )

        # Create refresh token payload
        refresh_token_payload = {
            "user_id": user_data["user_id"],
            "organization_id": user_data["organization_id"],
            "email": user_data["email"],
            "scopes": code_obj.scopes,
            "client_id": client_id,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=7)).timestamp()),
            "iss": self.issuer,
            "aud": client_id,
            "token_type": "refresh",
        }

        refresh_token = jwt.encode(
            refresh_token_payload,
            jwt_secret,
            algorithm=jwt_algorithm,
        )

        # Delete used authorization code
        code_obj.delete()

        # Link Claude.ai account to organization if not already linked
        self.link_claude_account_to_organization(user_data, client_id)

        print(f"[OAuth] Successfully exchanged code for tokens: {client_id}")

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 3600,  # 1 hour
            "scope": " ".join(code_obj.scopes),
        }

    def link_claude_account_to_organization(
        self, user_data: Dict[str, Any], client_id: str
    ):
        """Link Claude.ai account to ragstar organization"""
        if not user_data.get("organization_id"):
            return

        try:
            organisation = Organisation.objects.get(id=user_data["organization_id"])
        except Organisation.DoesNotExist:
            print(f"[OAuth] Organization not found: {user_data['organization_id']}")
            return

        # Check if Claude integration already exists for this organization
        integration, created = OrganisationIntegration.objects.get_or_create(
            organisation=organisation,
            integration_key="claude_mcp",
            defaults={
                "is_enabled": True,
                "configuration": {
                    "server_url": f"{self.authorization_base_url}/mcp",
                    "auth_provider": "claude",
                    "linked_at": datetime.now(timezone.utc).isoformat(),
                },
            },
        )

        # Store client_id and user info in credentials
        credentials = {
            "client_id": client_id,
            "linked_user_id": user_data["user_id"],
        }

        if not created:
            # Update existing integration
            credentials["last_linked_at"] = datetime.now(timezone.utc).isoformat()

        integration.set_credentials(credentials)

        print(
            f"[OAuth] {'Created' if created else 'Updated'} Claude MCP integration for organization: {organisation.name}"
        )


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

            # Create client in database
            client = MCPOAuthClient.create_client(
                client_name=data["client_name"],
                redirect_uris=data["redirect_uris"],
                grant_types=data.get("grant_types", ["authorization_code"]),
                response_types=data.get("response_types", ["code"]),
                scope=data.get("scope", "mcp:tools mcp:resources mcp:prompts"),
                auto_registered=False,
            )

            print(
                f"[OAuth] Registered new client: {client.client_id} ({client.client_name})"
            )

            return JsonResponse(
                {
                    "client_id": client.client_id,
                    "client_secret": client.client_secret,
                    "client_name": client.client_name,
                    "redirect_uris": client.redirect_uris,
                    "grant_types": client.grant_types,
                    "response_types": client.response_types,
                    "scope": client.scope,
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

        print(f"[OAuth] Authorization request received from client: {client_id}")

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

        if code_challenge_method != "S256":
            return JsonResponse(
                {
                    "error": "invalid_request",
                    "error_description": "Only 'S256' code challenge method is supported",
                },
                status=400,
            )

        # Check if client exists, if not auto-register it (like the working FastMCP version)
        try:
            client = MCPOAuthClient.objects.get(client_id=client_id)
            print(f"[OAuth] Client found: {client.client_name}")
        except MCPOAuthClient.DoesNotExist:
            print(f"[OAuth] Client not found: {client_id}, auto-registering...")

            # Auto-register client with reasonable defaults (same logic as FastMCP)
            client_name = "Claude.ai MCP Client"
            if "claude.ai" in redirect_uri:
                client_name = "Claude.ai MCP Client"
            elif "openai.com" in redirect_uri or "chatgpt.com" in redirect_uri:
                client_name = "ChatGPT MCP Client"
            elif "gemini.google.com" in redirect_uri:
                client_name = "Gemini MCP Client"
            else:
                client_name = f"MCP Client ({redirect_uri})"

            # Create client with the provided client_id
            client = MCPOAuthClient.objects.create(
                client_id=client_id,
                client_secret=secrets.token_urlsafe(32),
                client_name=client_name,
                redirect_uris=[redirect_uri],
                grant_types=["authorization_code"],
                response_types=["code"],
                scope=(
                    " ".join(scope) if scope else "mcp:tools mcp:resources mcp:prompts"
                ),
                auto_registered=True,
            )

            print(f"[OAuth] Auto-registered client: {client_id} ({client_name})")

        # Validate redirect URI
        if not client.is_redirect_uri_valid(redirect_uri):
            print(f"[OAuth] Invalid redirect_uri: {redirect_uri}")
            print(f"[OAuth] Allowed redirect_uris: {client.redirect_uris}")
            return JsonResponse(
                {
                    "error": "invalid_request",
                    "error_description": "Invalid redirect_uri",
                },
                status=400,
            )

        # Store authorization request for after authentication
        auth_request = MCPOAuthAuthorizationRequest.create_request(
            client_id=client_id,
            redirect_uri=redirect_uri,
            response_type=response_type,
            scope=" ".join(scope) if scope else "mcp:tools mcp:resources mcp:prompts",
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            state=state,
        )

        print(f"[OAuth] Created auth request: {auth_request.request_id}")

        # Check if user is authenticated
        if not request.user.is_authenticated:
            # Redirect to login with return URL (same as FastMCP logic)
            frontend_login_url = (
                getattr(settings, "NEXTAUTH_URL", "http://localhost:3000/")
                + "oauth-continue"
            )
            authorization_base_url = getattr(
                settings, "MCP_AUTHORIZATION_BASE_URL", "http://localhost:8000"
            )
            return_url = f"{authorization_base_url}/oauth/auth/callback?auth_request_id={auth_request.request_id}"

            print(
                f"[OAuth] Redirecting to frontend OAuth continuation: {frontend_login_url}"
            )
            return HttpResponseRedirect(f"{frontend_login_url}?next={return_url}")

        # User is authenticated, create authorization code directly
        oauth_server = MCPOAuthServer()
        auth_code = oauth_server.create_authorization_code(
            client=client,
            user=request.user,
            scopes=scope if scope else ["mcp:tools", "mcp:resources", "mcp:prompts"],
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
        )

        # Clean up auth request
        auth_request.delete()

        # Redirect back to client with authorization code
        redirect_params = f"code={auth_code}"
        if state:
            redirect_params += f"&state={state}"

        separator = "&" if "?" in redirect_uri else "?"
        return HttpResponseRedirect(f"{redirect_uri}{separator}{redirect_params}")


@method_decorator(csrf_exempt, name="dispatch")
class OAuthCallbackView(View):
    """OAuth callback endpoint to complete the authorization flow after Django authentication"""

    def get(self, request):
        auth_request_id = request.GET.get("auth_request_id")
        user_token = request.GET.get("user_token")

        if not auth_request_id:
            return JsonResponse(
                {
                    "error": "invalid_request",
                    "error_description": "Missing auth_request_id",
                },
                status=400,
            )

        # Retrieve stored authorization request
        try:
            auth_request = MCPOAuthAuthorizationRequest.objects.get(
                request_id=auth_request_id
            )
        except MCPOAuthAuthorizationRequest.DoesNotExist:
            return JsonResponse(
                {
                    "error": "invalid_request",
                    "error_description": "Invalid or expired authorization request",
                },
                status=400,
            )

        # Check if auth request has expired
        if auth_request.is_expired:
            auth_request.delete()
            return JsonResponse(
                {
                    "error": "invalid_request",
                    "error_description": "Authorization request has expired",
                },
                status=400,
            )

        # If no user token provided but user is authenticated, get user from session
        if not user_token and request.user.is_authenticated:
            user = request.user
        elif user_token:
            # Verify user token (same logic as FastMCP)
            try:
                from rest_framework_simplejwt.tokens import UntypedToken

                UntypedToken(user_token)

                from rest_framework_simplejwt.tokens import AccessToken

                access_token = AccessToken(user_token)
                user = User.objects.get(id=access_token.payload["user_id"])
            except Exception:
                return JsonResponse(
                    {
                        "error": "invalid_request",
                        "error_description": "Invalid user authentication",
                    },
                    status=401,
                )
        else:
            return JsonResponse(
                {
                    "error": "invalid_request",
                    "error_description": "User authentication required",
                },
                status=400,
            )

        # Get client
        try:
            client = MCPOAuthClient.objects.get(client_id=auth_request.client_id)
        except MCPOAuthClient.DoesNotExist:
            return JsonResponse(
                {"error": "invalid_client", "error_description": "Client not found"},
                status=400,
            )

        # Create authorization code
        oauth_server = MCPOAuthServer()
        auth_code = oauth_server.create_authorization_code(
            client=client,
            user=user,
            scopes=auth_request.scope.split(),
            code_challenge=auth_request.code_challenge,
            redirect_uri=auth_request.redirect_uri,
        )

        # Clean up auth request
        auth_request.delete()

        # Redirect back to client with authorization code
        redirect_params = f"code={auth_code}"
        if auth_request.state:
            redirect_params += f"&state={auth_request.state}"

        separator = "&" if "?" in auth_request.redirect_uri else "?"
        return HttpResponseRedirect(
            f"{auth_request.redirect_uri}{separator}{redirect_params}"
        )


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

        if not all([code, client_id, redirect_uri, code_verifier]):
            raise MCPAuthorizationError(
                "invalid_request", "Missing required parameters"
            )

        # Check if this is an auto-registered client (same logic as FastMCP)
        try:
            client = MCPOAuthClient.objects.get(client_id=client_id)
        except MCPOAuthClient.DoesNotExist:
            raise MCPAuthorizationError("invalid_client", "Client not found")

        if client.auto_registered:
            # For auto-registered clients, client_secret is optional
            effective_client_secret = client_secret or "auto_registered_client"
        else:
            # For manually registered clients, client_secret is required
            if not client_secret:
                raise MCPAuthorizationError(
                    "invalid_request", "client_secret is required"
                )
            effective_client_secret = client_secret

        oauth_server = MCPOAuthServer()
        tokens = oauth_server.exchange_code_for_tokens(
            code=code,
            client_id=client_id,
            client_secret=effective_client_secret,
            code_verifier=code_verifier,
            redirect_uri=redirect_uri,
        )

        return JsonResponse(tokens)

    def _handle_refresh_token_grant(self, request):
        """Handle refresh token grant"""
        # TODO: Implement refresh token logic
        raise MCPAuthorizationError(
            "unsupported_grant_type", "Refresh token not yet implemented"
        )
