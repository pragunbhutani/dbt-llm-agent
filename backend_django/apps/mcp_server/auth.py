"""Django-integrated authentication for MCP server."""

from typing import Any, Dict, Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from rest_framework_simplejwt.tokens import UntypedToken
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from django.contrib.auth import get_user_model
from asgiref.sync import sync_to_async

from apps.accounts.models import OrganisationSettings

User = get_user_model()

# Security
security = HTTPBearer(auto_error=False)


@sync_to_async
def _get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Get user information from JWT token."""
    try:
        # Validate the token
        UntypedToken(token)

        # Decode the token to get user info (we'll use Django's JWT handling)
        from rest_framework_simplejwt.tokens import AccessToken

        access_token = AccessToken(token)

        # Get user from token
        user = User.objects.get(id=access_token.payload["user_id"])

        # Get organization settings
        org_settings = OrganisationSettings.objects.filter(user=user).first()

        if not org_settings:
            return None

        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "organisation_id": org_settings.id,
            "organisation_name": getattr(org_settings, "name", "Default Organization"),
            "scopes": [
                "read:models",
                "search:models",
                "ask:questions",
                "read:projects",
            ],
        }

    except (InvalidToken, TokenError, User.DoesNotExist):
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Extract and validate user from JWT token."""
    if not credentials:
        return None

    try:
        return await _get_user_from_token(credentials.credentials)
    except Exception:
        return None


async def get_mcp_authenticated_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Extract and validate user from MCP OAuth token with organization linking."""
    if not credentials:
        return None

    try:
        user_data = await _get_user_from_token(credentials.credentials)
        if user_data:
            # Check if this user has a linked Claude MCP integration
            from apps.integrations.models import OrganisationIntegration

            claude_integration = OrganisationIntegration.objects.filter(
                organisation_id=user_data.get("organisation_id"),
                integration_key="claude_mcp",
                is_enabled=True,
            ).first()

            if claude_integration:
                user_data["claude_linked"] = True
                user_data["claude_client_id"] = claude_integration.credentials.get(
                    "client_id"
                )
            else:
                user_data["claude_linked"] = False

        return user_data
    except Exception:
        return None


async def require_auth(
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Require authentication for protected endpoints."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user
