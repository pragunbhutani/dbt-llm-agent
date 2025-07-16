"""Django-integrated authentication for MCP server."""

from typing import Any, Dict, Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from django.contrib.auth import get_user_model
from django.conf import settings
from asgiref.sync import sync_to_async
from jose import jwt, JWTError
from datetime import datetime, timezone

User = get_user_model()

# Security
security = HTTPBearer(auto_error=False)


@sync_to_async
def _get_user_from_oauth_token(token: str) -> Optional[Dict[str, Any]]:
    """Get user information from OAuth JWT token."""
    try:
        # Decode the JWT token using the same secret as the OAuth server
        jwt_secret = getattr(settings, "SECRET_KEY", "devsecret")
        jwt_algorithm = "HS256"

        payload = jwt.decode(token, jwt_secret, algorithms=[jwt_algorithm])

        # Check token expiration
        exp_ts = payload.get("exp")
        if exp_ts:
            exp_time = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
            if exp_time < datetime.now(timezone.utc):
                print(f"[MCP Auth] Token expired at {exp_time}")
                return None

        # Extract user information from JWT payload
        user_id = payload.get("user_id")
        organization_id = payload.get("organization_id")
        email = payload.get("email")
        scopes = payload.get("scopes", [])
        client_id = payload.get("client_id")

        if not user_id or not organization_id:
            print(f"[MCP Auth] Missing required claims in token")
            return None

        # Verify user exists
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            print(f"[MCP Auth] User {user_id} not found")
            return None

        # Check if Claude integration exists for this organization
        from apps.integrations.models import OrganisationIntegration

        claude_integration = OrganisationIntegration.objects.filter(
            organisation_id=organization_id,
            integration_key="claude_mcp",
            is_enabled=True,
        ).first()

        user_data = {
            "user_id": user_id,
            "username": user.username,
            "email": email,
            "organisation_id": organization_id,
            "organisation_name": (
                user.organisation.name if user.organisation else "Unknown"
            ),
            "scopes": scopes,
            "client_id": client_id,
            "access_token": token,  # Store the token for API calls
        }

        if claude_integration:
            user_data["claude_linked"] = True
            user_data["claude_client_id"] = claude_integration.credentials.get(
                "client_id"
            )
        else:
            user_data["claude_linked"] = False

        print(
            f"[MCP Auth] Successfully authenticated user {user_id} for organization {organization_id}"
        )
        return user_data

    except JWTError as e:
        print(f"[MCP Auth] JWT decode error: {e}")
        return None
    except Exception as e:
        print(f"[MCP Auth] Authentication error: {e}")
        return None


async def get_mcp_authenticated_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Extract and validate user from MCP OAuth token."""
    if not credentials:
        return None

    try:
        user_data = await _get_user_from_oauth_token(credentials.credentials)
        return user_data
    except Exception as e:
        print(f"[MCP Auth] Error getting authenticated user: {e}")
        return None


async def require_auth(
    user: Optional[Dict[str, Any]] = Depends(get_mcp_authenticated_user),
) -> Dict[str, Any]:
    """Require authentication for protected endpoints."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user
