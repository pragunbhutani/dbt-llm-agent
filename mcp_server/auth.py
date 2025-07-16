"""Standalone authentication module for the MCP FastAPI service.

This module validates JWT tokens via HTTP calls to the Django backend.
"""

import os
from typing import Any, Dict, Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx

# Configuration
DJANGO_BACKEND_URL = os.environ.get("DJANGO_BACKEND_URL", "http://localhost:8000")

# Security
security = HTTPBearer(auto_error=False)


async def _validate_token_with_backend(token: str) -> Optional[Dict[str, Any]]:
    """Validate token by calling Django backend API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DJANGO_BACKEND_URL}/api/mcp/auth/validate-token/",
                json={"token": token},
                timeout=10.0,
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"[MCP Auth] Token validation failed: {response.status_code}")
                return None

    except Exception as e:
        print(f"[MCP Auth] Error validating token with backend: {e}")
        return None


async def get_mcp_authenticated_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Extract and validate user from MCP OAuth token."""
    if not credentials:
        return None

    try:
        user_data = await _validate_token_with_backend(credentials.credentials)
        if user_data:
            print(
                f"[MCP Auth] Successfully authenticated user {user_data.get('user_id')} for organization {user_data.get('organisation_id')}"
            )
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
