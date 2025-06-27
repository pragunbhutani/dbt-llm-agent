"""Lightweight authentication extraction for the standalone MCP server.

We no longer have direct access to Django's database.  Instead of validating JWTs
locally, we treat them as opaque bearer tokens and forward them to the Django
backend (which will perform the real validation/authorisation).  This keeps the
MCP service completely stateless.
"""

from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# Security scheme – won't automatically error on missing creds
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Return a minimal user dict containing the raw access token (if any)."""

    if not credentials:
        return None

    return {"access_token": credentials.credentials}


async def require_auth(
    user: Optional[Dict[str, Any]] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Ensure the endpoint is called with Authentication."""

    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    return user
