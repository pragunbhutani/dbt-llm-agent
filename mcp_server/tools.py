"""Standalone implementation of MCP *tools* that proxies each call to the Django REST API.

This removes any direct dependency on Django models/ORM and cleanly decouples the MCP
service.  The Django backend remains the single source of truth for data; the MCP server
is essentially an HTTP façade that exposes that functionality via the Model Context
Protocol.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from .config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON Schemas exposed to LLM clients (unchanged from earlier version)
# ---------------------------------------------------------------------------


def get_tools() -> List[Dict[str, Any]]:
    """Return list of available MCP tools."""
    return [
        {
            "name": "list_dbt_models",
            "description": "List available dbt models in the knowledge base with optional filtering",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Filter by dbt project name (optional)",
                    },
                    "schema_name": {
                        "type": "string",
                        "description": "Filter by schema/dataset name (optional)",
                    },
                    "materialization": {
                        "type": "string",
                        "description": "Filter by materialization type (table, view, incremental, etc.) (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of models to return (default: 50, max: 200)",
                        "minimum": 1,
                        "maximum": 200,
                    },
                },
            },
        },
        {
            "name": "search_dbt_models",
            "description": "Search for relevant dbt models using natural language queries with semantic similarity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query to find relevant dbt models",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of models to return (default: 10, max: 50)",
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0 to 1.0, default: 0.7)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_model_details",
            "description": "Get detailed information about specific dbt models including SQL, documentation, and lineage",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of dbt model names to retrieve details for",
                    },
                    "include_sql": {
                        "type": "boolean",
                        "description": "Include raw and compiled SQL (default: true)",
                    },
                    "include_lineage": {
                        "type": "boolean",
                        "description": "Include upstream and downstream dependencies (default: true)",
                    },
                },
                "required": ["model_names"],
            },
        },
        {
            "name": "get_project_summary",
            "description": "Get a summary of connected dbt projects and their models",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Specific project name to get summary for (optional - returns all if not specified)",
                    }
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Internal helper – Generic proxy to the Ragstar Django REST API
# ---------------------------------------------------------------------------


_TOOL_ENDPOINTS = {
    "list_dbt_models": "/api/mcp/list-models/",
    "search_dbt_models": "/api/mcp/search-models/",
    "get_model_details": "/api/mcp/model-details/",
    "get_project_summary": "/api/mcp/project-summary/",
}


async def _call_backend_api(
    endpoint: str,
    payload: Dict[str, Any],
    access_token: Optional[str],
) -> Dict[str, Any]:
    """POST the payload to Django and return its JSON response."""

    url = settings.BACKEND_API_URL.rstrip("/") + endpoint
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Backend API error %s -> %s", url, exc.response.text)
        return {
            "error": f"Backend responded {exc.response.status_code}",
            "content": [
                {
                    "type": "text",
                    "text": f"The backend reported an error: {exc.response.text}",
                }
            ],
        }

    return resp.json()


# ---------------------------------------------------------------------------
# Public entry – called from main.py via FastMCP tool registration
# ---------------------------------------------------------------------------


async def handle_tool_call(
    name: str, arguments: Dict[str, Any], user: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Proxy the tool invocation to the Django REST API."""

    endpoint = _TOOL_ENDPOINTS.get(name)
    if not endpoint:
        return {
            "error": f"Unknown tool: {name}",
            "content": [{"type": "text", "text": f"Tool '{name}' is not supported."}],
        }

    # Extract bearer token if `user` passed from auth dependency
    access_token = user.get("access_token") if user else None

    return await _call_backend_api(endpoint, arguments, access_token)


__all__ = ["get_tools", "handle_tool_call"]
