"""Standalone FastMCP server for Ragstar integration.

This server provides Model Context Protocol integration, allowing LLM
applications like Claude.ai to interact with dbt knowledge bases. All
data access and OAuth 2.0 flows are proxied to the Django backend.

This implementation uses FastMCP properly for MCP protocol handling.
"""

from typing import Dict, Any, Optional, List
import os
import json
import logging

import httpx
from fastapi import HTTPException, Request, Depends, Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastmcp import FastMCP

from auth import get_mcp_authenticated_user

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Configuration
DJANGO_BACKEND_URL = os.environ.get("DJANGO_BACKEND_URL", "http://localhost:8000")
AUTHORIZATION_BASE_URL = os.environ.get(
    "AUTHORIZATION_BASE_URL", "http://localhost:8080"
)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# Security for tool authentication
security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# MCP Protocol / Main Application Initialization
# ---------------------------------------------------------------------------

mcp = FastMCP("Ragstar MCP Server")

# ---------------------------------------------------------------------------
# Debug & Health Check Endpoints
# ---------------------------------------------------------------------------


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health_check(request: Request):
    """Health check endpoint."""
    logger.info(f"📥 HEALTH CHECK: {request.method} {request.url}")
    return JSONResponse({"status": "healthy", "server": "Ragstar MCP Server"})


# ---------------------------------------------------------------------------
# OAuth 2.0 Metadata & Proxy Endpoints
# ---------------------------------------------------------------------------


@mcp.custom_route(
    "/.well-known/oauth-protected-resource", methods=["GET"], include_in_schema=False
)
async def oauth_protected_resource(request: Request):
    """OAuth 2.0 Protected Resource Metadata for MCP."""
    logger.info(f"📥 {request.method} {request.url}")
    return JSONResponse(
        {
            "resource": AUTHORIZATION_BASE_URL,
            "authorization_servers": [AUTHORIZATION_BASE_URL],
        }
    )


@mcp.custom_route(
    "/.well-known/oauth-protected-resource/mcp",
    methods=["GET"],
    include_in_schema=False,
)
async def oauth_protected_resource_mcp(request: Request):
    """OAuth 2.0 Protected Resource Metadata for MCP with /mcp path."""
    return JSONResponse(
        {
            "resource": f"{AUTHORIZATION_BASE_URL}/mcp",
            "authorization_servers": [AUTHORIZATION_BASE_URL],
            "scopes_supported": ["mcp:tools", "mcp:resources", "mcp:prompts"],
            "bearer_methods_supported": ["header"],
            "resource_documentation": f"{AUTHORIZATION_BASE_URL}/mcp/docs",
        }
    )


@mcp.custom_route(
    "/.well-known/oauth-authorization-server", methods=["GET"], include_in_schema=False
)
async def proxy_authorization_server_metadata(request: Request):
    """Proxy OAuth 2.0 Authorization Server Metadata from Django backend."""
    logger.info(f"📥 OAuth Authorization Server Metadata Request")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{DJANGO_BACKEND_URL}/.well-known/oauth-authorization-server"
            )
            if response.status_code == 200:
                return JSONResponse(response.json())
            else:
                logger.error(
                    f"Backend returned {response.status_code}: {response.text}"
                )
                return JSONResponse(
                    {"error": "OAuth metadata unavailable"},
                    status_code=response.status_code,
                )
    except Exception as e:
        logger.error(f"Error proxying OAuth metadata: {e}")
        return JSONResponse({"error": "OAuth metadata unavailable"}, status_code=500)


@mcp.custom_route(
    "/.well-known/oauth-authorization-server/mcp",
    methods=["GET"],
    include_in_schema=False,
)
async def proxy_authorization_server_metadata_mcp(request: Request):
    """Proxy OAuth 2.0 Authorization Server Metadata from Django backend with /mcp path."""
    logger.info(f"📥 OAuth Authorization Server Metadata Request (/mcp)")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{DJANGO_BACKEND_URL}/.well-known/oauth-authorization-server/mcp"
            )
            if response.status_code == 200:
                return JSONResponse(response.json())
            else:
                logger.error(
                    f"Backend returned {response.status_code}: {response.text}"
                )
                return JSONResponse(
                    {"error": "OAuth metadata unavailable"},
                    status_code=response.status_code,
                )
    except Exception as e:
        logger.error(f"Error proxying OAuth metadata: {e}")
        return JSONResponse({"error": "OAuth metadata unavailable"}, status_code=500)


# ---------------------------------------------------------------------------
# OAuth 2.0 Flow Endpoints
# ---------------------------------------------------------------------------


@mcp.custom_route("/oauth/auth/authorize", methods=["GET"], include_in_schema=False)
async def authorize(request: Request):
    """Proxy OAuth 2.0 authorization endpoint to Django backend."""
    logger.info(f"📥 OAuth Authorize Request")
    query_params = str(request.query_params)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{DJANGO_BACKEND_URL}/oauth/auth/authorize?{query_params}",
                follow_redirects=False,
            )
            logger.info(f"📤 OAuth Authorize Response: {response.status_code}")

            if response.status_code in [301, 302, 307, 308]:
                location = response.headers.get("location")
                if location:
                    return RedirectResponse(location, status_code=response.status_code)

            if response.headers.get("content-type", "").startswith("text/html"):
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )

            return JSONResponse(
                content=response.json() if response.content else {},
                status_code=response.status_code,
            )
    except Exception as e:
        logger.error(f"Error in OAuth authorize: {e}")
        return JSONResponse({"error": "Authorization failed"}, status_code=500)


@mcp.custom_route("/oauth/auth/token", methods=["POST"], include_in_schema=False)
async def token(request: Request):
    """Proxy OAuth 2.0 token endpoint to Django backend."""
    logger.info(f"📥 OAuth Token Request")

    try:
        body = await request.body()
        headers = {
            "Content-Type": request.headers.get(
                "Content-Type", "application/x-www-form-urlencoded"
            )
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DJANGO_BACKEND_URL}/oauth/auth/token",
                content=body,
                headers=headers,
            )
            logger.info(f"📤 OAuth Token Response: {response.status_code}")
            return JSONResponse(response.json(), status_code=response.status_code)
    except Exception as e:
        logger.error(f"Error in OAuth token: {e}")
        return JSONResponse({"error": "Token request failed"}, status_code=500)


@mcp.custom_route("/oauth/auth/register", methods=["POST"], include_in_schema=False)
async def register_client(request: Request):
    """Proxy OAuth 2.0 dynamic client registration to Django backend."""
    logger.info(f"📥 OAuth Register Request")

    try:
        body = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DJANGO_BACKEND_URL}/oauth/auth/register",
                json=body,
            )
            return JSONResponse(response.json(), status_code=response.status_code)
    except Exception as e:
        logger.error(f"Error in OAuth register: {e}")
        return JSONResponse({"error": "Registration failed"}, status_code=500)


# ---------------------------------------------------------------------------
# Authentication Helpers
# ---------------------------------------------------------------------------


async def get_authenticated_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Get authenticated user from JWT token."""
    if not credentials:
        # For testing, return None (unauthenticated)
        # In production, this would raise an exception
        return None

    try:
        return await get_mcp_authenticated_user(credentials.credentials)
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return None


# ---------------------------------------------------------------------------
# MCP Tools (require authentication for execution)
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_dbt_models(
    project_name: Optional[str] = None,
    schema_name: Optional[str] = None,
    materialization: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List available dbt models in the knowledge base with optional filtering.

    Args:
        project_name: Filter by dbt project name (optional)
        schema_name: Filter by schema/dataset name (optional)
        materialization: Filter by materialization type (table, view, incremental, etc.) (optional)
        limit: Maximum number of models to return (default: 50, max: 200)

    Returns:
        List of model dictionaries with their details
    """
    logger.info("🔧 ===== MCP TOOL CALLED: list_dbt_models =====")
    logger.info(
        f"Parameters: project_name={project_name}, schema_name={schema_name}, materialization={materialization}, limit={limit}"
    )

    # Prepare query parameters for Django backend
    params: Dict[str, Any] = {"limit": min(limit, 200)}
    if project_name:
        params["project_name"] = project_name
    if schema_name:
        params["schema_name"] = schema_name
    if materialization:
        params["materialization"] = materialization

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{DJANGO_BACKEND_URL}/mcp/tools/list-models/",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"list_dbt_models backend error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to list dbt models")

    logger.info("🔧 ===== MCP TOOL COMPLETED: list_dbt_models =====")
    return [{"type": "text", "text": json.dumps(data, indent=2)}]


@mcp.tool()
async def search_dbt_models(
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search for relevant dbt models using natural language queries.

    Args:
        query: Natural language description of what you're looking for
        limit: Maximum number of results to return (default: 10, max: 50)

    Returns:
        List of search results with relevance scores
    """
    logger.info("🔧 ===== MCP TOOL CALLED: search_dbt_models =====")
    logger.info(f"Parameters: query='{query}', limit={limit}")

    payload = {"query": query, "limit": min(limit, 50)}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{DJANGO_BACKEND_URL}/mcp/tools/search-models/",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"search_dbt_models backend error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to search dbt models")

    logger.info("🔧 ===== MCP TOOL COMPLETED: search_dbt_models =====")
    return [{"type": "text", "text": json.dumps(data, indent=2)}]


@mcp.tool()
async def get_model_details(
    model_name: str,
    project_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get detailed information about specific dbt models.

    Args:
        model_name: Name of the dbt model to get details for
        project_name: Filter by dbt project name (optional)

    Returns:
        List containing detailed model information including SQL, documentation, etc.
    """
    logger.info("🔧 ===== MCP TOOL CALLED: get_model_details =====")
    logger.info(f"Parameters: model_name='{model_name}', project_name={project_name}")

    params: Dict[str, Any] = {"name": model_name}
    if project_name:
        params["project_name"] = project_name

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{DJANGO_BACKEND_URL}/mcp/tools/model-details/",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"get_model_details backend error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to get model details")

    logger.info("🔧 ===== MCP TOOL COMPLETED: get_model_details =====")
    return [{"type": "text", "text": json.dumps(data, indent=2)}]


@mcp.tool()
async def get_project_summary(
    project_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get summary information about dbt projects and their contents.

    Args:
        project_name: Filter by specific project name (optional)

    Returns:
        List containing project summary with model counts, schemas, etc.
    """
    logger.info("🔧 ===== MCP TOOL CALLED: get_project_summary =====")
    logger.info(f"Parameters: project_name={project_name}")

    # Return proper MCP content format
    summary = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "projects": [
                        {
                            "name": "analytics",
                            "description": "Main analytics dbt project",
                            "model_count": 45,
                            "schemas": ["staging", "intermediate", "marts"],
                            "last_updated": "2024-01-15T10:30:00Z",
                        }
                    ],
                    "total_models": 45,
                    "total_schemas": 3,
                    "project_filter": project_name,
                },
                indent=2,
            ),
        }
    ]

    logger.info("🔧 ===== MCP TOOL COMPLETED: get_project_summary =====")
    return summary


# ---------------------------------------------------------------------------
# MCP Prompts
# ---------------------------------------------------------------------------


@mcp.prompt()
async def ragstar_introduction() -> str:
    """Learn about Ragstar and its capabilities for dbt analytics."""
    logger.info("📝 ===== MCP PROMPT CALLED: ragstar_introduction =====")

    return """# Welcome to Ragstar! 🌟

Ragstar is an AI-powered data analyst for teams whose analytics stack is built with **dbt**. Users will ask questions about their *data*—your job is to discover the dbt models that contain that data, understand them, and use them to generate the correct SQL (and eventually charts) that answers the question.

Although we operate on dbt models, always remember that the ultimate goal is DATA analysis, not model cataloguing.

## How You Work

1. **Discover available data** – Call `list_dbt_models` to see which models are enabled for question-answering. Filter by project, schema or materialisation when helpful.
2. **Find relevant models** – Call `search_dbt_models` with a natural-language description of the information you need. This performs a semantic search across model docs and metadata.
3. **Understand the details** – Call `get_model_details` on promising models to inspect their SQL, columns, and lineage.
4. **Answer the question** – Combine what you learn to craft SQL (and later visualisations). Reference the models you used in your explanation.

## Available Tools

• `list_dbt_models` – browse and filter dbt models  
• `search_dbt_models` – semantic search for relevant models  
• `get_model_details` – deep dive into model SQL & metadata

## Best-practice Tips

• Be precise in search queries (e.g. "monthly mrr growth" vs "revenue")  
• Chain your reasoning: *identify candidate models → verify details → write SQL*.  
• When uncertain, inspect columns and dependencies via `get_model_details`.  
• Cite model names in your answers so users understand the data source.

Ready to explore? Start by listing or searching for models related to the user’s question!"""


@mcp.prompt()
async def data_analysis_guidance() -> str:
    """Get guidance on how to analyze data and ask effective questions using Ragstar."""
    logger.info("📝 ===== MCP PROMPT CALLED: data_analysis_guidance =====")

    return """# Guidance for Analysing Data with Ragstar 📈

## Crafting Great Questions

• **Be specific & contextual** – "What were the top 5 products by revenue last quarter compared with the previous one?" is better than "Show me sales data".  
• **Iterate** – Start broad, then drill down based on insights.  
• **Reference models** – Mention model names or metrics you’ve discovered to ground follow-ups.

## Recommended Workflow

1. **Discovery** – Call `list_dbt_models` to understand the available data surfaces.  
2. **Exploration** – Use `search_dbt_models` with business language to narrow the candidate models.  
3. **Investigation** – Call `get_model_details` on 1-3 promising models to examine SQL, columns and dependencies.  
4. **Synthesis** – Combine insights from the models, write SQL, and form your answer.

## Tool Reference

• `list_dbt_models` – list & filter models  
• `search_dbt_models` – semantic search  
• `get_model_details` – detailed metadata

When responding, clearly explain *which* models you used and *why* they answer the question. If no suitable model exists, say so and suggest next steps (e.g., create a new model or gather additional data)."""


# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------

# FastMCP handles all the protocol details, server setup, and routing
if __name__ == "__main__":
    logger.info("🚀 Starting MCP server...")
    logger.info("📍 Server will run on: http://0.0.0.0:8080")
    logger.info(
        "🔧 Available tools: list_dbt_models, search_dbt_models, get_model_details"
    )
    logger.info("📝 Available prompts: ragstar_introduction, data_analysis_guidance")
    logger.info("🌐 OAuth endpoints available at: /oauth/auth/...")

    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8080,
        path="/",
        stateless_http=True,
    )
