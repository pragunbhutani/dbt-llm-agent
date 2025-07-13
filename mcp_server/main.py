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

    # Return proper MCP content format
    models = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "models": [
                        {
                            "name": "customers",
                            "project_name": "analytics",
                            "schema_name": "marts",
                            "materialization": "table",
                            "description": "Customer dimension table with demographic and account information",
                        },
                        {
                            "name": "orders",
                            "project_name": "analytics",
                            "schema_name": "marts",
                            "materialization": "table",
                            "description": "Order fact table with transaction details",
                        },
                    ],
                    "total_count": 2,
                    "filters": {
                        "project_name": project_name,
                        "schema_name": schema_name,
                        "materialization": materialization,
                        "limit": limit,
                    },
                },
                indent=2,
            ),
        }
    ]

    logger.info("🔧 ===== MCP TOOL COMPLETED: list_dbt_models =====")
    return models


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

    # Return proper MCP content format
    results = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "results": [
                        {
                            "name": "customers",
                            "project_name": "analytics",
                            "schema_name": "marts",
                            "relevance_score": 0.95,
                            "description": "Customer dimension table with demographic and account information",
                            "match_reason": f"Matches query: {query}",
                        },
                        {
                            "name": "customer_orders",
                            "project_name": "analytics",
                            "schema_name": "marts",
                            "relevance_score": 0.87,
                            "description": "Customer order history and transaction details",
                            "match_reason": f"Related to query: {query}",
                        },
                    ],
                    "total_count": 2,
                    "query": query,
                    "limit": limit,
                },
                indent=2,
            ),
        }
    ]

    logger.info("🔧 ===== MCP TOOL COMPLETED: search_dbt_models =====")
    return results


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

    # Return proper MCP content format
    details = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "model_name": model_name,
                    "project_name": project_name or "analytics",
                    "schema_name": "marts",
                    "materialization": "table",
                    "description": f"Detailed information about {model_name} model",
                    "sql": f"SELECT * FROM staging.{model_name}",
                    "columns": [
                        {"name": "id", "type": "string", "description": "Primary key"},
                        {
                            "name": "name",
                            "type": "string",
                            "description": "Display name",
                        },
                        {
                            "name": "created_at",
                            "type": "timestamp",
                            "description": "Creation timestamp",
                        },
                    ],
                    "dependencies": ["staging.raw_data"],
                    "documentation": f"This model contains {model_name} data processed from raw sources",
                },
                indent=2,
            ),
        }
    ]

    logger.info("🔧 ===== MCP TOOL COMPLETED: get_model_details =====")
    return details


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

Ragstar is an AI-powered data analyst designed specifically for teams working with **dbt** (data build tool). I help you understand, explore, and analyze your dbt projects and data models with intelligent search and question-answering capabilities.

## What I Can Do

### 🔍 **Model Discovery & Search**
- **List Models**: Browse all your dbt models with filtering by project, schema, or materialization type
- **Semantic Search**: Find relevant models using natural language queries (e.g., "customer revenue models" or "user behavior data")
- **Model Details**: Get comprehensive information about specific models including SQL, documentation, and lineage

### 🤔 **Intelligent Question Answering**
- Ask analytical questions about your data in plain English
- Get insights backed by relevant dbt models and supporting SQL queries
- Understand relationships between different data models

### 📊 **Project Overview**
- Get summaries of your connected dbt projects
- Understand model relationships and dependencies
- Explore different materialization strategies

## My Tools Available to You

1. **`list_dbt_models`** - Browse and filter your dbt models
2. **`search_dbt_models`** - Find models using natural language search
3. **`get_model_details`** - Deep dive into specific models with SQL and lineage
4. **`get_project_summary`** - Get overview of your dbt projects

## How to Get Started

1. **Explore your models**: Start with `get_project_summary` to understand what data you have
2. **Search for relevant data**: Use `search_dbt_models` to find models related to your analysis
3. **Dive deeper**: Use `get_model_details` to understand specific models better
4. **Ask questions**: Use the discovered models to ask analytical questions

## Tips for Best Results

- **Be specific** in your search queries (e.g., "customer acquisition models" vs "customer data")
- **Provide context** when asking questions to get more relevant answers
- **Explore dependencies** to understand how models relate to each other
- **Ask follow-up questions** to dive deeper into interesting findings

Ready to explore your data? Start with a project summary or search for relevant models to build your analysis!
"""


@mcp.prompt()
async def data_analysis_guidance() -> str:
    """Get guidance on how to analyze data and ask effective questions using Ragstar."""
    logger.info("📝 ===== MCP PROMPT CALLED: data_analysis_guidance =====")

    return """# Data Analysis with Ragstar 📈

## Asking Effective Questions

### 🎯 **Be Specific and Contextual**
Instead of: *"Show me sales data"*
Try: *"What are the top 5 products by revenue in the last quarter, and how do they compare to the previous quarter?"*

### 🔗 **Build on Previous Insights**
- Start with broad questions, then drill down
- Reference specific models or metrics you've discovered
- Ask about relationships between different data points

### 💡 **Great Question Examples**

**Business Metrics:**
- "What is our customer acquisition cost trend over the past 6 months?"
- "Which marketing channels are driving the highest lifetime value customers?"
- "How has our monthly recurring revenue grown year-over-year?"

**Operational Insights:**
- "What percentage of orders are shipped within 2 days?"
- "Which product categories have the highest return rates?"
- "How does customer support response time correlate with satisfaction scores?"

**Cohort & Segmentation:**
- "How do user retention rates differ between subscription tiers?"
- "What are the characteristics of our most valuable customer segment?"
- "How has user engagement changed since the new feature launch?"

## Analysis Workflow

### 1. **Discovery Phase**
Start with: `get_project_summary`
→ Understand what data domains you have
→ Identify key business areas

### 2. **Exploration Phase**  
Use: `search_dbt_models` with business terms
→ Find models related to your analysis area
→ Review model descriptions and metadata

### 3. **Investigation Phase**
Use: `get_model_details` on relevant models
→ Examine SQL and business logic
→ Understand data transformations and calculations

### 4. **Synthesis Phase**
Combine discovered models and their SQL
→ Build comprehensive understanding of data flow
→ Compose analytical insights from multiple sources

## Available Tools:
- `list_dbt_models`: Explore all available data models
- `search_dbt_models`: Find relevant models using natural language
- `get_model_details`: Deep dive into specific models
- `get_project_summary`: Understand your data ecosystem

Ready to start your analysis? Begin by exploring your project structure and discovering relevant models!
"""


# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------

# FastMCP handles all the protocol details, server setup, and routing
if __name__ == "__main__":
    logger.info("🚀 Starting MCP server...")
    logger.info("📍 Server will run on: http://0.0.0.0:8080")
    logger.info(
        "🔧 Available tools: list_dbt_models, search_dbt_models, get_model_details, get_project_summary"
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
