#!/usr/bin/env python3
"""
Ragstar MCP Server - Django-integrated version

This server provides Model Context Protocol integration for Ragstar,
allowing LLM applications like Claude.ai to interact with dbt knowledge bases.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastmcp import FastMCP
from pydantic import BaseModel

from .auth import get_current_user, require_auth
from .config import settings
from .prompts import get_prompts
from .resources import get_resources
from .tools import get_tools

# Configure logging
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager for startup/shutdown tasks."""
    logger.info("Starting Ragstar MCP Server...")
    yield
    logger.info("Shutting down Ragstar MCP Server...")


# Create FastAPI app
app = FastAPI(
    title="Ragstar MCP Server",
    description="Model Context Protocol server for Ragstar - AI Data Analyst for dbt projects",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Create MCP server instance using FastMCP
mcp_server = FastMCP("ragstar")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat(),
    )


# Register MCP tools using FastMCP decorators
@mcp_server.tool()
async def list_dbt_models(
    project_name: str = None,
    schema_name: str = None,
    materialization: str = None,
    limit: int = 50,
) -> str:
    """List available dbt models in the knowledge base with optional filtering"""
    from .tools import handle_tool_call

    arguments = {}
    if project_name:
        arguments["project_name"] = project_name
    if schema_name:
        arguments["schema_name"] = schema_name
    if materialization:
        arguments["materialization"] = materialization
    arguments["limit"] = limit

    result = await handle_tool_call("list_dbt_models", arguments)
    return result.get("content", [{}])[0].get("text", "No results found")


@mcp_server.tool()
async def search_models(query: str, limit: int = 10) -> str:
    """Search for dbt models using semantic similarity"""
    from .tools import handle_tool_call

    result = await handle_tool_call("search_models", {"query": query, "limit": limit})
    return result.get("content", [{}])[0].get("text", "No results found")


@mcp_server.tool()
async def get_model_details(model_name: str, project_name: str = None) -> str:
    """Get detailed information about a specific dbt model"""
    from .tools import handle_tool_call

    arguments = {"model_name": model_name}
    if project_name:
        arguments["project_name"] = project_name

    result = await handle_tool_call("get_model_details", arguments)
    return result.get("content", [{}])[0].get("text", "Model not found")


# Mount MCP server to FastAPI using FastMCP's http_app method
app.mount("/mcp", mcp_server.http_app())

# Note: OAuth 2.1 authorization endpoints are handled separately by Django
# This FastAPI app focuses on the MCP protocol endpoints
