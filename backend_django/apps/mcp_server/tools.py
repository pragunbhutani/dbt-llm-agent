"""MCP tools for exposing Ragstar functionality to LLM clients with organization scoping."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from asgiref.sync import sync_to_async
from django.db.models import Q
from pgvector.django import CosineDistance

from apps.knowledge_base.models import Model
from apps.data_sources.models import DbtProject
from apps.embeddings.models import ModelEmbedding
from apps.accounts.models import User, OrganisationSettings
from apps.llm_providers.services import EmbeddingService


from .config import settings

logger = logging.getLogger(__name__)


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


async def handle_tool_call(
    name: str, arguments: Dict[str, Any], user: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Handle MCP tool calls with organization scoping."""
    try:
        if name == "list_dbt_models":
            return await _list_dbt_models(arguments, user)
        elif name == "search_dbt_models":
            return await _search_dbt_models(arguments, user)
        elif name == "get_model_details":
            return await _get_model_details(arguments, user)

        elif name == "get_project_summary":
            return await _get_project_summary(arguments, user)
        else:
            return {
                "error": f"Unknown tool: {name}",
                "content": [
                    {"type": "text", "text": f"Tool '{name}' is not supported."}
                ],
            }
    except Exception as e:
        logger.error(f"Error handling tool call {name}: {e}", exc_info=True)
        return {
            "error": str(e),
            "content": [{"type": "text", "text": f"Error executing {name}: {str(e)}"}],
        }


async def _list_dbt_models(
    arguments: Dict[str, Any], user: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """List available dbt models with optional filtering and organization scoping."""
    if not user:
        return {
            "error": "Authentication required",
            "content": [
                {"type": "text", "text": "Please authenticate to access dbt models."}
            ],
        }

    project_name = arguments.get("project_name")
    schema_name = arguments.get("schema_name")
    materialization = arguments.get("materialization")
    limit = min(arguments.get("limit", 50), 200)
    organisation_id = user.get("organisation_id")

    @sync_to_async
    def _fetch_models():
        queryset = Model.objects.select_related("dbt_project").filter(
            organisation_id=organisation_id
        )

        # Apply filters
        if project_name:
            queryset = queryset.filter(dbt_project__name__icontains=project_name)
        if schema_name:
            queryset = queryset.filter(schema_name__icontains=schema_name)
        if materialization:
            queryset = queryset.filter(materialization__icontains=materialization)

        # Order by name and limit
        models = list(queryset.order_by("name")[:limit])

        return [
            {
                "name": model.name,
                "path": model.path,
                "schema": model.schema_name,
                "database": model.database,
                "materialization": model.materialization,
                "project": model.dbt_project.name if model.dbt_project else None,
                "description": model.yml_description or model.interpreted_description,
                "tags": model.tags or [],
                "depends_on": model.depends_on or [],
                "unique_id": model.unique_id,
                "created_at": (
                    model.created_at.isoformat() if model.created_at else None
                ),
                "updated_at": (
                    model.updated_at.isoformat() if model.updated_at else None
                ),
            }
            for model in models
        ]

    models = await _fetch_models()

    return {
        "content": [
            {
                "type": "text",
                "text": f"Found {len(models)} dbt models matching your criteria:\n\n"
                + "\n".join(
                    [
                        (
                            f"• **{model['name']}** ({model['materialization']}) - {model['description'][:100]}..."
                            if model["description"]
                            else f"• **{model['name']}** ({model['materialization']})"
                        )
                        for model in models
                    ]
                ),
            }
        ],
        "models": models,
    }


async def _search_dbt_models(
    arguments: Dict[str, Any], user: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Search for relevant dbt models using natural language queries with organization scoping."""
    if not user:
        return {
            "error": "Authentication required",
            "content": [
                {"type": "text", "text": "Please authenticate to search dbt models."}
            ],
        }

    query = arguments.get("query")
    limit = min(arguments.get("limit", 10), 50)
    similarity_threshold = arguments.get("similarity_threshold", 0.7)
    organisation_id = user.get("organisation_id")

    @sync_to_async
    def _get_org_settings():
        return OrganisationSettings.objects.filter(id=organisation_id).first()

    @sync_to_async
    def _search_models(query_embedding):
        # Use pgvector cosine distance search with organization scoping
        similar_models = (
            ModelEmbedding.objects.filter(organisation_id=organisation_id)
            .annotate(similarity=1 - CosineDistance("embedding", query_embedding))
            .filter(similarity__gte=similarity_threshold)
            .select_related("model", "model__dbt_project")
            .order_by("-similarity")[:limit]
        )

        results = []
        for embedding in similar_models:
            model = embedding.model
            results.append(
                {
                    "name": model.name,
                    "path": model.path,
                    "schema": model.schema_name,
                    "database": model.database,
                    "materialization": model.materialization,
                    "project": model.dbt_project.name if model.dbt_project else None,
                    "description": model.yml_description
                    or model.interpreted_description,
                    "tags": model.tags or [],
                    "depends_on": model.depends_on or [],
                    "unique_id": model.unique_id,
                    "similarity_score": float(embedding.similarity),
                    "created_at": (
                        model.created_at.isoformat() if model.created_at else None
                    ),
                    "updated_at": (
                        model.updated_at.isoformat() if model.updated_at else None
                    ),
                }
            )

        return results

    try:
        org_settings = await _get_org_settings()
        if not org_settings:
            return {
                "error": "Organization settings not found",
                "content": [
                    {"type": "text", "text": "Could not find organization settings."}
                ],
            }

        # Get embedding for the query
        embedding_service = EmbeddingService()
        query_embedding = await sync_to_async(embedding_service.get_embedding)(query)

        # Search for similar models
        similar_models = await _search_models(query_embedding)

        if not similar_models:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"No relevant models found for query: '{query}'. Try adjusting your search terms or lowering the similarity threshold.",
                    }
                ],
                "models": [],
            }

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Found {len(similar_models)} relevant dbt models for '{query}':\n\n"
                    + "\n".join(
                        [
                            (
                                f"• **{model['name']}** (similarity: {model['similarity_score']:.2f}) - {model['description'][:100]}..."
                                if model["description"]
                                else f"• **{model['name']}** (similarity: {model['similarity_score']:.2f})"
                            )
                            for model in similar_models
                        ]
                    ),
                }
            ],
            "models": similar_models,
        }

    except Exception as e:
        logger.error(f"Error in semantic search: {e}", exc_info=True)
        return {
            "error": str(e),
            "content": [{"type": "text", "text": f"Error performing search: {str(e)}"}],
        }


async def _get_model_details(
    arguments: Dict[str, Any], user: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Get detailed information about specific dbt models with organization scoping."""
    if not user:
        return {
            "error": "Authentication required",
            "content": [
                {"type": "text", "text": "Please authenticate to access model details."}
            ],
        }

    model_names = arguments.get("model_names", [])
    include_sql = arguments.get("include_sql", True)
    include_lineage = arguments.get("include_lineage", True)
    organisation_id = user.get("organisation_id")

    @sync_to_async
    def _fetch_model_details():
        queryset = Model.objects.select_related("dbt_project").filter(
            name__in=model_names, organisation_id=organisation_id
        )

        models = list(queryset)

        if not models:
            return []

        results = []
        for model in models:
            model_data = {
                "name": model.name,
                "unique_id": model.unique_id,
                "path": model.path,
                "schema": model.schema_name,
                "database": model.database,
                "materialization": model.materialization,
                "project": model.dbt_project.name if model.dbt_project else None,
                "description": model.yml_description or model.interpreted_description,
                "tags": model.tags or [],
                "meta": model.meta or {},
                "config": model.config or {},
                "created_at": (
                    model.created_at.isoformat() if model.created_at else None
                ),
                "updated_at": (
                    model.updated_at.isoformat() if model.updated_at else None
                ),
            }

            if include_sql:
                model_data.update(
                    {
                        "raw_sql": model.raw_sql,
                        "compiled_sql": model.compiled_sql,
                    }
                )

            if include_lineage:
                model_data.update(
                    {
                        "depends_on": model.depends_on or [],
                        "referenced_by": model.referenced_by or [],
                    }
                )

            results.append(model_data)

        return results

    models = await _fetch_model_details()

    if not models:
        return {
            "error": "Models not found",
            "content": [
                {
                    "type": "text",
                    "text": f"Could not find models: {', '.join(model_names)} in your organization.",
                }
            ],
        }

    return {
        "content": [
            {
                "type": "text",
                "text": f"Retrieved details for {len(models)} dbt models:\n\n"
                + "\n".join(
                    [
                        f"## {model['name']}\n"
                        f"- **Schema**: {model['schema']}\n"
                        f"- **Materialization**: {model['materialization']}\n"
                        f"- **Description**: {model['description'] or 'No description'}\n"
                        + (
                            f"- **Dependencies**: {', '.join(model['depends_on'])}\n"
                            if include_lineage and model["depends_on"]
                            else ""
                        )
                        + (
                            f"- **Referenced by**: {', '.join(model['referenced_by'])}\n"
                            if include_lineage and model["referenced_by"]
                            else ""
                        )
                        for model in models
                    ]
                ),
            }
        ],
        "models": models,
    }


async def _get_project_summary(
    arguments: Dict[str, Any], user: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Get a summary of connected dbt projects with organization scoping."""
    if not user:
        return {
            "error": "Authentication required",
            "content": [
                {
                    "type": "text",
                    "text": "Please authenticate to access project summary.",
                }
            ],
        }

    project_name = arguments.get("project_name")
    organisation_id = user.get("organisation_id")

    @sync_to_async
    def _fetch_project_summary():
        queryset = DbtProject.objects.filter(organisation_id=organisation_id)

        if project_name:
            queryset = queryset.filter(name__icontains=project_name)

        projects = list(queryset)

        results = []
        for project in projects:
            model_count = project.models.count()
            schema_names = list(
                project.models.values_list("schema_name", flat=True).distinct()
            )
            materializations = list(
                project.models.values_list("materialization", flat=True).distinct()
            )

            # Get some example models
            example_models = list(
                project.models.order_by("name")[:5].values(
                    "name", "schema_name", "materialization", "yml_description"
                )
            )

            project_data = {
                "id": project.id,
                "name": project.name,
                "connection_type": project.connection_type,
                "model_count": model_count,
                "schemas": schema_names,
                "materializations": materializations,
                "example_models": example_models,
                "dbt_cloud_url": project.dbt_cloud_url,
                "created_at": (
                    project.created_at.isoformat() if project.created_at else None
                ),
                "updated_at": (
                    project.updated_at.isoformat() if project.updated_at else None
                ),
            }

            results.append(project_data)

        return results

    projects = await _fetch_project_summary()

    if not projects:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "No dbt projects found in your organization. Please connect a dbt project first.",
                }
            ],
            "projects": [],
        }

    return {
        "content": [
            {
                "type": "text",
                "text": f"Found {len(projects)} dbt projects in your organization:\n\n"
                + "\n".join(
                    [
                        f"## {project['name']}\n"
                        f"- **Models**: {project['model_count']}\n"
                        f"- **Schemas**: {', '.join(project['schemas'])}\n"
                        f"- **Materializations**: {', '.join(project['materializations'])}\n"
                        f"- **Connection**: {project['connection_type']}\n"
                        for project in projects
                    ]
                ),
            }
        ],
        "projects": projects,
    }
