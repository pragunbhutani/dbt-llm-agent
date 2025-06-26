"""MCP resources for providing access to dbt project and model information with organization scoping."""

import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from asgiref.sync import sync_to_async
from django.db.models import Q

from apps.knowledge_base.models import Model
from apps.data_sources.models import DbtProject
from apps.embeddings.models import ModelEmbedding

logger = logging.getLogger(__name__)


def get_resources() -> List[Dict[str, Any]]:
    """Return list of available MCP resources."""
    return [
        {
            "uri": "ragstar://projects",
            "name": "dbt Projects",
            "description": "Overview of all connected dbt projects in your organization",
            "mimeType": "application/json",
        },
        {
            "uri": "ragstar://models",
            "name": "dbt Models",
            "description": "Browse all dbt models in your organization with optional filtering",
            "mimeType": "application/json",
        },
        {
            "uri": "ragstar://model/{model_name}",
            "name": "Model Details",
            "description": "Detailed information about a specific dbt model",
            "mimeType": "application/json",
        },
    ]


async def handle_resource_request(
    uri: str, user: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Handle MCP resource read requests with organization scoping."""
    try:
        if not user:
            return {
                "error": "Authentication required",
                "content": [
                    {"type": "text", "text": "Please authenticate to access resources."}
                ],
            }

        parsed_uri = urlparse(uri)
        if parsed_uri.scheme != "ragstar":
            return {
                "error": "Invalid URI scheme",
                "content": [
                    {
                        "type": "text",
                        "text": f"URI must start with 'ragstar://', got: {uri}",
                    }
                ],
            }

        path = parsed_uri.path.strip("/")
        query_params = parse_qs(parsed_uri.query)
        organisation_id = user.get("organisation_id")

        if path == "projects":
            return await _get_projects_resource(organisation_id)
        elif path == "models":
            return await _get_models_resource(query_params, organisation_id)
        elif path.startswith("model/"):
            model_name = path[6:]  # Remove "model/" prefix
            return await _get_model_resource(model_name, organisation_id)
        else:
            return {
                "error": "Unknown resource path",
                "content": [
                    {"type": "text", "text": f"Resource path not recognized: {path}"}
                ],
            }

    except Exception as e:
        logger.error(f"Error handling resource request {uri}: {e}", exc_info=True)
        return {
            "error": str(e),
            "content": [{"type": "text", "text": f"Error reading resource: {str(e)}"}],
        }


async def _get_projects_resource(organisation_id: int) -> Dict[str, Any]:
    """Get overview of all dbt projects with organization scoping."""

    @sync_to_async
    def _fetch_projects():
        projects = DbtProject.objects.filter(organisation_id=organisation_id)
        results = []
        for project in projects:
            model_count = project.models.count()
            project_data = {
                "id": project.id,
                "name": project.name,
                "connection_type": project.connection_type,
                "model_count": model_count,
                "created_at": (
                    project.created_at.isoformat() if project.created_at else None
                ),
            }
            results.append(project_data)
        return results

    projects = await _fetch_projects()

    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "projects": projects,
                        "total_count": len(projects),
                        "organization_scoped": True,
                    },
                    indent=2,
                ),
            }
        ],
        "mimeType": "application/json",
    }


async def _get_models_resource(
    query_params: Dict[str, List[str]], organisation_id: int
) -> Dict[str, Any]:
    """Get overview of all dbt models with organization scoping."""
    limit = min(int(query_params.get("limit", ["100"])[0]), 500)

    @sync_to_async
    def _fetch_models():
        queryset = Model.objects.select_related("dbt_project").filter(
            organisation_id=organisation_id
        )
        models = list(queryset.order_by("name")[:limit])

        results = []
        for model in models:
            model_data = {
                "name": model.name,
                "unique_id": model.unique_id,
                "schema": model.schema_name,
                "materialization": model.materialization,
                "description": model.yml_description or model.interpreted_description,
            }
            results.append(model_data)
        return results

    models = await _fetch_models()

    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "models": models,
                        "total_count": len(models),
                        "organization_scoped": True,
                    },
                    indent=2,
                ),
            }
        ],
        "mimeType": "application/json",
    }


async def _get_model_resource(model_name: str, organisation_id: int) -> Dict[str, Any]:
    """Get detailed information about a specific dbt model with organization scoping."""

    @sync_to_async
    def _fetch_model():
        try:
            model = Model.objects.select_related("dbt_project").get(
                name=model_name, organisation_id=organisation_id
            )
            return {
                "name": model.name,
                "unique_id": model.unique_id,
                "schema": model.schema_name,
                "materialization": model.materialization,
                "description": model.yml_description or model.interpreted_description,
                "raw_sql": model.raw_sql,
                "compiled_sql": model.compiled_sql,
                "depends_on": model.depends_on or [],
            }
        except Model.DoesNotExist:
            return None

    model_data = await _fetch_model()

    if not model_data:
        return {
            "error": "Model not found",
            "content": [
                {
                    "type": "text",
                    "text": f"Model '{model_name}' not found in your organization.",
                }
            ],
        }

    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "model": model_data,
                        "organization_scoped": True,
                    },
                    indent=2,
                ),
            }
        ],
        "mimeType": "application/json",
    }
