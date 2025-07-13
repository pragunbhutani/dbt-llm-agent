from django.shortcuts import render
from django.http import JsonResponse, HttpResponseRedirect
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from asgiref.sync import sync_to_async
from jose import jwt, JWTError
from datetime import datetime, timezone
from pgvector.django import CosineDistance
import json
import logging

from .oauth_server import MCPOAuthServer
from apps.knowledge_base.models import Model
from apps.data_sources.models import DbtProject
from apps.embeddings.models import ModelEmbedding
from apps.accounts.models import OrganisationSettings
from apps.llm_providers.services import EmbeddingService
from apps.integrations.models import OrganisationIntegration

logger = logging.getLogger(__name__)
User = get_user_model()

# Create your views here.


@method_decorator(csrf_exempt, name="dispatch")
class MCPInfoView(View):
    """MCP server information endpoint"""

    def get(self, request):
        return JsonResponse(
            {
                "name": "Ragstar MCP Server",
                "version": "0.1.0",
                "description": "Remote MCP server with OAuth 2.0 authentication for Ragstar integration",
                "capabilities": {"tools": True, "resources": True, "prompts": True},
                "authentication": {
                    "type": "oauth2",
                    "authorization_url": f"{request.build_absolute_uri('/oauth/auth/authorize')}",
                    "token_url": f"{request.build_absolute_uri('/oauth/auth/token')}",
                    "registration_url": f"{request.build_absolute_uri('/oauth/auth/register')}",
                },
            }
        )

    def post(self, request):
        """Forward POST requests to the FastAPI MCP server"""
        # Since the FastAPI app is mounted at /mcp/, redirect POST requests there
        return HttpResponseRedirect("/mcp/")


class MCPAuthorizationServerMetadataView(View):
    """OAuth 2.0 Authorization Server Metadata with MCP path"""

    def get(self, request):
        oauth_server = MCPOAuthServer()
        metadata = oauth_server.get_authorization_server_metadata()
        return JsonResponse(metadata)


class MCPProtectedResourceMetadataView(View):
    """OAuth 2.0 Protected Resource Metadata for MCP (RFC 8707)"""

    def get(self, request):
        authorization_base_url = getattr(
            settings, "MCP_AUTHORIZATION_BASE_URL", "http://localhost:8000"
        )
        oauth_server = MCPOAuthServer()

        metadata = {
            "resource": f"{authorization_base_url}/mcp",
            "authorization_servers": [oauth_server.issuer],
            "scopes_supported": ["mcp:tools", "mcp:resources", "mcp:prompts"],
            "bearer_methods_supported": ["header"],
            "resource_documentation": f"{authorization_base_url}/mcp/docs",
        }
        return JsonResponse(metadata)


# ---------------------------------------------------------------------------
# MCP API Endpoints for standalone MCP server
# ---------------------------------------------------------------------------


def _validate_jwt_token(token: str) -> dict:
    """Validate JWT token and return user data."""
    try:
        jwt_secret = getattr(settings, "SECRET_KEY", "devsecret")
        jwt_algorithm = "HS256"

        payload = jwt.decode(token, jwt_secret, algorithms=[jwt_algorithm])

        # Check token expiration
        exp_ts = payload.get("exp")
        if exp_ts:
            exp_time = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
            if exp_time < datetime.now(timezone.utc):
                raise ValueError("Token expired")

        # Extract user information from JWT payload
        user_id = payload.get("user_id")
        organization_id = payload.get("organization_id")
        email = payload.get("email")
        scopes = payload.get("scopes", [])
        client_id = payload.get("client_id")

        if not user_id or not organization_id:
            raise ValueError("Missing required claims in token")

        # Verify user exists
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise ValueError(f"User {user_id} not found")

        # Check if Claude integration exists for this organization
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
            "access_token": token,
        }

        if claude_integration:
            user_data["claude_linked"] = True
            user_data["claude_client_id"] = claude_integration.credentials.get(
                "client_id"
            )
        else:
            user_data["claude_linked"] = False

        return user_data

    except JWTError as e:
        raise ValueError(f"JWT decode error: {e}")
    except Exception as e:
        raise ValueError(f"Authentication error: {e}")


@api_view(["POST"])
@csrf_exempt
def mcp_validate_token(request):
    """Validate JWT token for MCP server authentication."""
    try:
        data = json.loads(request.body)
        token = data.get("token")

        if not token:
            return Response(
                {"error": "Token is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        user_data = _validate_jwt_token(token)
        return Response(user_data, status=status.HTTP_200_OK)

    except json.JSONDecodeError:
        return Response({"error": "Invalid JSON"}, status=status.HTTP_400_BAD_REQUEST)
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        logger.error(f"Token validation error: {e}", exc_info=True)
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@csrf_exempt
def mcp_list_models(request):
    """List dbt models with filtering and organization scoping for MCP server."""
    try:
        # Extract JWT token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                {"error": "Authorization header required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        token = auth_header[7:]  # Remove 'Bearer ' prefix
        user_data = _validate_jwt_token(token)
        organisation_id = user_data.get("organisation_id")

        # Get query parameters
        project_name = request.GET.get("project_name")
        schema_name = request.GET.get("schema_name")
        materialization = request.GET.get("materialization")
        limit = min(int(request.GET.get("limit", 50)), 200)

        # Build queryset with organization scoping
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
        models = queryset.order_by("name")[:limit]

        # Format response
        results = []
        for model in models:
            results.append(
                {
                    "name": model.name,
                    "path": model.path,
                    "schema": model.schema_name,
                    "database": model.database,
                    "materialization": model.materialization,
                    "project": (
                        {"name": model.dbt_project.name} if model.dbt_project else None
                    ),
                    "description": model.yml_description
                    or model.interpreted_description,
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
            )

        return Response(
            {"results": results, "count": len(results), "total": queryset.count()},
            status=status.HTTP_200_OK,
        )

    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        logger.error(f"List models error: {e}", exc_info=True)
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@csrf_exempt
def mcp_search_models(request):
    """Search dbt models using semantic similarity for MCP server."""
    try:
        # Extract JWT token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                {"error": "Authorization header required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        token = auth_header[7:]  # Remove 'Bearer ' prefix
        user_data = _validate_jwt_token(token)
        organisation_id = user_data.get("organisation_id")

        # Get request data
        data = json.loads(request.body)
        query = data.get("query")
        limit = min(data.get("limit", 10), 50)
        similarity_threshold = data.get("similarity_threshold", 0.7)

        if not query:
            return Response(
                {"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Get organization settings
        try:
            org_settings = OrganisationSettings.objects.get(
                organisation_id=organisation_id
            )
        except OrganisationSettings.DoesNotExist:
            return Response(
                {"error": "Organization settings not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Get embedding for the query
        embedding_service = EmbeddingService()
        query_embedding = embedding_service.get_embedding(query)

        # Use pgvector cosine distance search with organization scoping
        similar_models = (
            ModelEmbedding.objects.filter(organisation_id=organisation_id)
            .annotate(similarity=1 - CosineDistance("embedding", query_embedding))
            .filter(similarity__gte=similarity_threshold)
            .select_related("model", "model__dbt_project")
            .order_by("-similarity")[:limit]
        )

        # Format response
        results = []
        for embedding in similar_models:
            model = embedding.model
            results.append(
                {
                    "model": {
                        "name": model.name,
                        "path": model.path,
                        "schema": model.schema_name,
                        "database": model.database,
                        "materialization": model.materialization,
                        "project": (
                            {"name": model.dbt_project.name}
                            if model.dbt_project
                            else None
                        ),
                        "description": model.yml_description
                        or model.interpreted_description,
                        "tags": model.tags or [],
                        "depends_on": model.depends_on or [],
                        "unique_id": model.unique_id,
                        "created_at": (
                            model.created_at.isoformat() if model.created_at else None
                        ),
                        "updated_at": (
                            model.updated_at.isoformat() if model.updated_at else None
                        ),
                    },
                    "similarity": float(embedding.similarity),
                }
            )

        return Response(
            {"results": results, "count": len(results), "query": query},
            status=status.HTTP_200_OK,
        )

    except json.JSONDecodeError:
        return Response({"error": "Invalid JSON"}, status=status.HTTP_400_BAD_REQUEST)
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        logger.error(f"Search models error: {e}", exc_info=True)
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@csrf_exempt
def mcp_get_model_details(request):
    """Get detailed information about specific dbt models for MCP server."""
    try:
        # Extract JWT token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                {"error": "Authorization header required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        token = auth_header[7:]  # Remove 'Bearer ' prefix
        user_data = _validate_jwt_token(token)
        organisation_id = user_data.get("organisation_id")

        # Get query parameters
        name = request.GET.get("name")
        project_name = request.GET.get("project_name")

        if not name:
            return Response(
                {"error": "Model name is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Build queryset with organization scoping
        queryset = Model.objects.select_related("dbt_project").filter(
            organisation_id=organisation_id, name=name
        )

        if project_name:
            queryset = queryset.filter(dbt_project__name__icontains=project_name)

        models = list(queryset)

        if not models:
            return Response({"results": [], "count": 0}, status=status.HTTP_200_OK)

        # Format detailed response
        results = []
        for model in models:
            model_data = {
                "name": model.name,
                "unique_id": model.unique_id,
                "path": model.path,
                "schema": model.schema_name,
                "database": model.database,
                "materialization": model.materialization,
                "project": (
                    {"name": model.dbt_project.name} if model.dbt_project else None
                ),
                "description": model.yml_description or model.interpreted_description,
                "tags": model.tags or [],
                "meta": model.meta or {},
                "depends_on": model.depends_on or [],
                "created_at": (
                    model.created_at.isoformat() if model.created_at else None
                ),
                "updated_at": (
                    model.updated_at.isoformat() if model.updated_at else None
                ),
                "raw_sql": model.raw_sql,
                "compiled_sql": model.compiled_sql,
            }

            # Add column information if available
            if model.yml_columns:
                model_data["columns"] = model.yml_columns
            elif model.interpreted_columns:
                model_data["columns"] = model.interpreted_columns

            results.append(model_data)

        return Response(
            {"results": results, "count": len(results)}, status=status.HTTP_200_OK
        )

    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        logger.error(f"Get model details error: {e}", exc_info=True)
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@csrf_exempt
def mcp_get_project_summary(request):
    """Get summary of dbt projects for MCP server."""
    try:
        # Extract JWT token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return Response(
                {"error": "Authorization header required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        token = auth_header[7:]  # Remove 'Bearer ' prefix
        user_data = _validate_jwt_token(token)
        organisation_id = user_data.get("organisation_id")

        # Get query parameters
        project_name = request.GET.get("project_name")

        # Build queryset with organization scoping
        projects_queryset = DbtProject.objects.filter(organisation_id=organisation_id)

        if project_name:
            projects_queryset = projects_queryset.filter(name__icontains=project_name)

        projects = list(projects_queryset)

        # Format response
        results = []
        for project in projects:
            # Get model count for this project
            model_count = Model.objects.filter(
                organisation_id=organisation_id, dbt_project=project
            ).count()

            results.append(
                {
                    "name": project.name,
                    "connection_type": project.connection_type,
                    "model_count": model_count,
                    "created_at": (
                        project.created_at.isoformat() if project.created_at else None
                    ),
                    "updated_at": (
                        project.updated_at.isoformat() if project.updated_at else None
                    ),
                }
            )

        return Response(
            {"results": results, "count": len(results)}, status=status.HTTP_200_OK
        )

    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        logger.error(f"Get project summary error: {e}", exc_info=True)
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
