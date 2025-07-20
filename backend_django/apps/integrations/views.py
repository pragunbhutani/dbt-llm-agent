import json
import logging
from typing import Dict, Any

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from slack_bolt.request import BoltRequest
from slack_sdk.oauth.installation_store import Bot
from slack_sdk.oauth.state_store import FileOAuthStateStore

# Import necessary Bolt components (changed to sync)
from slack_bolt import App, BoltRequest  # Changed from AsyncApp, AsyncBoltRequest
from slack_bolt.oauth.oauth_settings import (
    OAuthSettings,
)  # Changed from AsyncOAuthSettings

# Slack app instances are now created dynamically per team

logger = logging.getLogger(__name__)


def to_bolt_request(request: HttpRequest) -> BoltRequest:  # Removed async
    """Helper function to convert Django HttpRequest to BoltRequest."""
    # Access request.body directly (it's bytes, not awaitable)
    body_bytes = request.body
    body_str = body_bytes.decode(request.encoding or "utf-8")

    # Extract headers correctly, handling potential list values from Django
    headers = {}
    for k, v in request.headers.items():
        # Slack headers typically aren't multi-valued, but handle just in case
        headers[k] = (
            v
            if isinstance(v, str)
            else v[0] if isinstance(v, list) and len(v) > 0 else ""
        )

    return BoltRequest(  # Changed from AsyncBoltRequest
        body=body_str,
        query=request.GET.urlencode(),  # Pass query string
        headers=headers,  # Pass headers dictionary
        context={},  # Start with empty context, Bolt fills it
        mode="http",  # We are receiving via HTTP
    )


def to_django_response(bolt_resp) -> HttpResponse:  # Removed async
    """Helper function to convert BoltResponse to Django HttpResponse."""
    # Get content-type from headers dictionary, defaulting to None or a standard type
    content_type = bolt_resp.headers.get(
        "content-type", "application/json"
    )  # Defaulting to JSON

    response = HttpResponse(
        status=bolt_resp.status,
        content=bolt_resp.body,
        content_type=content_type,  # Use the retrieved content type
    )
    # Copy other headers from BoltResponse if needed
    for k, v in bolt_resp.headers.items():
        # Handle potential multiple headers if necessary, though usually single for Bolt
        response[k] = v[0] if isinstance(v, list) else v
    return response


@csrf_exempt  # Slack requests don't have CSRF tokens
def slack_events_handler(request: HttpRequest):  # Removed async
    """Receives Slack event HTTP requests, converts to Bolt format, dispatches, and returns response."""
    if request.method != "POST":
        logger.warning(
            f"Received non-POST request to Slack events endpoint: {request.method}"
        )
        return HttpResponse(status=405, content="Method Not Allowed")

    # --- Handle Slack URL verification challenge requests ---------------------------------
    # Slack sends a special URL verification payload when you first configure the Events
    # endpoint. This payload does **not** contain a team_id and therefore will fail the
    # logic below that expects one. We short-circuit here and simply echo back the
    # provided challenge so Slack can confirm the endpoint.
    try:
        raw_body = request.body.decode(request.encoding or "utf-8")
        payload = json.loads(raw_body)
    except Exception:
        payload = None

    if payload and payload.get("type") == "url_verification":
        challenge = payload.get("challenge")
        if challenge:
            logger.info("Responding to Slack URL verification challenge")
            return HttpResponse(
                json.dumps({"challenge": challenge}),
                status=200,
                content_type="application/json",
            )

    logger.debug("slack_events_handler: Converting Django request to Bolt request")
    try:
        bolt_req: BoltRequest = to_bolt_request(request)  # Removed await
    except Exception as e:
        logger.error(f"Error converting request to BoltRequest: {e}", exc_info=True)
        return HttpResponse(
            status=500, content="Internal Server Error during request conversion"
        )

    # Log the bolt request details for debugging
    logger.info(f"Bolt request body: {bolt_req.body}")
    logger.info(f"Bolt request headers: {bolt_req.headers}")

    # Extract team ID from the request to get the appropriate Slack app instance
    try:
        # Parse the request body to get team_id
        if bolt_req.raw_body:
            if bolt_req.raw_body.startswith("payload="):
                # URL-encoded payload (interactive components)
                import urllib.parse

                payload_str = urllib.parse.unquote_plus(
                    bolt_req.raw_body[8:]
                )  # Remove "payload="
                payload = json.loads(payload_str)
                team_id = payload.get("team", {}).get("id") or payload.get("team_id")
            else:
                # JSON payload (events) - use parsed body since it's already a dict
                payload = bolt_req.body
                team_id = payload.get("team_id") or payload.get("event", {}).get("team")
        else:
            team_id = None

        if not team_id:
            logger.error("No team_id found in Slack request")
            return HttpResponse(status=400, content="Missing team_id in request")

        # Get the appropriate Slack app instance for this team
        from .slack.handlers import get_slack_app_for_team

        app = get_slack_app_for_team(team_id)
        if not app:
            logger.error(f"No Slack app configuration found for team: {team_id}")
            return HttpResponse(
                status=400, content="Slack integration not configured for this team"
            )

    except Exception as e:
        logger.error(f"Error getting Slack app for request: {e}", exc_info=True)
        return HttpResponse(
            status=500, content="Internal Server Error during app initialization"
        )

    # Dispatch the request to the Bolt app's internal handler
    logger.debug("slack_events_handler: Dispatching Bolt request to App")
    try:
        bolt_resp = app.dispatch(bolt_req)  # Changed from async_dispatch, removed await
    except Exception as e:
        logger.error(f"Error dispatching request within Bolt app: {e}", exc_info=True)
        return HttpResponse(
            status=500, content="Internal Server Error during Bolt dispatch"
        )

    # Convert the Bolt response back to a Django response
    logger.debug("slack_events_handler: Converting Bolt response to Django response")
    try:
        django_resp = to_django_response(bolt_resp)  # Removed await
        logger.debug(
            f"slack_events_handler: Returning Django response status: {django_resp.status_code}"
        )
        return django_resp
    except Exception as e:
        logger.error(
            f"Error converting BoltResponse to HttpResponse: {e}", exc_info=True
        )
        return HttpResponse(
            status=500, content="Internal Server Error during response conversion"
        )


# --- New Integration API Views ---

from .models import OrganisationIntegration
from .constants import get_active_integration_definitions, get_integration_definition
from .serializers import (
    IntegrationSerializer,
    OrganisationIntegrationSerializer,
    IntegrationStatusSerializer,
    ConnectionTestSerializer,
    IntegrationToolSerializer,
)
from .manager import IntegrationsManager


class IntegrationViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for available integrations (now from constants).
    """

    permission_classes = [IsAuthenticated]
    serializer_class = IntegrationSerializer

    def get_queryset(self):
        """Get integrations from constants instead of database."""
        integration_type = self.request.query_params.get("type")
        integrations = get_active_integration_definitions()

        if integration_type:
            integrations = [
                integration
                for integration in integrations
                if integration.integration_type == integration_type
            ]

        return integrations

    def list(self, request, *args, **kwargs):
        """Override list to work with constants instead of queryset."""
        integrations = self.get_queryset()
        serializer = self.get_serializer(
            [integration.to_dict() for integration in integrations], many=True
        )
        return Response(serializer.data)

    def retrieve(self, request, pk=None, *args, **kwargs):
        """Override retrieve to get integration by key from constants."""
        try:
            integration = get_integration_definition(pk)
            if not integration.is_active:
                return Response(
                    {"detail": "Integration not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )
            serializer = self.get_serializer(integration.to_dict())
            return Response(serializer.data)
        except ValueError:
            return Response(
                {"detail": "Integration not found."}, status=status.HTTP_404_NOT_FOUND
            )


class OrganisationIntegrationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for organization-specific integration configurations.
    """

    serializer_class = OrganisationIntegrationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """
        This view should return a list of all the integrations
        for the currently authenticated user's organisation.
        """
        user = self.request.user
        queryset = OrganisationIntegration.objects.filter(
            organisation=user.organisation
        ).order_by("integration_key")

        integration_keys_str = self.request.query_params.get("integration_keys")
        if integration_keys_str:
            integration_keys = [
                key.strip() for key in integration_keys_str.split(",") if key.strip()
            ]
            queryset = queryset.filter(integration_key__in=integration_keys)

        return queryset

    def perform_create(self, serializer):
        serializer.save(organisation=self.request.user.organisation)

    @action(detail=True, methods=["post"])
    def test_connection(self, request, pk=None):
        """Test connection for a specific integration."""
        org_integration = self.get_object()

        try:
            # Create integration manager for this organization
            manager = IntegrationsManager(request.user.organisation)
            integration = manager.get_integration(org_integration.integration_key)

            if not integration:
                return Response(
                    {"error": "Integration not loaded or configured"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Test the connection
            test_result = integration.test_connection()
            tested_at = timezone.now()
            test_result["tested_at"] = tested_at.isoformat()

            # Update the organization integration with test results
            org_integration.last_test_result = test_result
            org_integration.last_tested_at = tested_at
            org_integration.save(update_fields=["last_test_result", "last_tested_at"])

            serializer = ConnectionTestSerializer(test_result)
            return Response(serializer.data)

        except Exception as e:
            tested_at = timezone.now()
            error_result = {
                "success": False,
                "message": f"Test failed: {str(e)}",
                "tested_at": tested_at.isoformat(),
            }

            # Still save the error result
            org_integration.last_test_result = error_result
            org_integration.last_tested_at = tested_at
            org_integration.save(update_fields=["last_test_result", "last_tested_at"])

            return Response(error_result, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=["post"])
    def toggle_enable(self, request, pk=None):
        """Enable or disable an integration."""
        org_integration = self.get_object()
        org_integration.is_enabled = not org_integration.is_enabled
        org_integration.save(update_fields=["is_enabled"])

        serializer = self.get_serializer(org_integration)
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def refresh_link(self, request, pk=None):
        """Initiate re-link / OAuth refresh for integrations like GitHub."""
        org_integration = self.get_object()

        if org_integration.integration_key == "github":
            from apps.integrations.github.views import build_github_authorization_url

            url = build_github_authorization_url(request.user)
            return Response({"authorization_url": url})

        return Response(
            {"error": "Refresh not supported for this integration."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    @action(detail=False, methods=["get"])
    def status(self, request):
        """
        Returns the status of all or a subset of integrations for the user's organisation.
        """
        manager = IntegrationsManager(request.user.organisation)
        all_statuses = manager.get_integration_status()

        integration_keys_str = request.query_params.get("integration_keys")
        if integration_keys_str:
            integration_keys = [
                key.strip() for key in integration_keys_str.split(",") if key.strip()
            ]
            # Filter the statuses based on the provided keys
            statuses = [
                status
                for key, status in all_statuses.items()
                if key in integration_keys
            ]
        else:
            statuses = list(all_statuses.values())

        serializer = IntegrationStatusSerializer(statuses, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=["get"])
    def tools(self, request):
        """Get all available tools from enabled integrations."""
        try:
            manager = IntegrationsManager(request.user.organisation)
            tools = manager.get_all_tools()

            serializer = IntegrationToolSerializer(tools, many=True)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error getting integration tools: {e}")
            return Response(
                {"error": "Failed to get integration tools"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def test_all_connections(self, request):
        """Test connections for all enabled integrations."""
        try:
            manager = IntegrationsManager(request.user.organisation)
            test_results = manager.test_all_connections()

            # Update the database with test results
            for integration_key, result in test_results.items():
                try:
                    org_integration = OrganisationIntegration.objects.get(
                        organisation=request.user.organisation,
                        integration_key=integration_key,
                    )
                    tested_at = timezone.now()
                    result["tested_at"] = tested_at.isoformat()

                    org_integration.last_test_result = result
                    org_integration.last_tested_at = tested_at
                    org_integration.save(
                        update_fields=["last_test_result", "last_tested_at"]
                    )
                except OrganisationIntegration.DoesNotExist:
                    logger.warning(
                        f"OrganisationIntegration not found for key: {integration_key}"
                    )

            return Response(test_results)

        except Exception as e:
            logger.error(f"Error testing all connections: {e}")
            return Response(
                {"error": "Failed to test connections"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SlackIntegrationViewSet(viewsets.ViewSet):
    """
    Special viewset for Slack integration setup.
    """

    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=["post"])
    def setup(self, request):
        """Setup Slack integration with bot token, signing secret, and app token."""
        try:
            bot_token = request.data.get("bot_token")
            signing_secret = request.data.get("signing_secret")
            app_token = request.data.get("app_token")

            if not bot_token or not signing_secret:
                return Response(
                    {"error": "Bot token and signing secret are required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Validate bot token format
            if not bot_token.startswith("xoxb-"):
                return Response(
                    {"error": "Invalid bot token format. Should start with 'xoxb-'"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Test the bot token by getting team info
            from .slack.handlers import get_team_info_from_token

            team_info = get_team_info_from_token(bot_token)
            if not team_info:
                return Response(
                    {"error": "Invalid bot token or unable to connect to Slack"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Get or create organization integration (no longer need to create Integration record)
            org_integration, created = OrganisationIntegration.objects.get_or_create(
                organisation=request.user.organisation,
                integration_key="slack",
                defaults={
                    "is_enabled": True,
                    "configuration": {"team_id": team_info["team_id"]},
                },
            )

            if created:
                # Set credentials using the new method
                org_integration.set_credentials(
                    {
                        "bot_token": bot_token,
                        "signing_secret": signing_secret,
                        "app_token": app_token if app_token else "",
                    }
                )

            if not created:
                # Update existing integration
                org_integration.configuration.update({"team_id": team_info["team_id"]})
                org_integration.update_credentials(
                    {
                        "bot_token": bot_token,
                        "signing_secret": signing_secret,
                        "app_token": app_token if app_token else "",
                    }
                )
                org_integration.is_enabled = True
                org_integration.save()

            # Test the connection
            manager = IntegrationsManager(request.user.organisation)
            manager.reload_integrations()  # Reload to pick up new configuration
            integration = manager.get_integration("slack")

            if integration:
                test_result = integration.test_connection()
                tested_at = timezone.now()
                test_result["tested_at"] = tested_at.isoformat()

                org_integration.last_test_result = test_result
                org_integration.last_tested_at = tested_at
                org_integration.save(
                    update_fields=["last_test_result", "last_tested_at"]
                )
            else:
                test_result = {"success": False, "message": "Integration not loaded"}

            return Response(
                {
                    "success": True,
                    "message": f"Slack integration configured for workspace: {team_info['team_name']}",
                    "team_info": team_info,
                    "test_result": test_result,
                }
            )

        except Exception as e:
            logger.error(f"Error setting up Slack integration: {e}")
            return Response(
                {"error": f"Setup failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["get"])
    def status(self, request):
        """Get Slack integration status and team info."""
        try:
            # Get Slack integration for this organization
            org_integration = OrganisationIntegration.objects.filter(
                organisation=request.user.organisation,
                integration_key="slack",
            ).first()

            if not org_integration:
                return Response(
                    {
                        "configured": False,
                        "enabled": False,
                        "message": "Slack integration not configured",
                    }
                )

            # Get connection status
            connection_status = "unknown"
            team_info = None

            if org_integration.last_test_result:
                connection_status = (
                    "connected"
                    if org_integration.last_test_result.get("success")
                    else "error"
                )
                team_info = org_integration.last_test_result.get("team_info")

            return Response(
                {
                    "configured": org_integration.is_configured,
                    "enabled": org_integration.is_enabled,
                    "connection_status": connection_status,
                    "team_info": team_info,
                    "last_tested_at": (
                        org_integration.last_tested_at.isoformat()
                        if org_integration.last_tested_at
                        else None
                    ),
                    "test_result": org_integration.last_test_result,
                }
            )

        except Exception as e:
            logger.error(f"Error getting Slack integration status: {e}")
            return Response(
                {"error": f"Failed to get status: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SnowflakeIntegrationViewSet(viewsets.ViewSet):
    """
    ViewSet for Snowflake integration setup.
    """

    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=["post"])
    def setup(self, request):
        """
        Configure Snowflake integration for the user's organization.
        """
        user = request.user
        organisation = user.organisation

        # Extract configuration from request
        account = request.data.get("account")
        user_field = request.data.get("user")  # 'user' is a reserved name
        password = request.data.get("password")
        warehouse = request.data.get("warehouse")
        database = request.data.get("database")
        schema = request.data.get("schema", "PUBLIC")

        # Validate required fields
        if not all([account, user_field, password, warehouse]):
            return Response(
                {
                    "error": "Missing required fields: account, user, password, warehouse"
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Get or create organization integration
            org_integration, created = OrganisationIntegration.objects.get_or_create(
                organisation=organisation,
                integration_key="snowflake",
                defaults={
                    "is_enabled": True,
                    "configuration": {"schema": schema if database else None},
                },
            )

            if created:
                # Set credentials using the new method
                org_integration.set_credentials(
                    {
                        "account": account,
                        "user": user_field,
                        "password": password,
                        "warehouse": warehouse,
                        "database": database if database else None,
                        "schema": schema if database else None,
                    }
                )
            else:
                # Update existing integration
                org_integration.is_enabled = True
                org_integration.configuration = {"schema": schema if database else None}
                org_integration.set_credentials(
                    {
                        "account": account,
                        "user": user_field,
                        "password": password,
                        "warehouse": warehouse,
                        "database": database if database else None,
                        "schema": schema if database else None,
                    }
                )
                org_integration.save()

            # Test the connection
            from .manager import SnowflakeIntegration

            integration = SnowflakeIntegration(org_integration)
            test_result = integration.test_connection()

            if test_result["success"]:
                # Update test results
                org_integration.last_test_result = test_result
                org_integration.last_tested_at = timezone.now()
                org_integration.save()

                return Response(
                    {
                        "message": "Snowflake integration configured successfully",
                        "test_result": test_result,
                    }
                )
            else:
                # If test failed, still save the config but mark as error
                org_integration.last_test_result = test_result
                org_integration.last_tested_at = timezone.now()
                org_integration.save()

                return Response(
                    {
                        "error": "Snowflake configuration saved but connection test failed",
                        "test_result": test_result,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        except Exception as e:
            logger.error(f"Snowflake setup error: {e}", exc_info=True)
            return Response(
                {"error": f"Configuration failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class MetabaseIntegrationViewSet(viewsets.ViewSet):
    """
    ViewSet for Metabase integration setup.
    """

    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=["post"])
    def setup(self, request):
        """
        Configure Metabase integration for the user's organization.
        """
        user = request.user
        organisation = user.organisation

        # Extract configuration from request
        url = request.data.get("url")
        api_key = request.data.get("api_key")
        database_id = request.data.get("database_id")

        # Validate required fields
        if not all([url, api_key, database_id]):
            return Response(
                {"error": "Missing required fields: url, api_key, database_id"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate database_id is an integer
        try:
            database_id = int(database_id)
        except (ValueError, TypeError):
            return Response(
                {"error": "database_id must be a valid integer"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Clean up URL (remove trailing slash)
        url = url.rstrip("/")

        try:
            # Get or create organization integration
            org_integration, created = OrganisationIntegration.objects.get_or_create(
                organisation=organisation,
                integration_key="metabase",
                defaults={
                    "is_enabled": True,
                    "configuration": {"database_id": database_id},
                },
            )

            if created:
                # Set credentials using the new method
                org_integration.set_credentials(
                    {
                        "url": url,
                        "api_key": api_key,
                        "database_id": database_id,
                    }
                )
            else:
                # Update existing integration
                org_integration.is_enabled = True
                org_integration.configuration = {"database_id": database_id}
                org_integration.set_credentials(
                    {
                        "url": url,
                        "api_key": api_key,
                        "database_id": database_id,
                    }
                )
                org_integration.save()

            # Test the connection
            from .manager import MetabaseIntegration

            integration = MetabaseIntegration(org_integration)
            test_result = integration.test_connection()

            if test_result["success"]:
                # Update test results
                org_integration.last_test_result = test_result
                org_integration.last_tested_at = timezone.now()
                org_integration.save()

                return Response(
                    {
                        "message": "Metabase integration configured successfully",
                        "test_result": test_result,
                    }
                )
            else:
                # If test failed, still save the config but mark as error
                org_integration.last_test_result = test_result
                org_integration.last_tested_at = timezone.now()
                org_integration.save()

                return Response(
                    {
                        "error": "Metabase configuration saved but connection test failed",
                        "test_result": test_result,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        except Exception as e:
            logger.error(f"Metabase setup error: {e}", exc_info=True)
            return Response(
                {"error": f"Configuration failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
