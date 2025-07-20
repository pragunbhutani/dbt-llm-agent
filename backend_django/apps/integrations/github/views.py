# apps/integrations/github/views.py
import logging
import secrets
from django.conf import settings
from django.core.cache import cache
from django.http import HttpResponseRedirect, JsonResponse
from rest_framework.response import Response
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated, AllowAny
import requests
from django.views import View

from apps.integrations.models import OrganisationIntegration
from apps.accounts.models import Organisation
from django.core.management.base import CommandError

logger = logging.getLogger(__name__)


def build_github_authorization_url(user):
    """Helper to create authorization URL and cache state."""
    client_id = settings.GITHUB_APP_CLIENT_ID
    redirect_uri = f"{settings.NEXT_PUBLIC_API_URL}/api/integrations/github/callback/"
    state = secrets.token_urlsafe(32)
    cache.set(
        f"github_oauth_state:{state}",
        {"user_id": user.id, "organisation_id": user.organisation.id},
        timeout=600,
    )
    return f"https://github.com/login/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}&scope=repo&state={state}"


def github_oauth_callback(request):
    """
    Handles the callback from GitHub after installation.
    This endpoint is open but validates the state parameter to ensure security.
    """
    code = request.GET.get("code")
    state = request.GET.get("state")

    if not code or not state:
        return JsonResponse(
            {"error": "Missing code or state parameter"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Retrieve user/org info from cache using state
    cached_data = cache.get(f"github_oauth_state:{state}")
    if not cached_data:
        return JsonResponse(
            {"error": "Invalid or expired state token."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    cache.delete(f"github_oauth_state:{state}")

    client_id = settings.GITHUB_APP_CLIENT_ID
    client_secret = settings.GITHUB_APP_CLIENT_SECRET

    token_url = "https://github.com/login/oauth/access_token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
    }
    headers = {"Accept": "application/json"}

    try:
        response = requests.post(token_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        access_token = data.get("access_token")

        if not access_token:
            return JsonResponse(
                {"error": "Could not retrieve access token"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Get organisation from cached data
        organisation = Organisation.objects.get(id=cached_data["organisation_id"])

        # Create or update the integration (without credentials field)
        org_integration, _ = OrganisationIntegration.objects.update_or_create(
            organisation=organisation,
            integration_key="github",
            defaults={
                "is_enabled": True,
                "configuration": {},
            },
        )
        # Try secret manager; fallback to configuration field in dev/local
        if not org_integration.set_credentials({"access_token": access_token}):
            cfg = org_integration.configuration or {}
            cfg["access_token"] = access_token
            org_integration.configuration = cfg
            org_integration.save(update_fields=["configuration"])

        # Redirect user back to the new project page
        redirect_url = f"{settings.FRONTEND_URL}/dashboard?github_status=success"
        return HttpResponseRedirect(redirect_url)

    except Organisation.DoesNotExist:
        return JsonResponse(
            {"error": "Organisation not found."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting GitHub access token: {e}")
        return JsonResponse(
            {"error": "Failed to get access token from GitHub"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class GitHubIntegrationViewSet(viewsets.ViewSet):
    """
    ViewSet for GitHub integration.
    """

    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=["get"])
    def install(self, request):
        """
        Returns the GitHub App installation URL.
        """
        client_id = settings.GITHUB_APP_CLIENT_ID
        redirect_uri = (
            f"{settings.NEXT_PUBLIC_API_URL}/api/integrations/github/callback/"
        )

        state = secrets.token_urlsafe(32)
        # Store user and organisation context in cache, keyed by state
        cache.set(
            f"github_oauth_state:{state}",
            {
                "user_id": request.user.id,
                "organisation_id": request.user.organisation.id,
            },
            timeout=600,
        )  # 10 minute expiry

        authorization_url = (
            f"https://github.com/login/oauth/authorize?client_id={client_id}"
            f"&redirect_uri={redirect_uri}&scope=repo&state={state}"
        )
        return JsonResponse({"authorization_url": authorization_url})

    # ---------------- Owners -----------------
    @action(detail=False, methods=["get"])
    def owners(self, request):
        """Return list of owners (user + orgs) associated with the access token."""
        organisation = request.user.organisation
        try:
            integration = OrganisationIntegration.objects.get(
                organisation=organisation, integration_key="github", is_enabled=True
            )
            access_token = integration.get_credential(
                "access_token"
            ) or integration.configuration.get("access_token")
            if not access_token:
                return Response(
                    {"error": "GitHub access token not found."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except OrganisationIntegration.DoesNotExist:
            return Response(
                {"error": "GitHub integration not configured."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if not access_token:
            return Response(
                {"error": "GitHub access token missing."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            user_resp = requests.get("https://api.github.com/user", headers=headers)
            user_resp.raise_for_status()
            user_data = user_resp.json()

            orgs_resp = requests.get(
                "https://api.github.com/user/orgs", headers=headers
            )
            orgs_resp.raise_for_status()
            orgs_data = orgs_resp.json()

            owners = [
                {"login": user_data.get("login"), "type": user_data.get("type", "User")}
            ]
            owners += [
                {"login": org.get("login"), "type": org.get("type", "Organization")}
                for org in orgs_data
            ]

            return Response(owners)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching owners from GitHub: {e}")
            return Response(
                {"error": "Failed to fetch owners from GitHub."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["get"])
    def repositories(self, request):
        """
        Fetches the list of repositories for the authenticated user's organisation.
        """
        organisation = request.user.organisation
        if not organisation:
            return Response(
                {"error": "User is not associated with an organisation"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            integration = OrganisationIntegration.objects.get(
                organisation=organisation, integration_key="github", is_enabled=True
            )
            access_token = integration.credentials.get("access_token")
        except OrganisationIntegration.DoesNotExist:
            return Response(
                {"error": "GitHub integration not found or not enabled."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if not access_token:
            return Response(
                {"error": "Access token not found for GitHub integration."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        owner_login = request.query_params.get("owner")

        if owner_login:
            repos_url = f"https://api.github.com/users/{owner_login}/repos"
        else:
            repos_url = "https://api.github.com/user/repos"

        headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            response = requests.get(repos_url, headers=headers)
            response.raise_for_status()
            repos = response.json()
            # We only need the name and full_name for the user to select from
            repo_list = [
                {"name": repo["name"], "full_name": repo["full_name"]} for repo in repos
            ]
            return Response(repo_list)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching repositories from GitHub: {e}")
            return Response(
                {"error": "Failed to fetch repositories from GitHub."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
