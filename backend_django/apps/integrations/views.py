import logging
from django.http import HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt

# Import necessary Bolt components (changed to sync)
from slack_bolt import App, BoltRequest  # Changed from AsyncApp, AsyncBoltRequest
from slack_bolt.oauth.oauth_settings import (
    OAuthSettings,
)  # Changed from AsyncOAuthSettings

# Import your App instance (now sync)
from .slack.handlers import app

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


# Slack team ID detection is now integrated into the OrganisationSettings save flow
# No separate endpoint needed
