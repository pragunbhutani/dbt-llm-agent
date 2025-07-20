"""
Integration constants and definitions.

This file contains the definitions of all available integrations that can be
configured by organizations. Moving these to constants eliminates the need
to seed the database on every deployment.
"""

from typing import Dict, List, Any


class IntegrationType:
    INBOUND = "inbound"  # Services that send requests to us (Slack, MCP)
    OUTBOUND = "outbound"  # Services we send requests to (Snowflake, Metabase)

    CHOICES = [
        (INBOUND, "Inbound"),
        (OUTBOUND, "Outbound"),
    ]


class IntegrationDefinition:
    """Class to hold integration definition data."""

    def __init__(
        self,
        key: str,
        name: str,
        description: str,
        integration_type: str,
        configuration_schema: Dict[str, Any] = None,
        icon_url: str = None,
        documentation_url: str = None,
        is_active: bool = True,
    ):
        self.key = key
        self.name = name
        self.description = description
        self.integration_type = integration_type
        self.configuration_schema = configuration_schema or {}
        self.icon_url = icon_url
        self.documentation_url = documentation_url
        self.is_active = is_active

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "integration_type": self.integration_type,
            "configuration_schema": self.configuration_schema,
            "icon_url": self.icon_url,
            "documentation_url": self.documentation_url,
            "is_active": self.is_active,
        }


# Available integrations
AVAILABLE_INTEGRATIONS: Dict[str, IntegrationDefinition] = {
    "slack": IntegrationDefinition(
        key="slack",
        name="Slack",
        description="Connect your Slack workspace to interact with ragstar through mentions and DMs.",
        integration_type=IntegrationType.INBOUND,
        configuration_schema={
            "type": "object",
            "properties": {
                "team_id": {"type": "string", "title": "Team ID"},
                "team_name": {"type": "string", "title": "Team Name"},
                "team_domain": {"type": "string", "title": "Team Domain"},
            },
            "required": ["team_id"],
        },
        icon_url="https://a.slack-edge.com/80588/marketing/img/icons/icon_slack_hash_colored.png",
        documentation_url="https://api.slack.com/",
        is_active=True,
    ),
    "snowflake": IntegrationDefinition(
        key="snowflake",
        name="Snowflake",
        description="Connect to your Snowflake data warehouse to run queries and analyze data.",
        integration_type=IntegrationType.OUTBOUND,
        configuration_schema={
            "type": "object",
            "properties": {
                "database": {"type": "string", "title": "Default Database"},
                "schema": {"type": "string", "title": "Default Schema"},
            },
        },
        icon_url="https://companieslogo.com/img/orig/SNOW-35164165.png",
        documentation_url="https://docs.snowflake.com/",
        is_active=True,
    ),
    "metabase": IntegrationDefinition(
        key="metabase",
        name="Metabase",
        description="Integrate with Metabase to create dashboards and visualizations.",
        integration_type=IntegrationType.OUTBOUND,
        configuration_schema={
            "type": "object",
            "properties": {
                "database_id": {
                    "type": "integer",
                    "title": "Database ID",
                    "default": 1,
                },
            },
        },
        icon_url="https://www.metabase.com/images/logo.svg",
        documentation_url="https://www.metabase.com/docs/",
        is_active=True,
    ),
    "mcp": IntegrationDefinition(
        key="mcp",
        name="MCP Server",
        description="Expose ragstar as a remote MCP server for Claude, OpenAI, and other LLM applications.",
        integration_type=IntegrationType.INBOUND,
        configuration_schema={
            "type": "object",
            "properties": {
                "server_url": {"type": "string", "title": "Server URL"},
                "auth_provider": {
                    "type": "string",
                    "title": "Auth Provider",
                    "enum": ["claude", "openai", "gemini"],
                    "default": "claude",
                },
            },
            "required": ["server_url", "auth_provider"],
        },
        icon_url=None,  # No official MCP icon yet
        documentation_url="https://modelcontextprotocol.io/",
        is_active=True,
    ),
    "github": IntegrationDefinition(
        key="github",
        name="GitHub",
        description="Connect a GitHub repository containing your dbt Core project.",
        integration_type=IntegrationType.OUTBOUND,
        configuration_schema={},
        icon_url="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
        documentation_url="https://docs.github.com/en/developers/apps",
        is_active=True,
    ),
}


def get_integration_definition(key: str) -> IntegrationDefinition:
    """Get integration definition by key."""
    integration = AVAILABLE_INTEGRATIONS.get(key)
    if not integration:
        raise ValueError(f"Unknown integration key: {key}")
    return integration


def get_all_integration_definitions() -> List[IntegrationDefinition]:
    """Get all available integration definitions."""
    return list(AVAILABLE_INTEGRATIONS.values())


def get_active_integration_definitions() -> List[IntegrationDefinition]:
    """Get all active integration definitions."""
    return [
        integration
        for integration in AVAILABLE_INTEGRATIONS.values()
        if integration.is_active
    ]


def get_integrations_by_type(integration_type: str) -> List[IntegrationDefinition]:
    """Get integrations filtered by type."""
    return [
        integration
        for integration in AVAILABLE_INTEGRATIONS.values()
        if integration.integration_type == integration_type and integration.is_active
    ]
