"""
Integrations Manager for handling dynamic integrations with external systems.

This module provides a framework for managing different types of integrations
(Snowflake, Metabase, etc.) that can be enabled/disabled per organization.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type
import logging
from django.conf import settings
import json

logger = logging.getLogger(__name__)


class BaseIntegration(ABC):
    """Base class for all integrations."""

    def __init__(self, org_integration):
        """
        Initialize integration with organization integration instance.

        Args:
            org_integration: OrganisationIntegration instance
        """
        self.org_integration = org_integration
        self.organisation_settings = org_integration.organisation.settings
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the integration."""
        pass

    @property
    @abstractmethod
    def key(self) -> str:
        """Unique key for the integration."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if integration is properly configured."""
        pass

    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the external system.

        Returns:
            Dict with 'success' boolean and 'message' string
        """
        pass

    @abstractmethod
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of tools/functions this integration provides.

        Returns:
            List of tool definitions compatible with LLM function calling
        """
        pass

    @abstractmethod
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a specific tool/function.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool-specific parameters

        Returns:
            Dict with execution results
        """
        pass


class MCPIntegration(BaseIntegration):
    """Integration for exposing ragstar as a remote MCP server."""

    @property
    def name(self) -> str:
        return "MCP Server"

    @property
    def key(self) -> str:
        return "mcp"

    def is_configured(self) -> bool:
        """Check if MCP server configuration is complete."""
        config = self.org_integration.configuration
        credentials = self.org_integration.credentials

        # Check if basic server configuration exists
        server_url = config.get("server_url")
        auth_provider = config.get("auth_provider")  # claude, openai, etc.

        # Check if OAuth credentials are configured for the auth provider
        oauth_configured = bool(credentials.get(f"{auth_provider}_oauth_config"))

        return bool(server_url and auth_provider and oauth_configured)

    def test_connection(self) -> Dict[str, Any]:
        """Test MCP server accessibility."""
        if not self.is_configured():
            return {
                "success": False,
                "message": "MCP server integration not properly configured",
            }

        try:
            config = self.org_integration.configuration
            server_url = config.get("server_url")

            # Test if the MCP server endpoint is accessible
            import requests

            response = requests.get(f"{server_url}/health", timeout=10)

            if response.status_code == 200:
                return {
                    "success": True,
                    "message": f"MCP server accessible at {server_url}",
                }
            else:
                return {
                    "success": False,
                    "message": f"MCP server returned status {response.status_code}",
                }

        except Exception as e:
            self.logger.error(f"MCP server connection test failed: {e}")
            return {"success": False, "message": f"Connection failed: {str(e)}"}

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tools that MCP server exposes to external LLMs."""
        return [
            {
                "name": "search_dbt_models",
                "description": "Search and retrieve information about dbt models in the knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for finding relevant dbt models",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_conversation_history",
                "description": "Retrieve past conversations and insights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of recent conversations to retrieve",
                            "default": 5,
                        },
                        "search_query": {
                            "type": "string",
                            "description": "Optional search query to filter conversations",
                        },
                    },
                },
            },
            {
                "name": "ask_data_question",
                "description": "Ask a question about the organization's data and get insights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Data question to analyze",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context for the question",
                        },
                    },
                    "required": ["question"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute MCP tools - these will be called by external LLMs."""
        if not self.is_configured():
            return {"success": False, "error": "MCP integration not configured"}

        try:
            if tool_name == "search_dbt_models":
                return self._search_dbt_models(**kwargs)
            elif tool_name == "get_conversation_history":
                return self._get_conversation_history(**kwargs)
            elif tool_name == "ask_data_question":
                return self._ask_data_question(**kwargs)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            self.logger.error(f"MCP tool execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _search_dbt_models(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for dbt models in the knowledge base."""
        from apps.knowledge_base.models import Model
        from django.db.models import Q

        models = Model.objects.filter(
            organisation=self.org_integration.organisation
        ).filter(
            Q(name__icontains=query)
            | Q(description__icontains=query)
            | Q(sql__icontains=query)
        )[
            :limit
        ]

        results = []
        for model in models:
            results.append(
                {
                    "name": model.name,
                    "description": model.description,
                    "schema": model.schema,
                    "database": model.database,
                    "columns": [
                        {"name": col.name, "type": col.data_type}
                        for col in model.columns.all()
                    ],
                }
            )

        return {
            "success": True,
            "results": results,
            "total_found": len(results),
        }

    def _get_conversation_history(
        self, limit: int = 5, search_query: str = None
    ) -> Dict[str, Any]:
        """Get conversation history."""
        from apps.workflows.models import Conversation

        conversations = Conversation.objects.filter(
            organisation=self.org_integration.organisation
        ).order_by("-created_at")

        if search_query:
            conversations = conversations.filter(
                question__question_text__icontains=search_query
            )

        conversations = conversations[:limit]

        results = []
        for conv in conversations:
            results.append(
                {
                    "id": str(conv.id),
                    "question": conv.question.question_text,
                    "created_at": conv.created_at.isoformat(),
                    "status": conv.status,
                    "summary": conv.conversation_context.get("summary", ""),
                }
            )

        return {
            "success": True,
            "conversations": results,
        }

    def _ask_data_question(self, question: str, context: str = None) -> Dict[str, Any]:
        """Process a data question through the workflow system."""
        from apps.workflows.tasks import run_conversation_workflow

        # This would typically queue a conversation workflow
        # For MCP, we might want a synchronous version or polling mechanism
        return {
            "success": True,
            "message": "Question submitted for processing",
            "question": question,
            "status": "processing",
        }


class SnowflakeIntegration(BaseIntegration):
    """Integration for Snowflake data warehouse."""

    @property
    def name(self) -> str:
        return "Snowflake"

    @property
    def key(self) -> str:
        return "snowflake"

    def is_configured(self) -> bool:
        """Check if Snowflake credentials are configured."""
        credentials = self.org_integration.credentials
        return all(
            [
                credentials.get("account"),
                credentials.get("user"),
                credentials.get("password"),
                credentials.get("warehouse"),
            ]
        )

    def test_connection(self) -> Dict[str, Any]:
        """Test Snowflake connection."""
        if not self.is_configured():
            return {
                "success": False,
                "message": "Snowflake integration not properly configured",
            }

        try:
            import snowflake.connector

            credentials = self.org_integration.credentials
            conn = snowflake.connector.connect(
                account=credentials.get("account"),
                user=credentials.get("user"),
                password=credentials.get("password"),
                warehouse=credentials.get("warehouse"),
                database=(
                    credentials.get("database") if credentials.get("database") else None
                ),
                schema=(
                    credentials.get("schema", "PUBLIC")
                    if credentials.get("database")
                    else None
                ),
            )

            cursor = conn.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            version = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return {
                "success": True,
                "message": f"Connected to Snowflake version {version}",
            }

        except Exception as e:
            self.logger.error(f"Snowflake connection test failed: {e}")
            return {"success": False, "message": f"Connection failed: {str(e)}"}

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get Snowflake tools for LLM."""
        return [
            {
                "name": "execute_snowflake_query",
                "description": "Execute a SQL query against Snowflake and return results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of rows to return",
                            "default": 100,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_snowflake_schema",
                "description": "Get schema information for tables and columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_pattern": {
                            "type": "string",
                            "description": "Pattern to match table names (optional)",
                        }
                    },
                },
            },
            {
                "name": "sample_snowflake_data",
                "description": "Get sample data from a specific table",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to sample",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of sample rows",
                            "default": 10,
                        },
                    },
                    "required": ["table_name"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute Snowflake tools."""
        if not self.is_configured():
            return {"success": False, "error": "Snowflake integration not configured"}

        try:
            if tool_name == "execute_snowflake_query":
                return self._execute_query(
                    kwargs.get("query"), kwargs.get("limit", 100)
                )
            elif tool_name == "get_snowflake_schema":
                return self._get_schema(kwargs.get("table_pattern"))
            elif tool_name == "sample_snowflake_data":
                return self._sample_data(
                    kwargs.get("table_name"), kwargs.get("limit", 10)
                )
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            self.logger.error(f"Snowflake tool execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_connection(self):
        """Get Snowflake connection."""
        import snowflake.connector

        return snowflake.connector.connect(
            account=self.org_integration.credentials.get("account"),
            user=self.org_integration.credentials.get("user"),
            password=self.org_integration.credentials.get("password"),
            warehouse=self.org_integration.credentials.get("warehouse"),
            database=(
                self.org_integration.credentials.get("database")
                if self.org_integration.credentials.get("database")
                else None
            ),
            schema=(
                self.org_integration.credentials.get("schema", "PUBLIC")
                if self.org_integration.credentials.get("database")
                else None
            ),
        )

    def _execute_query(self, query: str, limit: int = 100) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Add LIMIT if not present
            if limit > 0 and "LIMIT" not in query.upper():
                query = f"{query.strip().rstrip(';')} LIMIT {limit}"

            cursor.execute(query)

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Get results
            rows = cursor.fetchall()

            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }

        finally:
            cursor.close()
            conn.close()

    def _get_schema(self, table_pattern: Optional[str] = None) -> Dict[str, Any]:
        """Get schema information."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Build query to get table and column info
            query = """
            SELECT 
                table_name,
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns 
            WHERE table_schema = CURRENT_SCHEMA()
            """

            if table_pattern:
                query += f" AND table_name ILIKE '%{table_pattern}%'"

            query += " ORDER BY table_name, ordinal_position"

            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            # Group by table
            tables = {}
            for row in rows:
                table_name = row[0]
                if table_name not in tables:
                    tables[table_name] = []

                tables[table_name].append(
                    {
                        "column_name": row[1],
                        "data_type": row[2],
                        "is_nullable": row[3],
                        "column_default": row[4],
                    }
                )

            return {"success": True, "tables": tables}

        finally:
            cursor.close()
            conn.close()

    def _sample_data(self, table_name: str, limit: int = 10) -> Dict[str, Any]:
        """Get sample data from table."""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self._execute_query(query, limit)


class MetabaseIntegration(BaseIntegration):
    """Integration for Metabase analytics platform."""

    @property
    def name(self) -> str:
        return "Metabase"

    @property
    def key(self) -> str:
        return "metabase"

    def is_configured(self) -> bool:
        """Check if Metabase credentials are configured."""
        credentials = self.org_integration.credentials
        return all(
            [
                credentials.get("url"),
                credentials.get("api_key"),
                credentials.get("database_id"),
            ]
        )

    def test_connection(self) -> Dict[str, Any]:
        """Test Metabase connection."""
        if not self.is_configured():
            return {
                "success": False,
                "message": "Metabase integration not properly configured",
            }

        try:
            from apps.integrations.metabase.client import MetabaseClient

            credentials = self.org_integration.credentials
            client = MetabaseClient(
                metabase_url=credentials.get("url"),
                api_key=credentials.get("api_key"),
                database_id=credentials.get("database_id"),
            )

            # Test by making a simple API call
            collections = client.list_collections()

            return {
                "success": True,
                "message": f"Connected to Metabase successfully. Found {len(collections)} collections.",
            }

        except Exception as e:
            self.logger.error(f"Metabase connection test failed: {e}")
            return {"success": False, "message": f"Connection failed: {str(e)}"}

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get Metabase tools for LLM."""
        return [
            {
                "name": "list_metabase_dashboards",
                "description": "List available Metabase dashboards",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search": {
                            "type": "string",
                            "description": "Search term to filter dashboards",
                        }
                    },
                },
            },
            {
                "name": "create_metabase_dashboard",
                "description": "Create a new Metabase dashboard with questions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Dashboard name"},
                        "description": {
                            "type": "string",
                            "description": "Dashboard description",
                        },
                        "questions": {
                            "type": "array",
                            "description": "List of SQL queries to add as questions",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "sql": {"type": "string"},
                                    "visualization_type": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["name", "questions"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute Metabase tools."""
        if not self.is_configured():
            return {"success": False, "error": "Metabase integration not configured"}

        try:
            from apps.integrations.metabase.client import MetabaseClient

            credentials = self.org_integration.credentials
            client = MetabaseClient(
                metabase_url=credentials.get("url"),
                api_key=credentials.get("api_key"),
                database_id=credentials.get("database_id"),
            )

            if tool_name == "list_metabase_dashboards":
                return self._list_dashboards(client, kwargs.get("search"))
            elif tool_name == "create_metabase_dashboard":
                return self._create_dashboard(
                    client,
                    kwargs.get("name"),
                    kwargs.get("description", ""),
                    kwargs.get("questions", []),
                )
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            self.logger.error(f"Metabase tool execution failed: {e}")
            return {"success": False, "error": str(e)}

    def _list_dashboards(self, client, search: Optional[str] = None) -> Dict[str, Any]:
        """List Metabase dashboards."""
        dashboards = client.get_dashboards()

        if search:
            search_lower = search.lower()
            dashboards = [
                d
                for d in dashboards
                if search_lower in d.get("name", "").lower()
                or search_lower in d.get("description", "").lower()
            ]

        return {"success": True, "dashboards": dashboards}

    def _create_dashboard(
        self, client, name: str, description: str, questions: List[Dict]
    ) -> Dict[str, Any]:
        """Create Metabase dashboard."""
        dashboard = client.create_dashboard(name, description)

        # Add questions to dashboard
        for question_data in questions:
            question = client.create_question(
                name=question_data["name"],
                sql=question_data["sql"],
                visualization_type=question_data.get("visualization_type", "table"),
            )
            client.add_question_to_dashboard(dashboard["id"], question["id"])

        credentials = self.org_integration.credentials
        return {
            "success": True,
            "dashboard": dashboard,
            "dashboard_url": f"{credentials.get('url')}/dashboard/{dashboard['id']}",
        }


class SlackIntegration(BaseIntegration):
    """Integration for Slack workspace communication."""

    @property
    def name(self) -> str:
        return "Slack"

    @property
    def key(self) -> str:
        return "slack"

    def is_configured(self) -> bool:
        """Check if Slack integration is properly configured."""
        credentials = self.org_integration.credentials
        bot_token = credentials.get("bot_token")
        return bool(bot_token and bot_token.startswith("xoxb-"))

    def test_connection(self) -> Dict[str, Any]:
        """Test Slack bot token and workspace connection."""
        if not self.is_configured():
            return {
                "success": False,
                "message": "Slack integration not properly configured. Bot token required.",
            }

        try:
            credentials = self.org_integration.credentials
            bot_token = credentials.get("bot_token")

            # Test the bot token by calling Slack's auth.test API
            from .slack.handlers import get_team_info_from_token

            team_info = get_team_info_from_token(bot_token)

            if team_info:
                return {
                    "success": True,
                    "message": f"Connected to Slack workspace: {team_info.get('team_name', 'Unknown')}",
                    "team_info": team_info,
                }
            else:
                return {
                    "success": False,
                    "message": "Invalid bot token or unable to connect to Slack workspace",
                }

        except Exception as e:
            self.logger.error(f"Slack connection test failed: {e}")
            return {"success": False, "message": f"Connection test failed: {str(e)}"}

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get tools that Slack integration provides."""
        return [
            {
                "name": "send_slack_message",
                "description": "Send a message to a Slack channel or user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel": {
                            "type": "string",
                            "description": "Channel ID or user ID to send message to",
                        },
                        "message": {
                            "type": "string",
                            "description": "Message content to send",
                        },
                        "thread_ts": {
                            "type": "string",
                            "description": "Thread timestamp to reply in thread (optional)",
                        },
                    },
                    "required": ["channel", "message"],
                },
            },
            {
                "name": "get_slack_channels",
                "description": "Get list of Slack channels the bot has access to",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "types": {
                            "type": "string",
                            "description": "Channel types to include (public_channel, private_channel, mpim, im)",
                            "default": "public_channel,private_channel",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of channels to return",
                            "default": 100,
                        },
                    },
                },
            },
        ]

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute Slack-specific tools."""
        if not self.is_configured():
            return {
                "success": False,
                "error": "Slack integration not configured",
            }

        try:
            if tool_name == "send_slack_message":
                return self._send_message(**kwargs)
            elif tool_name == "get_slack_channels":
                return self._get_channels(**kwargs)
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                }

        except Exception as e:
            self.logger.error(f"Error executing Slack tool {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
            }

    def _send_message(
        self, channel: str, message: str, thread_ts: str = None
    ) -> Dict[str, Any]:
        """Send a message to Slack channel or user."""
        try:
            from slack_sdk import WebClient

            credentials = self.org_integration.credentials
            client = WebClient(token=credentials.get("bot_token"))

            response = client.chat_postMessage(
                channel=channel, text=message, thread_ts=thread_ts
            )

            return {
                "success": True,
                "message_ts": response["ts"],
                "channel": response["channel"],
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to send message: {str(e)}",
            }

    def _get_channels(
        self, types: str = "public_channel,private_channel", limit: int = 100
    ) -> Dict[str, Any]:
        """Get list of Slack channels."""
        try:
            from slack_sdk import WebClient

            credentials = self.org_integration.credentials
            client = WebClient(token=credentials.get("bot_token"))

            response = client.conversations_list(types=types, limit=limit)

            channels = []
            for channel in response["channels"]:
                channels.append(
                    {
                        "id": channel["id"],
                        "name": channel["name"],
                        "is_private": channel.get("is_private", False),
                        "is_member": channel.get("is_member", False),
                    }
                )

            return {
                "success": True,
                "channels": channels,
                "total": len(channels),
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get channels: {str(e)}",
            }


class IntegrationsManager:
    """
    Manager for handling all integrations for an organization.

    This class dynamically loads and manages enabled integrations,
    providing a unified interface for the conversation agent.
    """

    # Registry of available integrations
    AVAILABLE_INTEGRATIONS: Dict[str, Type[BaseIntegration]] = {
        "slack": SlackIntegration,
        "snowflake": SnowflakeIntegration,
        "metabase": MetabaseIntegration,
        "mcp": MCPIntegration,
    }

    def __init__(self, organisation):
        """
        Initialize integrations manager.

        Args:
            organisation: Organisation instance
        """
        self.organisation = organisation
        self.logger = logging.getLogger(__name__)
        self._loaded_integrations = {}
        self._load_enabled_integrations()

    def _load_enabled_integrations(self):
        """Load all enabled and configured integrations."""
        from .models import OrganisationIntegration

        enabled_integrations = OrganisationIntegration.objects.filter(
            organisation=self.organisation, is_enabled=True
        )

        for org_integration in enabled_integrations:
            integration_key = org_integration.integration_key

            if integration_key in self.AVAILABLE_INTEGRATIONS:
                try:
                    integration_class = self.AVAILABLE_INTEGRATIONS[integration_key]
                    integration = integration_class(org_integration)

                    if integration.is_configured():
                        self._loaded_integrations[integration_key] = integration
                        self.logger.info(f"Loaded integration: {integration_key}")
                    else:
                        self.logger.warning(
                            f"Integration {integration_key} is enabled but not configured"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Failed to load integration {integration_key}: {e}"
                    )
            else:
                self.logger.warning(f"Unknown integration key: {integration_key}")

    def get_available_integrations(self) -> List[str]:
        """Get list of available integration keys."""
        return list(self.AVAILABLE_INTEGRATIONS.keys())

    def get_enabled_integrations(self) -> List[BaseIntegration]:
        """Get list of enabled and configured integrations."""
        return list(self._loaded_integrations.values())

    def get_integration(self, key: str) -> Optional[BaseIntegration]:
        """Get specific integration by key."""
        return self._loaded_integrations.get(key)

    def is_integration_enabled(self, key: str) -> bool:
        """Check if integration is enabled and configured."""
        return key in self._loaded_integrations

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools from enabled integrations.

        Returns:
            List of tool definitions with integration metadata
        """
        all_tools = []

        for integration_key, integration in self._loaded_integrations.items():
            try:
                tools = integration.get_available_tools()
                for tool in tools:
                    tool["integration_key"] = integration_key
                    tool["integration_name"] = integration.name
                    all_tools.append(tool)

            except Exception as e:
                self.logger.error(f"Failed to get tools from {integration_key}: {e}")

        return all_tools

    def execute_tool(
        self, tool_name: str, integration_key: str = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool from a specific integration.

        Args:
            tool_name: Name of the tool to execute
            integration_key: Key of the integration to use
            **kwargs: Tool parameters

        Returns:
            Dict with execution results
        """
        if integration_key:
            integration = self.get_integration(integration_key)
            if not integration:
                return {
                    "success": False,
                    "error": f"Integration {integration_key} not available",
                }

            return integration.execute_tool(tool_name, **kwargs)

        else:
            # Try to find the tool in any enabled integration
            for integration in self._loaded_integrations.values():
                try:
                    tools = integration.get_available_tools()
                    tool_names = [tool["name"] for tool in tools]

                    if tool_name in tool_names:
                        return integration.execute_tool(tool_name, **kwargs)

                except Exception as e:
                    self.logger.error(f"Error checking tools in {integration.key}: {e}")

            return {"success": False, "error": f"Tool {tool_name} not found"}

    def test_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """Test connections for all enabled integrations."""
        results = {}

        for integration_key, integration in self._loaded_integrations.items():
            try:
                results[integration_key] = integration.test_connection()
            except Exception as e:
                results[integration_key] = {
                    "success": False,
                    "message": f"Test failed: {str(e)}",
                }

        return results

    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all integrations."""
        from .constants import get_active_integration_definitions

        # Get all available integrations from constants
        available_integrations = {
            defn.key: defn for defn in get_active_integration_definitions()
        }

        # Get organization integrations from database
        from .models import OrganisationIntegration

        org_integrations = {
            oi.integration_key: oi
            for oi in OrganisationIntegration.objects.filter(
                organisation=self.organisation
            )
        }

        status = {}
        for integration_key, integration_defn in available_integrations.items():
            org_integration = org_integrations.get(integration_key)
            loaded_integration = self._loaded_integrations.get(integration_key)

            if org_integration:
                status[integration_key] = {
                    "id": org_integration.id,
                    "key": integration_key,
                    "name": integration_defn.name,
                    "integration_type": integration_defn.integration_type,
                    "is_enabled": org_integration.is_enabled,
                    "is_configured": org_integration.is_configured,
                    "connection_status": org_integration.connection_status,
                    "last_tested_at": org_integration.last_tested_at,
                    "tools_count": (
                        len(loaded_integration.get_available_tools())
                        if loaded_integration
                        else 0
                    ),
                }
            else:
                # Integration not configured for this organization
                status[integration_key] = {
                    "id": None,
                    "key": integration_key,
                    "name": integration_defn.name,
                    "integration_type": integration_defn.integration_type,
                    "is_enabled": False,
                    "is_configured": False,
                    "connection_status": "not_configured",
                    "last_tested_at": None,
                    "tools_count": 0,
                }

        return status

    def reload_integrations(self):
        """Reload all integrations."""
        self._loaded_integrations.clear()
        self._load_enabled_integrations()

    @classmethod
    def register_integration(cls, key: str, integration_class: Type[BaseIntegration]):
        """Register a new integration class."""
        cls.AVAILABLE_INTEGRATIONS[key] = integration_class

    def __str__(self):
        enabled_count = len(self._loaded_integrations)
        return f"IntegrationsManager({enabled_count} integrations)"
