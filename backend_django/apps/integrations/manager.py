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

    def __init__(self, organisation_settings):
        """
        Initialize integration with organization settings.

        Args:
            organisation_settings: OrganisationSettings instance
        """
        self.organisation_settings = organisation_settings
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
        return all(
            [
                self.organisation_settings.snowflake_account,
                self.organisation_settings.snowflake_user,
                self.organisation_settings.snowflake_password,
                self.organisation_settings.snowflake_warehouse,
                self.organisation_settings.snowflake_database,
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

            conn = snowflake.connector.connect(
                account=self.organisation_settings.snowflake_account,
                user=self.organisation_settings.snowflake_user,
                password=self.organisation_settings.snowflake_password,
                warehouse=self.organisation_settings.snowflake_warehouse,
                database=self.organisation_settings.snowflake_database,
                schema=self.organisation_settings.snowflake_schema or "PUBLIC",
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
            account=self.organisation_settings.snowflake_account,
            user=self.organisation_settings.snowflake_user,
            password=self.organisation_settings.snowflake_password,
            warehouse=self.organisation_settings.snowflake_warehouse,
            database=self.organisation_settings.snowflake_database,
            schema=self.organisation_settings.snowflake_schema or "PUBLIC",
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
        return all(
            [
                self.organisation_settings.metabase_url,
                self.organisation_settings.metabase_username,
                self.organisation_settings.metabase_password,
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

            client = MetabaseClient(
                url=self.organisation_settings.metabase_url,
                username=self.organisation_settings.metabase_username,
                password=self.organisation_settings.metabase_password,
            )

            # Test by getting user info
            user_info = client.get_user_info()

            return {
                "success": True,
                "message": f'Connected to Metabase as {user_info.get("email", "unknown")}',
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

            client = MetabaseClient(
                url=self.organisation_settings.metabase_url,
                username=self.organisation_settings.metabase_username,
                password=self.organisation_settings.metabase_password,
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

        return {
            "success": True,
            "dashboard": dashboard,
            "dashboard_url": f"{self.organisation_settings.metabase_url}/dashboard/{dashboard['id']}",
        }


class IntegrationsManager:
    """
    Manager for handling all integrations for an organization.

    This class dynamically loads and manages enabled integrations,
    providing a unified interface for the conversation agent.
    """

    # Registry of available integrations
    AVAILABLE_INTEGRATIONS: Dict[str, Type[BaseIntegration]] = {
        "snowflake": SnowflakeIntegration,
        "metabase": MetabaseIntegration,
    }

    def __init__(self, organisation_settings):
        """
        Initialize integrations manager.

        Args:
            organisation_settings: OrganisationSettings instance
        """
        self.organisation_settings = organisation_settings
        self.logger = logging.getLogger(__name__)
        self._loaded_integrations = {}
        self._load_enabled_integrations()

    def _load_enabled_integrations(self):
        """Load all enabled and configured integrations."""
        enabled_integrations = self.organisation_settings.enabled_integrations

        for integration_key in enabled_integrations:
            if integration_key in self.AVAILABLE_INTEGRATIONS:
                try:
                    integration_class = self.AVAILABLE_INTEGRATIONS[integration_key]
                    integration = integration_class(self.organisation_settings)

                    if integration.is_configured():
                        self._loaded_integrations[integration_key] = integration
                        self.logger.info(f"Loaded integration: {integration.name}")
                    else:
                        self.logger.warning(
                            f"Integration {integration.name} enabled but not configured"
                        )

                except Exception as e:
                    self.logger.error(
                        f"Failed to load integration {integration_key}: {e}"
                    )

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
        Get all tools from all enabled integrations.

        Returns:
            List of tool definitions for LLM function calling
        """
        all_tools = []

        for integration in self._loaded_integrations.values():
            try:
                tools = integration.get_available_tools()
                # Prefix tool names with integration key to avoid conflicts
                for tool in tools:
                    tool["integration_key"] = integration.key
                all_tools.extend(tools)
            except Exception as e:
                self.logger.error(f"Failed to get tools from {integration.name}: {e}")

        return all_tools

    def execute_tool(
        self, tool_name: str, integration_key: str = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool from a specific integration.

        Args:
            tool_name: Name of the tool to execute
            integration_key: Key of the integration (optional, inferred from tool name)
            **kwargs: Tool parameters

        Returns:
            Dict with execution results
        """
        # If integration_key not provided, try to find it from tool metadata
        if not integration_key:
            # This would require storing integration_key in tool metadata
            # For now, require explicit integration_key
            return {
                "success": False,
                "error": "Integration key required for tool execution",
            }

        integration = self.get_integration(integration_key)
        if not integration:
            return {
                "success": False,
                "error": f"Integration {integration_key} not available",
            }

        return integration.execute_tool(tool_name, **kwargs)

    def test_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """Test connections for all enabled integrations."""
        results = {}

        for key, integration in self._loaded_integrations.items():
            try:
                results[key] = integration.test_connection()
            except Exception as e:
                results[key] = {"success": False, "message": f"Test failed: {str(e)}"}

        return results

    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all integrations (enabled, configured, etc.)."""
        status = {}

        for key, integration_class in self.AVAILABLE_INTEGRATIONS.items():
            integration = integration_class(self.organisation_settings)

            status[key] = {
                "name": integration.name,
                "enabled": key in self.organisation_settings.enabled_integrations,
                "configured": integration.is_configured(),
                "loaded": key in self._loaded_integrations,
            }

        return status

    def reload_integrations(self):
        """Reload all integrations (useful after configuration changes)."""
        self._loaded_integrations.clear()
        self._load_enabled_integrations()

    @classmethod
    def register_integration(cls, key: str, integration_class: Type[BaseIntegration]):
        """Register a new integration class."""
        cls.AVAILABLE_INTEGRATIONS[key] = integration_class

    def __str__(self):
        enabled = list(self._loaded_integrations.keys())
        return f"IntegrationsManager(enabled={enabled})"
