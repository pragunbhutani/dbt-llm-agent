import logging
import json
import asyncio
import re
from typing import Dict, List, Any, Optional, TypedDict
from typing_extensions import Annotated

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver

from django.conf import settings as django_settings
from asgiref.sync import sync_to_async

# Project specific imports
from apps.llm_providers.services import ChatService
from apps.accounts.models import OrganisationSettings
from apps.integrations.manager import IntegrationsManager
from .prompts import (
    create_sql_verifier_debug_prompt,
    create_schema_validation_prompt,
    create_style_check_prompt,
)
from .models import (
    SQLDebugResult,
    SQLSchemaValidationResult,
    SQLStyleCheckResult,
)
from apps.workflows.services import trigger_model_interpretation, ConversationLogger

# Snowflake connector (ensure it's listed in requirements if not already)
try:
    import snowflake.connector
except ImportError:
    snowflake = None
    logging.warning(
        "Snowflake connector not installed. SQLVerifier will not work with Snowflake."
    )

# Shared response schema
from apps.workflows.schemas import SQLVerificationResponse

logger = logging.getLogger(__name__)


# --- State Definition ---
class SQLVerifierState(TypedDict):
    original_sql_query: str
    current_sql_query: str
    warehouse_type: str
    is_valid: Optional[bool]
    execution_error: Optional[str]
    corrected_sql_query: Optional[str]
    debug_attempts: int
    max_debug_attempts: int
    dbt_models_info: Optional[List[Dict[str, Any]]]
    # Removed live_schema_context and live_column_values_context, will be managed by agent tools
    # live_schema_context: Optional[Dict[str, Any]]
    # live_column_values_context: Optional[Dict[str, Dict[str, List[Any]]]]

    debug_explanation: Optional[str]  # Explanation from LLM for the correction
    debugging_log: List[str]  # Log of actions and observations during verification

    # Agent-specific state
    messages: Annotated[
        List[BaseMessage], add_messages
    ]  # For conversational history with the agent
    agent_described_tables_info: Dict[
        str, Any
    ]  # Store schema info from agent's describe_table tool calls
    agent_listed_column_values: Dict[
        str, Dict[str, List[Any]]
    ]  # Store column values from agent's list_column_values tool calls
    agent_tool_calls: List[Any]  # To store pending tool calls from the agent

    # Style checking fields
    is_style_compliant: Optional[bool]
    style_violations: Optional[List[str]]
    # Potentially add a field for db_connection_details if it can vary


class SQLVerifierInput(BaseModel):
    sql_query: str
    max_debug_attempts: Optional[int] = Field(default=3)
    dbt_models_info: Optional[List[Dict[str, Any]]] = Field(default=None)
    # Potentially: db_connection_info: Optional[Dict[str, Any]] # If not globally configured


# --- SQL Verifier Workflow Class ---
class SQLVerifierWorkflow:
    """Workflow for verifying and optionally debugging SQL queries."""

    def __init__(
        self,
        org_settings: OrganisationSettings,
        memory: Optional[BaseCheckpointSaver] = None,  # For potential checkpointing
        max_debug_loops: int = 3,
    ):
        self.verbose = django_settings.RAGSTAR_LOG_LEVEL == "DEBUG"
        if self.verbose:
            logger.info(f"SQLVerifierWorkflow initialized with verbose={self.verbose}")

        try:
            self.chat_service = ChatService(org_settings=org_settings)
            self.llm = self.chat_service.get_client()
            if not self.llm:
                logger.error(
                    "SQLVerifierWorkflow: LLM client could not be initialized. Debugging LLM will not work."
                )
        except Exception as e:
            logger.error(
                f"SQLVerifierWorkflow: Failed to initialize chat service or LLM client: {e}",
                exc_info=self.verbose,
            )
            self.chat_service = None
            self.llm = None

        self.memory = memory
        self.default_max_debug_loops = max_debug_loops

        try:
            # Get Snowflake credentials from integration system instead of environment variables
            integrations_manager = IntegrationsManager(org_settings.organisation)
            snowflake_integration = integrations_manager.get_integration("snowflake")

            if snowflake_integration and snowflake_integration.is_configured():
                # Get credentials from the integration
                credentials = snowflake_integration.org_integration.credentials
                self.snowflake_creds = {
                    "account": credentials.get("account"),
                    "user": credentials.get("user"),
                    "password": credentials.get("password"),
                    "warehouse": credentials.get("warehouse"),
                    "database": credentials.get("database"),
                    "schema": credentials.get("schema"),
                }
                self.warehouse_type = "snowflake"
                logger.info(
                    "SQLVerifierWorkflow: Snowflake credentials loaded from integration system."
                )
            else:
                self.snowflake_creds = None
                self.warehouse_type = None
                logger.warning(
                    "SQLVerifierWorkflow: Snowflake integration not configured."
                )

        except Exception as e:
            logger.error(
                f"SQLVerifierWorkflow: Failed to load Snowflake credentials from integration system: {e}. Workflow may not function for Snowflake."
            )
            self.snowflake_creds = None
            self.warehouse_type = None

        self._define_tools()  # Define tools after credentials are initialized

        # No specific tools to define for external LLM calls yet, debugging is an internal LLM call node.
        # self.graph_app = self._build_graph() # Will be uncommented once nodes are defined

        # Note: self.tools is initialized inside _define_tools; avoid overriding here.

        # Optional conversation logger (propagated by orchestrator)
        self.conversation_logger: Optional[ConversationLogger] = None

    # --- Core SQL Execution Logic (adapted from QueryExecutorWorkflow) ---
    def _execute_sql_query_sync(
        self, sql_query: str
    ) -> tuple[Optional[List[tuple]], Optional[str]]:
        """
        Synchronously executes a SQL query against Snowflake.
        Returns a tuple: (results, error_message)
        Results being non-None indicates success, error_message non-None indicates failure.
        A simple execution that doesn't return rows (e.g., DDL, some DML) is a success if error_message is None.
        For verification, we mainly care if it *parses and runs* without error.
        """
        if self.warehouse_type != "snowflake" or not snowflake:
            logger.warning(
                f"Attempted to execute query on unsupported or unavailable warehouse: {self.warehouse_type}"
            )
            # For non-Snowflake, we can't execute. We might consider a dry-run parser if available.
            # For now, assume valid if not Snowflake, or return a specific error.
            # This part needs refinement based on how to handle non-Snowflake DBs for verification.
            # Option 1: Return (None, "Verification only supported for Snowflake")
            # Option 2: Assume valid, return ([], None) - Risky
            # Option 3: Attempt a generic SQL parse (needs a library)
            return (
                None,
                f"SQL execution/verification currently only supports Snowflake. Warehouse type: {self.warehouse_type}",
            )

        if not self.snowflake_creds:
            err_msg = "Snowflake credentials not available for query execution."
            logger.error(err_msg)
            return None, err_msg

        conn = None
        try:
            conn_params = {
                "user": self.snowflake_creds["user"],
                "password": self.snowflake_creds["password"],
                "account": self.snowflake_creds["account"],
            }
            if (
                "warehouse" in self.snowflake_creds
                and self.snowflake_creds["warehouse"]
            ):
                conn_params["warehouse"] = self.snowflake_creds["warehouse"]
            if "database" in self.snowflake_creds and self.snowflake_creds["database"]:
                conn_params["database"] = self.snowflake_creds["database"]
            if "schema" in self.snowflake_creds and self.snowflake_creds["schema"]:
                conn_params["schema"] = self.snowflake_creds["schema"]

            # For verification, we might want to run a "dry run" or "parse only" command
            # Snowflake has `VALIDATE()` function or `EXPLAIN`. Let's try a simple execute.
            # A more robust verification might be to prefix with `EXPLAIN USING TABULAR`
            # or use `SELECT PARSE_JSON('{}')` on a query part if it's just about syntax for some dialects.
            # For now, direct execution attempt is fine for initial version.

            conn = snowflake.connector.connect(**conn_params)
            cursor = conn.cursor()
            try:
                # For pure validation, we don't need to fetch results, just see if it errors.
                # However, some errors only occur on fetch.
                # A common way to validate is to try to EXPLAIN the query.
                # `EXPLAIN USING TABULAR ${sql_query}`
                # For now, let's stick to direct execution and catch errors.
                # We can refine this to use EXPLAIN later if direct execution is too slow or risky.

                logger.info(
                    f"Attempting to execute/validate SQL (first 200 chars): {sql_query[:200]}..."
                )
                cursor.execute(sql_query)

                # If execute() doesn't throw an error, the query is syntactically valid and runnable.
                # We don't need to fetch rows for basic validation.
                # row_count = cursor.rowcount
                logger.info(f"SQL query executed successfully without errors.")
                return [], None  # Success, no results needed for validation, no error.

            except snowflake.connector.Error as sf_err:
                error_msg = f"Snowflake SQL Error: {sf_err}"
                logger.warning(error_msg)  # Warning, as this is expected in a verifier
                return None, error_msg
            except Exception as e:
                error_msg = (
                    f"Unexpected error during query execution for validation: {e}"
                )
                logger.error(error_msg, exc_info=True)
                return None, error_msg
            finally:
                if cursor:
                    cursor.close()
        except snowflake.connector.Error as conn_err:
            error_msg = f"Snowflake Connection Error: {conn_err}"
            logger.error(error_msg, exc_info=self.verbose)
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error setting up Snowflake connection: {e}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
        finally:
            if conn and not conn.is_closed():
                conn.close()

    async def _execute_sql_query(
        self, sql_query: str
    ) -> tuple[Optional[List[tuple]], Optional[str]]:
        return await sync_to_async(self._execute_sql_query_sync)(sql_query=sql_query)

    def _fetch_table_schema_sync(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Synchronously fetches schema for a given table from Snowflake."""
        if (
            self.warehouse_type != "snowflake"
            or not snowflake
            or not self.snowflake_creds
        ):
            logger.warning(
                f"Cannot fetch live schema for {table_name}: Snowflake not configured or unavailable."
            )
            return None

        conn = None
        try:
            conn_params = {
                "user": self.snowflake_creds["user"],
                "password": self.snowflake_creds["password"],
                "account": self.snowflake_creds["account"],
            }
            if self.snowflake_creds.get("warehouse"):
                conn_params["warehouse"] = self.snowflake_creds["warehouse"]
            if self.snowflake_creds.get("database"):
                conn_params["database"] = self.snowflake_creds["database"]
            if self.snowflake_creds.get("schema"):
                conn_params["schema"] = self.snowflake_creds["schema"]

            conn = snowflake.connector.connect(**conn_params)
            cursor = conn.cursor()
            try:
                # Using DESCRIBE TABLE to get schema information
                describe_query = f"DESCRIBE TABLE {table_name};"
                logger.info(
                    f"Fetching schema for table: {table_name} with query: {describe_query}"
                )
                cursor.execute(describe_query)
                columns_data = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]

                schema_info = {"table_name": table_name, "columns": []}
                for row_data in columns_data:
                    col_dict = dict(zip(column_names, row_data))
                    schema_info["columns"].append(
                        {
                            "name": col_dict.get("name"),
                            "type": col_dict.get("type"),
                            "null?": col_dict.get("null?"),
                            "default": col_dict.get("default"),
                            "primary key": col_dict.get("primary key") == "Y",
                            "unique key": col_dict.get("unique key") == "Y",
                            "comment": col_dict.get("comment"),
                        }
                    )
                logger.info(
                    f"Successfully fetched schema for {table_name}: {len(schema_info['columns'])} columns."
                )
                return schema_info
            except snowflake.connector.Error as sf_err:
                logger.error(
                    f"Snowflake error fetching schema for {table_name}: {sf_err}"
                )
                return None
            finally:
                if cursor:
                    cursor.close()
        except Exception as e:
            logger.error(
                f"Error connecting to Snowflake or fetching schema for {table_name}: {e}"
            )
            return None
        finally:
            if conn and not conn.is_closed():
                conn.close()

    async def _fetch_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        return await sync_to_async(self._fetch_table_schema_sync)(table_name=table_name)

    def _fetch_column_values_sync(
        self, table_name: str, column_name: str, limit: int = 20
    ) -> tuple[Optional[List[Any]], Optional[str]]:
        """Synchronously fetches distinct values for a given column from Snowflake."""
        if (
            self.warehouse_type != "snowflake"
            or not snowflake
            or not self.snowflake_creds
        ):
            logger.warning(
                f"Cannot fetch column values for {table_name}.{column_name}: Snowflake not configured or unavailable."
            )
            return None, "Snowflake not configured or unavailable."

        conn = None
        try:
            conn_params = {
                "user": self.snowflake_creds["user"],
                "password": self.snowflake_creds["password"],
                "account": self.snowflake_creds["account"],
            }
            if self.snowflake_creds.get("warehouse"):
                conn_params["warehouse"] = self.snowflake_creds["warehouse"]
            if self.snowflake_creds.get("database"):
                conn_params["database"] = self.snowflake_creds["database"]
            if self.snowflake_creds.get("schema"):
                conn_params["schema"] = self.snowflake_creds["schema"]

            conn = snowflake.connector.connect(**conn_params)
            cursor = conn.cursor()

            # Ensure table_name and column_name are properly escaped if they can contain special characters.
            # For simplicity, assuming they are valid identifiers here.
            query = f'SELECT DISTINCT "{column_name}" FROM {table_name} LIMIT {limit}'

            if self.verbose:
                logger.info(f"Fetching column values with query: {query}")
            cursor.execute(query)
            rows = cursor.fetchall()
            column_values = [row[0] for row in rows]
            return column_values, None

        except snowflake.connector.Error as sf_err:
            error_msg = f"Snowflake SQL Error fetching column values for {table_name}.{column_name}: {sf_err}"
            logger.error(error_msg, exc_info=self.verbose)
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error fetching column values for {table_name}.{column_name}: {e}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
        finally:
            if conn and not conn.is_closed():
                conn.close()

    async def _fetch_column_values(
        self, table_name: str, column_name: str, limit: int = 20
    ) -> tuple[Optional[List[Any]], Optional[str]]:
        return await sync_to_async(self._fetch_column_values_sync)(
            table_name=table_name, column_name=column_name, limit=limit
        )

    def _define_tools(self):
        """Defines the tools available to the debugging agent."""

        self.tools = []

        # Only expose Snowflake-dependent tools when credentials are available. This prevents
        # the agent from hallucinating calls in organisations that have not configured a
        # warehouse connection.
        if self.snowflake_creds:

            @tool
            async def describe_table_tool(table_name: str) -> Dict[str, Any]:
                """Get the schema description for a specific table.
                Input must be a fully-qualified table name.
                """
                if not isinstance(table_name, str) or not table_name.strip():
                    return {
                        "error": "Invalid table_name provided. Must be a non-empty string."
                    }

                logger.info(
                    f"Agent tool: describe_table_tool called for table: {table_name}"
                )

                # Conversation logging
                if getattr(self, "conversation_logger", None):
                    await sync_to_async(self.conversation_logger.log_tool_call)(
                        tool_name="describe_table_tool",
                        tool_input={"table_name": table_name},
                    )

                schema_info = await self._fetch_table_schema(table_name)
                if schema_info:
                    return schema_info
                return {
                    "error": f"Could not fetch schema for table '{table_name}'. It might not exist or you may lack permissions."
                }

            @tool
            async def list_column_values_tool(
                table_name: str, column_name: str, limit: Optional[int] = 10
            ) -> Dict[str, Any]:
                """Return distinct values for a column. Useful for debugging joins and filters."""

                if not isinstance(table_name, str) or not table_name.strip():
                    return {
                        "error": "Invalid table_name provided. Must be a non-empty string."
                    }
                if not isinstance(column_name, str) or not column_name.strip():
                    return {
                        "error": "Invalid column_name provided. Must be a non-empty string."
                    }

                effective_limit = limit if limit is not None and limit > 0 else 10
                logger.info(
                    f"Agent tool: list_column_values_tool called for {table_name}.{column_name} with limit {effective_limit}"
                )

                # Conversation logging
                if getattr(self, "conversation_logger", None):
                    await sync_to_async(self.conversation_logger.log_tool_call)(
                        tool_name="list_column_values_tool",
                        tool_input={
                            "table_name": table_name,
                            "column_name": column_name,
                            "limit": effective_limit,
                        },
                    )

                values, error = await self._fetch_column_values(
                    table_name, column_name, limit=effective_limit
                )
                if error:
                    return {
                        "error": f"Error fetching column values for {table_name}.{column_name}: {error}"
                    }
                if values is not None:
                    return {
                        "table_name": table_name,
                        "column_name": column_name,
                        "distinct_values": values,
                    }
                return {
                    "error": f"No distinct values returned or an unknown error occurred for {table_name}.{column_name}"
                }

                # Bind tools (empty list is fine – LLM just gets no tool schema)
                if self.llm:
                    self.llm_with_tools = self.llm.bind_tools(self.tools)
                else:
                    self.llm_with_tools = None
                    logger.error(
                        "SQLVerifierWorkflow: LLM not available, cannot bind tools for agentic debugging."
                    )

            # Register the tools so the agent schema includes them
            self.tools.extend([describe_table_tool, list_column_values_tool])

        # Outside the Snowflake-credentials block so attribute always exists
        # Bind tools (empty list is fine – LLM just gets no tool schema)
        if self.llm:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = None
            logger.error(
                "SQLVerifierWorkflow: LLM not available, cannot bind tools for agentic debugging."
            )

    # --- Node Functions (to be defined) ---
    async def start_verification_node(self, state: SQLVerifierState) -> Dict[str, Any]:
        """
        Initializes the state for the verification process.
        This node sets up the initial query, debug attempts, and other necessary fields.
        Resets agent conversation history for a new verification pass.
        """
        current_query = state["original_sql_query"]
        max_attempts = state["max_debug_attempts"]
        current_debugging_log = state.get("debugging_log", [])
        current_debugging_log.append(
            f"Starting verification for query: {current_query[:200]}..."
        )

        if self.verbose:
            logger.info(
                f"start_verification_node: Initializing with query: {current_query[:100]}..."
            )
            logger.info(f"start_verification_node: Max debug attempts: {max_attempts}")

        updates = {
            "current_sql_query": current_query,
            "debug_attempts": 0,  # Reset debug attempts for a new run
            "execution_error": None,
            "corrected_sql_query": None,
            "is_valid": None,
            "debug_explanation": None,
            "dbt_models_info": state.get("dbt_models_info"),
            "max_debug_attempts": max_attempts,
            "debugging_log": current_debugging_log,
            # Initialize/reset agent state fields
            "messages": [],
            "agent_described_tables_info": {},
            "agent_listed_column_values": {},
            "agent_tool_calls": [],
        }

        # Log a summary of dbt_models_info if present
        updates_log_copy = updates.copy()
        if (
            "dbt_models_info" in updates_log_copy
            and updates_log_copy["dbt_models_info"] is not None
        ):
            updates_log_copy["dbt_models_info"] = (
                f"<Present: {len(updates_log_copy['dbt_models_info'])} models>"
            )
        else:
            updates_log_copy["dbt_models_info"] = "<Not Present>"

        # Also truncate current_sql_query for logging
        if "current_sql_query" in updates_log_copy and isinstance(
            updates_log_copy["current_sql_query"], str
        ):
            updates_log_copy["current_sql_query"] = (
                f"{updates_log_copy['current_sql_query'][:100]}..."
            )

        logger.info(
            f"start_verification_node: Updates being returned: {updates_log_copy}"
        )
        return updates

    async def execute_sql_node(self, state: SQLVerifierState) -> Dict[str, Any]:
        """
        Executes the current SQL query and updates the state with the result.
        """
        query = state["current_sql_query"]
        current_debugging_log = state.get("debugging_log", [])
        current_debugging_log.append(
            f"Attempting to execute SQL (attempt {state['debug_attempts'] + 1}): {query[:200]}..."
        )
        if self.verbose:
            logger.info(
                f"execute_sql_node: Executing query (attempt {state['debug_attempts'] + 1}): {query[:100]}..."
            )

        # NEW: Log the execution attempt as a tool call **before** running the query so the code path is always hit
        if getattr(self, "conversation_logger", None):
            try:
                await sync_to_async(self.conversation_logger.log_tool_call)(
                    tool_name="execute_sql_query",
                    tool_input={"sql_query": query[:400]},
                )
            except Exception as e:
                logger.error(
                    f"Failed to log execute_sql_query tool call: {e}",
                    exc_info=self.verbose,
                )

        _, error = await self._execute_sql_query(query)

        if error:
            current_debugging_log.append(f"Execution failed: {error}")
            if self.verbose:
                logger.warning(f"execute_sql_node: Execution error: {error}")
            return {
                "execution_error": error,
                "is_valid": False,
                "debugging_log": current_debugging_log,
            }
        else:
            current_debugging_log.append("Execution successful.")
            if self.verbose:
                logger.info("execute_sql_node: Execution successful.")
            return {
                "execution_error": None,
                "is_valid": True,
                "debugging_log": current_debugging_log,
            }

    async def attempt_debug_or_fail_node(
        self, state: SQLVerifierState
    ) -> Dict[str, Any]:
        """
        Determines if debugging should be attempted or if the process should fail.
        Increments debug attempts.
        """
        current_debugging_log = state.get("debugging_log", [])
        debug_attempts = state.get("debug_attempts", 0) + 1
        max_attempts = state.get("max_debug_attempts", self.default_max_debug_loops)

        current_debugging_log.append(
            f"Attempting debug: {debug_attempts} of {max_attempts}."
        )
        logger.info(
            f"attempt_debug_or_fail_node: Debug attempt {debug_attempts}, Max attempts: {max_attempts}"
        )
        logger.warning(  # Log the current error that triggered this debug attempt
            f"attempt_debug_or_fail_node: Current execution error: {state.get('execution_error')}"
        )

        updates: Dict[str, Any] = {
            "debug_attempts": debug_attempts,
            "debugging_log": current_debugging_log,
        }

        # Add a new HumanMessage to the messages list describing the current failure.
        # This ensures the LLM is aware of the latest error it needs to fix.
        # The 'messages' field in the state uses `add_messages`, so this new message will be appended by LangGraph.
        new_human_message_content = (
            f"The previously attempted SQL query:\\n```sql\\n{state['current_sql_query']}\\n```\\n"
            f"Failed with the following error:\\n{state['execution_error']}\\n\\n"
            f"Please analyze this error and provide a corrected SQL query and explanation."
        )
        updates["messages"] = [HumanMessage(content=new_human_message_content)]

        return updates

    async def agent_tool_executor_node(self, state: SQLVerifierState) -> Dict[str, Any]:
        """Executes tools called by the agent and returns their outputs as ToolMessages."""
        tool_calls = state.get("agent_tool_calls", [])
        current_debugging_log = state.get("debugging_log", [])
        messages = list(state.get("messages", []))

        agent_described_tables_info = dict(state.get("agent_described_tables_info", {}))
        agent_listed_column_values = dict(state.get("agent_listed_column_values", {}))

        if not tool_calls:
            current_debugging_log.append(
                "Agent tool executor: No tool calls to execute."
            )
            return {"debugging_log": current_debugging_log, "agent_tool_calls": []}

        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")

            current_debugging_log.append(
                f"Agent tool executor: Executing tool '{tool_name}' with args: {tool_args}"
            )
            if self.verbose:
                logger.info(f"Agent tool executor: Executing tool call: {tool_call}")

            # Find the tool in self.tools
            executable_tool = next((t for t in self.tools if t.name == tool_name), None)

            if executable_tool:
                try:
                    # Note: Langchain tools expect arguments as a single dict or individual args.
                    # Here, assuming tool_args is the dict for the tool.
                    tool_output_content = await executable_tool.ainvoke(tool_args)

                    # Store results if they are from our specific data-fetching tools
                    if (
                        tool_name == "describe_table_tool"
                        and isinstance(tool_output_content, dict)
                        and not tool_output_content.get("error")
                    ):
                        table_name_arg = tool_args.get("table_name")
                        if table_name_arg:
                            agent_described_tables_info[table_name_arg] = (
                                tool_output_content
                            )
                            current_debugging_log.append(
                                f"Stored schema for '{table_name_arg}' from describe_table_tool."
                            )

                    elif (
                        tool_name == "list_column_values_tool"
                        and isinstance(tool_output_content, dict)
                        and not tool_output_content.get("error")
                    ):
                        table_name_arg = tool_args.get("table_name")
                        column_name_arg = tool_args.get("column_name")
                        if (
                            table_name_arg
                            and column_name_arg
                            and "distinct_values" in tool_output_content
                        ):
                            if table_name_arg not in agent_listed_column_values:
                                agent_listed_column_values[table_name_arg] = {}
                            agent_listed_column_values[table_name_arg][
                                column_name_arg
                            ] = tool_output_content["distinct_values"]
                            current_debugging_log.append(
                                f"Stored column values for '{table_name_arg}.{column_name_arg}' from list_column_values_tool."
                            )

                except Exception as e:
                    tool_output_content = f"Error executing tool {tool_name}: {e}"
                    logger.error(
                        f"Agent tool executor: Error executing tool {tool_name}: {e}",
                        exc_info=True,
                    )
                    current_debugging_log.append(
                        f"Agent tool executor: Error executing tool {tool_name}: {e}"
                    )

                tool_messages.append(
                    ToolMessage(
                        content=(
                            json.dumps(tool_output_content)
                            if isinstance(tool_output_content, dict)
                            else str(tool_output_content)
                        ),
                        tool_call_id=tool_id,
                    )
                )
            else:
                current_debugging_log.append(
                    f"Agent tool executor: Tool '{tool_name}' not found."
                )
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: Tool '{tool_name}' not found.",
                        tool_call_id=tool_id,
                    )
                )

        messages.extend(tool_messages)
        current_debugging_log.append(
            f"Agent tool executor: Added {len(tool_messages)} tool messages to history."
        )

        return {
            "messages": messages,
            "agent_tool_calls": [],  # Clear executed calls
            "debugging_log": current_debugging_log,
            "agent_described_tables_info": agent_described_tables_info,
            "agent_listed_column_values": agent_listed_column_values,
        }

    async def invoke_debug_llm_node(self, state: SQLVerifierState) -> Dict[str, Any]:
        """
        Invokes the SQL Debugging Agent.
        Manages message history and tool call requests.
        """
        current_debugging_log = state.get("debugging_log", [])
        messages = list(state.get("messages", []))  # Get a mutable copy

        # Ensure SystemMessage is present, typically the first message.
        # attempt_debug_or_fail_node now adds a HumanMessage for retries,
        # so `messages` list won't be empty on a retry path here.
        # The initial population of SystemMessage + HumanMessage (for first error)
        # happens in start_verification_node which transitions to invoke_debug_llm_node (or execute_sql_node first).
        # For SQLVerifier, the first call to invoke_debug_llm_node comes after a failure in execute_sql_node,
        # via attempt_debug_or_fail_node. So messages should already have System + Human(err1).
        # Let's ensure SystemMessage is at the start if we are building messages from scratch.

        if not messages or not isinstance(messages[0], SystemMessage):
            # This case should ideally not be hit if graph is structured correctly,
            # but as a safeguard, prepend SystemMessage.
            system_prompt_str = create_sql_verifier_debug_prompt(
                warehouse_type=state.get("warehouse_type", "snowflake"),
                dbt_models_info=state.get("dbt_models_info"),
            )
            messages.insert(0, SystemMessage(content=system_prompt_str))
            current_debugging_log.append(
                "Safeguard: Prepended SystemMessage in invoke_debug_llm_node as it was missing or not first."
            )
            logger.warning(
                "invoke_debug_llm_node: Safeguard triggered for SystemMessage."
            )

        # Prepare the actual list of messages to be sent to the LLM for this invocation.
        llm_actual_input_messages = list(messages)

        # Problem 1 Fix: If the message before the current Human error message
        # (which should now be the actual last one due to attempt_debug_or_fail_node's addition)
        # was an AIMessage with no tool_calls (i.e., a JSON response from the LLM that failed),
        # we should remove it to prevent confusing the LLM.
        if (
            len(llm_actual_input_messages) >= 2
        ):  # Need at least [..., PrevAI, CurrentHumanError]
            # The last message is HumanMessage(current_error) added by attempt_debug_or_fail_node
            # The one before it (-2) is potentially the problematic AI JSON response
            potential_prev_ai_msg = llm_actual_input_messages[-2]
            if (
                isinstance(potential_prev_ai_msg, AIMessage)
                and not potential_prev_ai_msg.tool_calls
            ):
                logger.info(
                    "invoke_debug_llm_node: Removing second-to-last message as it was likely a "
                    "failed AI JSON response from the previous debug attempt."
                )
                llm_actual_input_messages.pop(
                    -2
                )  # Remove the AI's previous JSON output

        current_debugging_log.append(
            f"Invoking SQL Debugging Agent with {len(llm_actual_input_messages)} messages for LLM (after potential pruning). Full history length: {len(messages)}."
        )
        if self.verbose:
            # Log detailed message content previews for the messages being sent to LLM
            detailed_message_log = []
            for i, m in enumerate(llm_actual_input_messages):
                content_preview = ""
                if hasattr(m, "content"):
                    if isinstance(m.content, list):
                        preview_parts = []
                        for part in m.content:
                            if isinstance(part, str):
                                preview_parts.append(
                                    part[:200] + "..." if len(part) > 200 else part
                                )
                            elif isinstance(part, dict) and "text" in part:
                                preview_parts.append(
                                    part["text"][:200] + "..."
                                    if len(part["text"]) > 200
                                    else part["text"]
                                )
                            else:
                                preview_parts.append(str(part)[:200] + "...")
                        content_preview = " | ".join(preview_parts)
                    elif isinstance(m.content, str):
                        content_preview = (
                            m.content[:300] + "..."
                            if len(m.content) > 300
                            else m.content
                        )
                    else:
                        content_preview = (
                            f"(Non-string/list content: {type(m.content)})"
                        )
                else:
                    content_preview = "(No content attribute)"

                tool_calls_summary = ""
                if hasattr(m, "tool_calls") and m.tool_calls:
                    tool_calls_summary = f", ToolCalls: {m.tool_calls}"

                detailed_message_log.append(
                    f"  LLM_INPUT {i}. {type(m).__name__}: {content_preview}{tool_calls_summary}"
                )
            logger.info(
                f"invoke_debug_llm_node: Messages being sent to LLM ({len(llm_actual_input_messages)} total):\n"
                + "\n".join(detailed_message_log)
            )

        if not self.llm_with_tools:
            error_msg = (
                "LLM with tools not available. Cannot proceed with agentic debugging."
            )
            logger.error(f"invoke_debug_llm_node: {error_msg}")
            current_debugging_log.append(error_msg)
            return {
                "corrected_sql_query": None,
                "debug_explanation": error_msg,
                "debugging_log": current_debugging_log,
                "messages": messages,
                "agent_tool_calls": [],
            }

        try:
            # Invoke the agent using the pruned messages
            ai_response_message = await self.llm_with_tools.ainvoke(
                llm_actual_input_messages
            )

            # --- NEW: Log token usage for this LLM call ---
            if getattr(self, "conversation_logger", None):
                try:
                    usage_meta = (
                        getattr(ai_response_message, "additional_kwargs", {}).get(
                            "usage", {}
                        )
                        or getattr(ai_response_message, "response_metadata", {}).get(
                            "usage", {}
                        )
                        or getattr(ai_response_message, "usage_metadata", {})
                    )

                    prompt_tokens = int(
                        usage_meta.get("prompt_tokens")
                        or usage_meta.get("input_tokens")
                        or 0
                    )
                    completion_tokens = int(
                        usage_meta.get("completion_tokens")
                        or usage_meta.get("output_tokens")
                        or 0
                    )

                    if prompt_tokens:
                        await sync_to_async(
                            self.conversation_logger.log_agent_response
                        )(
                            content="[LLM Prompt – SQL Verifier]",
                            tokens_used=prompt_tokens,
                            metadata={"type": "llm_input"},
                        )

                    await sync_to_async(self.conversation_logger.log_agent_response)(
                        content=(
                            ai_response_message.content
                            if hasattr(ai_response_message, "content")
                            else "[LLM Response]"
                        ),
                        tokens_used=completion_tokens,
                        metadata={"type": "llm_output"},
                    )
                except Exception as log_err:
                    logger.warning(
                        f"SQLVerifierWorkflow: failed to record LLM token usage: {log_err}"
                    )

            # IMPORTANT: Append the new AI response to the original `messages` list
            # (which reflects the full state history being built up), NOT to `llm_actual_input_messages`.
            messages.append(ai_response_message)
            current_debugging_log.append(
                f"Agent responded. Total messages: {len(messages)}"
            )
            if self.verbose:
                # Summarize AI response to avoid verbose logging
                ai_response_summary = f"{type(ai_response_message).__name__}(content_length={len(ai_response_message.content) if hasattr(ai_response_message, 'content') else 'N/A'}, tool_calls_count={len(ai_response_message.tool_calls) if hasattr(ai_response_message, 'tool_calls') else 'N/A'})"
                logger.info(
                    f"invoke_debug_llm_node: Agent AIMessage summary: {ai_response_summary}"
                )

            # Check for tool calls
            if ai_response_message.tool_calls:
                current_debugging_log.append(
                    f"Agent requested tool calls: {ai_response_message.tool_calls}"
                )
                return {
                    "messages": messages,
                    "agent_tool_calls": ai_response_message.tool_calls,  # Pass tool calls to ToolNode
                    "debugging_log": current_debugging_log,
                    # No corrected_sql or explanation yet, agent needs tools executed
                    "corrected_sql_query": None,
                    "debug_explanation": None,
                }
            else:
                # Agent provided a final answer (hopefully JSON)
                current_debugging_log.append(
                    "Agent provided final answer without tool calls."
                )
                response_content = ai_response_message.content

                # --- Start of corrected JSON parsing for invoke_debug_llm_node ---
                response_content_str = ""
                if isinstance(response_content, list):
                    # Handle list content, potentially from models like Anthropic
                    processed_parts = []
                    for item in response_content:
                        if isinstance(item, str):
                            processed_parts.append(item)
                        elif (
                            isinstance(item, dict) and "text" in item
                        ):  # Common pattern for text parts
                            processed_parts.append(item["text"])
                        else:
                            # Fallback: convert other types to string, though ideally content is text
                            try:
                                processed_parts.append(str(item))
                            except Exception:
                                # If str() conversion fails, log and skip this part
                                logger.warning(
                                    f"Could not convert item to string: {type(item)}"
                                )
                    response_content_str = "\n".join(processed_parts)
                elif isinstance(response_content, str):
                    response_content_str = response_content
                else:
                    # If content is neither list nor string, log an error and prepare for failure
                    logger.error(
                        f"invoke_debug_llm_node: Unexpected AI response content type: {type(response_content)}. Content: {response_content}"
                    )
                    response_content_str = ""  # Ensure it's a string to avoid downstream errors with re.search

                parsed_response_data = {}
                corrected_sql = None
                explanation = None
                json_parsing_error = None

                try:
                    if not response_content_str.strip():
                        raise json.JSONDecodeError(
                            "AI response content is empty after processing.",
                            response_content_str,
                            0,
                        )

                    match = re.search(
                        r"```json\\s*(\\{.*?\\})\\s*```",
                        response_content_str,
                        re.DOTALL,
                    )
                    if match:
                        json_str = match.group(1)
                    else:
                        # Fallback if no markdown fences, try to find JSON directly
                        start_index = response_content_str.find("{")
                        end_index = response_content_str.rfind("}")
                        if (
                            start_index != -1
                            and end_index != -1
                            and end_index > start_index
                        ):
                            json_str = response_content_str[start_index : end_index + 1]
                        else:
                            json_str = ""

                    if not json_str.strip():
                        raise json.JSONDecodeError(
                            "No JSON found in agent's final response after attempting to strip markdown.",
                            response_content_str,
                            0,
                        )

                    parsed_response_data = json.loads(json_str)
                    corrected_sql = parsed_response_data.get("corrected_sql")
                    explanation = parsed_response_data.get("explanation")
                    current_debugging_log.append(
                        f"Agent final answer parsed. Corrected SQL: {bool(corrected_sql)}, Explanation provided: {bool(explanation)}"
                    )
                except json.JSONDecodeError as e:
                    json_parsing_error = f"Failed to parse agent's final JSON response: {e}. Response: {response_content_str[:500]}..."
                    logger.error(f"invoke_debug_llm_node: {json_parsing_error}")
                    current_debugging_log.append(json_parsing_error)
                    # Set to None, error will be handled below
                    corrected_sql = None
                    explanation = json_parsing_error  # Pass error as explanation

                # --- End of corrected JSON parsing for invoke_debug_llm_node ---

                # If parsing failed, corrected_sql will be None and explanation will contain the error.
                # If parsing succeeded but keys are missing, they will be None.
                return {
                    "corrected_sql_query": corrected_sql,
                    "debug_explanation": explanation,  # This will be the error if JSON parsing failed
                    "messages": messages,
                    "agent_tool_calls": [],
                    "debugging_log": current_debugging_log,
                }

        except Exception as e:  # This is the outer try-except for the whole node
            error_msg = f"Error invoking SQL Debugging Agent (outer try-except): {e}"
            logger.error(f"invoke_debug_llm_node: {error_msg}", exc_info=True)
            current_debugging_log.append(error_msg)
            return {
                "corrected_sql_query": None,
                "debug_explanation": error_msg,
                "messages": messages,  # Return current messages for continuity if possible
                "agent_tool_calls": [],
                "debugging_log": current_debugging_log,
            }

    async def update_query_from_llm_suggestion_node(
        self, state: SQLVerifierState
    ) -> Dict[str, Any]:
        """
        Updates the current_sql_query with the LLM's suggestion if available.
        Resets execution_error to allow re-evaluation.
        """
        corrected_sql = state.get("corrected_sql_query")
        current_debugging_log = state.get("debugging_log", [])

        if corrected_sql:
            current_debugging_log.append(
                f"Updating current query with LLM suggestion: {corrected_sql[:200]}..."
            )
            if self.verbose:
                logger.info(
                    f"update_query_from_llm_suggestion_node: Updating current_sql_query to: {corrected_sql[:100]}..."
                )
            return {
                "current_sql_query": corrected_sql,
                "execution_error": None,  # Reset error before retrying
                "is_valid": None,  # Reset validity, it needs to be re-checked
                "debugging_log": current_debugging_log,
            }
        else:
            current_debugging_log.append(
                "No LLM suggestion available, keeping original query for this attempt."
            )
            if self.verbose:
                logger.warning(
                    "update_query_from_llm_suggestion_node: No corrected_sql_query found in state. Query not updated."
                )
            # If there's no correction, we typically wouldn't retry with the same query unless some other state changed.
            # The graph logic (should_retry_execution_after_llm_debug) handles whether to proceed.
            return {"debugging_log": current_debugging_log}

    async def pre_validate_schema_node(self, state: SQLVerifierState) -> Dict[str, Any]:
        logger.info("Node: pre_validate_schema_node")
        # Log a summary of dbt_models_info
        dbt_models_info_state = state.get("dbt_models_info")
        if dbt_models_info_state is not None:
            logger.info(
                f"pre_validate_schema_node: Current state dbt_models_info: <Present: {len(dbt_models_info_state)} models>"
            )
        else:
            logger.info(
                f"pre_validate_schema_node: Current state dbt_models_info: <Not Present>"
            )

        updates: Dict[str, Any] = {"is_valid": None, "execution_error": None}
        current_query = state.get("current_sql_query", "")
        dbt_models = state.get("dbt_models_info")

        # Run pre-validation if we have dbt_models_info and LLM is available
        if dbt_models and self.llm_with_tools:
            logger.info(
                "dbt_models_info available, attempting LLM-based schema pre-validation."
            )
            try:
                prompt_content = create_schema_validation_prompt(
                    sql_query=current_query, dbt_models_info=dbt_models
                )
                messages_to_llm = [HumanMessage(content=prompt_content)]
                if self.verbose:
                    logger.info(
                        f"Schema Validation LLM Prompt: {prompt_content[:500]}..."
                    )

                response = await self.llm_with_tools.ainvoke(messages_to_llm)
                if self.verbose:
                    logger.info(f"Schema Validation LLM Response: {response.content}")

                parsed_response = None
                if isinstance(response.content, str):
                    response_content_str = response.content
                    try:
                        # Attempt to parse the LLM response (expecting JSON)
                        # Handle potential markdown code fences around JSON
                        match = re.search(
                            r"```json\\s*(\\{.*?\\})\\s*```",
                            response_content_str,
                            re.DOTALL,
                        )
                        if match:
                            json_str = match.group(1)
                        else:
                            # Fallback if no markdown fences, try to find JSON directly
                            start_index = response_content_str.find("{")
                            end_index = response_content_str.rfind("}")
                            if (
                                start_index != -1
                                and end_index != -1
                                and end_index > start_index
                            ):
                                json_str = response_content_str[
                                    start_index : end_index + 1
                                ]
                            else:
                                json_str = (
                                    ""  # Set to empty if no valid JSON object is found
                                )

                        if not json_str.strip():
                            raise json.JSONDecodeError(
                                "No JSON content found in LLM response after attempting to strip markdown.",
                                response_content_str,
                                0,
                            )

                        parsed_response_data = json.loads(json_str)
                        parsed_response = SQLSchemaValidationResult(
                            **parsed_response_data
                        )

                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse schema validation LLM JSON response: {e}. Response: {response_content_str}"
                        )
                        updates["execution_error"] = (
                            f"LLM response for schema validation was not valid JSON: {e}"
                        )
                        updates["is_valid"] = False
                        return updates  # Return early
                    except (
                        Exception
                    ) as e:  # Catch Pydantic validation errors or other unexpected issues
                        logger.error(
                            f"Failed to validate schema validation LLM response or other error: {e}. Response: {response_content_str}",
                            exc_info=True,
                        )
                        updates["execution_error"] = (
                            f"LLM response structure for schema validation was invalid or other error: {e}"
                        )
                        updates["is_valid"] = False
                        return updates  # Return early
                else:
                    logger.error(
                        f"Unexpected response type from schema validation LLM: {type(response.content)}"
                    )
                    updates["execution_error"] = (
                        "Received unexpected response type from schema validation LLM."
                    )
                    updates["is_valid"] = False
                    # No return here, will be handled by the `if parsed_response:` check

                if parsed_response:
                    if parsed_response.is_valid:
                        logger.info("LLM-based schema pre-validation passed.")
                        updates["is_valid"] = True
                    else:
                        error_msg = f"LLM-based schema pre-validation failed. Discrepancies: {', '.join(parsed_response.discrepancies) if parsed_response.discrepancies else 'No specific discrepancies provided.'}"
                        logger.warning(error_msg)
                        updates["execution_error"] = error_msg
                        updates["is_valid"] = False
                else:
                    # This case is hit if response.content was not a string, or if parsed_response is still None
                    # (e.g. if json_str was empty after stripping and no error was raised to return early)
                    # and an error wasn't already set and returned.
                    if not updates.get(
                        "execution_error"
                    ):  # Avoid overwriting a more specific error
                        logger.error(
                            "Failed to get a parsed response from schema validation LLM and no specific error was set."
                        )
                        updates["execution_error"] = (
                            "Failed to get a parsed response from schema validation LLM."
                        )
                    updates["is_valid"] = (
                        False  # Ensure is_valid is False if we couldn't parse
                    )

            except Exception as e:
                logger.error(
                    f"Outer error during LLM-based schema pre-validation: {e}",
                    exc_info=True,
                )
                updates["execution_error"] = (
                    f"Error during LLM-based schema pre-validation: {e}"
                )
                updates["is_valid"] = False
        elif not self.llm_with_tools and dbt_models:
            logger.warning(
                "LLM not available for schema pre-validation. Skipping this step."
            )
        else:
            logger.info(
                "Skipping schema pre-validation (no dbt_models_info or LLM not available)."
            )

        return updates

    async def enhance_context_for_debug_node(
        self, state: SQLVerifierState
    ) -> Dict[str, Any]:
        """
        This node is now a pass-through. Context enhancement is handled by the agent using tools.
        """
        current_debugging_log = state.get("debugging_log", [])
        current_debugging_log.append(
            "enhance_context_for_debug_node: Node called (now a pass-through, agent handles context)."
        )
        if self.verbose:
            logger.info(
                "enhance_context_for_debug_node: Pass-through node, no context fetching here."
            )
        return {"debugging_log": current_debugging_log}  # No state changes

    # --- Conditional Edge Functions (to be defined) ---
    def should_attempt_debug(self, state: SQLVerifierState) -> str:
        if state.get("is_valid"):
            logger.info("Conditional: SQL is valid, ending.")
            return "end_workflow"  # Or a success node

        current_attempts = state.get("debug_attempts", 0)
        max_attempts = state.get("max_debug_attempts", self.default_max_debug_loops)

        if current_attempts < max_attempts:
            logger.info(
                "Conditional: SQL is invalid, attempts remaining. Routing to attempt_debug_or_fail."
            )
            # No longer goes to enhance_context. Agent will fetch if needed.
            return "attempt_debug_or_fail"
        else:
            logger.info(
                "Conditional: SQL is invalid, max debug attempts reached. Routing to failure/end."
            )
            return "end_workflow"  # Or a failure node

    def should_retry_execution_after_llm_debug(self, state: SQLVerifierState) -> str:
        # This node is entered after update_query_from_llm_suggestion_node.
        # It checks if the LLM provided a new query to try.
        corrected_sql = state.get("corrected_sql_query")
        current_query_before_update = state.get(
            "current_sql_query"
        )  # This is query that was just executed

        # The update_query_from_llm_suggestion_node would have updated current_sql_query
        # if corrected_sql was provided by the agent.
        # Here, we check if corrected_sql (from agent's final answer) is present and non-empty.
        if corrected_sql and corrected_sql.strip():
            logger.info(
                "Conditional: LLM provided a corrected SQL query. Routing back to execute_sql."
            )
            return "retry_execute_sql"
        else:
            # If LLM (agent) did not provide a corrected_sql, or it was empty,
            # it means the agent either couldn't fix it or there was an issue.
            # In this case, we don't retry execution with the same (or empty) query.
            # We proceed to style checking and finalization.
            logger.info(
                "Conditional: LLM did not provide a usable corrected SQL. Proceeding to style check/finalize."
            )
            return "proceed_to_style_check"

    def should_call_agent_tools_or_proceed(self, state: SQLVerifierState) -> str:
        """Determines if the agent requested tool calls or provided a final answer."""
        if state.get("agent_tool_calls"):
            logger.info(
                "Conditional: Agent requested tool calls. Routing to agent_tool_executor."
            )
            return "call_agent_tools"
        else:
            logger.info(
                "Conditional: Agent provided final answer (no tool calls). Routing to update_query_from_llm."
            )
            return "proceed_with_agent_answer"

    # --- Graph Building ---
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(SQLVerifierState)

        # Define Nodes
        workflow.add_node("start_verification", self.start_verification_node)
        workflow.add_node("pre_validate_schema", self.pre_validate_schema_node)
        workflow.add_node("execute_sql", self.execute_sql_node)
        # This node increments attempt count and decides if max attempts reached *before* LLM call
        workflow.add_node("attempt_debug_or_fail", self.attempt_debug_or_fail_node)
        workflow.add_node("invoke_debug_llm", self.invoke_debug_llm_node)
        workflow.add_node(
            "update_query_from_llm", self.update_query_from_llm_suggestion_node
        )
        workflow.add_node(  # Simplified node
            "enhance_context_for_debug", self.enhance_context_for_debug_node
        )
        workflow.add_node(
            "agent_tool_executor", self.agent_tool_executor_node
        )  # New tool executor
        workflow.add_node("finalize_verification", self.finalize_verification_node)

        # Define Edges
        workflow.set_entry_point("start_verification")
        workflow.add_edge("start_verification", "pre_validate_schema")

        # Conditional edge after pre-validation
        workflow.add_conditional_edges(
            "pre_validate_schema",
            lambda state: (
                "attempt_debug_or_fail"
                if state.get("is_valid") is False
                and state.get("max_debug_attempts", 0) > 0
                else (
                    "finalize_verification"
                    if state.get("is_valid") is False
                    else "execute_sql"
                )
            ),
            {
                "attempt_debug_or_fail": "attempt_debug_or_fail",
                "finalize_verification": "finalize_verification",
                "execute_sql": "execute_sql",
            },
        )

        # After execution, check if valid or if debugging is needed/possible
        workflow.add_conditional_edges(
            "execute_sql",
            self.should_attempt_debug,  # Checks is_valid and debug_attempts vs max_debug_attempts
            {
                "attempt_debug_or_fail": "attempt_debug_or_fail",
                "end_workflow": "finalize_verification",
            },
        )

        # After incrementing debug_attempts, decide to call LLM or fail out
        workflow.add_conditional_edges(
            "attempt_debug_or_fail",  # This node increments debug_attempts
            lambda state: (
                "invoke_debug_llm"
                if state["debug_attempts"] <= state["max_debug_attempts"]
                else "finalize_verification"
            ),
            {
                "invoke_debug_llm": "invoke_debug_llm",
                "finalize_verification": "finalize_verification",
            },
        )

        # After invoke_debug_llm, check if agent wants to use tools or gave an answer
        workflow.add_conditional_edges(
            "invoke_debug_llm",
            self.should_call_agent_tools_or_proceed,
            {
                "call_agent_tools": "agent_tool_executor",
                "proceed_with_agent_answer": "update_query_from_llm",
            },
        )

        # After tools are executed, go back to the agent to process results
        workflow.add_edge("agent_tool_executor", "invoke_debug_llm")

        # After update_query_from_llm (which processes agent's final answer),
        # decide whether to retry execution with a new query or proceed to end.
        workflow.add_conditional_edges(
            "update_query_from_llm",
            self.should_retry_execution_after_llm_debug,
            {
                "retry_execute_sql": "execute_sql",  # If agent gave a new SQL, try it
                "proceed_to_style_check": "finalize_verification",  # If agent couldn't fix, or error, go to style check/end
            },
        )

        # Final node that prepares the output
        workflow.add_edge("finalize_verification", END)

        # Compile the graph
        compile_kwargs = {}
        if self.memory:
            compile_kwargs["checkpointer"] = self.memory
        return workflow.compile(**compile_kwargs)

    # --- Main Workflow Runner ---
    async def run(
        self,
        sql_query: str,
        warehouse_type: str = "snowflake",
        max_debug_attempts: Optional[int] = None,
        dbt_models_info: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None,  # For checkpointing
    ) -> SQLVerificationResponse:
        """
        Asynchronously runs the SQL verification and debugging workflow.
        """
        if self.verbose:
            logger.info(
                f"SQLVerifierWorkflow run called with query: {sql_query[:100]}..., warehouse: {warehouse_type}"
            )

        effective_max_debug_attempts = (
            max_debug_attempts
            if max_debug_attempts is not None
            else self.default_max_debug_loops
        )

        # Initialize debugging_log as an empty list if it's None
        initial_debugging_log = []

        # Check for LLM availability if debugging is possible
        if not self.llm_with_tools and effective_max_debug_attempts > 0:
            logger.error(
                "SQLVerifierWorkflow: LLM client not available, but debugging is requested. Cannot proceed with debugging."
            )
            # Return a structured failure response
            fallback_state = SQLVerifierState(
                original_sql_query=sql_query,
                current_sql_query=sql_query,
                warehouse_type=warehouse_type,
                is_valid=False,
                execution_error="LLM client not available.",
                corrected_sql_query=None,
                debug_attempts=0,
                max_debug_attempts=effective_max_debug_attempts,
                dbt_models_info=dbt_models_info,
                agent_described_tables_info={},
                agent_listed_column_values={},
                debug_explanation="LLM client not available.",
                debugging_log=["Attempted to run workflow without LLM for debugging."],
                is_style_compliant=True,  # Assume compliant as style check is removed
                style_violations=[],  # No violations as style check is removed
                messages=[],
                agent_tool_calls=[],
            )

            fallback_response = SQLVerificationResponse(
                success=False,
                is_valid=False,
                verified_sql=sql_query,
                error=fallback_state["execution_error"],
                explanation=fallback_state["debug_explanation"],
                debugging_log=fallback_state["debugging_log"],
                was_executed=False,
            )
            return fallback_response.dict()

        # Handle missing credentials gracefully by falling back to offline verification.
        if not self.snowflake_creds and warehouse_type.lower() == "snowflake":
            logger.warning(
                "SQLVerifierWorkflow: Snowflake credentials not available – performing offline (non-executed) verification."
            )

            offline_state = SQLVerifierState(
                original_sql_query=sql_query,
                current_sql_query=sql_query,
                warehouse_type=warehouse_type,
                is_valid=False,  # Cannot guarantee validity without execution
                execution_error="Query was not executed – warehouse connection not configured.",
                corrected_sql_query=sql_query,  # No correction attempted in offline mode
                debug_attempts=0,
                max_debug_attempts=0,
                dbt_models_info=dbt_models_info,
                agent_described_tables_info={},
                agent_listed_column_values={},
                debug_explanation="Performed static/offline verification only.",
                debugging_log=[
                    "Offline verification: Snowflake credentials unavailable; query not executed."
                ],
                is_style_compliant=True,
                style_violations=[],
                messages=[],
                agent_tool_calls=[],
            )

            # Include an explicit flag so callers know the query wasn't run.
            offline_state["was_executed"] = False  # type: ignore

            offline_response = SQLVerificationResponse(
                success=False,
                is_valid=False,
                verified_sql=sql_query,
                error=offline_state["execution_error"],
                explanation=offline_state["debug_explanation"],
                debugging_log=offline_state["debugging_log"],
                was_executed=False,
            )
            return offline_response.dict()

        # Ensure the graph is built
        if not hasattr(self, "graph_app") or self.graph_app is None:
            self.graph_app = self._build_graph()

        initial_state_dict = SQLVerifierState(
            original_sql_query=sql_query,
            current_sql_query=sql_query,
            warehouse_type=warehouse_type,
            is_valid=None,
            execution_error=None,
            corrected_sql_query=None,
            debug_attempts=0,
            max_debug_attempts=effective_max_debug_attempts,
            dbt_models_info=dbt_models_info,
            agent_described_tables_info={},
            agent_listed_column_values={},
            debug_explanation=None,
            debugging_log=initial_debugging_log,  # Use initialized list
            is_style_compliant=True,  # Assume compliant as style check is removed
            style_violations=[],  # No violations as style check is removed
            messages=[],
            agent_tool_calls=[],
        )

        # Configuration for the graph run
        config = {}
        if conversation_id:
            config = {"configurable": {"thread_id": conversation_id}}
        else:
            # Generate a default conversation_id if needed for checkpointing, or run without if no memory
            if self.memory:
                import uuid

                conv_id = f"sql-verifier-{str(uuid.uuid4())}"
                logger.info(
                    f"SQLVerifierWorkflow: No conversation_id provided, using generated: {conv_id}"
                )
                config = {"configurable": {"thread_id": conv_id}}

        ###########################
        # BEGIN ERROR HANDLING WRAP
        ###########################
        try:
            final_state = await self.graph_app.ainvoke(
                initial_state_dict, config=config
            )

            # The final_state from ainvoke will be the state of the graph when it reached END.
            # This state should contain all the necessary output fields.
            result_dict = {
                "original_sql_query": final_state["original_sql_query"],
                "corrected_sql_query": final_state["corrected_sql_query"],
                "is_valid": final_state["is_valid"],
                "execution_error": final_state["execution_error"],
                "debug_explanation": final_state["debug_explanation"],
                "debug_attempts_made": final_state["debug_attempts"],
                "is_style_compliant": final_state["is_style_compliant"],
                "style_violations": final_state["style_violations"],
                "verified_sql": final_state["current_sql_query"],
                "debugging_log": final_state["debugging_log"],
                "status_message": (
                    "SQL query is valid."
                    if final_state["is_valid"]
                    else f"SQL query is invalid after {final_state['debug_attempts']} attempt(s)."
                ),
                "messages": final_state["messages"],
                "agent_described_tables_info": final_state[
                    "agent_described_tables_info"
                ],
                "agent_listed_column_values": final_state["agent_listed_column_values"],
                "agent_tool_calls": final_state["agent_tool_calls"],
                "success": final_state["is_valid"],
            }
            response = SQLVerificationResponse(**result_dict)
            return response.dict()
        except Exception as e:
            # Catch **all** unexpected errors so that upstream workflows never break.
            err_msg = f"SQLVerifierWorkflow crashed: {e}"
            logger.exception(err_msg)

            safe_state: SQLVerifierState = SQLVerifierState(
                original_sql_query=sql_query,
                current_sql_query=sql_query,
                warehouse_type=warehouse_type,
                is_valid=False,
                execution_error=str(e),
                corrected_sql_query=None,
                debug_attempts=0,
                max_debug_attempts=effective_max_debug_attempts,
                dbt_models_info=dbt_models_info,
                agent_described_tables_info={},
                agent_listed_column_values={},
                debug_explanation="Verifier encountered an unexpected error and could not complete automatic checks.",
                debugging_log=[err_msg],
                is_style_compliant=True,
                style_violations=[],
                messages=[],
                agent_tool_calls=[],
            )

            # Include a flag so callers know we bailed out early
            safe_state["was_executed"] = False  # type: ignore

            result_dict = {
                "original_sql_query": safe_state["original_sql_query"],
                "corrected_sql_query": safe_state["corrected_sql_query"],
                "is_valid": safe_state["is_valid"],
                "execution_error": safe_state["execution_error"],
                "debug_explanation": safe_state["debug_explanation"],
                "debug_attempts_made": safe_state["debug_attempts"],
                "is_style_compliant": safe_state["is_style_compliant"],
                "style_violations": safe_state["style_violations"],
                "verified_sql": safe_state["current_sql_query"],
                "debugging_log": safe_state["debugging_log"],
                "status_message": (
                    "SQL query is invalid."
                    if not safe_state["is_valid"]
                    else "SQL query is valid."
                ),
                "messages": safe_state["messages"],
                "agent_described_tables_info": safe_state[
                    "agent_described_tables_info"
                ],
                "agent_listed_column_values": safe_state["agent_listed_column_values"],
                "agent_tool_calls": safe_state["agent_tool_calls"],
                "success": safe_state["is_valid"],
            }
            response = SQLVerificationResponse(**result_dict)
            return response.dict()
        #########################
        # END ERROR HANDLING WRAP
        #########################

    async def finalize_verification_node(
        self, state: SQLVerifierState
    ) -> Dict[str, Any]:
        """
        Finalizes the verification process and prepares the output.
        This node is now the definitive end for all paths, including those that
        previously went to style checking. It ensures that `is_style_compliant`
        and `style_violations` are set to default values indicating compliance
        since the style check step is removed.
        """
        current_debugging_log = list(state.get("debugging_log", []))
        current_debugging_log.append("Finalizing verification process.")

        # if self.verbose: # Comment out or remove the full state dump
        #     logger.info(f"finalize_verification_node: State entering finalize: {state}")

        original_query = state["original_sql_query"]
        # Use current_sql_query if corrected_sql_query is not set (e.g., direct success or failed debug)
        final_query = state.get("corrected_sql_query") or state["current_sql_query"]
        is_valid = state.get("is_valid")
        error = state.get("execution_error")
        explanation = state.get("debug_explanation")
        attempts = state.get("debug_attempts", 0)

        # Since style check is removed, assume compliance.
        style_compliant = True
        style_violations = []
        current_debugging_log.append(
            "SQL style check step has been removed. Assuming style compliance."
        )

        status_message = ""
        if is_valid and final_query == original_query:
            current_debugging_log.append(
                f"Finalizing verification. SQL is valid. Final query: {final_query[:200]}..."
            )
            status_message = "SQL query is valid."
            if final_query != original_query:
                status_message += f" Query was corrected after {attempts} attempt(s)."
        else:
            current_debugging_log.append(
                f"Finalizing verification. SQL is invalid. Error: {error}"
            )
            status_message = f"SQL query is invalid after {attempts} attempt(s)."
            if error:
                status_message += f" Last error: {error}"

        if self.verbose:
            logger.info(
                "finalize_verification_node: Finalizing with the following key state values:"
            )
            logger.info(f"  Is valid: {is_valid}")
            logger.info(f"  Final query (preview): {final_query[:200]}...")
            logger.info(f"  Original query (preview): {original_query[:200]}...")
            logger.info(f"  Execution Error: {error}")
            logger.info(f"  Debug Explanation: {explanation}")
            logger.info(f"  Debug Attempts: {attempts}")
            logger.info(
                f"  Number of messages in history: {len(state.get('messages', []))}"
            )
            # Avoid logging full dbt_models_info or messages content here unless specifically needed for deep debug

        # Return all relevant fields that QueryExecutorWorkflow might need.
        # This structure should align with what SQLDebugResult expects, more or less.
        result_dict = {
            "original_sql_query": original_query,
            "corrected_sql_query": (
                final_query
                if (is_valid and final_query != original_query)
                else final_query if is_valid else None
            ),
            "is_valid": is_valid,
            "execution_error": error,
            "debug_explanation": explanation,
            "debug_attempts_made": attempts,
            "is_style_compliant": style_compliant,
            "style_violations": style_violations,
            "verified_sql": final_query,
            "debugging_log": current_debugging_log,  # Include the log
            "status_message": status_message,  # A human-readable summary
            "messages": state.get("messages", []),
            "agent_described_tables_info": state.get("agent_described_tables_info", {}),
            "agent_listed_column_values": state.get("agent_listed_column_values", {}),
            "agent_tool_calls": state.get("agent_tool_calls", []),
            "success": is_valid,
        }
        # Ensure all keys expected by SQLDebugResult are present, even if None
        # This might be better handled by constructing an SQLDebugResult Pydantic model
        # and then converting to dict, but for now, let's be explicit.

        # Clean up None values if SQLDebugResult model doesn't like them or for cleaner output
        # final_result_cleaned = {k: v for k, v in result.items() if v is not None}
        # However, QueryExecutor might expect all keys.

        response = SQLVerificationResponse(**result_dict)
        return response.dict()  # Return canonical shape as a dict for graph state


# Example usage (for testing, not part of the class)
async def main_test():
    # This requires Django settings and appropriate env vars for Snowflake and LLM
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    # Mock Django settings if not running in a Django context for standalone testing
    if not django_settings.configured:
        django_settings.configure(
            RAGSTAR_LOG_LEVEL="DEBUG",
            # Add other necessary settings mocks here if your workflow depends on them
            # e.g., LLM provider settings, data warehouse settings.
            # For SQLVerifier, we might need Snowflake creds.
            # These would typically come from environment variables.
            # Example (ensure these are properly managed in a real test setup):
            # RAGSTAR_DEFAULT_LLM_PROVIDER="...",
            # RAGSTAR_OPENAI_API_KEY="...",
            # RAGSTAR_SNOWFLAKE_USER="...",
            # ... etc.
        )

    # --- Mocked dbt_models_info for testing ---
    mock_dbt_models_info = [
        {
            "model_name": "customers",
            "description": "Table containing customer data.",
            "columns": [
                {"name": "id", "type": "INTEGER", "description": "Customer ID"},
                {"name": "name", "type": "VARCHAR", "description": "Customer name"},
                {
                    "name": "email",
                    "type": "VARCHAR",
                    "description": "Customer email",
                },
            ],
            "table_type": "BASE TABLE",  # Example field
            "sample_data": [],  # Example field
        },
        {
            "model_name": "orders",
            "description": "Table containing order data.",
            "columns": [
                {"name": "id", "type": "INTEGER", "description": "Order ID"},
                {
                    "name": "customer_id",
                    "type": "INTEGER",
                    "description": "ID of the customer who placed the order.",
                },
                {
                    "name": "order_date",
                    "type": "DATE",
                    "description": "Date the order was placed.",
                },
                {
                    "name": "amount",
                    "type": "DECIMAL",
                    "description": "Total amount of the order.",
                },
            ],
            "table_type": "BASE TABLE",
            "sample_data": [],
        },
    ]

    # --- Test Cases ---
    test_cases = [
        {
            "name": "Valid Query - Simple Select",
            "query": "SELECT id, name FROM customers LIMIT 10",
            "warehouse_type": "snowflake",  # Assuming Snowflake for testing execution
            "max_debug_attempts": 0,  # No debugging needed
            "dbt_models_info": mock_dbt_models_info,
            "expected_is_valid": True,
            "expected_error": None,
        },
        # {
        #     "name": "Invalid Query - Non-existent column",
        #     "query": "SELECT non_existent_col FROM customers", # This query will fail
        #     "warehouse_type": "snowflake",
        #     "max_debug_attempts": 1, # Allow debugging
        #     "dbt_models_info": mock_dbt_models_info,
        #     "expected_is_valid": False, # Expect it to remain invalid if LLM can't fix
        #     # "expected_corrected_sql_contains": "SELECT" # if LLM fixes it
        # },
        # {
        #     "name": "Invalid Query - Syntax Error (missing comma)",
        #     "query": "SELECT id name FROM customers",
        #     "warehouse_type": "snowflake",
        #     "max_debug_attempts": 1,
        #     "dbt_models_info": mock_dbt_models_info,
        #     "expected_is_valid": False,
        # },
        # Add more test cases:
        # - Query that needs schema from dbt_models_info to be fixed
        # - Query with a join that is incorrect
        # - Query that is valid but could be improved (style check - though style node removed)
        # - Query for a non-Snowflake DB (if you want to test that path, e.g., "duckdb")
        #   (current implementation primarily supports Snowflake execution)
    ]

    # Initialize the workflow
    # Note: For testing, you might need to ensure Snowflake credentials and LLM are configured
    # or mock the parts of the workflow that interact with them if focusing on graph logic.
    # For this test, we assume Snowflake connection and LLM are available or mocked by env.
    workflow_instance = SQLVerifierWorkflow()

    logger.info("\n--- Running SQL Verifier Workflow Tests ---")
    for i, test_case in enumerate(test_cases):
        logger.info(f"\n--- Test Case {i+1}: {test_case['name']} ---")
        logger.info(f"Query: {test_case['query']}")

        try:
            result = await workflow_instance.run(
                sql_query=test_case["query"],
                warehouse_type=test_case["warehouse_type"],
                max_debug_attempts=test_case["max_debug_attempts"],
                dbt_models_info=test_case["dbt_models_info"],
                conversation_id=f"test-conv-{i+1}",
            )

            logger.info(f"Workflow Result for '{test_case['name']}':")
            logger.info(f"  Is Valid: {result.get('is_valid')}")
            logger.info(f"  Corrected SQL: {result.get('corrected_sql_query')}")
            logger.info(f"  Execution Error: {result.get('execution_error')}")
            logger.info(f"  Debug Explanation: {result.get('debug_explanation')}")
            logger.info(f"  Debug Attempts: {result.get('debug_attempts_made')}")
            logger.info(f"  Style Compliant: {result.get('is_style_compliant')}")
            # logger.info(f"  Debugging Log: {result.get('debugging_log')}")

            # Assertions
            assert result.get("is_valid") == test_case["expected_is_valid"], (
                f"Test '{test_case['name']}' failed: is_valid mismatch. "
                f"Expected {test_case['expected_is_valid']}, got {result.get('is_valid')}"
            )
            if test_case["expected_error"]:
                assert test_case["expected_error"] in (
                    result.get("execution_error") or ""
                ), (
                    f"Test '{test_case['name']}' failed: error message mismatch. "
                    f"Expected contains '{test_case['expected_error']}', got '{result.get('execution_error')}'"
                )
            # Add more assertions as needed, e.g., for corrected_sql_query if applicable

            logger.info(f"--- Test Case '{test_case['name']}' PASSED ---")

        except Exception as e:
            logger.error(
                f"--- Test Case '{test_case['name']}' FAILED with exception: {e} ---",
                exc_info=True,
            )


if __name__ == "__main__":
    # This allows running the test function directly if the script is executed.
    # Note: Ensure Django settings are configured or mocked appropriately before running.

    asyncio.run(main_test())
