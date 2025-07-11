# This file will be rewritten with the LangGraph implementation.
# The old content from agent.py is being replaced.

import logging
import uuid
import json
import csv
import io
from typing import Dict, List, Any, Optional, TypedDict, Set
from typing_extensions import Annotated
import re

# Langchain & LangGraph Imports
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver  # For potential checkpointing
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# Django & Project Imports
from django.conf import settings as django_settings  # To access Django global settings
from asgiref.sync import sync_to_async

# LLM Service import (similar to QuestionAnswerer)
from apps.llm_providers.services import ChatService
from apps.accounts.models import OrganisationSettings

# Query Executor specific imports
# REMOVED: from . import tools as query_executor_tools

from .prompts import SQL_DEBUG_PROMPT_TEMPLATE
from apps.workflows.rules_loader import get_agent_rules  # Updated import

# Import the new centralized Slack client getter
from apps.workflows.services import get_slack_web_client

# NEW: Import SQLVerifierWorkflow
from apps.workflows.sql_verifier.workflow import SQLVerifierWorkflow
from apps.workflows.services import ConversationLogger
from apps.integrations.manager import IntegrationsManager

# Import SQLDebugResult from the centralized SQLVerifierWorkflow models
from apps.workflows.sql_verifier.models import SQLDebugResult

# Other schemas specific to QueryExecutor can remain imported from .schemas
from .schemas import (
    PostMessageInput,
    PostCsvInput,
    DescribeTableInput,
    ListColumnValuesInput,
)

try:
    from slack_sdk.web.async_client import AsyncWebClient
    from slack_sdk.errors import SlackApiError

    SLACK_SDK_AVAILABLE = True
except ImportError:
    AsyncWebClient = None
    SlackApiError = None
    SLACK_SDK_AVAILABLE = False

# Imports moved from tools.py
import snowflake.connector
import aiohttp


logger = logging.getLogger(__name__)


# --- Tool Input Schemas (for tools the LLM might call, or for clarity) ---
# REMOVED PostMessageInput, PostCsvInput, DescribeTableInput, ListColumnValuesInput, SQLDebugResult definitions
# All Pydantic models are now in schemas.py


# --- LangGraph State Definition ---
class QueryExecutorState(TypedDict):
    # Input/Context from Slack event
    channel_id: str
    thread_ts: str
    user_id: str
    original_trigger_message_ts: (
        str  # The TS of the user message that might contain/refer to the SQL
    )

    # Query processing
    fetched_sql_query: Optional[str]
    current_sql_query: Optional[str]

    # Execution results
    execution_result_rows: Optional[List[tuple]]  # Raw rows from DB, first is header
    execution_error: Optional[str]
    csv_content: Optional[str]  # String content of the CSV

    # Debugging loop
    debug_attempts: int
    max_debug_attempts: int
    # For LLM-based debugging context
    described_tables_info: Dict[
        str, Any
    ]  # Store schema info from describe_table tool calls
    listed_column_values: Dict[
        str, List[Any]
    ]  # Store column values from list_column_values tool calls

    # NEW: Log for debugging and execution steps
    debugging_log: List[str]

    # General message history for LLM interactions (primarily for debugging LLM)
    messages: Annotated[List[BaseMessage], add_messages]

    # Final outcome
    final_status_message: Optional[str]
    is_success: Optional[bool]
    posted_response_ts: Optional[str]
    posted_file_ts: Optional[str]

    # Checkpointing/run management
    conversation_id: Optional[str]  # For langgraph checkpointing


# --- Query Executor Workflow Class ---
class QueryExecutorWorkflow:
    """Workflow for executing SQL queries and handling results/debugging via LangGraph."""

    # --- Helper methods moved from tools.py ---
    async def _async_fetch_query_from_thread(
        self, client: AsyncWebClient, channel_id: str, thread_ts: str
    ) -> Optional[str]:
        """
        Fetches the latest SQL query from a file in a given Slack thread using an AsyncWebClient.
        Returns the query string or None if not found or an error occurs.
        (Moved from tools.py)
        """
        if not SLACK_SDK_AVAILABLE:
            logger.error(
                "slack_sdk is not installed. _async_fetch_query_from_thread cannot work."
            )
            return None
        if not client:  # self.slack_client should be used if this is a method
            logger.error(
                "AsyncWebClient not provided to _async_fetch_query_from_thread."
            )
            return None

        try:
            logger.info(
                f"Fetching replies in thread {thread_ts} from channel {channel_id}"
            )
            result = await client.conversations_replies(
                channel=channel_id, ts=thread_ts, limit=200
            )

            messages = result.get("messages", [])
            if not messages:
                logger.info(f"No messages found in thread {thread_ts}.")
                return None

            sql_file_content = None
            latest_sql_file_ts = 0

            for message in reversed(messages):  # Check most recent messages first
                if "files" in message:
                    for file_info in message["files"]:
                        file_type = file_info.get("filetype", "").lower()
                        file_name = file_info.get("name", "").lower()
                        is_sql_file = (file_type == "sql") or file_name.endswith(".sql")

                        message_ts_float = float(message.get("ts", 0))

                        if is_sql_file and message_ts_float > latest_sql_file_ts:
                            logger.info(
                                f"Found potential SQL file: {file_name} (type: {file_type}) at ts {message['ts']}"
                            )
                            file_url_private = file_info.get("url_private_download")
                            if not file_url_private:
                                logger.warning(
                                    f"SQL file {file_name} has no private download URL. Skipping."
                                )
                                continue

                            # Ensure client.token is valid for aiohttp.
                            # The provided client instance should have its token properly configured.
                            request_headers = {
                                "Authorization": f"Bearer {client.token}"
                            }
                            async with aiohttp.ClientSession() as session:
                                async with session.get(
                                    file_url_private, headers=request_headers
                                ) as resp:
                                    if resp.status == 200:
                                        query_content_bytes = await resp.read()
                                        sql_file_content = query_content_bytes.decode(
                                            "utf-8"
                                        ).strip()
                                        latest_sql_file_ts = message_ts_float
                                        logger.info(
                                            f"Successfully downloaded content from {file_name}."
                                        )
                                        break
                                    else:
                                        logger.error(
                                            f"Failed to download file {file_name}. Status: {resp.status}, Body: {await resp.text()}"
                                        )
                                        continue
                    if sql_file_content and message_ts_float == latest_sql_file_ts:
                        break

            if sql_file_content:
                logger.info(f"Successfully fetched SQL query from thread {thread_ts}.")
                return sql_file_content
            else:
                logger.info(f"No SQL file found in thread {thread_ts}.")
                return None

        except SlackApiError as e:  # Make sure SlackApiError is imported
            logger.error(
                f"Slack API Error fetching thread replies for {thread_ts}: {e.response['error'] if e.response else str(e)}",
                exc_info=True,
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error fetching query from thread {thread_ts}: {e}",
                exc_info=True,
            )
            return None

    def _execute_sql_query(
        self, sql_query: str
    ) -> tuple[Optional[List[tuple]], Optional[str]]:
        """
        Executes a SQL query against Snowflake using credentials and settings from the class instance.
        Returns a tuple: (results, error_message)
        (Moved from tools.py and adapted to be a method)
        """
        if self.warehouse_type != "snowflake":
            logger.error(f"Unsupported warehouse type: {self.warehouse_type}")
            return None, f"Unsupported warehouse type: {self.warehouse_type}"

        if not self.snowflake_creds:
            err_msg = "Snowflake credentials not available for query execution (check __init__)."
            logger.error(err_msg)
            return None, err_msg

        conn = None
        try:
            # Construct connection parameters from self.snowflake_creds
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

            conn = snowflake.connector.connect(
                **conn_params
            )  # Ensure snowflake.connector is imported
            logger.info(
                f"Successfully connected to Snowflake account: {self.snowflake_creds.get('account')}"
            )

            cursor = conn.cursor()
            try:
                logger.info(
                    f"Executing SQL query (first 500 chars): {sql_query[:500]}..."
                )
                cursor.execute(sql_query)

                results = []
                if cursor.description:
                    column_names = [col[0] for col in cursor.description]
                    results.append(tuple(column_names))
                    for row in cursor:
                        results.append(row)
                else:
                    logger.info(
                        f"Query executed successfully (no rows returned). Row count: {cursor.rowcount}"
                    )
                    results = [("Status",), (f"OK, {cursor.rowcount} rows affected.",)]

                logger.info(
                    f"Query executed successfully. Result rows (excluding headers): {len(results)-1 if results and len(results) > 0 else 0}"
                )
                return results, None
            except snowflake.connector.Error as sf_err:
                error_msg = f"Snowflake Error: {sf_err}"
                logger.error(
                    error_msg, exc_info=self.verbose
                )  # Add exc_info for better debug
                return None, error_msg
            except Exception as e:
                error_msg = f"An unexpected error occurred during query execution: {e}"
                logger.error(error_msg, exc_info=True)
                return None, error_msg
            finally:
                if cursor:
                    cursor.close()
        except (
            snowflake.connector.Error
        ) as conn_err:  # Catch connection specific errors
            error_msg = f"Snowflake Connection Error: {conn_err}"
            logger.error(
                error_msg, exc_info=self.verbose
            )  # Add exc_info for better debug
            return None, error_msg
        except Exception as e:  # Catch other unexpected errors during connection setup
            error_msg = f"An unexpected error occurred while setting up Snowflake connection: {e}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
        finally:
            if conn and not conn.is_closed():
                conn.close()
                logger.info("Snowflake connection closed.")

    # --- End of Helper methods ---

    def __init__(
        self,
        org_settings: OrganisationSettings,
        slack_client: Optional[AsyncWebClient] = None,
        memory: Optional[BaseCheckpointSaver] = None,
        max_debug_loops: int = 3,
    ):
        self.verbose = django_settings.RAGSTAR_LOG_LEVEL == "DEBUG"
        if self.verbose:
            logger.info(
                f"QueryExecutorWorkflow initialized with verbose={self.verbose} (LogLevel: {django_settings.RAGSTAR_LOG_LEVEL})"
            )

        self.slack_client = (
            slack_client if slack_client is not None else get_slack_web_client()
        )
        if not self.slack_client:
            logger.warning(
                "QueryExecutorWorkflow: Slack client is not available. Slack operations will fail."
            )

        try:
            self.chat_service = ChatService(org_settings=org_settings)
            self.llm = self.chat_service.get_client()
            if not self.llm:
                logger.error(
                    "QueryExecutorWorkflow: LLM client could not be initialized. Debugging LLM will not work for its tools."
                )
        except Exception as e:
            logger.error(
                f"QueryExecutorWorkflow: Failed to initialize chat service or LLM client: {e}",
                exc_info=self.verbose,
            )
            self.chat_service = None
            self.llm = None

        self.memory = memory
        self.max_debug_loops = max_debug_loops  # Max debug loops for the SQL Verifier

        # Instantiate the SQLVerifierWorkflow
        self.sql_verifier = SQLVerifierWorkflow(org_settings=org_settings)

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
                    "QueryExecutorWorkflow: Snowflake credentials loaded from integration system."
                )
            else:
                self.snowflake_creds = None
                self.warehouse_type = None
                logger.warning(
                    "QueryExecutorWorkflow: Snowflake integration not configured."
                )

        except Exception as e:
            logger.error(
                f"QueryExecutorWorkflow: Failed to load Snowflake credentials from integration system: {e}. Workflow may not function."
            )
            self.snowflake_creds = None  # Ensure it's None if loading failed
            self.warehouse_type = None  # Ensure it's None
            # Depending on strictness, could raise an exception here to halt initialization.

        self._define_tools()
        # Graph is compiled within _build_graph, which includes checkpointer logic
        self.graph_app = self._build_graph()

    def _define_tools(self):
        """Defines the tools available to this workflow, including internal and Slack tools."""

        # --- Slack Interaction Tools ---
        @tool
        async def post_message_to_thread(
            message_text: str,
            state: QueryExecutorState,  # state must be passed if tool needs it.
        ) -> Dict[str, Any]:
            """Posts a text message to the current Slack thread in context."""
            # This tool needs access to self.slack_client, channel_id, thread_ts from state.
            # It's called by nodes that have access to `self` and `state`.
            # If called by ToolNode, it won't have `self` directly.
            # For direct calls from methods: self.slack_client
            # For calls via ToolNode: Need to ensure slack_client is available.
            # One way: if self.slack_client is consistently available, tools can use it.

            # Let's assume this tool is designed to be called from a context where self.slack_client is available.
            # For ToolNode usage, if the LangChain tool object is `self.post_message_tool_func`,
            # it's implicitly bound to the instance if defined as a method.
            # If defined as a local function within _define_tools, it needs client passed or from closure.

            # Current structure: `self.post_message_tool_func = post_message_to_thread`
            # `post_message_to_thread` is an inner function here. It needs `self.slack_client`.
            # And it gets `state` from its arguments when called by a node.

            _slack_client = (
                self.slack_client
            )  # From outer scope (QueryExecutorWorkflow instance)

            if not _slack_client:
                return {"success": False, "error": "Slack client not available."}

            current_channel_id = state.get("channel_id")
            current_thread_ts = state.get("thread_ts")

            if not current_channel_id or not current_thread_ts:
                logger.error(
                    "post_message_to_thread: channel_id or thread_ts missing in state."
                )
                return {
                    "success": False,
                    "error": "Channel or thread information missing.",
                }
            try:
                result = (
                    await _slack_client.chat_postMessage(  # Use captured _slack_client
                        channel=current_channel_id,
                        thread_ts=current_thread_ts,
                        text=message_text,
                    )
                )
                if result["ok"]:
                    return {"success": True, "ts": result.get("ts")}
                return {
                    "success": False,
                    "error": result.get("error", "Slack API error"),
                }
            except Exception as e:
                logger.error(
                    f"Error in post_message_to_thread: {e}", exc_info=self.verbose
                )
                return {"success": False, "error": str(e)}

        self.post_message_tool_func = post_message_to_thread

        @tool
        async def post_csv_to_thread(
            csv_data: str,
            initial_comment: str,
            filename: str,
            state: QueryExecutorState,  # Needs state for channel/thread
        ) -> Dict[str, Any]:
            """Posts a CSV file to the current Slack thread in context."""
            _slack_client = self.slack_client  # From outer scope
            if not _slack_client:
                return {"success": False, "error": "Slack client not available."}

            # Get channel_id and thread_ts from state
            current_channel_id = state.get("channel_id")
            current_thread_ts = state.get("thread_ts")

            if not current_channel_id or not current_thread_ts:
                logger.error("post_csv_to_thread: channel_id or thread_ts missing.")
                return {"success": False, "error": "Channel/thread info missing."}

            try:
                result = (
                    await _slack_client.files_upload_v2(  # Use captured _slack_client
                        channel=current_channel_id,
                        thread_ts=current_thread_ts,
                        content=csv_data.encode("utf-8"),
                        filename=filename,
                        initial_comment=initial_comment,
                    )
                )
                if result["ok"]:
                    file_obj = result.get("file")
                    return {
                        "success": True,
                        "file_id": file_obj.get("id") if file_obj else None,
                    }
                return {
                    "success": False,
                    "error": result.get("error", "Slack API error"),
                }
            except Exception as e:
                logger.error(f"Error in post_csv_to_thread: {e}", exc_info=self.verbose)
                return {"success": False, "error": str(e)}

        self.post_csv_tool_func = (
            post_csv_to_thread  # Assign for direct calls if needed
        )

        @tool
        async def post_text_file_to_thread(
            text_content: str,
            initial_comment: str,
            filename: str,
            state: QueryExecutorState,
        ) -> Dict[str, Any]:
            """Posts a text file to the current Slack thread in context."""
            _slack_client = self.slack_client
            if not _slack_client:
                return {"success": False, "error": "Slack client not available."}

            current_channel_id = state.get("channel_id")
            current_thread_ts = state.get("thread_ts")

            if not current_channel_id or not current_thread_ts:
                logger.error(
                    "post_text_file_to_thread: channel_id or thread_ts missing."
                )
                return {"success": False, "error": "Channel/thread info missing."}

            try:
                result = await _slack_client.files_upload_v2(
                    channel=current_channel_id,
                    thread_ts=current_thread_ts,
                    content=text_content.encode("utf-8"),
                    filename=filename,
                    initial_comment=initial_comment,
                )
                if result["ok"]:
                    file_obj = result.get("file")
                    return {
                        "success": True,
                        "file_id": file_obj.get("id") if file_obj else None,
                    }
                return {
                    "success": False,
                    "error": result.get("error", "Slack API error"),
                }
            except Exception as e:
                logger.error(
                    f"Error in post_text_file_to_thread: {e}", exc_info=self.verbose
                )
                return {"success": False, "error": str(e)}

        self.post_text_file_tool_func = post_text_file_to_thread

        # Tools for LLM-assisted debugging (using self._execute_sql_query)
        @tool(args_schema=DescribeTableInput)
        async def describe_table_tool(table_name: str) -> Dict[str, Any]:
            """Describes the schema of a given table in the data warehouse (for LLM use).
            (Logic moved from tools.py and uses internal _execute_sql_query)
            """
            # Log the tool call
            if getattr(self, "conversation_logger", None):
                await sync_to_async(self.conversation_logger.log_tool_call)(
                    tool_name="describe_table",
                    tool_input={"table_name": table_name},
                )

            if not re.match(
                r"^[a-zA-Z_][a-zA-Z0-9_.]*$", table_name
            ):  # Basic validation
                return {"error": "Invalid table name format."}

            sql_query = f"DESCRIBE TABLE {table_name};"

            # self._execute_sql_query is a sync method, needs sync_to_async
            results, error = await sync_to_async(self._execute_sql_query)(
                sql_query=sql_query
            )

            if error:
                logger.error(f"Error describing table {table_name}: {error}")
                return {"error": error}

            if not results or len(results) < 2:  # Header + data
                logger.warning(
                    f"No schema information returned for table {table_name}. Results: {results}"
                )
                return {"schema": []}  # Return schema: [] for consistency if no info

            header = results[0]
            column_data = results[1:]
            schema_info = []
            try:
                # Ensure these column names match Snowflake's DESCRIBE TABLE output
                name_idx = header.index("name")
                type_idx = header.index("type")
                nullable_idx = header.index("null?")
            except ValueError as e:  # If a column name is not found
                logger.error(
                    f"Could not find expected columns in DESCRIBE TABLE output for {table_name}. Error: {e}. Header: {header}"
                )
                return {
                    "error": f"Malformed DESCRIBE TABLE output, missing expected column: {e}"
                }

            for row in column_data:
                try:
                    col_info = {
                        "name": row[name_idx],
                        "type": row[type_idx],
                        "nullable": row[nullable_idx] == "Y",  # Snowflake uses 'Y'/'N'
                    }
                    schema_info.append(col_info)
                except IndexError as e:  # If a row doesn't have enough elements
                    logger.error(
                        f"Error parsing row for DESCRIBE TABLE output of {table_name}. Row: {row}. Error: {e}"
                    )
                    continue  # Skip malformed row
            logger.info(
                f"Successfully described table {table_name}. Columns found: {len(schema_info)}"
            )
            return {"schema": schema_info}

        @tool(args_schema=ListColumnValuesInput)
        async def list_column_values_tool(
            table_name: str, column_name: str, limit: Optional[int] = None
        ) -> Dict[str, Any]:
            """Lists distinct values for a given column in a table (for LLM use).
            (Logic moved from tools.py and uses internal _execute_sql_query)
            """
            # Log the tool call
            if getattr(self, "conversation_logger", None):
                await sync_to_async(self.conversation_logger.log_tool_call)(
                    tool_name="list_column_values",
                    tool_input={
                        "table_name": table_name,
                        "column_name": column_name,
                        "limit": limit,
                    },
                )

            # Basic validation for table and column names
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", table_name) or not re.match(
                r"^[a-zA-Z_][a-zA-Z0-9_]*$", column_name
            ):  # Column name more restrictive
                return {"error": "Invalid table or column name format."}

            if (
                not limit or not isinstance(limit, int) or limit <= 0
            ):  # Updated condition
                limit = 20  # Default to a safe limit (as per original schema)
                logger.warning(
                    f"Invalid or missing limit specified for list_column_values, defaulting to {limit}"
                )

            # Snowflake is case-insensitive by default for unquoted identifiers,
            # but quoting them (e.g. f'"{column_name}"') makes it case-sensitive.
            # Assuming column_name should be treated as potentially case-sensitive as per best practice.
            sql_query = (
                f'SELECT DISTINCT "{column_name}" FROM {table_name} LIMIT {limit};'
            )

            results, error = await sync_to_async(self._execute_sql_query)(
                sql_query=sql_query
            )

            if error:
                logger.error(
                    f"Error listing column values for {table_name}.{column_name}: {error}"
                )
                return {"error": error}

            if not results or len(results) < 2:  # Expecting header and data rows
                logger.info(
                    f"No distinct values found for {table_name}.{column_name} or query returned no data."
                )
                return {"values": []}  # Return empty list if no values

            distinct_values = [
                row[0] for row in results[1:]
            ]  # Data is in the first column of subsequent rows
            logger.info(
                f"Successfully listed {len(distinct_values)} distinct values for {table_name}.{column_name}."
            )
            return {"values": distinct_values}

        # Only expose Snowflake-dependent debugging tools when credentials are configured.
        if self.snowflake_creds:
            self.debug_llm_tools = [describe_table_tool, list_column_values_tool]
        else:
            self.debug_llm_tools = []

        # self.all_defined_tools combines all tools that might be registered with a ToolNode
        self.all_defined_tools = self.debug_llm_tools  # + any other LLM-callable tools

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(QueryExecutorState)

        # Define Nodes
        workflow.add_node("fetch_sql_from_slack", self.fetch_sql_from_slack_node)
        workflow.add_node(
            "execute_sql", self.execute_sql_node
        )  # This now calls SQLVerifier internally
        workflow.add_node(
            "process_execution_results", self.process_execution_results_node
        )
        workflow.add_node("prepare_csv", self.prepare_csv_node)
        workflow.add_node("post_success_to_slack", self.post_success_to_slack_node)
        # workflow.add_node("attempt_debug_or_fail", self.attempt_debug_or_fail_node) # Removed

        # LLM Debugging nodes are removed as SQLVerifierWorkflow handles this.
        # workflow.add_node("invoke_debug_llm", self.invoke_debug_llm_node) # Removed

        # The debug_tool_node might still be relevant if QueryExecutor's own LLM uses tools like describe_table.
        # For now, let's assume the debug_llm_tools list might be used by a general purpose LLM in QE.
        # If self.debug_llm_tools is empty or only contained SQL debugging tools, this ToolNode can be removed.
        # Based on current _define_tools, self.debug_llm_tools = [self.describe_table_tool_func, self.list_column_values_tool_func]
        # These tools might still be useful for a general QE LLM, so we keep the node definition but it won't be part of the main SQL flow.
        if hasattr(self, "debug_llm_tools") and self.debug_llm_tools:
            debug_tool_node = ToolNode(self.debug_llm_tools, handle_tool_errors=True)
            workflow.add_node(
                "debug_tools", debug_tool_node
            )  # Kept, but not linked in main SQL flow

        # workflow.add_node(
        # "update_state_after_debug_tool", self.update_state_after_debug_tool_node # Removed
        # )
        # workflow.add_node(
        # "update_query_from_llm_suggestion",
        # self.update_query_from_llm_suggestion_node, # Removed
        # )

        workflow.add_node("post_failure_to_slack", self.post_failure_to_slack_node)
        workflow.add_node("finalize_workflow", self.finalize_workflow_node)

        # Define Edges
        workflow.set_entry_point("fetch_sql_from_slack")
        workflow.add_edge("fetch_sql_from_slack", "execute_sql")
        workflow.add_edge("execute_sql", "process_execution_results")

        workflow.add_conditional_edges(
            "process_execution_results",
            self.should_prepare_csv_or_post_failure,
            {
                "prepare_csv": "prepare_csv",
                "post_failure_to_slack": "post_failure_to_slack",
            },
        )
        workflow.add_edge("prepare_csv", "post_success_to_slack")
        workflow.add_edge("post_success_to_slack", "finalize_workflow")

        workflow.add_edge("post_failure_to_slack", "finalize_workflow")
        workflow.add_edge("finalize_workflow", END)

        compile_kwargs = {}
        if self.memory:
            compile_kwargs["checkpointer"] = self.memory
        return workflow.compile(**compile_kwargs)

    # --- Node Functions ---
    async def fetch_sql_from_slack_node(
        self, state: QueryExecutorState
    ) -> Dict[str, Any]:
        logger.info("Node: fetch_sql_from_slack_node")
        state["debugging_log"].append(
            "Attempting to fetch SQL query from Slack thread..."
        )
        if not self.slack_client:  # Use self.slack_client here
            logger.error("Slack client not available in fetch_sql_from_slack_node.")
            error_msg = "Slack client not configured, cannot fetch query."
            state["debugging_log"].append(f"Error: {error_msg}")
            return {"execution_error": error_msg}

        channel_id = state["channel_id"]
        thread_ts_for_replies = state["original_trigger_message_ts"]

        try:
            # Call the internal helper method
            sql_query = await self._async_fetch_query_from_thread(
                client=self.slack_client,  # Pass the instance's client
                channel_id=channel_id,
                thread_ts=thread_ts_for_replies,
            )

            if sql_query:
                logger.info(
                    f"Successfully fetched SQL from Slack: {sql_query[:100]}..."
                )
                state["debugging_log"].append(
                    "Successfully fetched SQL from Slack thread."
                )
                return {
                    "fetched_sql_query": sql_query,
                    "current_sql_query": sql_query,
                    "execution_error": None,
                }
            else:
                logger.warning(
                    f"Could not find a SQL query in thread {channel_id}/{thread_ts_for_replies}"
                )
                no_query_message = "I couldn't find a .sql file in the specified message or its thread. Please make sure a .sql file was attached."
                state["debugging_log"].append("No SQL query file found in the thread.")

                # Use the tool directly for posting messages.
                # self.post_message_tool_func is the @tool decorated object.
                # To call its underlying function with `self` correctly handled by Python,
                # if it's an instance method: self.post_message_tool_func._raw_function(self, ...)
                # or if it's from @tool: tool_obj.__wrapped__(self_or_first_arg, ...)
                # Simpler: call the tool as if an LLM would, it needs state.
                # The tool `post_message_to_thread` is defined within `_define_tools` and captures `self`.
                # So direct call should work.

                # Correction: The tool `post_message_to_thread` is already an async function
                # and defined to take `state`.
                post_tool_result = await self.post_message_tool_func.ainvoke(
                    {"message_text": no_query_message, "state": state}
                )

                if not post_tool_result.get(
                    "success"
                ):  # ainvoke returns the tool's output dict directly
                    logger.error(
                        f"Failed to post 'no query found' message: {post_tool_result.get('error')}"
                    )
                return {
                    "execution_error": "No SQL query found in Slack thread.",
                    "is_success": False,  # Mark as not successful
                    "final_status_message": no_query_message,
                }

        except Exception as e:
            logger.error(
                f"Error in fetch_sql_from_slack_node: {e}", exc_info=self.verbose
            )
            error_msg = f"Failed to fetch SQL from Slack: {e}"
            state["debugging_log"].append(f"Exception during SQL fetch: {error_msg}")
            return {"execution_error": error_msg}

    async def execute_sql_node(self, state: QueryExecutorState) -> Dict[str, Any]:
        # current_attempt_number = state.get("debug_attempts", 0) + 1 # Debug attempts now handled by SQLVerifier
        logger.info(f"Node: execute_sql_node (delegating to SQLVerifierWorkflow)")
        state["debugging_log"].append(
            f"Attempting to verify and execute SQL via SQLVerifierWorkflow."
        )

        current_query = state.get("current_sql_query")
        if not current_query:
            logger.error("No SQL query found in state for execution/verification.")
            internal_error_msg = "Internal error: No query to execute."
            state["debugging_log"].append(
                f"Error: {internal_error_msg} - Current query is missing in state."
            )
            # This is an internal error, user message might not be needed or could be generic.
            # await self.post_message_tool_func.ainvoke( # Consider if a message should be posted
            #     {"message_text": internal_error_msg, "state": state}
            # )
            return {
                "execution_error": internal_error_msg,
                "is_success": False,
                "final_status_message": internal_error_msg,
                "execution_result_rows": None,  # Ensure this is set for downstream checks
            }

        # Call the SQLVerifierWorkflow
        # The conversation_id for the verifier can be derived or a new one created.
        # For simplicity, let's use the QueryExecutor's conversation_id if available, or let verifier make one.
        verifier_conversation_id = state.get("conversation_id")
        # Ensure max_debug_attempts from QueryExecutorState is passed to the verifier
        max_debug_attempts_for_verifier = state.get(
            "max_debug_attempts", self.max_debug_loops
        )

        from apps.workflows.schemas import SQLVerificationResponse

        verifier_result_raw = await self.sql_verifier.run(
            sql_query=current_query,
            max_debug_attempts=max_debug_attempts_for_verifier,
            dbt_models_info=None,  # TODO: QueryExecutor doesn't have dbt_models_info, consider fetching if needed
            conversation_id=verifier_conversation_id,
        )

        # Normalise verifier_result to a dictionary for downstream logic
        if isinstance(verifier_result_raw, SQLVerificationResponse):
            verifier_result = verifier_result_raw.dict()
        else:
            verifier_result = verifier_result_raw

        updates: Dict[str, Any] = {}
        if verifier_result.get("is_valid"):
            logger.info(
                f"SQLVerifierWorkflow confirmed query is valid. Original or Corrected SQL: {verifier_result.get('corrected_sql_query')[:200]}..."
            )
            state["debugging_log"].append(
                f"SQLVerifierWorkflow successful. Valid SQL: {verifier_result.get('corrected_sql_query')[:100]}..."
            )
            updates["current_sql_query"] = verifier_result["corrected_sql_query"]
            updates["execution_error"] = None

            # Now, actually execute the validated/corrected query to get results for QueryExecutor
            # The SQLVerifier only validated it, QueryExecutor needs to run it for results.
            # We re-use the _execute_sql_query method which was originally here for execution.
            # This means _execute_sql_query should remain in QueryExecutorWorkflow.

            validated_sql_to_execute = verifier_result["corrected_sql_query"]
            results, error = await sync_to_async(self._execute_sql_query)(
                sql_query=validated_sql_to_execute
            )

            if error:
                logger.error(
                    f"SQL execution failed AFTER successful validation/correction by verifier: {error}"
                )
                state["debugging_log"].append(
                    f"Execution of verifier-approved SQL failed. Error: {error}"
                )
                updates["execution_result_rows"] = None
                updates["execution_error"] = (
                    error  # This error is from the actual data-fetching execution
                )
                updates["is_success"] = False  # Mark as not successful overall for QE
                updates["final_status_message"] = (
                    f"Query was validated but failed during final execution: {error}"
                )
            else:
                num_rows = len(results) - 1 if results and len(results) > 0 else 0
                logger.info(
                    f"SQL execution successful after verifier. Rows returned: {num_rows}"
                )
                state["debugging_log"].append(
                    f"Successfully executed verifier-approved SQL. Rows: {num_rows}."
                )
                updates["execution_result_rows"] = results
                updates["execution_error"] = (
                    None  # Clear any prior verifier errors if final exec is good
                )
                # is_success will be determined later based on results processing
        else:
            error_msg_from_verifier = verifier_result.get(
                "execution_error",
                "SQLVerifierWorkflow failed to validate or correct the query.",
            )
            debug_explanation = verifier_result.get("debug_explanation", "")
            logger.warning(
                f"SQLVerifierWorkflow could not validate/correct the query. Error: {error_msg_from_verifier}. Explanation: {debug_explanation}"
            )
            state["debugging_log"].append(
                f"SQLVerifierWorkflow failed. Error: {error_msg_from_verifier}. Explanation: {debug_explanation}"
            )
            updates["execution_result_rows"] = None
            updates["execution_error"] = error_msg_from_verifier
            updates["current_sql_query"] = verifier_result.get(
                "current_sql_query"
            )  # The last attempted query by verifier
            updates["is_success"] = False  # Mark as not successful overall for QE
            updates["final_status_message"] = (
                f"Failed to validate/execute SQL: {error_msg_from_verifier}. {debug_explanation}".strip()
            )

        return updates

    async def process_execution_results_node(
        self, state: QueryExecutorState
    ) -> Dict[str, Any]:
        logger.info("Node: process_execution_results_node")
        # This node mainly serves as a branching point.
        # The actual decision is in `should_prepare_csv_or_debug`.
        # No state changes are made here directly.
        return {}

    async def prepare_csv_node(self, state: QueryExecutorState) -> Dict[str, Any]:
        logger.info("Node: prepare_csv_node")
        rows = state.get("execution_result_rows")

        if not rows:
            logger.warning("prepare_csv_node called without execution_result_rows.")
            return {
                "csv_content": None,
                "execution_error": state.get("execution_error")
                or "No data to format into CSV.",  # Preserve or set error
            }

        try:
            output = io.StringIO()
            csv_writer = csv.writer(output)
            for row in rows:
                csv_writer.writerow(row)
            csv_data = output.getvalue()
            output.close()
            logger.info(f"CSV prepared successfully. Length: {len(csv_data)}")
            return {
                "csv_content": csv_data,
                "execution_error": None,  # Clear error as CSV prep was successful
            }
        except Exception as e:
            logger.error(f"Error preparing CSV: {e}", exc_info=self.verbose)
            return {
                "csv_content": None,
                "execution_error": f"Failed to prepare CSV: {e}",
            }

    async def post_success_to_slack_node(
        self, state: QueryExecutorState
    ) -> Dict[str, Any]:
        logger.info("Node: post_success_to_slack_node")

        csv_data = state.get("csv_content")
        user_id = state.get("user_id", "user")  # Get user_id for mention
        debugging_log_entries = state.get("debugging_log", [])
        formatted_debug_log = ""
        if debugging_log_entries:
            # Keep the header concise for the separate message
            formatted_debug_log = "--- Debugging & Execution Log ---\n" + "\n".join(
                f"- {entry}" for entry in debugging_log_entries
            )

        if not self.slack_client:
            error_msg = "Slack client is not available. Cannot post results."
            logger.error(error_msg)
            return {
                "is_success": False,
                "final_status_message": error_msg,
                "execution_error": state.get("execution_error") or error_msg,
            }

        if not csv_data:
            # This path implies an issue if prepare_csv was supposed to run and create data.
            error_msg = "No CSV content available to post. This might indicate an earlier issue."
            logger.error(error_msg)
            # Post a message to user about this internal issue.
            await self.post_message_tool_func.ainvoke(
                {
                    "message_text": f"<@{user_id}>, I prepared to send the results, but the data was missing. Please check for earlier errors.",
                    "state": state,
                }
            )
            return {
                "is_success": False,
                "final_status_message": error_msg,
                "execution_error": state.get("execution_error") or error_msg,
            }

        comment = f"Successfully executed your query, <@{user_id}>! Results attached."
        # MODIFICATION: Use the concise comment for the CSV upload
        initial_comment_for_csv = comment

        # Ensure a unique filename to prevent Slack caching issues or overwrites if names were identical.
        filename = f"query_results_{state.get('original_trigger_message_ts', 'unknown_ts')}_{uuid.uuid4().hex[:8]}.csv"

        try:
            # Call the post_csv_tool_func (which is the @tool decorated post_csv_to_thread)
            tool_result = await self.post_csv_tool_func.ainvoke(
                {
                    "csv_data": csv_data,
                    "initial_comment": initial_comment_for_csv,  # MODIFIED: Use concise comment
                    "filename": filename,
                    "state": state,  # Pass the state object
                }
            )

            if tool_result.get("success"):
                logger.info(
                    f"Successfully posted CSV to Slack. File ID (if any): {tool_result.get('file_id')}"
                )
                updates_for_state = {
                    "is_success": True,
                    "final_status_message": "Query results successfully posted to Slack.",
                    "posted_file_ts": tool_result.get(
                        "file_id"
                    ),  # Store file ID or similar identifier
                    "execution_error": None,  # Clear error on success
                }

                # MODIFICATION: Post the debug log as a separate message if it exists
                if formatted_debug_log:
                    try:
                        log_post_result = await self.post_message_tool_func.ainvoke(
                            {
                                "message_text": formatted_debug_log,
                                "state": state,  # Use the same state for channel/thread context
                            }
                        )
                        if log_post_result.get("success"):
                            logger.info("Successfully posted debugging log to Slack.")
                        else:
                            logger.warning(
                                f"Failed to post debugging log to Slack: {log_post_result.get('error')}"
                            )
                            # Optionally, update final_status_message or log this minor failure
                            # For now, just logging the warning.
                    except Exception as e_log_post:
                        logger.error(
                            f"Exception posting debugging log: {e_log_post}",
                            exc_info=self.verbose,
                        )

                # Check if the query was modified and post the corrected SQL
                fetched_sql = state.get("fetched_sql_query")
                current_sql = state.get("current_sql_query")
                if current_sql and fetched_sql != current_sql:
                    logger.info(
                        "Original query was modified. Posting corrected SQL to Slack."
                    )
                    corrected_sql_filename = f"corrected_query_for_{state.get('original_trigger_message_ts', 'unknown_ts')}.sql"
                    corrected_sql_comment = "Here is the corrected SQL query that was successfully executed:"
                    # Also append debug log to the corrected SQL message if it's not already part of the main success message
                    # For now, the main success message (initial_comment_for_csv) already contains it.
                    # If we wanted a separate post for the SQL with its own context, we could add it here.
                    # For simplicity, we assume the main message's log is sufficient.
                    # If we decide to post the log with the corrected SQL as well, it would be:
                    # corrected_sql_comment_with_log = corrected_sql_comment + formatted_debug_log
                    # However, this might be redundant if the CSV post already has the log.
                    # Let's keep it to the initial comment for now to avoid double posting the log.

                    try:
                        sql_post_result = await self.post_text_file_tool_func.ainvoke(
                            {
                                "text_content": current_sql,
                                "initial_comment": corrected_sql_comment,
                                "filename": corrected_sql_filename,
                                "state": state,
                            }
                        )
                        if sql_post_result.get("success"):
                            logger.info(
                                f"Successfully posted corrected SQL file: {corrected_sql_filename}"
                            )
                        else:
                            logger.warning(
                                f"Failed to post corrected SQL file: {sql_post_result.get('error')}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Exception calling post_text_file_tool_func for corrected SQL: {e}",
                            exc_info=self.verbose,
                        )
                return updates_for_state
            else:
                error_from_tool = tool_result.get(
                    "error", "Failed to post CSV to Slack."
                )
                logger.error(f"post_csv_to_thread tool failed: {error_from_tool}")
                # Notify user that posting the file failed.
                await self.post_message_tool_func.ainvoke(
                    {
                        "message_text": f"<@{user_id}>, I executed the query, but failed to upload the results file: {error_from_tool}",
                        "state": state,
                    }
                )
                return {
                    "is_success": False,  # Overall success is false if file not posted
                    "final_status_message": f"Failed to post CSV results to Slack: {error_from_tool}",
                    "execution_error": error_from_tool,
                }
        except Exception as e:  # Catch errors from ainvoke or tool execution itself
            logger.error(
                f"Exception calling post_csv_tool_func: {e}", exc_info=self.verbose
            )
            await self.post_message_tool_func.ainvoke(
                {
                    "message_text": f"<@{user_id}>, an unexpected error occurred while trying to upload your query results: {e}",
                    "state": state,
                }
            )
            return {
                "is_success": False,
                "final_status_message": f"An unexpected error occurred while posting CSV to Slack: {e}",
                "execution_error": str(e),
            }

    async def post_failure_to_slack_node(
        self, state: QueryExecutorState
    ) -> Dict[str, Any]:
        logger.info("Node: post_failure_to_slack_node")

        user_id = state.get("user_id", "user")
        error_from_state = state.get(
            "execution_error", "an unspecified technical difficulty"
        )
        attempts = state.get(
            "debug_attempts", 0
        )  # This includes the current failed attempt

        # Construct the main failure message first
        base_final_message = f"Sorry, <@{user_id}>, I encountered an issue trying to execute the SQL query. "
        if attempts > 0:
            base_final_message += (
                f"After {attempts} attempt(s) to fix it, the problem remains. "
            )
        base_final_message += f"The last error was: `{error_from_state}`"

        posted_ts = None
        # Post the main failure message
        if self.slack_client:
            try:
                main_message_result = await self.post_message_tool_func.ainvoke(
                    {
                        "message_text": base_final_message,
                        "state": state,
                    }
                )
                if main_message_result.get("success"):
                    posted_ts = main_message_result.get("ts")
                    logger.info(
                        f"Successfully posted main failure message to Slack. TS: {posted_ts}"
                    )
                else:
                    tool_error = main_message_result.get(
                        "error", "Unknown error posting main failure message"
                    )
                    logger.error(
                        f"Failed to post main failure message to Slack: {tool_error}"
                    )
                    # Even if main message fails, we might still want to log the error for the return state
            except Exception as e_main_msg:
                logger.error(
                    f"Exception posting main failure message: {e_main_msg}",
                    exc_info=self.verbose,
                )
        else:
            logger.error(
                "Slack client not available. Cannot post failure message to user."
            )

        # Now, handle the debugging log as a follow-up
        debugging_log_entries = state.get("debugging_log", [])
        if debugging_log_entries:
            full_log_text = "--- Debugging & Execution Log ---\n" + "\n".join(
                f"- {entry}" for entry in debugging_log_entries
            )

            # Check length and decide to post as file or message
            if len(full_log_text) > 3000 and hasattr(self, "post_text_file_tool_func"):
                try:
                    log_filename = f"debug_log_{state.get('original_trigger_message_ts', 'unknown_ts')}_{uuid.uuid4().hex[:8]}.txt"
                    log_file_comment = "Detailed debugging and execution log:"
                    file_post_result = await self.post_text_file_tool_func.ainvoke(
                        {
                            "text_content": full_log_text,
                            "initial_comment": log_file_comment,
                            "filename": log_filename,
                            "state": state,  # Ensures it goes to the same thread
                        }
                    )
                    if file_post_result.get("success"):
                        logger.info(
                            f"Successfully posted lengthy debug log as file: {log_filename}"
                        )
                    else:
                        logger.warning(
                            f"Failed to post lengthy debug log as file: {file_post_result.get('error')}. Attempting to post truncated log as message."
                        )
                        # Fallback: try to post a truncated version as a message
                        truncated_log = full_log_text[:2800] + "... (log truncated)"
                        await self.post_message_tool_func.ainvoke(
                            {"message_text": truncated_log, "state": state}
                        )
                except Exception as e_log_file:
                    logger.error(
                        f"Exception posting debug log as file: {e_log_file}",
                        exc_info=self.verbose,
                    )
                    # Fallback: try to post a truncated version as a message
                    truncated_log = (
                        full_log_text[:2800]
                        + "... (log truncated due to error posting as file)"
                    )
                    await self.post_message_tool_func.ainvoke(
                        {"message_text": truncated_log, "state": state}
                    )
            else:  # Log is not too long, post as a direct message
                try:
                    log_message_result = await self.post_message_tool_func.ainvoke(
                        {"message_text": full_log_text, "state": state}
                    )
                    if log_message_result.get("success"):
                        logger.info(
                            "Successfully posted debug log as follow-up message."
                        )
                    else:
                        logger.warning(
                            f"Failed to post debug log as follow-up message: {log_message_result.get('error')}"
                        )
                except Exception as e_log_msg:
                    logger.error(
                        f"Exception posting debug log as follow-up message: {e_log_msg}",
                        exc_info=self.verbose,
                    )

        # The final_status_message in the state should reflect the main error, not the log content.
        return {
            "is_success": False,
            "final_status_message": base_final_message,  # The message sent to user about the failure
            "posted_response_ts": posted_ts,  # TS of the main failure message
        }

    async def finalize_workflow_node(self, state: QueryExecutorState) -> Dict[str, Any]:
        logger.info("Node: finalize_workflow_node")
        # This node is the formal end. Log final state.
        success_status = state.get("is_success", False)  # Default to False if not set
        final_msg = state.get("final_status_message", "Workflow ended.")
        logger.info(
            f"Workflow finished. Success: {success_status}. Final Message: {final_msg}"
        )
        # No state changes here, it's just a terminal point.
        return {}

    # --- Conditional Edge Functions ---
    def should_prepare_csv_or_post_failure(self, state: QueryExecutorState) -> str:
        # This condition is checked after execute_sql_node (which runs the verifier and then actual execution)
        # and after process_execution_results_node.
        # execute_sql_node is responsible for setting execution_error and potentially is_success = False
        # if the SQLVerifierWorkflow fails or if the subsequent data-fetching execution fails.

        execution_error = state.get("execution_error")
        is_success_flag = state.get(
            "is_success"
        )  # This should be set by execute_sql_node or process_execution_results

        # If there was any execution error flagged by execute_sql_node (either from verifier or final run)
        if execution_error:
            logger.info(
                f"Conditional: Execution error is present ('{execution_error}'). Routing to post_failure_to_slack."
            )
            return "post_failure_to_slack"

        # If is_success is explicitly False, even without an error message (though unlikely)
        if is_success_flag is False:
            logger.info(
                f"Conditional: is_success is explicitly False. Routing to post_failure_to_slack."
            )
            return "post_failure_to_slack"

        # If no execution_error and is_success is not False (i.e. True, or None which implies success if no error)
        # A successful run through execute_sql_node means:
        # 1. SQLVerifier found/made the query valid.
        # 2. The subsequent _execute_sql_query to fetch data also succeeded.
        # In this case, execution_result_rows should be populated and execution_error should be None.
        logger.info(
            "Conditional: No execution error and is_success is not False. Routing to prepare_csv."
        )
        return "prepare_csv"

    # --- Main Workflow Runner ---
    async def run_workflow(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        trigger_message_ts: str,
    ) -> Dict[str, Any]:
        """Runs the query executor workflow with the given Slack event context."""
        if not self.slack_client:
            # This case should ideally be caught by the calling view/service first.
            err_msg = "QueryExecutorWorkflow: Slack client not initialized. Cannot run workflow."
            logger.error(err_msg)
            return {"error": err_msg, "is_success": False}

        if not self.snowflake_creds:
            # This is a critical configuration error.
            err_msg = "QueryExecutorWorkflow: Snowflake credentials not loaded. Cannot execute queries."
            logger.error(err_msg)
            # Attempt to notify Slack if possible
            # Construct a minimal state for the post_message tool
            minimal_state_for_message = QueryExecutorState(
                channel_id=channel_id,
                thread_ts=thread_ts,
                user_id=user_id,
                original_trigger_message_ts=trigger_message_ts,
                # Fill other required fields with defaults if `PostMessageInput` or `state` for the tool needs them
                fetched_sql_query=None,
                current_sql_query=None,
                execution_result_rows=None,
                execution_error=err_msg,
                csv_content=None,
                debug_attempts=0,
                max_debug_attempts=self.max_debug_loops,
                described_tables_info={},
                listed_column_values={},
                messages=[],
                final_status_message=err_msg,
                is_success=False,
                posted_response_ts=None,
                posted_file_ts=None,
                conversation_id="",  # Minimal, won't be used for checkpointing in this error path
                debugging_log=[f"CRITICAL ERROR: {err_msg}"],  # Log the critical error
            )
            if self.slack_client and hasattr(self, "post_message_tool_func"):
                try:
                    await self.post_message_tool_func.ainvoke(
                        {"message_text": err_msg, "state": minimal_state_for_message}
                    )
                except Exception as post_err:
                    logger.error(
                        f"Failed to post Snowflake credential error to Slack: {post_err}"
                    )

            return {
                "error": err_msg,
                "is_success": False,
                "final_status_message": err_msg,
                "conversation_id": f"query-executor-error-{str(uuid.uuid4())}",
            }

        # Use a unique conversation ID for each run if not provided or for better tracking
        # This helps in separating logs and potentially checkpointing if ever enabled.
        conversation_id = f"query-executor-{channel_id}-{thread_ts}-{trigger_message_ts}-{str(uuid.uuid4())[:8]}"
        config = {
            "configurable": {"thread_id": conversation_id}
        }  # For LangGraph checkpointing

        initial_state: QueryExecutorState = {
            "channel_id": channel_id,
            "thread_ts": thread_ts,  # Parent thread for replies
            "user_id": user_id,
            "original_trigger_message_ts": trigger_message_ts,  # Specific message with .sql or link
            "fetched_sql_query": None,
            "current_sql_query": None,
            "execution_result_rows": None,
            "execution_error": None,
            "csv_content": None,
            "debug_attempts": 0,
            "max_debug_attempts": self.max_debug_loops,
            "described_tables_info": {},
            "listed_column_values": {},
            "messages": [],
            "final_status_message": None,
            "is_success": None,
            "posted_response_ts": None,
            "posted_file_ts": None,
            "conversation_id": conversation_id,
            "debugging_log": [],  # Initialize the debugging log
        }

        graph = self._build_graph()
        if not self.graph_app:  # Should have been built in __init__
            logger.error(
                "QueryExecutorWorkflow: graph_app not initialized. Cannot run workflow."
            )
            return {
                "error": "Internal error: workflow graph not initialized",
                "is_success": False,
            }

        final_graph_output_map = {}  # To store the output of the graph execution
        try:
            logger.info(
                f"Invoking QueryExecutorWorkflow graph for {channel_id}/{thread_ts}"
            )
            # Use astream to get events and the final state.
            # The final result of astream is the full state object if the graph ends.
            async for event in self.graph_app.astream(
                initial_state, stream_mode="values"
            ):
                # stream_mode="values" should yield the full state object at each step.
                # The last one will be the final state.
                if self.verbose:
                    # Log event structure to understand what keys are present
                    event_keys = list(event.keys())
                    logger.info(f"Graph Event Keys: {event_keys}")
                    # Example: Log specific fields from the event if they exist
                    # current_node = event.get("__name__") # This depends on stream_mode and LangGraph version
                    # logger.info(f"  Current Node (if available): {current_node}")
                    # logger.info(f"  Current State (partial): execution_error: {event.get('execution_error')}, current_sql_query: {str(event.get('current_sql_query'))[:50]}")

                final_graph_output_map = event  # Keep the latest state

            logger.info(
                f"QueryExecutorWorkflow graph finished for {channel_id}/{thread_ts}. Final state collected."
            )

        except Exception as e:
            logger.error(
                f"Critical error running QueryExecutorWorkflow graph for {channel_id}/{thread_ts}: {e}",
                exc_info=True,  # Log full traceback
            )
            final_status_message = (
                f"A critical internal error occurred in the workflow: {e}"
            )
            # Try to post to Slack
            # Construct a minimal state for the post_message tool
            minimal_state_for_error_post = QueryExecutorState(
                channel_id=channel_id,
                thread_ts=thread_ts,
                user_id=user_id,
                original_trigger_message_ts=trigger_message_ts,
                fetched_sql_query=initial_state.get("fetched_sql_query"),
                current_sql_query=initial_state.get("current_sql_query"),
                execution_result_rows=None,
                execution_error=str(e),
                csv_content=None,
                debug_attempts=initial_state.get("debug_attempts", 0),
                max_debug_attempts=self.max_debug_loops,
                described_tables_info={},
                listed_column_values={},
                messages=[],
                final_status_message=final_status_message,
                is_success=False,
                posted_response_ts=None,
                posted_file_ts=None,
                conversation_id=str(uuid.uuid4()),
                debugging_log=[],
            )
            if self.slack_client and hasattr(self, "post_message_tool_func"):
                try:
                    await self.post_message_tool_func.ainvoke(
                        {
                            "message_text": final_status_message,
                            "state": minimal_state_for_error_post,
                        }
                    )
                except Exception as post_err:
                    logger.error(
                        f"Failed to send critical graph error to Slack: {post_err}"
                    )

            return {  # Return a structured error response
                "error": str(e),
                "is_success": False,
                "final_status_message": final_status_message,
                "conversation_id": str(uuid.uuid4()),
            }

        # Extract results from the final graph state
        return {
            "is_success": final_graph_output_map.get("is_success"),
            "final_status_message": final_graph_output_map.get("final_status_message"),
            "posted_response_ts": final_graph_output_map.get("posted_response_ts"),
            "posted_file_ts": final_graph_output_map.get("posted_file_ts"),
            "conversation_id": str(uuid.uuid4()),
            "error": (
                final_graph_output_map.get("execution_error")
                if not final_graph_output_map.get("is_success")
                else None
            ),
        }


# Example of how this might be called by a Slack handler (conceptual)
async def handle_slack_event_for_query_executor(payload: Dict):  # Added type hint
    # ... (conceptual, needs actual payload structure and async setup)
    pass


if __name__ == "__main__":
    # Basic setup for local testing - requires environment variables for Slack & DB
    # Needs Django settings (os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yourproject.settings'); django.setup())
    # import asyncio
    # asyncio.run(handle_slack_event_for_query_executor({}))
    print("Query Executor Workflow module loaded. Define test scenarios to run.")
