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
from pydantic import BaseModel, Field  # For tool input schemas
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver  # For potential checkpointing
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# Django & Project Imports
from django.conf import settings as django_settings  # To access Django global settings
from asgiref.sync import sync_to_async

# LLM Service import (similar to QuestionAnswerer)
from apps.llm_providers.services import default_chat_service

# Query Executor specific imports
# REMOVED: from . import tools as query_executor_tools
from . import (
    settings as query_executor_app_settings,
)  # Our app-specific settings (for Snowflake)
from .prompts import SQL_DEBUG_PROMPT_TEMPLATE

# Import the new centralized Slack client getter
from apps.workflows.services import get_slack_web_client

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
class PostMessageInput(BaseModel):
    message_text: str = Field(
        description="The text message to post to the Slack thread."
    )


class PostCsvInput(BaseModel):
    csv_data: str = Field(description="The CSV data as a string.")
    initial_comment: str = Field(description="A comment to accompany the CSV file.")
    filename: str = Field(
        default="query_results.csv", description="The filename for the CSV."
    )


# Potentially for LLM-assisted debugging, if it needs to call schema tools:
class DescribeTableInput(BaseModel):
    table_name: str = Field(description="The name of the table to describe.")


class ListColumnValuesInput(BaseModel):
    table_name: str = Field(description="The name of the table.")
    column_name: str = Field(
        description="The name of the column to list distinct values for."
    )
    limit: int = Field(
        default=20, description="Max number of distinct values to return."
    )


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
            self.chat_service = default_chat_service
            self.llm = self.chat_service.get_client()
            if not self.llm:
                logger.error(
                    "QueryExecutorWorkflow: LLM client could not be initialized from default_chat_service. Debugging LLM will not work."
                )
        except Exception as e:
            logger.error(
                f"QueryExecutorWorkflow: Failed to initialize chat service or LLM client: {e}",
                exc_info=self.verbose,
            )
            self.chat_service = None
            self.llm = None

        self.memory = memory
        self.max_debug_loops = max_debug_loops

        try:
            self.snowflake_creds = (
                query_executor_app_settings.get_snowflake_credentials()
            )
            self.warehouse_type = query_executor_app_settings.get_data_warehouse_type()
        except ValueError as e:
            logger.error(
                f"QueryExecutorWorkflow: Critical - Failed to load Snowflake credentials: {e}. Workflow may not function."
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

        # Tools for LLM-assisted debugging (using self._execute_sql_query)
        @tool(args_schema=DescribeTableInput)
        async def describe_table_tool(table_name: str) -> Dict[str, Any]:
            """Describes the schema of a given table in the data warehouse (for LLM use).
            (Logic moved from tools.py and uses internal _execute_sql_query)
            """
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
            table_name: str, column_name: str, limit: int
        ) -> Dict[str, Any]:
            """Lists distinct values for a given column in a table (for LLM use).
            (Logic moved from tools.py and uses internal _execute_sql_query)
            """
            # Basic validation for table and column names
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", table_name) or not re.match(
                r"^[a-zA-Z_][a-zA-Z0-9_]*$", column_name
            ):  # Column name more restrictive
                return {"error": "Invalid table or column name format."}

            if not isinstance(limit, int) or limit <= 0:
                limit = 20  # Default to a safe limit (as per original schema)
                logger.warning(
                    f"Invalid limit specified for list_column_values, defaulting to {limit}"
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

        # self.slack_tools was used before, let's ensure consistency
        # The tools for ToolNode are those callable by the LLM.
        self.debug_llm_tools = [  # These are for the debugging LLM
            describe_table_tool,
            list_column_values_tool,
        ]
        # If post_message_to_thread and post_csv_to_thread can be called by an LLM, add them.
        # For now, they seem to be called directly by nodes.
        # self.all_defined_tools combines all tools that might be registered with a ToolNode
        self.all_defined_tools = self.debug_llm_tools  # + any other LLM-callable tools

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(QueryExecutorState)

        # Define Nodes
        workflow.add_node("fetch_sql_from_slack", self.fetch_sql_from_slack_node)
        workflow.add_node("execute_sql", self.execute_sql_node)
        workflow.add_node(
            "process_execution_results", self.process_execution_results_node
        )
        workflow.add_node("prepare_csv", self.prepare_csv_node)
        workflow.add_node("post_success_to_slack", self.post_success_to_slack_node)
        workflow.add_node("attempt_debug_or_fail", self.attempt_debug_or_fail_node)

        workflow.add_node("invoke_debug_llm", self.invoke_debug_llm_node)
        # ToolNode uses self.debug_llm_tools (or self.all_defined_tools if more general)
        debug_tool_node = ToolNode(self.debug_llm_tools, handle_tool_errors=True)
        workflow.add_node("debug_tools", debug_tool_node)
        workflow.add_node(
            "update_state_after_debug_tool", self.update_state_after_debug_tool_node
        )
        workflow.add_node(
            "update_query_from_llm_suggestion",
            self.update_query_from_llm_suggestion_node,
        )

        workflow.add_node("post_failure_to_slack", self.post_failure_to_slack_node)
        workflow.add_node("finalize_workflow", self.finalize_workflow_node)

        # Define Edges
        workflow.set_entry_point("fetch_sql_from_slack")
        workflow.add_edge("fetch_sql_from_slack", "execute_sql")
        workflow.add_edge("execute_sql", "process_execution_results")

        workflow.add_conditional_edges(
            "process_execution_results",
            self.should_prepare_csv_or_debug,
            {
                "prepare_csv": "prepare_csv",
                "attempt_debug_or_fail": "attempt_debug_or_fail",
            },
        )
        workflow.add_edge("prepare_csv", "post_success_to_slack")
        workflow.add_edge("post_success_to_slack", "finalize_workflow")

        workflow.add_conditional_edges(
            "attempt_debug_or_fail",
            self.should_invoke_debug_llm_or_post_failure,
            {
                "invoke_debug_llm": "invoke_debug_llm",
                "post_failure_to_slack": "post_failure_to_slack",
            },
        )

        workflow.add_conditional_edges(
            "invoke_debug_llm",
            tools_condition,
            {
                "debug_tools": "debug_tools",
                END: "update_query_from_llm_suggestion",
            },
        )
        workflow.add_edge("debug_tools", "update_state_after_debug_tool")
        workflow.add_edge("update_state_after_debug_tool", "invoke_debug_llm")

        workflow.add_edge("update_query_from_llm_suggestion", "execute_sql")

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
        if not self.slack_client:  # Use self.slack_client here
            logger.error("Slack client not available in fetch_sql_from_slack_node.")
            return {
                "execution_error": "Slack client not configured, cannot fetch query."
            }

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
            return {"execution_error": f"Failed to fetch SQL from Slack: {e}"}

    async def execute_sql_node(self, state: QueryExecutorState) -> Dict[str, Any]:
        logger.info(
            f"Node: execute_sql_node (Attempt: {state.get('debug_attempts', 0) + 1})"
        )
        if not self.snowflake_creds:
            logger.error("Snowflake credentials not available in execute_sql_node.")
            error_message_to_user = "Critical error: Snowflake connection details are not configured. Cannot execute query."
            if self.slack_client:
                # Call the tool, ensuring `state` is passed.
                await self.post_message_tool_func.ainvoke(
                    {"message_text": error_message_to_user, "state": state}
                )
            return {
                "execution_error": error_message_to_user,
                "is_success": False,
                "final_status_message": error_message_to_user,
            }

        current_query = state.get("current_sql_query")
        if not current_query:
            logger.error("No SQL query found in state for execution.")
            # This is an internal error, user message might not be needed or could be generic.
            internal_error_msg = "Internal error: No query to execute."
            await self.post_message_tool_func.ainvoke(
                {"message_text": internal_error_msg, "state": state}
            )
            return {
                "execution_error": internal_error_msg,
                "is_success": False,
                "final_status_message": internal_error_msg,
            }

        logger.info(f"Executing SQL query (first 500 chars): {current_query[:500]}...")

        # Call the internal synchronous helper method using sync_to_async
        results, error = await sync_to_async(self._execute_sql_query)(
            sql_query=current_query
        )

        if error:
            logger.warning(f"SQL execution failed: {error}")
            return {"execution_result_rows": None, "execution_error": error}
        else:
            logger.info(
                f"SQL execution successful. Rows returned: {len(results) -1 if results and len(results) > 0 else 0} (excluding header)"
            )
            return {"execution_result_rows": results, "execution_error": None}

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
        # Ensure a unique filename to prevent Slack caching issues or overwrites if names were identical.
        filename = f"query_results_{state.get('original_trigger_message_ts', 'unknown_ts')}_{uuid.uuid4().hex[:8]}.csv"

        try:
            # Call the post_csv_tool_func (which is the @tool decorated post_csv_to_thread)
            tool_result = await self.post_csv_tool_func.ainvoke(
                {
                    "csv_data": csv_data,
                    "initial_comment": comment,
                    "filename": filename,
                    "state": state,  # Pass the state object
                }
            )

            if tool_result.get("success"):
                logger.info(
                    f"Successfully posted CSV to Slack. File ID (if any): {tool_result.get('file_id')}"
                )
                return {
                    "is_success": True,
                    "final_status_message": "Query results successfully posted to Slack.",
                    "posted_file_ts": tool_result.get(
                        "file_id"
                    ),  # Store file ID or similar identifier
                    "execution_error": None,  # Clear error on success
                }
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

    async def attempt_debug_or_fail_node(
        self, state: QueryExecutorState
    ) -> Dict[str, Any]:
        logger.info("Node: attempt_debug_or_fail_node")
        current_attempts = state.get("debug_attempts", 0)
        updates = {"debug_attempts": current_attempts + 1}
        # The decision to proceed to debug or fail is handled by the conditional edge
        # `should_invoke_debug_llm_or_post_failure` based on `debug_attempts`.
        return updates

    async def invoke_debug_llm_node(self, state: QueryExecutorState) -> Dict[str, Any]:
        logger.info("Node: invoke_debug_llm_node")

        if not self.llm:
            error_msg = "LLM client not available for SQL debugging. Cannot attempt to fix query."
            logger.error(error_msg)
            # This AIMessage will be added to state['messages'] and handled by update_query_from_llm_suggestion_node
            return {"messages": [AIMessage(content=f"CriticalError: {error_msg}")]}

        current_query = state.get("current_sql_query", "")
        error_message_from_db = state.get("execution_error", "Unknown execution error.")

        table_schemas_str = json.dumps(state.get("described_tables_info", {}), indent=2)
        column_values_str = json.dumps(state.get("listed_column_values", {}), indent=2)

        prompt_content = SQL_DEBUG_PROMPT_TEMPLATE.format(
            error_message=error_message_from_db,
            sql_query=current_query,
            table_schemas=(
                table_schemas_str
                if table_schemas_str not in ["{}", "null"]  # Check for empty or null
                else "No schema information available for this attempt."
            ),
            column_values=(
                column_values_str
                if column_values_str not in ["{}", "null"]  # Check for empty or null
                else "No column values available for this attempt."
            ),
        )

        # Debug LLM messages should be isolated for each debug attempt if possible,
        # or carefully managed if accumulated.
        # For simplicity, using state["messages"] which accumulates.
        # A more advanced setup might use a sub-graph for debugging with its own message history.
        messages_for_llm: List[BaseMessage] = [SystemMessage(content=prompt_content)]

        # Add relevant past messages from this debug interaction if any.
        # `state.get("messages", [])` currently holds ALL messages.
        # We might only want messages since the last `execute_sql_node` failure for this specific debug.
        # For now, we'll keep it simple and pass the system prompt. If more context is needed,
        # the `messages` field in the state would need more careful management for the debug loop.
        # Let's just pass the system prompt and let the LLM decide if it needs tools.
        # The `tools_condition` will route to `debug_tools` if the LLM calls any.

        if self.verbose:
            logger.info(
                f"SQL Debug System Prompt (first 500 chars): {prompt_content[:500]}..."
            )
            # Log messages for LLM
            log_llm_messages_content = [f"  0. System: {prompt_content[:150]}..."]
            # If passing more messages:
            # for i, msg in enumerate(state.get("messages", [])): # Example if passing history
            #     log_llm_messages_content.append(f"  {i+1}. Type: {type(msg).__name__}, Content: {str(msg.content)[:100]}...")
            logger.info(
                "Messages for Debug LLM (invoke_debug_llm_node):\n"
                + "\n".join(log_llm_messages_content)
            )

        # Bind the debug-specific tools
        llm_with_tools = self.llm.bind_tools(self.debug_llm_tools)
        config = {"run_name": "DebugSQLLLMNode"}

        try:
            # Invoke the LLM. Messages are added to state by add_messages mechanism.
            response_ai_message = await llm_with_tools.ainvoke(
                messages_for_llm,
                config=config,  # Pass only the fresh system prompt for this attempt
            )
            if self.verbose:
                logger.info(f"Debug LLM Response: {response_ai_message}")
            # The graph will add this response to state['messages'] via add_messages.
            # This node's output should be structured to be merged into the state.
            return {"messages": [response_ai_message]}
        except Exception as e:
            logger.error(f"Error invoking SQL Debug LLM: {e}", exc_info=self.verbose)
            return {
                "messages": [
                    AIMessage(content=f"Error during LLM call for SQL debug: {e}")
                ]
            }

    async def update_state_after_debug_tool_node(
        self, state: QueryExecutorState
    ) -> Dict[str, Any]:
        logger.info("Node: update_state_after_debug_tool_node")
        updates: Dict[str, Any] = {}
        last_message = state["messages"][-1] if state.get("messages") else None

        if not isinstance(last_message, ToolMessage):
            logger.warning(
                "update_state_after_debug_tool_node: Last message not a ToolMessage."
            )
            return updates  # No tool message to process

        tool_name = last_message.name
        tool_content_str = (
            last_message.content
        )  # Content is usually stringified JSON from ToolNode

        try:
            # ToolNode stringifies the dict output of the tool.
            tool_data = (
                json.loads(tool_content_str)
                if isinstance(tool_content_str, str)
                else tool_content_str
            )
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool content JSON: {tool_content_str}")
            # updates["execution_error"] = f"Debug tool {tool_name} output was not valid JSON." # Potentially set an error
            return updates

        if tool_name == "describe_table_tool":
            if tool_data and "schema" in tool_data:
                # Assuming tool_data["schema"] is the schema list.
                # We need to store this against the table name it was for.
                # The input arguments to the tool (table_name) are in the AIMessage that called the tool.
                # This requires looking back further in messages or having tool calls include inputs.
                # For simplicity, if only one table is described at a time in debug:
                # Find corresponding AIMessage:
                ai_message_with_tool_call = None
                for msg in reversed(state.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc.get("id") == last_message.tool_call_id:
                                ai_message_with_tool_call = msg
                                break
                        if ai_message_with_tool_call:
                            break

                table_name_described = "unknown_table"
                if ai_message_with_tool_call:
                    for tc in ai_message_with_tool_call.tool_calls:
                        if (
                            tc.get("id") == last_message.tool_call_id
                            and tc.get("name") == "describe_table_tool"
                        ):
                            table_name_described = tc.get("args", {}).get(
                                "table_name", "unknown_table"
                            )
                            break

                current_described_tables = state.get("described_tables_info", {}).copy()
                current_described_tables[table_name_described] = tool_data["schema"]
                updates["described_tables_info"] = current_described_tables
                logger.info(
                    f"Updated described_tables_info for {table_name_described} from debug tool."
                )

        elif tool_name == "list_column_values_tool":
            if tool_data and "values" in tool_data:
                # Similar logic to find table_name and column_name from AIMessage tool call
                ai_message_with_tool_call = None
                for msg in reversed(state.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc.get("id") == last_message.tool_call_id:
                                ai_message_with_tool_call = msg
                                break
                        if ai_message_with_tool_call:
                            break

                key_for_values = "unknown_column_values"
                if ai_message_with_tool_call:
                    for tc in ai_message_with_tool_call.tool_calls:
                        if (
                            tc.get("id") == last_message.tool_call_id
                            and tc.get("name") == "list_column_values_tool"
                        ):
                            args = tc.get("args", {})
                            tn = args.get("table_name", "unk_table")
                            cn = args.get("column_name", "unk_col")
                            key_for_values = f"{tn}.{cn}"
                            break

                current_listed_values = state.get("listed_column_values", {}).copy()
                current_listed_values[key_for_values] = tool_data["values"]
                updates["listed_column_values"] = current_listed_values
                logger.info(
                    f"Updated listed_column_values for {key_for_values} from debug tool."
                )

        if self.verbose and updates:
            logger.info(f"State updates after debug tool: {updates.keys()}")
        return updates

    async def update_query_from_llm_suggestion_node(
        self, state: QueryExecutorState
    ) -> Dict[str, Any]:
        logger.info("Node: update_query_from_llm_suggestion_node")
        updates: Dict[str, Any] = {}
        # The LLM's response (AIMessage) should be the last message if tools_condition routed here.
        last_message = state["messages"][-1] if state.get("messages") else None

        if not isinstance(last_message, AIMessage):
            logger.warning(
                "update_query_from_llm_suggestion_node: Last message is not an AIMessage. Cannot extract SQL suggestion."
            )
            # This implies LLM failed or tools_condition had an unexpected path.
            # Keep existing error and query. Debug loop might retry or fail.
            return updates  # No changes if no valid AIMessage

        llm_content = last_message.content  # AIMessage.content

        # Handle cases where content might be list (e.g. from some models)
        if isinstance(llm_content, list):
            llm_content = "\\n".join(
                str(c) for c in llm_content
            )  # Join if it's a list of content blocks

        if not isinstance(llm_content, str):
            logger.warning(
                f"update_query_from_llm_suggestion_node: AIMessage content is not a string or list of strings (type: {type(llm_content)}). Cannot parse SQL."
            )
            return updates  # No changes if content is not parseable

        # Check for explicit error messages from the LLM itself (e.g., "CriticalError: LLM not available")
        if "CriticalError:" in llm_content or "Error during LLM call" in llm_content:
            logger.error(f"LLM reported an internal error: {llm_content}")
            # updates["execution_error"] = llm_content # Propagate LLM's critical error
            # Don't change the query. Let the debug loop handle max attempts.
            return updates

        # Attempt to parse SQL from a markdown block ```sql ... ```
        sql_block_match = re.search(
            r"```(?:sql)?\\n(.*?)(?:\\n```|```$)",
            llm_content,
            re.DOTALL | re.IGNORECASE,  # Made sql optional in tag
        )
        corrected_sql = None
        if sql_block_match:
            corrected_sql = sql_block_match.group(1).strip()
            if corrected_sql:
                logger.info(
                    f"LLM suggested corrected SQL query (from markdown block): {corrected_sql[:300]}..."
                )
            else:
                logger.warning("LLM provided an empty SQL block.")
                corrected_sql = None  # Explicitly mark as None if block is empty
        else:
            # If no markdown, consider if the whole content is SQL.
            # Avoid taking instructions or refusals as SQL.
            potential_sql = llm_content.strip()
            is_likely_sql = any(
                potential_sql.lower().startswith(kw)
                for kw in [
                    "select",
                    "with",
                    "insert",
                    "update",
                    "delete",
                    "create",
                    "--",
                ]
            )
            is_refusal = any(
                ref_text in potential_sql.lower()
                for ref_text in ["unable to", "cannot fix", "i cannot", "sorry"]
            )

            if (
                is_likely_sql and not is_refusal and len(potential_sql) > 10
            ):  # Arbitrary length check
                logger.info(
                    f"LLM provided content directly as SQL query (no markdown): {potential_sql[:300]}..."
                )
                corrected_sql = potential_sql
            else:
                logger.warning(
                    "LLM response did not contain a parsable SQL block or direct SQL query. Original query may be retried or workflow may fail."
                )

        if corrected_sql:
            updates["current_sql_query"] = corrected_sql
            updates["execution_error"] = (
                None  # Clear previous error before retrying new query
            )
            # Clear debug context for the new attempt with corrected query
            updates["described_tables_info"] = {}
            updates["listed_column_values"] = {}
            # Optionally, clear messages related to the previous failed attempt's debug cycle.
            # For now, `add_messages` accumulates. A new SystemPrompt will be generated on next `invoke_debug_llm`.
            logger.info(
                "SQL query updated from LLM suggestion. Ready for re-execution."
            )
        else:
            logger.warning(
                "No valid SQL correction found in LLM response. No query update."
            )
            # If no correction, the existing query and error will persist.
            # The debug loop will either retry (if attempts left) or fail.

        return updates

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

        final_message_to_user = f"Sorry, <@{user_id}>, I encountered an issue trying to execute the SQL query. "
        if (
            attempts > 0
        ):  # If debug_attempts > 0, it means at least one attempt was made.
            final_message_to_user += (
                f"After {attempts} attempt(s) to fix it, the problem remains. "
            )
        final_message_to_user += (
            f"The last error was: `{error_from_state}`"  # Use backticks for error
        )

        posted_ts = None
        if self.slack_client:
            try:
                # Call the post_message_tool_func
                tool_result = await self.post_message_tool_func.ainvoke(
                    {
                        "message_text": final_message_to_user,
                        "state": state,
                    }
                )

                if tool_result.get("success"):
                    posted_ts = tool_result.get("ts")
                    logger.info(
                        f"Successfully posted failure message to Slack. TS: {posted_ts}"
                    )
                else:
                    tool_error = tool_result.get(
                        "error", "Unknown error from post_message_to_thread"
                    )
                    logger.error(
                        f"Failed to post failure message to Slack: {tool_error}"
                    )
            except Exception as e:  # Catch error from ainvoke itself
                logger.error(
                    f"Exception calling post_message_tool_func in post_failure_to_slack_node: {e}",
                    exc_info=self.verbose,
                )
        else:
            logger.error(
                "Slack client not available. Cannot post failure message to user."
            )

        # The workflow is considered a failure regarding the SQL query.
        return {
            "is_success": False,
            "final_status_message": final_message_to_user,
            "posted_response_ts": posted_ts,
            # execution_error (original SQL error) is already in state and should persist.
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
    def should_prepare_csv_or_debug(self, state: QueryExecutorState) -> str:
        if state.get("execution_error"):  # If there's an error after execute_sql_node
            logger.info(
                "Conditional: Execution error found after SQL execution, routing to attempt_debug_or_fail."
            )
            return "attempt_debug_or_fail"
        # If no error, and execution_result_rows might be present (or empty if query returned nothing)
        logger.info("Conditional: No SQL execution error, routing to prepare_csv.")
        return "prepare_csv"

    def should_invoke_debug_llm_or_post_failure(self, state: QueryExecutorState) -> str:
        # debug_attempts is incremented in attempt_debug_or_fail_node *before* this condition is checked.
        if state.get("debug_attempts", 0) >= self.max_debug_loops:
            logger.info(
                f"Conditional: Max debug attempts ({state.get('debug_attempts',0)}/{self.max_debug_loops}) reached, routing to post_failure_to_slack."
            )
            return "post_failure_to_slack"

        # Check if the error is something non-recoverable by LLM (e.g. creds issue from _execute_sql_query)
        # This check could be more sophisticated.
        error_msg = state.get("execution_error", "")
        if (
            "Snowflake credentials not available" in error_msg
            or "Unsupported warehouse type" in error_msg
        ):
            logger.warning(
                f"Conditional: Non-recoverable configuration error ('{error_msg}'), routing to post_failure_to_slack without LLM debug."
            )
            return "post_failure_to_slack"

        logger.info(
            f"Conditional: Debug attempts ({state.get('debug_attempts',0)}/{self.max_debug_loops}) not exhausted, routing to invoke_debug_llm."
        )
        return "invoke_debug_llm"

    # --- Main Workflow Runner ---
    async def run_workflow(
        self,
        channel_id: str,
        thread_ts: str,
        user_id: str,
        trigger_message_ts: str,
    ) -> Dict[str, Any]:
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
                conversation_id="",
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

            return {  # Return a structured error response
                "error": str(e),
                "is_success": False,
                "final_status_message": final_status_message,
                "conversation_id": conversation_id,
            }

        conversation_id = (
            f"query-executor-{channel_id}-{thread_ts}-{trigger_message_ts}"
        )
        config = {"configurable": {"thread_id": conversation_id}}

        initial_state = QueryExecutorState(
            channel_id=channel_id,
            thread_ts=thread_ts,  # This is the parent thread for replies
            user_id=user_id,
            original_trigger_message_ts=trigger_message_ts,  # This is the specific message with the .sql or link
            fetched_sql_query=None,
            current_sql_query=None,
            execution_result_rows=None,
            execution_error=None,
            csv_content=None,
            debug_attempts=0,
            max_debug_attempts=self.max_debug_loops,
            described_tables_info={},
            listed_column_values={},
            messages=[],  # For the debug LLM mainly
            final_status_message=None,
            is_success=None,
            posted_response_ts=None,
            posted_file_ts=None,
            conversation_id=conversation_id,
        )

        final_graph_output_map = {}  # To store the output of the graph execution
        try:
            logger.info(f"Invoking QueryExecutorWorkflow graph for {conversation_id}")
            # Use astream to get events and the final state.
            # The final result of astream is the full state object if the graph ends.
            async for event in self.graph_app.astream(
                initial_state, config=config, stream_mode="values"
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
                f"QueryExecutorWorkflow graph finished for {conversation_id}. Final state collected."
            )

        except Exception as e:
            logger.error(
                f"Critical error running QueryExecutorWorkflow graph for {conversation_id}: {e}",
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
                conversation_id=conversation_id,
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
                "conversation_id": conversation_id,
            }

        # Extract results from the final graph state
        return {
            "is_success": final_graph_output_map.get("is_success"),
            "final_status_message": final_graph_output_map.get("final_status_message"),
            "posted_response_ts": final_graph_output_map.get("posted_response_ts"),
            "posted_file_ts": final_graph_output_map.get("posted_file_ts"),
            "conversation_id": conversation_id,
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
