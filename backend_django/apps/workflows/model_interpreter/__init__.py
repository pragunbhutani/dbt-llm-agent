"""Agent for interpreting dbt models within the Django app."""

import logging
import re
import yaml  # Keep yaml for potential use, but maybe remove later
import uuid
import json  # <-- Add json import
from typing import Dict, Any, Optional, List, Set, TypedDict, Sequence
from typing_extensions import Annotated

# Langchain/LangGraph imports
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

# Removed PostgresSaver and psycopg_pool imports - no persistent memory for now

# Django imports
from apps.knowledge_base.models import Model as DjangoModel
from apps.llm_providers.services import ChatService
from django.conf import settings

# Local agent imports
from .prompts import create_system_prompt, create_initial_human_message

# Removed old ragstar imports
# from ragstar.core.llm.client import LLMClient
# from ragstar.storage.model_storage import ModelStorage
# from ragstar.storage.model_embedding_storage import ModelEmbeddingStorage
# from ragstar.core.models import ModelTable # For saving
# from ragstar.utils.cli_utils import get_config_value

logger = logging.getLogger(__name__)


# --- Tool Schemas (Keep Pydantic models for tool inputs/outputs) ---
class UpstreamModelInput(BaseModel):
    model_names: List[str] = Field(
        description="List of upstream dbt model names to fetch raw SQL for."
    )


class ColumnDocumentation(BaseModel):
    name: str = Field(description="The name of the column.")
    description: str = Field(description="The description of the column.")


class ModelDocumentation(BaseModel):
    name: str = Field(description="The name of the dbt model.")
    description: str = Field(
        description="A brief, 1-2 sentence description summarizing the model's purpose, suitable for a dbt description field."
    )
    columns: List[ColumnDocumentation] = Field(
        description="A detailed list of all columns produced by the final SELECT statement (including aliases) with concise descriptions."
    )


class FinishInterpretationInput(BaseModel):
    documentation: ModelDocumentation = Field(
        description="Final documentation for the interpreted model as a structured object, including name, description, and columns. This signals the end of the workflow."
    )


# --- Refined State (Keep as is) ---
class InterpretationState(TypedDict):
    """Represents the state of the model interpretation workflow."""

    model_name: str
    raw_sql: str
    messages: Annotated[List[BaseMessage], add_messages]
    fetched_sql_map: Dict[str, Optional[str]]
    final_documentation: Optional[Dict[str, Any]]
    is_finished: bool


# --- Model Interpreter Agent ---
class ModelInterpreterAgent:
    """Agent specialized in interpreting dbt models using LangGraph within Django."""

    def __init__(
        self,
        chat_service: ChatService,
        verbosity: int = 0,
        # Removed model_storage, vector_store, openai_api_key, memory
        # Console might be useful for debugging within command
        console: Optional[Any] = None,
    ):
        """Initializes the agent using Django's ChatService."""
        logger.debug(f"Initializing ModelInterpreterAgent with verbosity: {verbosity}")
        self.chat_service = chat_service
        self.llm_client: Optional[BaseChatModel] = self.chat_service.get_client()
        self.verbosity = verbosity
        self.console = console  # Store console if provided

        if not self.llm_client:
            # Log error or raise exception? Agent unusable without LLM.
            logger.error(
                "ModelInterpreterAgent failed to get LLM client from ChatService."
            )
            raise ValueError("LLM Client not available for ModelInterpreterAgent")

        # Define tools using instance methods bound to self
        self._define_tools()
        # Build graph without checkpointer
        self.graph_app = self._build_graph()

    # --- Tool Definitions --- #
    def _define_tools(self):
        """Defines tools available to the agent."""

        # Need to define tools as methods or ensure they have access to 'self'
        # or necessary Django context if defined outside.
        # Defining them here allows access to self.verbosity etc.

        @tool(args_schema=UpstreamModelInput)
        def get_models_raw_sql(model_names: List[str]) -> Dict[str, Optional[str]]:
            """Fetches the raw SQL code for specified dbt models using the Django ORM.
            Use this tool to retrieve the source code of upstream models referenced via `ref()`.
            Returns a dictionary mapping model name to its raw SQL string, or an error message string.
            """
            results: Dict[str, Optional[str]] = {}
            if self.verbosity >= 1 and self.console:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: get_models_raw_sql(model_names={model_names})[/bold magenta]"
                )
            logger.info(f"Executing tool: get_models_raw_sql for models: {model_names}")

            # Query Django DB for models
            found_models = DjangoModel.objects.filter(name__in=model_names).values(
                "name", "raw_sql"
            )
            found_map = {m["name"]: m["raw_sql"] for m in found_models}

            for name in model_names:
                if name not in found_map:
                    msg = f"Error: Model '{name}' not found in database."
                    results[name] = msg
                    logger.warning(f"Model '{name}' not found via Django ORM.")
                    if self.verbosity >= 1 and self.console:
                        self.console.print(f"[yellow] -> Tool Warning: {msg}[/yellow]")
                elif not found_map[name]:  # Check if raw_sql is empty or null
                    msg = f"Error: Raw SQL not available for model '{name}'."
                    results[name] = msg
                    logger.warning(f"Model '{name}' found but has no raw SQL.")
                    if self.verbosity >= 1 and self.console:
                        self.console.print(f"[yellow] -> Tool Warning: {msg}[/yellow]")
                else:
                    sql = found_map[name]
                    results[name] = sql
                    if self.verbosity >= 1 and self.console:
                        self.console.print(
                            f"[dim] -> Fetched raw SQL for model '{name}' (Length: {len(sql)}).[/dim]"
                        )

            # Log summary
            result_summary = {
                name: (
                    "Length: ..."
                    if isinstance(sql, str) and not sql.startswith("Error:")
                    else "Error"
                )
                for name, sql in results.items()
            }
            logger.info(
                f"Tool get_models_raw_sql finished. Results summary: {result_summary}"
            )
            # Added debug log before returning
            if self.verbosity >= 2:
                logger.debug(
                    f"Tool get_models_raw_sql returning: type={type(results)}, value={results}"
                )
            return results

        @tool(args_schema=FinishInterpretationInput)
        def finish_interpretation(documentation: ModelDocumentation) -> str:
            """Completes the interpretation process with the final model documentation object.
            Use ONLY when analysis of target and ALL upstream SQL is complete.
            The input MUST be a valid ModelDocumentation object.
            Calling this signals the end of the interpretation workflow.
            """
            documentation_dict = documentation.dict()
            if self.verbosity >= 1 and self.console:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: finish_interpretation (model: {documentation_dict.get('name')})[/bold magenta]"
                )
            logger.info(
                f"Executing tool: finish_interpretation for model {documentation_dict.get('name')}"
            )
            # The actual documentation is stored in the state by update_state_node
            return "Final interpretation documentation received. Workflow completion signaled."

        # Assign tools to self
        self._tools = [get_models_raw_sql, finish_interpretation]
        self._tool_executor = ToolNode(self._tools, handle_tool_errors=True)

    # --- NEW: LLM Helper --- #
    def _get_agent_llm(self) -> BaseChatModel:
        """Returns the LLM client instance, binding tools."""
        if not self.llm_client:
            # This case should ideally not be reached due to __init__ check
            logger.error("LLM client is not available in _get_agent_llm.")
            raise ValueError("LLM Client not available.")

        # Check if the client supports binding tools (common pattern)
        if hasattr(self.llm_client, "bind_tools"):
            return self.llm_client.bind_tools(self._tools)
        else:
            logger.warning(
                "LLM chat client does not support 'bind_tools'. Tool calling might be impaired."
            )
            return self.llm_client

    # --- END NEW --- #

    # --- Graph Definition --- #
    def _build_graph(self):
        """Builds the LangGraph workflow without persistence."""
        workflow = StateGraph(InterpretationState)

        workflow.add_node("agent", self.agent_node)
        workflow.add_node("execute_tools", self._tool_executor)
        workflow.add_node("update_state", self.update_state_node)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "execute_tools", END: END},
        )
        workflow.add_edge("execute_tools", "update_state")
        workflow.add_conditional_edges(
            "update_state",
            lambda state: END if state.get("is_finished") else "agent",
            {END: END, "agent": "agent"},
        )

        # Compile without checkpointer
        return workflow.compile()

    # --- Agent Node --- #
    def agent_node(self, state: InterpretationState) -> Dict[str, Any]:
        """Invokes the agent LLM to decide the next action or generate the final response."""
        if self.verbosity >= 2:
            logger.debug(f"Entering agent_node for model: {state.get('model_name')}")
        if self.verbosity >= 1 and self.console:
            self.console.print(f"\n[bold green]🚀 Agent Node[/bold green]")

        messages = state.get("messages", [])
        agent_llm = self._get_agent_llm()

        # --- NEW: Verbosity 3 Logging ---
        if self.verbosity >= 3:
            logger.debug(f"--- Agent State (Start of agent_node) ---")
            logger.debug(f"Model Name: {state.get('model_name')}")
            logger.debug(f"Raw SQL Provided: {bool(state.get('raw_sql'))}")
            logger.debug(
                f"Fetched SQL Map Keys: {list(state.get('fetched_sql_map', {}).keys())}"
            )
            logger.debug(
                f"Final Documentation Set: {bool(state.get('final_documentation'))}"
            )
            logger.debug(f"Is Finished Flag: {state.get('is_finished')}")
            logger.debug(f"--- Message History ({len(messages)} messages) ---")
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                content_preview = (
                    str(getattr(msg, "content", ""))[:200].replace("\n", " ") + "..."
                )
                tool_calls_info = ""
                tool_id_info = ""
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_info = f" (Tool Calls: {len(msg.tool_calls)})"
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    tool_id_info = f" (Tool Call ID: {msg.tool_call_id})"

                logger.debug(
                    f"  [{i}] {msg_type}{tool_calls_info}{tool_id_info}: {content_preview}"
                )
            logger.debug(f"--- End Agent State/Messages ---")
        # --- END NEW ---

        if self.verbosity >= 1 and self.console:
            # Simple console log for lower verbosity
            self.console.print(f"[dim]  Target Model: {state['model_name']}[/dim]")
            # Removed fetched_sql log as it's less reliable - agent checks history

        # Invoke the agent LLM
        response = agent_llm.invoke(messages)

        # --- NEW: Verbosity 3 Logging for Response ---
        if self.verbosity >= 3:
            logger.debug(f"--- LLM Response ---")
            response_type = type(response).__name__
            content_preview = (
                str(getattr(response, "content", ""))[:200].replace("\n", " ") + "..."
            )
            tool_calls_info = ""
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls_info = f" (Tool Calls: {[tc.get('name', 'unknown') for tc in response.tool_calls]})"  # Show tool names
            logger.debug(f"  Type: {response_type}")
            logger.debug(f"  Content Preview: {content_preview}")
            logger.debug(f"  Tool Calls Info:{tool_calls_info}")
            logger.debug(f"--- End LLM Response ---")
        # --- END NEW ---

        return {"messages": [response]}

    # --- State Update Node --- #
    def update_state_node(self, state: InterpretationState) -> Dict[str, Any]:
        """Processes tool messages and updates the state."""
        if self.verbosity >= 2:
            logger.debug(
                f"Entering update_state_node for model: {state.get('model_name')}"
            )
        if self.verbosity >= 1 and self.console:
            self.console.print(f"\n[bold cyan]🔄 Update State Node[/bold cyan]")

        updates: Dict[str, Any] = {}
        finish_called = False
        fetched_sql_updates = state.get("fetched_sql_map", {}).copy()

        # Simplified logic to find the latest tool messages corresponding to the last AI call
        last_ai_message = None
        for i in range(len(state["messages"]) - 1, -1, -1):
            if isinstance(state["messages"][i], AIMessage):
                last_ai_message = state["messages"][i]
                break

        if not last_ai_message or not getattr(last_ai_message, "tool_calls", None):
            logger.debug("No recent AIMessage with tool calls found to process.")
            return {"is_finished": False}

        latest_tool_call_ids = {tc.get("id") for tc in last_ai_message.tool_calls}
        latest_tool_messages = [
            msg
            for msg in reversed(state["messages"])  # Check recent messages
            if isinstance(msg, ToolMessage)
            and getattr(msg, "tool_call_id", None) in latest_tool_call_ids
        ]

        if self.verbosity >= 1 and self.console:
            self.console.print(
                f"[dim]  Processing {len(latest_tool_messages)} tool result(s)...[/dim]"
            )

        for tool_msg in latest_tool_messages:
            tool_name = getattr(tool_msg, "name", "unknown_tool")
            tool_content = tool_msg.content
            tool_call_id = getattr(tool_msg, "tool_call_id", None)

            # Added detailed tool message logging at verbosity >= 2
            if self.verbosity >= 2:
                logger.debug(
                    f"Processing ToolMessage in update_state: name='{tool_name}', id='{tool_call_id}', content_type={type(tool_content)}, content_value={tool_content}"
                )

            # Added detailed tool message logging at verbosity >= 3
            if self.verbosity >= 3:
                logger.debug(
                    f"Processing ToolMessage: name='{tool_name}', id='{tool_call_id}', content_type={type(tool_content)}"
                )

            # Log tool execution result summary (optional)
            if self.verbosity >= 1 and self.console:
                content_summary = f"Result type: {type(tool_content)}"
                if isinstance(tool_content, dict):
                    content_summary = f"Result keys: {list(tool_content.keys())}"
                elif isinstance(tool_content, str):
                    content_summary = (
                        f'Result (string preview): "{tool_content[:50]}..."'
                    )
                self.console.print(
                    f"[dim]  - Tool: '{tool_name}', {content_summary}[/dim]"
                )

            # Process specific tools
            if tool_name == "get_models_raw_sql":
                sql_results = None
                if isinstance(tool_content, dict):
                    # If it's already a dict, use it directly
                    sql_results = tool_content
                elif isinstance(tool_content, str):
                    # If it's a string, try parsing it as JSON
                    try:
                        parsed_content = json.loads(tool_content)
                        if isinstance(parsed_content, dict):
                            sql_results = parsed_content
                            logger.debug(
                                f"Parsed string tool content for '{tool_name}' into dict."
                            )
                        else:
                            # Parsed JSON wasn't a dict
                            logger.warning(
                                f"Tool '{tool_name}' content was a string, but parsed JSON was not a dict. Type: {type(parsed_content)}. Ignored."
                            )
                    except json.JSONDecodeError as e:
                        # String wasn't valid JSON
                        logger.warning(
                            f"Tool '{tool_name}' content was a string, but failed to parse as JSON: {e}. Content Preview: {tool_content[:200]}. Ignored."
                        )

                # Now process sql_results if it's a valid dictionary
                if isinstance(sql_results, dict):
                    fetched_sql_updates.update(sql_results)
                    logger.info(
                        f"Updated state with fetched SQL: {list(sql_results.keys())}"
                    )
                    if self.verbosity >= 1 and self.console:
                        self.console.print(
                            f"[green dim]    -> Updated fetched SQL map for {len(sql_results)} models.[/green dim]"
                        )
                else:
                    # Log if we ended up without a dict (either original type wasn't dict/str, or parsing failed)
                    if (
                        sql_results is None
                    ):  # Only log if we haven't already logged a parsing warning
                        warning_msg = (
                            f"Tool '{tool_name}' result was not processed into a dict. Ignored. "
                            f"Original Type: {type(tool_content)}, Content Preview: {str(tool_content)[:200]}"
                        )
                        logger.warning(warning_msg)
                        if self.verbosity >= 1 and self.console:
                            self.console.print(
                                f"[yellow]    -> Tool '{tool_name}' result ignored (not a dict/parsable string). Type: {type(tool_content)}[/yellow]"
                            )

            elif tool_name == "finish_interpretation":
                finish_called = True
                # Find the corresponding tool call in the last AI message to get args
                documentation_dict = None
                for tc in last_ai_message.tool_calls:
                    if tc.get("id") == tool_call_id:
                        args = tc.get("args", {})
                        if isinstance(args, dict):
                            documentation_dict = args.get("documentation")
                        break

                if isinstance(documentation_dict, dict):
                    updates["final_documentation"] = documentation_dict
                    logger.info(
                        "Stored final documentation from finish_interpretation tool call."
                    )
                    if self.verbosity >= 1 and self.console:
                        self.console.print(
                            "[green dim]    -> Stored final documentation object.[/green dim]"
                        )
                else:
                    logger.warning(
                        "Could not extract documentation dict from finish_interpretation args."
                    )

                updates["is_finished"] = True  # Signal completion
                if self.verbosity >= 1 and self.console:
                    self.console.print(
                        "[green bold]    -> Signaling workflow completion.[/green bold]"
                    )

        # Update fetched_sql_map if it changed
        if fetched_sql_updates != state.get("fetched_sql_map", {}):
            updates["fetched_sql_map"] = fetched_sql_updates

        # Ensure is_finished is set correctly (should be set if finish_called)
        if "is_finished" not in updates:
            updates["is_finished"] = finish_called

        return updates

    # --- Workflow Execution --- #
    def run_interpretation_workflow(
        self, model_name: str, raw_sql: str
    ) -> Dict[str, Any]:
        """Runs the interpretation workflow for a given model and its SQL."""
        if self.verbosity >= 2:
            logger.debug(
                f"Entering run_interpretation_workflow for model: {model_name}"
            )
        logger.info(f"Starting agent interpretation workflow for model: {model_name}")
        if self.verbosity >= 1 and self.console:
            self.console.print(
                f"\n[bold underline]🚀 Starting Agent Interpretation: {model_name}[/bold underline]"
            )

        # Use a unique ID for each run, no persistence needed unless specified later
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Create initial messages
        system_prompt_text = create_system_prompt(model_name=model_name)
        initial_human_message_text = create_initial_human_message(
            model_name=model_name, raw_sql=raw_sql
        )
        initial_messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=initial_human_message_text),
        ]

        # Define initial state
        initial_state = InterpretationState(
            model_name=model_name,
            raw_sql=raw_sql,
            messages=initial_messages,
            fetched_sql_map={},
            final_documentation=None,
            is_finished=False,
        )

        try:
            # Invoke the graph
            if self.verbosity >= 2:
                logger.debug(f"Invoking agent graph for model: {model_name}")
            final_state = self.graph_app.invoke(initial_state, config=config)

            if self.verbosity >= 2:
                logger.debug(
                    f"Agent graph finished for model: {model_name}. Final state keys: {list(final_state.keys())}"
                )

            final_documentation = final_state.get("final_documentation")
            success = bool(final_documentation)

            logger.info(
                f"Agent workflow for '{model_name}' finished. Success: {success}"
            )

            # Return dictionary expected by the management command
            return {
                "model_name": model_name,
                "documentation": final_documentation,
                "success": success,
                "error": (
                    None if success else "Agent failed to produce final documentation."
                ),
                # Optionally include messages for debugging failure cases
                "messages": final_state.get("messages", []) if not success else [],
            }

        except Exception as e:
            logger.error(
                f"Error during agent workflow for {model_name}: {e}", exc_info=True
            )
            if self.verbosity >= 1 and self.console:
                self.console.print(
                    f"[bold red]\n💥 Error during agent workflow: {e}[/bold red]"
                )
            # Return error structure
            return {
                "model_name": model_name,
                "documentation": None,
                "success": False,
                "error": str(e),
                "messages": [],  # State retrieval might be complex without checkpointer
            }

    # Removed save_interpreted_documentation method - saving is done by caller


# Example of how to potentially export (if needed, depends on Django structure)
# __all__ = ["ModelInterpreterAgent", "ModelDocumentation", "ColumnDocumentation"]
