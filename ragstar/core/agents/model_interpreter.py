"""Agent for interpreting dbt models."""

import logging
import re
import yaml
import uuid
from typing import Dict, Any, Optional, List, Set, TypedDict, Sequence
from typing_extensions import Annotated

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field, RootModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from rich.markdown import Markdown

from ragstar.core.llm.client import LLMClient
from ragstar.core.llm.prompts import MODEL_INTERPRETATION_PROMPT
from ragstar.storage.model_storage import ModelStorage
from ragstar.storage.model_embedding_storage import ModelEmbeddingStorage
from ragstar.core.models import ModelTable  # For saving
from ragstar.utils.cli_utils import get_config_value

logger = logging.getLogger(__name__)


# --- Tool Schemas ---
class UpstreamModelInput(BaseModel):
    model_names: List[str] = Field(
        description="List of upstream dbt model names to fetch raw SQL for."
    )

# Define the structure for the documentation dictionary
class ColumnDocumentation(BaseModel):
    name: str = Field(description="The name of the column.")
    description: str = Field(description="The description of the column.")

class ModelDocumentation(BaseModel):
    name: str = Field(description="The name of the dbt model.")
    description: str = Field(
        description="A comprehensive description synthesizing insights from the target and ALL analyzed upstream SQL."
    )
    columns: List[ColumnDocumentation] = Field(
        description="A detailed list of all columns produced by the final SELECT statement (including aliases) with descriptions."
    )

class FinishInterpretationInput(BaseModel):
    documentation: ModelDocumentation = Field(
        description="Final documentation for the interpreted model as a structured object, including name, description, and columns."
    )


# --- Refined State ---
class InterpretationState(TypedDict):
    """Represents the state of the model interpretation workflow."""

    # Input values
    model_name: str
    raw_sql: str  # Store raw SQL of the target model

    # Dynamic values updated during the workflow
    messages: Annotated[List[BaseMessage], add_messages]
    fetched_sql_map: Dict[
        str, Optional[str]
    ]  # Stores raw SQL for fetched models (name -> SQL or None/Error)
    final_documentation: Optional[Dict[str, Any]] # Stores the documentation dict from finish_interpretation
    is_finished: bool  # Flag set by update_state_node after finish_interpretation runs


class ModelInterpreter:
    """Agent specialized in interpreting dbt models and saving the interpretation."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_storage: ModelStorage,
        vector_store: ModelEmbeddingStorage,
        verbose: bool = False,
        console: Optional["Console"] = None,
        temperature: float = 0.0,
        openai_api_key: Optional[str] = None,
        memory: Optional[PostgresSaver] = None,
    ):
        self.llm = llm_client
        self.model_storage = model_storage
        self.vector_store = vector_store
        self.verbose = verbose
        from rich.console import Console as RichConsole

        self.console = console or RichConsole()
        self.temperature = temperature

        # Setup Postgres connection for memory
        pg_conn_string = get_config_value("POSTGRES_URI")

        # Create connection pool with proper parameters
        connection_kwargs = {
            "autocommit": True,  # Critical for CREATE INDEX CONCURRENTLY
            "prepare_threshold": 0,
        }

        pool = ConnectionPool(
            conninfo=pg_conn_string,
            kwargs=connection_kwargs,
            max_size=20,
            min_size=5,
        )

        # Initialize PostgresSaver with the pool
        self.memory = memory or PostgresSaver(conn=pool)

        # Call setup() on the saver instance
        self.memory.setup()

        self._define_tools()
        self.graph_app = self._build_graph()

    def _define_tools(self):
        @tool(args_schema=UpstreamModelInput)
        def get_models_raw_sql(model_names: List[str]) -> Dict[str, Optional[str]]:
            """Fetches the raw SQL code for a list of specified dbt models.
            Use this tool to retrieve the source code of upstream models referenced via `ref()`
            in SQL code you are analyzing. This helps understand data lineage.
            Provide a list of all model names you need SQL for in a single call.
            Returns a dictionary mapping each requested model name to its raw SQL string,
            or an error message string if the model or its SQL is not found.
            """
            results: Dict[str, Optional[str]] = {}
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: get_models_raw_sql(model_names={model_names})[/bold magenta]"
                )
            logger.info(f"Executing tool: get_models_raw_sql for models: {model_names}")

            for name in model_names:
                model = self.model_storage.get_model(name)
                if not model:
                    logger.warning(f"Model '{name}' not found in storage.")
                    results[name] = f"Error: Model '{name}' not found."
                    if self.verbose:
                        self.console.print(
                            f"[yellow] -> Tool Warning: Model '{name}' not found.[/yellow]"
                        )
                elif not model.raw_sql:
                    logger.warning(f"Model '{name}' found but has no raw SQL.")
                    results[name] = f"Error: Raw SQL not available for model '{name}'."
                    if self.verbose:
                        self.console.print(
                            f"[yellow] -> Tool Warning: Model '{name}' has no raw SQL.[/yellow]"
                        )
                else:
                    # Model found and has raw SQL
                    results[name] = model.raw_sql

                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Fetched raw SQL for model '{name}' (Length: {len(model.raw_sql)}).[/dim]"
                        )

            # Log only the keys (model names) instead of the full SQL content at INFO level
            result_summary = {name: (f"Length: {len(sql)}" if isinstance(sql, str) and not sql.startswith("Error:") else ("Error" if isinstance(sql, str) else type(sql).__name__)) for name, sql in results.items()}
            logger.info(f"Tool get_models_raw_sql finished. Results summary: {result_summary}")
            return results

        @tool(args_schema=FinishInterpretationInput)
        def finish_interpretation(documentation: ModelDocumentation) -> str:
            """Completes the interpretation process with the final model documentation object.
            Use this tool ONLY when:
            1. You have fully analyzed the target model's SQL.
            2. You have fetched and analyzed the raw SQL for ALL upstream models necessary
               to understand the target model's logic and data lineage completely.
            3. The provided documentation object includes the model's name, a comprehensive description synthesizing insights
               from the SQL analysis (target and upstream), and a detailed list of all columns
               produced by the final SELECT statement (including aliases) with descriptions.
            The input MUST be a valid JSON object matching the required structure.
            Calling this tool signals the end of the interpretation workflow.
            """
            # Convert Pydantic model to dict for logging/preview if needed
            documentation_dict = documentation.dict()

            if self.verbose:
                # Create a concise preview for logging
                preview_dict = {
                    "name": documentation_dict.get("name", "N/A"),
                    "description_preview": documentation_dict.get("description", "")[:100] + "...",
                    "num_columns": len(documentation_dict.get("columns", [])),
                }
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: finish_interpretation(documentation={preview_dict}) [/bold magenta]"
                )
            logger.info(f"Executing tool: finish_interpretation for model {documentation_dict.get('name')}")

            # Basic validation is now handled by Pydantic during tool call parsing
            # We trust the input `documentation` is a valid ModelDocumentation object here

            return (
                "Final interpretation documentation received and workflow completion signaled."
            )

        self._tools = [get_models_raw_sql, finish_interpretation]
        self._tool_executor = ToolNode(self._tools, handle_tool_errors=True)

    def _build_graph(self):
        """Builds the LangGraph workflow."""
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

        return workflow.compile(checkpointer=self.memory)

    def _get_agent_llm(self):
        """Returns the LLM client instance, binding tools if possible."""
        chat_client_instance = self.llm.chat_client
        if hasattr(chat_client_instance, "bind_tools"):
            return chat_client_instance.bind_tools(self._tools)
        else:
            logger.warning(
                "LLM chat client does not have 'bind_tools'. Tool calling might not work as expected."
            )
            return chat_client_instance

    def agent_node(self, state: InterpretationState) -> Dict[str, Any]:
        """Agent node that decides the next action based on the current state."""
        logger.info("Entering agent node.")
        if self.verbose:
            self.console.print(f"\n[bold green]🚀 Agent Node[/bold green]")
            self.console.print(f"[dim]  Target Model: {state['model_name']}[/dim]")
            # This log might be less accurate now as we rely on the agent checking history
            # fetched_sql_keys = list(state.get("fetched_sql_map", {}).keys())
            # self.console.print(f"[dim]  SQL fetched for (state perspective): {fetched_sql_keys}[/dim]")

        messages = state.get("messages", [])
        agent_llm = self._get_agent_llm()

        messages_to_send = list(messages) # Start with existing messages from state

        # === Existing Logging ===
        if self.verbose:
            self.console.print("[dim]Message structure *before* sending to LLM:[/dim]")
            for i, msg in enumerate(messages_to_send):
                msg_type = type(msg).__name__
                content_preview = str(getattr(msg, 'content', ''))[:70].replace("\n", " ") + "..." if len(str(getattr(msg, 'content', ''))) > 70 else str(getattr(msg, 'content', ''))
                tool_calls_info = ""
                tool_id_info = ""
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                     tool_calls_info = f" [bold](with {len(msg.tool_calls)} tool_calls)[/bold]"
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                     tool_id_info = f" [bold](for tool_call_id: {msg.tool_call_id})[/bold]"
                self.console.print(
                    f"  [{i}] {msg_type}{tool_calls_info}{tool_id_info}: {content_preview}"
                )

        # === END Logging ===

        response = agent_llm.invoke(messages_to_send)

        # === Existing Logging ===
        if self.verbose:
            self.console.print(f"[dim]  LLM Response received.[/dim]")
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_names = [
                    (
                        tc.get("name")
                        if isinstance(tc, dict)
                        else getattr(tc, "name", "unknown")
                    )
                    for tc in response.tool_calls
                ]
                self.console.print(f"[dim]  LLM requested tools: {tool_names}[/dim]")
        # === END Logging ===

        # Return the *single* new response message to be added to the state by add_messages
        return {"messages": [response]}

    def update_state_node(self, state: InterpretationState) -> Dict[str, Any]:
        """Processes the most recent tool messages and updates the state."""
        logger.info("Entering state update node.")
        if self.verbose:
            self.console.print(f"\n[bold cyan]🔄 Update State Node[/bold cyan]")

        updates: Dict[str, Any] = {}
        finish_called = False
        fetched_sql_updates = state.get("fetched_sql_map", {}).copy()

        # Find the most recent AI message with tool calls
        # Ensure we only process the *latest* set of tool calls and results
        last_message = state["messages"][-1]
        if not isinstance(last_message, ToolMessage):
             # If the last message isn't a ToolMessage, look for the preceding AIMessage
             last_ai_message_index = -1
             for i in range(len(state["messages"]) - 1, -1, -1):
                 if isinstance(state["messages"][i], AIMessage):
                     last_ai_message_index = i
                     break
             if last_ai_message_index == -1 or not getattr(state["messages"][last_ai_message_index], 'tool_calls', None):
                  logger.debug("No recent AIMessage with tool calls found to process.")
                  # Loop back if the agent just sent a text response without intending to finish
                  return {"is_finished": False}

             latest_ai_msg = state["messages"][last_ai_message_index]
             # Find tool messages corresponding to *this* AI message's tool calls
             latest_tool_call_ids = {
                 tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                 for tc in latest_ai_msg.tool_calls
             }
             latest_tool_messages = [
                 msg for msg in state["messages"][last_ai_message_index + 1:] # Look *after* the AI message
                 if isinstance(msg, ToolMessage) and getattr(msg, 'tool_call_id', None) in latest_tool_call_ids
             ]
        else:
             # This case should be less common if the graph adds ToolMessages correctly,
             # but handle it defensively. Assume the last ToolMessage is the one to process.
             latest_tool_messages = [last_message]
             # Try to find the AIMessage that invoked it
             latest_ai_msg = None
             for i in range(len(state["messages"]) - 2, -1, -1):
                 msg = state["messages"][i]
                 if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                      call_ids = {tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None) for tc in msg.tool_calls}
                      if last_message.tool_call_id in call_ids:
                           latest_ai_msg = msg
                           break
             if not latest_ai_msg:
                  logger.warning("Could not find invoking AIMessage for the last ToolMessage.")
                  return {"is_finished": False} # Cannot process finish_interpretation without args

        if self.verbose:
            self.console.print(
                f"[dim]  Processing {len(latest_tool_messages)} tool result(s) from latest turn...[/dim]"
            )

        for tool_msg in latest_tool_messages:
            tool_name = getattr(tool_msg, "name", "unknown_tool")
            tool_content = tool_msg.content
            tool_call_id = getattr(tool_msg, "tool_call_id", None)

            if self.verbose:
                # Modify how tool content is summarized to avoid large dumps
                content_summary = ""
                parsed_content = None
                # Try parsing the content as JSON first, as it might be a stringified dict
                if isinstance(tool_content, str):
                    try:
                        import json
                        parsed_content = json.loads(tool_content)
                    except json.JSONDecodeError:
                        # If it's not JSON, treat it as a plain string
                        parsed_content = tool_content
                else:
                     # If it's not a string, use it directly (e.g., if it's already a dict)
                     parsed_content = tool_content

                if isinstance(parsed_content, dict):
                    # For dicts (like get_models_raw_sql result), show keys only
                    content_summary = f"Result keys: {list(parsed_content.keys())}"
                elif isinstance(parsed_content, str):
                    # For strings, show a truncated preview
                    content_summary = f'Result (string, length {len(parsed_content)}): "{parsed_content[:80].replace("\n", " ")}..."'
                else:
                    # Fallback for other types
                    content_summary = f"Result type: {type(parsed_content)}"

                self.console.print(
                    f"[dim]  - Tool: '{tool_name}', Result Summary: {content_summary}[/dim]"
                )

            if tool_name == "get_models_raw_sql":
                try:
                    # ToolNode might return the dict directly or as a JSON string
                    sql_results = {}
                    if isinstance(tool_content, dict):
                        sql_results = tool_content
                    elif isinstance(tool_content, str):
                         try:
                              import json
                              sql_results = json.loads(tool_content)
                              if not isinstance(sql_results, dict):
                                   raise ValueError("Parsed JSON is not a dict")
                         except (json.JSONDecodeError, ValueError) as json_e:
                              logger.warning(f"Failed to parse get_models_raw_sql result string as dict: {json_e}. Content: {tool_content[:200]}")
                              sql_results = {"error": f"Failed to parse tool result: {tool_content[:100]}"} # Store an error instead

                    if isinstance(sql_results, dict):
                         fetched_sql_updates.update(sql_results)
                         logger.info(f"Updated state with fetched SQL for models: {list(sql_results.keys())}")
                         if self.verbose:
                            count = len(sql_results)
                            errors = sum(
                                1
                                for v in sql_results.values()
                                if isinstance(v, str) and v.startswith("Error:")
                            )
                            self.console.print(
                                f"[dim]    -> Stored/updated SQL for {count} models ({errors} errors).[/dim]"
                            )
                    else:
                         logger.warning(f"Tool '{tool_name}' result was not processed into a dictionary. Type: {type(tool_content)}")
                         if self.verbose:
                             self.console.print(f"[yellow dim]    -> Result was not a dict. Not stored. Type: {type(tool_content)}[/yellow dim]")

                except Exception as e:
                    logger.error(
                        f"Failed to process '{tool_name}' result: {e}",
                        exc_info=self.verbose,
                    )
                    if self.verbose:
                        self.console.print(
                            f"[red]    -> Error processing tool result: {e}[/red]"
                        )

            elif tool_name == "finish_interpretation":
                finish_called = True
                documentation_dict = None
                # Find the documentation dict argument from the invoking AIMessage's tool call
                if latest_ai_msg and getattr(latest_ai_msg, 'tool_calls', None):
                    for tc in latest_ai_msg.tool_calls:
                        call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        call_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)

                        if call_name == "finish_interpretation" and call_id == tool_call_id:
                            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                            if isinstance(args, dict):
                                # The argument should be named 'documentation' and already be a dict
                                documentation_dict = args.get("documentation")
                            break # Found the matching call

                if isinstance(documentation_dict, dict):
                    updates["final_documentation"] = documentation_dict
                    logger.info(f"Stored final documentation dictionary from finish_interpretation.")
                    if self.verbose:
                        # Log keys or structure instead of full YAML
                        doc_preview = {k: type(v).__name__ for k, v in documentation_dict.items()}
                        self.console.print(
                            f"[green dim]    -> Stored final documentation object. Structure: {doc_preview}[/green dim]"
                        )
                else:
                    logger.warning("Couldn't extract documentation dictionary from finish_interpretation call args.")
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim]    -> Couldn't retrieve documentation dict argument from tool call.[/yellow dim]"
                        )

                updates["is_finished"] = True
                if self.verbose:
                    self.console.print("[green bold]    -> Signaling workflow completion.[/green bold]")

        # Update the state with the accumulated SQL map if changed
        if fetched_sql_updates != state.get("fetched_sql_map", {}):
            updates["fetched_sql_map"] = fetched_sql_updates

        # Ensure 'is_finished' is set correctly
        if "is_finished" not in updates:
            updates["is_finished"] = finish_called

        if self.verbose and not updates:
            self.console.print("[dim]  No state changes detected in this update step.[/dim]")

        return updates

    def run_interpretation_workflow(self, model_name: str) -> Dict[str, Any]:
        """Runs the interpretation workflow for a given model name."""
        logger.info(f"Starting interpretation workflow for model: {model_name}")
        self.console.print(
            f"\n[bold underline]🚀 Starting Interpretation: {model_name}[/bold underline]"
        )

        model = self.model_storage.get_model(model_name)
        if not model:
            logger.error(f"Model '{model_name}' not found. Cannot start workflow.")
            self.console.print(
                f"[bold red]Error: Model '{model_name}' not found in storage.[/bold red]"
            )
            return {"success": False, "error": f"Model {model_name} not found."}
        if not model.raw_sql:
            logger.error(f"Model '{model_name}' has no raw SQL. Cannot interpret.")
            self.console.print(
                f"[bold red]Error: Model '{model_name}' has no SQL content.[/bold red]"
            )
            return {"success": False, "error": f"Model {model_name} has no SQL."}

        raw_sql = model.raw_sql
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        if self.verbose:
            self.console.print(f"[dim]Workflow Thread ID: {thread_id}[/dim]")

        # --- Create Initial Messages Here ---
        system_prompt = """You are an expert dbt model interpreter. Your task is to analyze the SQL code for a target dbt model, recursively explore its upstream dependencies by fetching their raw SQL, and generate CONCISE documentation (as a structured object) suitable for dbt YAML files for the original target model.

Process:
1. **Analyze SQL:** Carefully analyze the provided SQL (from the initial Human message or subsequent Tool results for `get_models_raw_sql`) to understand its logic and identify ALL models referenced via `ref()`.
2. **Check History:** Look back through the conversation history. Identify all models whose SQL has been successfully returned in `ToolMessage` results from the `get_models_raw_sql` tool. Also, remember the SQL for the target model (`{model_name}`) was provided initially.
3. **Identify Needed SQL:** Determine which models referenced in the *most recently analyzed SQL* are NOT among those whose SQL you've already seen (from step 2).
4. **Fetch Upstream SQL:** If there are unfetched referenced models needed for the analysis, use the `get_models_raw_sql` tool ONCE with a list of ALL such model names.
5. **Recursive Analysis:** Analyze the newly fetched SQL (from the latest `ToolMessage`), identify further `ref()` calls, and repeat steps 2-4 until you have analyzed all necessary upstream SQL to fully understand the target model's data lineage and column derivations.
6. **Synthesize Documentation:** Once your analysis is complete (meaning you've seen the SQL for the target and all recursive dependencies mentioned via `ref`), create the final documentation object for the *original target model* (`{model_name}`). Include:
   - Accurate model name.
   - A **brief, 1-2 sentence description** summarizing the model's purpose, suitable for a dbt `description:` field.
   - A complete list of all output columns from the target model's final SELECT statement, with **concise descriptions** for each column, suitable for dbt column documentation.
7. **Finish:** Call the `finish_interpretation` tool with the complete documentation object.

**IMPORTANT:**
- Generate **concise** descriptions. Avoid long paragraphs.
- Use `get_models_raw_sql` only when needed, requesting all necessary SQL in a single batch per turn based on your analysis of the *latest* SQL and conversation history.
- Do not re-request SQL for models already provided in previous `ToolMessage` results.
- Only call `finish_interpretation` when you are certain you have analyzed the SQL for the target model and *all* its upstream dependencies referenced directly or indirectly via `ref()`.
- Ensure the final output to `finish_interpretation` is a structured object with 'name', 'description', and 'columns' (each column having 'name' and 'description')."""

        initial_prompt = f"""Please interpret the dbt model '{model_name}'. Its raw SQL is:

```sql
{raw_sql}
```

Follow the interpretation process outlined in the system message. Start by analyzing this initial SQL."""

        initial_system_message = SystemMessage(
            content=system_prompt.format(model_name=model_name)
        )
        initial_human_message = HumanMessage(content=initial_prompt)
        # --- End Initial Message Creation ---

        # Define the initial state dictionary, including messages directly
        initial_state = InterpretationState(
            model_name=model_name,
            raw_sql=raw_sql,
            messages=[initial_system_message, initial_human_message], # Include messages directly
            fetched_sql_map={},
            final_documentation=None,
            is_finished=False,
        )

        try:
            final_state_result = self.graph_app.invoke(initial_state, config=config)
            # Handle potential variations in how the final state is returned
            if isinstance(final_state_result, dict):
                 final_state = final_state_result
            elif hasattr(final_state_result, 'values'): # Check if it's a StateSnapshot
                 final_state = final_state_result.values
            else:
                  logger.error(f"Unexpected final state format: {type(final_state_result)}. Cannot extract results.")
                  final_state = {} # Fallback to empty dict


            final_documentation = final_state.get("final_documentation") # Use new key
            success = bool(final_documentation)
            fetched_sql_map = final_state.get("fetched_sql_map", {})

            logger.info(
                f"Interpretation workflow for '{model_name}' finished. Success: {success}. Fetched SQL for {len(fetched_sql_map)} upstream models."
            )

            # Return the result dictionary containing the documentation object
            return {
                "model_name": model_name,
                "documentation": final_documentation, # Use new key
                "fetched_sql_map": fetched_sql_map,
                "success": success,
                # Optionally pass back messages or other debug info if needed by CLI
                "messages": final_state.get("messages", []) if not success else [] # Only include messages on failure?
            }

        except Exception as e:
            # ... (keep existing error handling) ...
            logger.error(
                f"Error during interpretation workflow for {model_name}: {e}",
                exc_info=True,
            )
            self.console.print(
                f"[bold red]\n💥 Error during interpretation workflow for {model_name}:[/bold red]\n{e}"
            )
            # Attempt to get state on error for debugging
            error_state = {}
            try:
                # Use get_state method which returns a StateSnapshot
                state_snapshot = self.graph_app.get_state(config)
                error_state = state_snapshot.values if state_snapshot else {}
                logger.info(f"State at time of error: {error_state}")
            except Exception as state_err:
                logger.error(f"Could not retrieve state after error: {state_err}")

            return {
                "model_name": model_name,
                "documentation": None,
                "fetched_sql_map": error_state.get("fetched_sql_map", {}),
                "success": False,
                "error": str(e),
                "messages": error_state.get("messages", []) # Include messages on error
            }

    def save_interpreted_documentation(
        self, model_name: str, documentation: Optional[Dict[str, Any]], embed: bool = False
    ) -> Dict[str, Any]:
        """Save interpreted documentation (provided as a dictionary) for a model."""
        if not documentation:
            logger.warning(
                f"Attempted to save empty documentation for {model_name}. Skipping save."
            )
            return {
                "success": False,
                "error": "No documentation dictionary provided to save.",
                "model_name": model_name,
            }

        logger.info(f"Saving interpreted documentation for model {model_name}")
        session = self.model_storage.Session()
        try:
            # Use the documentation dictionary directly
            model_data = documentation # Rename for clarity within this scope

            # Basic validation of the dictionary structure
            if not isinstance(model_data, dict):
                 raise ValueError("Provided documentation is not a dictionary.")

            if (
                "name" not in model_data
                or "description" not in model_data
                or "columns" not in model_data
            ):
                logger.warning(
                    f"Provided documentation dict for {model_name} might be missing name, description, or columns."
                )
                # Allow saving even if keys are missing, but log warning

            dict_model_name = model_data.get("name")
            if dict_model_name != model_name:
                logger.warning(
                    f"Model name mismatch in documentation dict ('{dict_model_name}') and function call ('{model_name}'). Using '{model_name}' for database lookup."
                )


            # Extract interpretation data
            interpreted_description = model_data.get("description", "") # Default to empty string if missing
            interpreted_columns = {}

            # Process columns, expecting a list of dictionaries
            if "columns" in model_data and isinstance(model_data["columns"], list):
                for col_data in model_data["columns"]:
                    # Check if column entry is a dict with 'name'
                    if isinstance(col_data, dict) and "name" in col_data:
                        col_name = col_data["name"]
                        # Get description, default to empty string
                        col_desc = col_data.get("description", "")
                        interpreted_columns[col_name] = col_desc
                    else:
                        logger.warning(
                            f"Skipping invalid column entry in documentation dict for {model_name}: {col_data}"
                        )
            elif "columns" in model_data:
                 logger.warning(f"'columns' field in documentation for {model_name} is not a list: {type(model_data['columns'])}")


            # Fetch the existing ModelTable record
            model_record = (
                session.query(ModelTable)
                .filter(ModelTable.name == model_name)
                .one_or_none()
            )

            if not model_record:
                logger.error(f"Model {model_name} not found in database for saving.")
                return {
                    "success": False,
                    "error": f"Model {model_name} not found in database",
                    "model_name": model_name,
                }

            # Update the record directly
            model_record.interpreted_description = interpreted_description
            model_record.interpreted_columns = interpreted_columns # Assumes JSON field type in DB

            session.add(model_record)
            session.commit()
            logger.info(f"Successfully saved interpretation for model {model_name}")

            # Optional: Re-embed the model
            if embed:
                logger.info(
                    f"Re-embedding model {model_name} after saving interpretation"
                )
                session.refresh(model_record) # Ensure we have the latest data

                try:
                    # Construct text representation for embedding using updated record
                    text_parts = [
                        f"Model: {model_record.name}",
                        f"Description: {model_record.interpreted_description or ''}", # Use new description
                        "Columns:",
                    ]
                    # Use new columns
                    for col_name, col_desc in (
                        model_record.interpreted_columns or {}
                    ).items():
                        text_parts.append(f"  - {col_name}: {col_desc or ''}")
                    if model_record.raw_sql: # Still include raw SQL if available in DB
                        text_parts.append("SQL:")
                        text_parts.append(model_record.raw_sql)

                    model_text_for_embedding = "\n".join(text_parts)

                    if model_text_for_embedding.strip(): # Check if there's actual content
                        self.vector_store.store_model_embedding(
                            model_name=model_record.name,
                            model_text=model_text_for_embedding,
                        )
                        logger.info(f"Successfully re-embedded model {model_name}")
                    else:
                        logger.warning(
                            f"Could not generate non-empty text representation for {model_name} from DB record for embedding."
                        )

                except Exception as embed_err:
                    logger.error(
                        f"Error during re-embedding of {model_name}: {embed_err}",
                        exc_info=True,
                    )
                    # Report success for saving, but warn about embedding failure
                    return {
                        "success": True,
                        "warning": f"Interpretation saved, but failed to re-embed model {model_name}: {embed_err}",
                        "model_name": model_name,
                    }


            return {"success": True, "model_name": model_name}

        except ValueError as ve: # Catch specific validation errors
             logger.error(f"Validation error saving documentation for {model_name}: {ve}")
             session.rollback()
             return {
                 "success": False,
                 "error": f"Invalid documentation format: {ve}",
                 "model_name": model_name,
             }
        except Exception as e:
            logger.error(
                f"Error saving interpreted documentation for {model_name}: {e}",
                exc_info=True,
            )
            session.rollback()
            return {
                "success": False,
                "error": f"Internal error saving interpretation: {e}",
                "model_name": model_name,
            }
        finally:
            session.close()
