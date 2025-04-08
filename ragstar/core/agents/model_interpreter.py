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
from pydantic import BaseModel, Field
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


class FinishInterpretationInput(BaseModel):
    yaml_doc: str = Field(
        description="Final YAML documentation for the interpreted model, including name, description, and columns."
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
    final_yaml: Optional[str]  # Stores the YAML from finish_interpretation
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
                    if model and model.raw_sql:
                        self.console.print(
                            f"[dim] -> Fetched raw SQL for model '{name}' (Length: {len(model.raw_sql)}).[/dim]"
                        )

            logger.info(f"Tool get_models_raw_sql returning: {results}")
            return results

        @tool(args_schema=FinishInterpretationInput)
        def finish_interpretation(yaml_doc: str) -> str:
            """Completes the interpretation process with the final YAML documentation.
            Use this tool ONLY when:
            1. You have fully analyzed the target model's SQL.
            2. You have fetched and analyzed the raw SQL for ALL upstream models necessary
               to understand the target model's logic and data lineage completely.
            3. The provided YAML includes the model's name, a comprehensive description synthesizing insights
               from the SQL analysis (target and upstream), and a detailed list of all columns
               produced by the final SELECT statement (including aliases) with descriptions.
            The input MUST be a single string containing the complete, valid dbt model YAML.
            Calling this tool signals the end of the interpretation workflow.
            """
            if self.verbose:
                yaml_preview = yaml_doc.strip()[:100].replace("\n", " ") + "..."
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: finish_interpretation(yaml_doc='{yaml_preview}') [/bold magenta]"
                )
            logger.info(f"Executing tool: finish_interpretation")
            try:
                # Basic validation
                parsed = yaml.safe_load(yaml_doc)
                if (
                    not isinstance(parsed, dict)
                    or not parsed.get("name")
                    or not parsed.get("columns")
                ):
                    logger.warning(
                        "YAML passed to finish_interpretation seems incomplete or invalid."
                    )
            except yaml.YAMLError as e:
                logger.error(f"Invalid YAML provided to finish_interpretation: {e}")
                return f"YAML received, but parsing failed: {e}"

            return (
                "Final interpretation YAML received and workflow completion signaled."
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
            fetched_sql_keys = list(state.get("fetched_sql_map", {}).keys())
            self.console.print(f"[dim]  SQL fetched for: {fetched_sql_keys}[/dim]")

        messages = state.get("messages", [])
        agent_llm = self._get_agent_llm()

        # Identify models whose SQL has already been fetched
        already_fetched_sql_for = set(state.get("fetched_sql_map", {}).keys())
        # Also include the target model itself, as its SQL is provided initially
        already_fetched_sql_for.add(state["model_name"])

        if not messages:  # First turn - generate initial prompt
            system_prompt = """You are an expert dbt model interpreter. Your task is to analyze the SQL code for a target dbt model, recursively explore its upstream dependencies by fetching their raw SQL, and generate comprehensive YAML documentation for the original target model.

Process:
1. **Analyze SQL:** Carefully analyze the provided SQL (either the target model's or fetched upstream SQL) to understand its logic and identify ALL models referenced via `ref()`.
2. **Identify Needed SQL:** Determine which referenced models you haven't seen the SQL for yet.
3. **Fetch Upstream SQL:** If there are unfetched referenced models, use the `get_models_raw_sql` tool ONCE with a list of ALL model names you need SQL for in this step. The tool returns only the raw SQL code or an error message for each model.
4. **Recursive Analysis:** Analyze the newly fetched SQL, identify further `ref()` calls, and repeat steps 2-3 until you have analyzed all necessary upstream SQL to fully understand the target model's data lineage and column derivations.
5. **Synthesize YAML:** Once your analysis is complete, create detailed YAML documentation for the *original target model* (`{model_name}`). Include:
   - Accurate model name.
   - Comprehensive description synthesizing insights from the target and ALL analyzed upstream SQL.
   - Complete list of all output columns from the target model's final SELECT statement, with clear descriptions derived from your analysis.
6. **Finish:** Call the `finish_interpretation` tool with the complete YAML string.

**IMPORTANT:**
- The `get_models_raw_sql` tool takes a LIST of model names and returns a dictionary mapping names to their raw SQL string (or an error string).
- Call `get_models_raw_sql` only when needed, and request all necessary SQL in a single batch per turn.
- Focus your final YAML on the *target* model, using upstream insights to enrich its description and column definitions."""

            initial_prompt = f"""Please interpret the dbt model '{state['model_name']}'. Its raw SQL is:

```sql
{state['raw_sql']}
```

Follow the interpretation process:
1. Analyze this SQL. Identify all `ref()` calls.
2. Use `get_models_raw_sql` to fetch the raw SQL for all referenced models you haven't seen yet. Provide the list of model names to the tool.
3. Analyze the fetched SQL, identify more `ref()`s, and fetch their SQL recursively until you understand the lineage.
4. Synthesize the final YAML documentation for '{state['model_name']}' and call `finish_interpretation`."""

            messages_to_send = [
                SystemMessage(
                    content=system_prompt.format(model_name=state["model_name"])
                ),
                HumanMessage(content=initial_prompt),
            ]

            if self.verbose:
                self.console.print("[dim]  Generating initial prompt...[/dim]")

        else:  # Subsequent turns
            # Provide context about which models' SQL has been fetched
            guidance_content = f"""Guidance: You are interpreting '{state['model_name']}'.

You have ALREADY fetched the raw SQL for these models: {list(already_fetched_sql_for)}

**Your NEXT step is CRITICAL:**
1. **Analyze the NEWLY fetched SQL** (from the last `get_models_raw_sql` call) very carefully.
2. Identify ALL `ref()` calls within that new SQL that point to models whose SQL you DO NOT YET HAVE (check against the list above).
3. If you find any such *new* upstream dependencies, use `get_models_raw_sql` AGAIN to fetch their SQL in a batch.
4. **DO NOT call `finish_interpretation` yet.** Only call `finish_interpretation` when you have analyzed ALL necessary SQL up the entire dependency chain and are certain you have the complete context for '{state['model_name']}'."""

            # Check if the last message is already providing guidance
            last_message = messages[-1] if messages else None
            if isinstance(
                last_message, SystemMessage
            ) and last_message.content.startswith("Guidance:"):
                messages_to_send = messages  # Guidance already present
                if self.verbose:
                    self.console.print(
                        "[dim]  Guidance already present in last message.[/dim]"
                    )
            else:
                messages_to_send = messages + [SystemMessage(content=guidance_content)]
                if self.verbose:
                    self.console.print(
                        f"[dim blue]  Injecting guidance: {guidance_content}[/dim blue]"
                    )

        if self.verbose:
            self.console.print(
                f"[dim]  Sending {len(messages_to_send)} messages to LLM...[/dim]"
            )
            # Optional: Log message content previews

        response = agent_llm.invoke(messages_to_send)

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

            return {"messages": [response]}

    def update_state_node(self, state: InterpretationState) -> Dict[str, Any]:
        """Processes the most recent tool messages and updates the state."""
        logger.info("Entering state update node.")
        if self.verbose:
            self.console.print(f"\n[bold cyan]🔄 Update State Node[/bold cyan]")

        updates: Dict[str, Any] = {}
        finish_called = False
        # Use the correct state key: fetched_sql_map
        fetched_sql_updates = state.get("fetched_sql_map", {}).copy()

        # Find the most recent AI message with tool calls
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        if not ai_messages:
            logger.warning("No AI messages found in state.")
            return {"is_finished": False}

        latest_ai_msg = ai_messages[-1]

        if not hasattr(latest_ai_msg, "tool_calls") or not latest_ai_msg.tool_calls:
            # This might happen if the agent decides to finish without a tool call
            # Or if the last message wasn't an AIMessage with tool calls (shouldn't happen in normal flow)
            logger.debug(
                "Latest AI message has no tool calls. Checking if it's a final answer."
            )
            # If the agent *intended* to finish, it should have used the finish_interpretation tool.
            # If it just sent a message, we loop back.
            return {"is_finished": False}

        # Find corresponding tool messages
        tool_messages = [
            msg for msg in state["messages"] if isinstance(msg, ToolMessage)
        ]
        latest_tool_call_ids = {
            tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            for tc in latest_ai_msg.tool_calls
        }
        latest_tool_messages = [
            tm
            for tm in tool_messages
            if hasattr(tm, "tool_call_id") and tm.tool_call_id in latest_tool_call_ids
        ]

        if self.verbose:
            self.console.print(
                f"[dim]  Processing {len(latest_tool_messages)} tool result(s)...[/dim]"
            )

        for tool_msg in latest_tool_messages:
            tool_name = getattr(tool_msg, "name", "unknown_tool")
            tool_content = (
                tool_msg.content
            )  # This should be Dict[str, Optional[str]] for the SQL tool
            tool_call_id = tool_msg.tool_call_id

            if self.verbose:
                content_summary = str(tool_content)[:150].replace("\n", " ") + "..."
                self.console.print(
                    f"[dim]  - Tool: '{tool_name}', Result Summary: {content_summary}[/dim]"
                )

            # Updated tool name check
            if tool_name == "get_models_raw_sql":
                try:
                    if isinstance(tool_content, dict):
                        sql_results = tool_content
                    else:
                        # Attempt to parse if it's a string (less likely now)
                        import json

                        sql_results = (
                            json.loads(tool_content)
                            if isinstance(tool_content, str)
                            else {}
                        )

                    if isinstance(sql_results, dict):
                        # Merge the fetched SQL into the state
                        fetched_sql_updates.update(sql_results)
                        logger.info(
                            f"Updated state with fetched SQL for models: {list(sql_results.keys())}"
                        )
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
                        logger.warning(
                            f"Tool '{tool_name}' result was not a dictionary: {type(tool_content)}"
                        )
                        if self.verbose:
                            self.console.print(
                                f"[yellow dim]    -> Result was not a dict. Not stored.[/yellow dim]"
                            )

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
                # Find the YAML argument from the invoking AIMessage's tool call
                yaml_doc = None
                for tc in latest_ai_msg.tool_calls:
                    call_name = (
                        tc.get("name")
                        if isinstance(tc, dict)
                        else getattr(tc, "name", None)
                    )
                    call_id = (
                        tc.get("id")
                        if isinstance(tc, dict)
                        else getattr(tc, "id", None)
                    )

                    if call_name == "finish_interpretation" and call_id == tool_call_id:
                        args = (
                            tc.get("args")
                            if isinstance(tc, dict)
                            else getattr(tc, "args", None)
                        )
                        if isinstance(args, dict):
                            yaml_doc = args.get("yaml_doc")
                        break  # Found the matching call

                if yaml_doc:
                    updates["final_yaml"] = yaml_doc
                    logger.info(
                        f"Stored final YAML (length: {len(yaml_doc)}) from finish_interpretation."
                    )
                    if self.verbose:
                        self.console.print(
                            f"[green dim]    -> Stored final YAML (length: {len(yaml_doc)}).[/green dim]"
                        )
                else:
                    logger.warning(
                        "Couldn't extract YAML from finish_interpretation call args."
                    )
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim]    -> Couldn't retrieve YAML argument from tool call.[/yellow dim]"
                        )

                updates["is_finished"] = True  # Signal workflow completion
                if self.verbose:
                    self.console.print(
                        "[green bold]    -> Signaling workflow completion.[/green bold]"
                    )

        # Update the state with the accumulated SQL map if changed
        # Use the correct key: fetched_sql_map
        if fetched_sql_updates != state.get("fetched_sql_map", {}):
            updates["fetched_sql_map"] = fetched_sql_updates

        # Ensure 'is_finished' is set correctly
        if "is_finished" not in updates:
            updates["is_finished"] = finish_called

        if self.verbose and not updates:
            self.console.print(
                "[dim]  No state changes detected in this update step.[/dim]"
            )

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
            self.console.print(
                f"[dim]Target Model Raw SQL:\n```sql\n{raw_sql[:500]}...\n```[/dim]"
            )

        # Define the initial state dictionary
        initial_state = InterpretationState(
            model_name=model_name,
            raw_sql=raw_sql,
            messages=[],
            fetched_sql_map={},  # Initialize as empty dict
            final_yaml=None,
            is_finished=False,
        )

        try:
            final_state = self.graph_app.invoke(initial_state, config=config)

            final_yaml = final_state.get("final_yaml")
            success = bool(final_yaml)
            # Use correct key for fetched SQL map
            fetched_sql_map = final_state.get("fetched_sql_map", {})

            logger.info(
                f"Interpretation workflow for '{model_name}' finished. Success: {success}. Fetched SQL for {len(fetched_sql_map)} upstream models."
            )

            if self.verbose or not success:
                self.console.print(
                    f"\n[bold {'green' if success else 'red'}]🏁 Interpretation Workflow Finished for {model_name} {'✅' if success else '❌'}[/bold {'green' if success else 'red'}]"
                )
                final_messages = final_state.get("messages", [])
                self.console.print(
                    f"[dim]  Total messages in history: {len(final_messages)}"
                )
                self.console.print(
                    f"[dim]  Fetched SQL for {len(fetched_sql_map)} models: {list(fetched_sql_map.keys())}"
                )

                if success:
                    self.console.print(
                        "\n[bold blue]📝 Final YAML Interpretation:[/bold blue]"
                    )
                    self.console.print(Markdown(f"```yaml\n{final_yaml}\n```"))
                else:
                    self.console.print(Markdown("*No final YAML was produced.*"))
                    # Optionally print last few messages on failure
                    if final_messages:
                        self.console.print(
                            "[yellow dim]Last few messages:[/yellow dim]"
                        )
                        for msg in final_messages[-3:]:
                            content_str = getattr(msg, "content", "N/A")
                            self.console.print(
                                f"[yellow dim] - {type(msg).__name__}: {str(content_str)[:100]}...[/yellow dim]"
                            )

            return {
                "model_name": model_name,
                "yaml_documentation": final_yaml,
                "fetched_sql_map": fetched_sql_map,  # Renamed key
                "success": success,
            }

        except Exception as e:
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
                state_result = self.graph_app.get_state(config)
                error_state = state_result.values if state_result else {}
                logger.info(f"State at time of error: {error_state}")
            except Exception as state_err:
                logger.error(f"Could not retrieve state after error: {state_err}")

            return {
                "model_name": model_name,
                "yaml_documentation": None,
                "fetched_sql_map": error_state.get(
                    "fetched_sql_map", {}
                ),  # Renamed key
                "success": False,
                "error": str(e),
            }

    def save_interpreted_documentation(
        self, model_name: str, yaml_documentation: str, embed: bool = False
    ) -> Dict[str, Any]:
        """Save interpreted documentation for a model."""
        if not yaml_documentation:
            logger.warning(
                f"Attempted to save empty documentation for {model_name}. Skipping save."
            )
            return {
                "success": False,
                "error": "No documentation provided to save.",
                "model_name": model_name,
            }

        logger.info(f"Saving interpreted documentation for model {model_name}")
        session = self.model_storage.Session()
        try:
            # Parse the YAML
            try:
                model_data = yaml.safe_load(yaml_documentation)
                if not isinstance(model_data, dict):
                    raise ValueError("YAML content is not a dictionary.")

                if (
                    "name" not in model_data
                    or "description" not in model_data
                    or "columns" not in model_data
                ):
                    logger.warning(
                        f"Parsed YAML for {model_name} might be missing name, description, or columns."
                    )

                yaml_model_name = model_data.get("name")
                if yaml_model_name != model_name:
                    logger.warning(
                        f"Model name mismatch in YAML ('{yaml_model_name}') and function call ('{model_name}'). Using '{model_name}' for database lookup."
                    )

            except (yaml.YAMLError, ValueError) as e:
                logger.error(f"Error parsing YAML for {model_name}: {e}")
                return {
                    "success": False,
                    "error": f"Invalid YAML format: {e}",
                    "model_name": model_name,
                }

            # Extract interpretation data
            interpreted_description = model_data.get("description", "")
            interpreted_columns = {}

            if "columns" in model_data and isinstance(model_data["columns"], list):
                for col_data in model_data["columns"]:
                    if isinstance(col_data, dict) and "name" in col_data:
                        col_name = col_data["name"]
                        col_desc = col_data.get("description", "")
                        interpreted_columns[col_name] = col_desc
                    else:
                        logger.warning(
                            f"Skipping invalid column entry in YAML for {model_name}: {col_data}"
                        )

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
            model_record.interpreted_columns = (
                interpreted_columns  # Assumes JSON field type
            )

            session.add(model_record)
            session.commit()
            logger.info(f"Successfully saved interpretation for model {model_name}")

            # Optional: Re-embed the model
            if embed:
                logger.info(
                    f"Re-embedding model {model_name} after saving interpretation"
                )
                session.refresh(model_record)

                try:
                    # Construct text representation for embedding using updated record
                    text_parts = [
                        f"Model: {model_record.name}",
                        f"Description: {model_record.interpreted_description}",  # Use new description
                        "Columns:",
                    ]
                    # Use new columns
                    for col_name, col_desc in (
                        model_record.interpreted_columns or {}
                    ).items():
                        text_parts.append(f"  - {col_name}: {col_desc}")
                    if model_record.raw_sql:  # Still include raw SQL if available in DB
                        text_parts.append("SQL:")
                        text_parts.append(model_record.raw_sql)

                    model_text_for_embedding = "\n".join(text_parts)

                    if model_text_for_embedding:
                        self.vector_store.store_model_embedding(
                            model_name=model_record.name,
                            model_text=model_text_for_embedding,
                        )
                        logger.info(f"Successfully re-embedded model {model_name}")
                    else:
                        logger.warning(
                            f"Could not generate text representation for {model_name} from DB record for embedding."
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
