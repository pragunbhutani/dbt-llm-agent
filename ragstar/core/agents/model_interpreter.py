"""Agent for interpreting dbt models."""

import logging
import re
import yaml
import uuid
from typing import Dict, Any, Optional, List, Set, TypedDict
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
        memory: Optional["PostgresSaver"] = None,
    ):
        self.llm = llm_client
        self.model_storage = model_storage
        self.vector_store = vector_store
        self.verbose = verbose
        from rich.console import Console as RichConsole

        self.console = console or RichConsole()
        self.temperature = temperature

        from ragstar.utils.cli_utils import get_config_value
        from psycopg_pool import ConnectionPool
        from langgraph.checkpoint.postgres import PostgresSaver

        pg_conn_string = get_config_value("POSTGRES_URI")
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        pool = ConnectionPool(
            conninfo=pg_conn_string,
            kwargs=connection_kwargs,
            max_size=20,
            min_size=5,
        )
        self.memory = memory or PostgresSaver(conn=pool)
        self.memory.setup()

        self._define_tools()
        self.graph_app = self._build_graph()

    # --- LangGraph Imports ---
    # REMOVED IMPORTS FROM HERE

    # --- Tool Schemas ---
    class UpstreamModelInput(BaseModel):
        model_name: str = Field(
            description="Name of the upstream dbt model to fetch details for."
        )

    class FinishInterpretationInput(BaseModel):
        yaml_doc: str = Field(
            description="Final YAML documentation for the interpreted model."
        )

    # --- State ---
    class InterpretationState(TypedDict):
        model_name: str
        messages: Annotated[List[BaseMessage], add_messages]
        upstream_models_info: Dict[str, Dict[str, Any]]
        explored_models: Set[str]
        final_yaml: Optional[str]

    def _define_tools(self):
        @tool(args_schema=self.UpstreamModelInput)
        def get_upstream_model_details(model_name: str) -> Dict[str, Any]:
            """Fetches details for a specified upstream dbt model.
            Use this tool to understand the data source, columns, and description of models referenced in the main model's SQL.
            """
            logger.info(
                f"Executing tool: get_upstream_model_details for '{model_name}'"
            )
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: get_upstream_model_details(model_name='{model_name}')[/bold magenta]"
                )
            model = self.model_storage.get_model(model_name)
            if not model:
                if self.verbose:
                    self.console.print(
                        f"[dim] -> Model '{model_name}' not found in storage.[/dim]"
                    )
                return {"error": f"Model {model_name} not found."}

            result = {
                "name": model.name,
                "description": model.interpreted_description or model.description or "",
                "raw_sql": model.raw_sql or "",
                "columns": model.interpreted_columns
                or {k: v.description for k, v in (model.columns or {}).items()},
                "all_upstream_models": model.all_upstream_models or [],
            }
            if self.verbose:
                self.console.print(
                    f"[dim] -> Fetched details for model '{model_name}'.[/dim]"
                )
            return result

        @tool(args_schema=self.FinishInterpretationInput)
        def finish_interpretation(yaml_doc: str) -> str:
            """Completes the interpretation process and returns the final YAML documentation.
            Use this tool ONLY when the model and its columns are fully understood, including information gathered from upstream models.
            The input should be a single string containing the complete YAML document.
            """
            logger.info(f"Executing tool: finish_interpretation")
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: finish_interpretation(yaml_doc='{yaml_doc[:100]}...') [/bold magenta]"
                )
            return yaml_doc

        self._tools = [get_upstream_model_details, finish_interpretation]

    def _build_graph(self):
        workflow = StateGraph(self.InterpretationState)
        tool_node = ToolNode(self._tools, handle_tool_errors=True)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("update_state", self.update_state_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END,
            },
        )
        workflow.add_edge("tools", "update_state")
        workflow.add_edge("update_state", "agent")
        return workflow.compile(checkpointer=self.memory)

    def _get_agent_llm(self):
        chat_client_instance = self.llm.chat_client
        if hasattr(chat_client_instance, "bind_tools"):
            return chat_client_instance.bind_tools(self._tools)
        else:
            logger.warning(
                "LLM chat client does not have 'bind_tools'. Tool calling might not work."
            )
            return chat_client_instance

    def agent_node(
        self, state: "ModelInterpreter.InterpretationState"
    ) -> Dict[str, Any]:
        logger.info("Entering agent node.")
        if self.verbose:
            incoming_messages = state.get("messages", [])
            self.console.print(
                f"[bold yellow]\n>>> Entering agent_node <<<[/bold yellow]"
            )
            self.console.print(f"[dim]State Model: {state['model_name']}[/dim]")
            self.console.print(
                f"[dim]Received {len(incoming_messages)} messages in state:[/dim]"
            )
            for i, msg in enumerate(incoming_messages):
                msg_type = type(msg).__name__
                content_preview = (
                    str(msg.content)[:80] + "..."
                    if len(str(msg.content)) > 80
                    else str(msg.content)
                )
                tool_calls_info = (
                    " [bold]with tool_calls[/bold]"
                    if hasattr(msg, "tool_calls") and msg.tool_calls
                    else ""
                )
                tool_id_info = (
                    f" [bold]for tool_call_id: {msg.tool_call_id}[/bold]"
                    if hasattr(msg, "tool_call_id") and msg.tool_call_id
                    else ""
                )
                self.console.print(
                    f"  [dim]State[{i}] {msg_type}{tool_calls_info}{tool_id_info}: {content_preview}[/dim]"
                )

        messages = state.get("messages", [])
        if not messages:
            model = self.model_storage.get_model(state["model_name"])
            if not model:
                # Log error if model not found even at the start
                logger.error(
                    f"Model {state['model_name']} not found for interpretation."
                )
                return {
                    "messages": [
                        AIMessage(
                            content=f"Critical Error: Model {state['model_name']} not found in storage."
                        )
                    ]
                }
            initial_prompt = f"""
You are an expert data analyst tasked with interpreting a dbt model called '{model.name}'.

Your goal is to produce a complete YAML documentation for this model, including:
- A clear description of what the model represents.
- A detailed list of all columns it produces, with explanations.

You have access to a tool `get_upstream_model_details` to fetch details of any upstream models used in this model's SQL.

You should:
- Analyze the model's raw SQL (including Jinja).
- Identify all columns produced, including those from upstream models.
- Call `get_upstream_model_details` as needed to understand upstream models.
- Recursively explore upstreams if necessary.
- When ready, call `finish_interpretation` with the final YAML documentation.

Here is the raw SQL for the model '{model.name}':
```sql
{model.raw_sql}
```
"""
            messages_for_llm = [
                SystemMessage(content="You are an expert dbt model interpreter."),
                HumanMessage(content=initial_prompt),
            ]
            if self.verbose:
                self.console.print("[dim]Generated initial prompt for LLM.[/dim]")
        else:
            messages_for_llm = list(messages)
            if self.verbose:
                self.console.print("[dim]Using existing messages for LLM.[/dim]")

        agent_llm = self._get_agent_llm()
        if self.verbose:
            self.console.print(
                f"[blue dim]Sending {len(messages_for_llm)} messages to LLM...[/blue dim]"
            )

        try:
            response = agent_llm.invoke(messages_for_llm)
            # Log tool calls if present (non-verbose)
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_names = [
                    tc.get("name")
                    for tc in response.tool_calls
                    if isinstance(tc, dict) and tc.get("name")
                ]
                if tool_names:
                    logger.info(f"LLM requested tool calls: {tool_names}")
            elif self.verbose:
                self.console.print("[dim]LLM Response has no tool calls.[/dim]")

            if self.verbose:
                self.console.print(
                    f"[dim]LLM Response: {response.content[:150]}...[/dim]"
                )
                if hasattr(response, "tool_calls") and response.tool_calls:
                    self.console.print(
                        f"[dim]LLM Tool Calls: {response.tool_calls}[/dim]"
                    )
                # Removed redundant verbose log for no tool calls here

            return {"messages": [response]}
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=self.verbose)
            if self.verbose:
                self.console.print(f"[bold red]LLM invocation failed: {e}[/bold red]")
            return {"messages": [AIMessage(content=f"LLM invocation failed: {str(e)}")]}

    def update_state_node(
        self, state: "ModelInterpreter.InterpretationState"
    ) -> Dict[str, Any]:
        logger.info("Entering state update node.")
        if self.verbose:
            self.console.print(
                "[bold cyan]\n--- Updating State After Tool Execution ---[/bold cyan]"
            )
        updates: Dict[str, Any] = {}
        messages = state.get("messages", [])
        if not messages:
            if self.verbose:
                self.console.print(
                    "[yellow dim] -> No messages in state, skipping update.[/yellow dim]"
                )
            return updates

        recent_tool_messages = []
        i = len(messages) - 1
        last_assistant_with_tool_calls = None
        while i >= 0:
            msg = messages[i]
            if isinstance(msg, ToolMessage):
                recent_tool_messages.append(msg)
            elif (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                last_assistant_with_tool_calls = msg
                break
            i -= 1

        if not recent_tool_messages:
            if self.verbose:
                self.console.print(
                    "[yellow dim] -> No recent tool messages found, skipping update.[/yellow dim]"
                )
            return updates
        if self.verbose:
            self.console.print(
                f"[dim] -> Processing {len(recent_tool_messages)} recent tool messages[/dim]"
            )

        for tool_message in recent_tool_messages:
            tool_name = tool_message.name
            content = tool_message.content
            tool_call_id = (
                tool_message.tool_call_id
                if hasattr(tool_message, "tool_call_id")
                else "N/A"
            )
            if self.verbose:
                content_summary = str(content)[:150] + (
                    "..." if len(str(content)) > 150 else ""
                )
                self.console.print(
                    f"[dim] -> Processing result from tool '{tool_name}' (ID: {tool_call_id}): {content_summary}[/dim]"
                )

            processed_info = f"tool '{tool_name}'"
            if tool_name == "get_upstream_model_details":
                try:
                    import json

                    model_info = (
                        json.loads(content) if isinstance(content, str) else content
                    )
                    if "error" in model_info:
                        if self.verbose:
                            self.console.print(
                                f"[yellow dim] -> Tool '{tool_name}' reported error: {model_info['error']}[/yellow dim]"
                            )
                        processed_info = (
                            f"tool '{tool_name}' result (error: {model_info['error']})"
                        )
                        continue  # Skip updating state for this error

                    model_name_from_tool = model_info.get("name")
                    if model_name_from_tool:
                        upstreams = state.get("upstream_models_info", {}).copy()
                        upstreams[model_name_from_tool] = model_info
                        updates["upstream_models_info"] = upstreams
                        processed_info = (
                            f"upstream details for model '{model_name_from_tool}'"
                        )
                        if self.verbose:
                            self.console.print(
                                f"[dim] -> Updated state with upstream info for '{model_name_from_tool}'. Total upstreams: {len(upstreams)}.[/dim]"
                            )
                    else:
                        processed_info = f"tool '{tool_name}' result (missing name)"
                        if self.verbose:
                            self.console.print(
                                f"[yellow dim] -> Tool '{tool_name}' result missing 'name' field.[/yellow dim]"
                            )

                except Exception as e:
                    logger.warning(
                        f"Failed to parse upstream model details: {e}",
                        exc_info=self.verbose,
                    )
                    processed_info = f"tool '{tool_name}' result (parsing error)"
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim] -> Error parsing '{tool_name}' result: {e}[/yellow dim]"
                        )
                    continue  # Skip update on parsing error
            elif tool_name == "finish_interpretation":
                if isinstance(content, str):
                    updates["final_yaml"] = content
                    processed_info = (
                        f"final YAML interpretation (length: {len(content)})"
                    )
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with final_yaml (length: {len(content)}).[/dim]"
                        )
                else:
                    processed_info = f"tool '{tool_name}' result (non-string content)"
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim] -> finish_interpretation tool returned non-string content: {type(content)}[/yellow dim]"
                        )

            logger.info(f"Processed {processed_info}.")

        if not updates and self.verbose:
            self.console.print(
                f"[dim] -> No state updates generated from tool messages."
            )

        return updates

    def run_interpretation_workflow(self, model_name: str) -> Dict[str, Any]:
        logger.info(f"Starting interpretation workflow for model: {model_name}")
        self.console.print(
            f"[bold]🚀 Starting interpretation workflow for model:[/bold] {model_name}"
        )
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        if self.verbose:
            self.console.print(f"[dim]Generated thread ID: {thread_id}[/dim]")

        initial_state = self.InterpretationState(
            model_name=model_name,
            messages=[],
            upstream_models_info={},
            explored_models=set(),  # Consider adding logic to update explored_models
            final_yaml=None,
        )

        try:
            final_state = self.graph_app.invoke(initial_state, config=config)
            final_yaml = final_state.get("final_yaml", "")
            success = bool(final_yaml)
            logger.info(
                f"Interpretation workflow for {model_name} finished. Success: {success}"
            )
            if self.verbose:
                self.console.print(
                    f"[green]✅ Interpretation workflow finished for {model_name}.[/green]"
                )
                message_count = len(final_state.get("messages", []))
                upstream_count = len(final_state.get("upstream_models_info", {}))
                self.console.print(
                    f"[dim]Final state contains {message_count} messages."
                )
                self.console.print(
                    f"[dim]Gathered info for {upstream_count} upstream models."
                )
                self.console.print(
                    "\n[bold blue]📝 Final YAML Interpretation:[/bold blue]"
                )
                self.console.print(Markdown(final_yaml or "*No YAML produced*"))
            else:
                # Log YAML presence even if not verbose
                if success:
                    logger.debug(
                        f"Final YAML generated for {model_name} (length: {len(final_yaml)})"
                    )
                else:
                    logger.warning(f"No final YAML produced for {model_name}.")

            return {
                "model_name": model_name,
                "yaml_documentation": final_yaml,
                "upstream_models_info": final_state.get("upstream_models_info", {}),
                "success": success,
            }
        except Exception as e:
            logger.error(
                f"Error during interpretation workflow for {model_name}: {e}",
                exc_info=True,
            )
            self.console.print(
                f"[bold red]Error during interpretation workflow for {model_name}:[/bold red] {e}"
            )
            return {
                "model_name": model_name,
                "yaml_documentation": None,
                "upstream_models_info": {},
                "success": False,
                "error": str(e),
            }

    def save_interpreted_documentation(
        self, model_name: str, yaml_documentation: str, embed: bool = False
    ) -> Dict[str, Any]:
        """Save interpreted documentation for a model.

        Args:
            model_name: Name of the model
            yaml_documentation: YAML documentation as a string. This should include
                any column additions, removals, or modifications that were recommended
                during the verification phase.
            embed: Whether to embed the model in the vector store

        Returns:
            Dict containing the result of the operation
        """
        logger.info(f"Saving interpreted documentation for model {model_name}")
        session = self.model_storage.Session()  # Get a session from storage
        try:
            # Parse the YAML
            try:
                parsed_yaml = yaml.safe_load(yaml_documentation)
                if (
                    not isinstance(parsed_yaml, dict)
                    or "models" not in parsed_yaml
                    or not isinstance(parsed_yaml["models"], list)
                    or not parsed_yaml["models"]
                ):
                    raise ValueError("Invalid YAML structure")
                model_data = parsed_yaml["models"][0]  # Assuming one model per YAML
                if model_data.get("name") != model_name:
                    logger.warning(
                        f"Model name mismatch in YAML ({model_data.get('name')}) and function call ({model_name}). Using {model_name}."
                    )

            except (yaml.YAMLError, ValueError) as e:
                logger.error(f"Error parsing YAML for {model_name}: {e}")
                return {
                    "success": False,
                    "error": f"Invalid YAML format: {e}",
                    "model_name": model_name,
                }

            # Extract interpretation data
            interpreted_description = model_data.get("description")
            interpreted_columns = {}
            if "columns" in model_data and isinstance(model_data["columns"], list):
                for col_data in model_data["columns"]:
                    if isinstance(col_data, dict) and "name" in col_data:
                        interpreted_columns[col_data["name"]] = col_data.get(
                            "description", ""
                        )

            # Fetch the existing ModelTable record
            model_record = (
                session.query(ModelTable).filter(ModelTable.name == model_name).first()
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
            model_record.interpreted_columns = interpreted_columns

            session.add(model_record)
            session.commit()
            logger.info(f"Successfully saved interpretation for model {model_name}")

            # Optional: Re-embed the model
            if embed:
                logger.info(
                    f"Re-embedding model {model_name} after saving interpretation"
                )
                updated_model = self.model_storage.get_model(model_name)
                if updated_model:
                    model_text_for_embedding = updated_model.get_text_representation(
                        include_documentation=True
                    )
                    self.vector_store.store_model_embedding(
                        model_name=updated_model.name,
                        model_text=model_text_for_embedding,
                    )
                    logger.info(f"Successfully re-embedded model {model_name}")
                else:
                    logger.error(
                        f"Failed to fetch updated model {model_name} for re-embedding"
                    )
                    return {
                        "success": True,
                        "warning": f"Interpretation saved, but failed to re-embed model {model_name}",
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
