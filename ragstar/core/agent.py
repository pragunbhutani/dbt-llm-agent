"""Agent for answering questions about dbt models."""

import logging
import re
import json
import yaml
import sys
import os  # Import os
import openai  # Import openai
import functools  # Add functools import
from typing import Dict, List, Any, Optional, Union, Set, TypedDict
import uuid

# Langchain & LangGraph Imports
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
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from ragstar.storage.question_storage import QuestionStorage

from ragstar.core.llm.client import LLMClient
from ragstar.core.llm.prompts import (
    MODEL_INTERPRETATION_PROMPT,
)
from ragstar.storage.model_storage import ModelStorage
from ragstar.storage.model_embedding_storage import ModelEmbeddingStorage
from ragstar.core.models import DBTModel, ModelTable
from ragstar.utils.cli_utils import get_config_value

from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


# --- Tool Input Schemas ---
class SearchModelsInput(BaseModel):
    query: str = Field(
        description="Concise, targeted query focusing on specific information needed to find relevant dbt models."
    )


class SearchFeedbackInput(BaseModel):
    query: str = Field(
        description="The original user question to search for similar past questions and feedback."
    )


class FinishWorkflowInput(BaseModel):
    final_answer: str = Field(
        description="The comprehensive final answer to the user's original question, synthesized from all gathered information and feedback. This should include the final SQL query and explanation."
    )


# --- Standalone Tool Logic ---

# Note: We pass dependencies like console, verbose, storages explicitly
# These functions return the raw data, not formatted strings for the LLM state update
# The @tool decorator will be applied later to partial versions of these functions


def _search_dbt_models_logic(
    query: str,
    *,  # Force keyword args for dependencies
    console: Console,
    verbose: bool,
    vector_store: ModelEmbeddingStorage,
    model_storage: ModelStorage,
) -> List[Dict[str, Any]]:
    """Core logic for searching dbt models."""
    if verbose:
        console.print(
            f"[bold magenta]🛠️ Executing Tool Logic: _search_dbt_models_logic(query='{query}')[/bold magenta]"
        )
    console.print(f"[bold blue]🔍 Searching models relevant to: '{query}'[/bold blue]")

    search_results = vector_store.search_models(query=query, n_results=5)
    newly_added_model_details = []

    if not search_results:
        if verbose:
            console.print(
                "[dim] -> No models found by vector store for this query.[/dim]"
            )
        return []  # Return empty list

    for result in search_results:
        model_name = result["model_name"]
        similarity = result.get("similarity_score", 0)

        if isinstance(similarity, (int, float)) and similarity > 0.3:
            model = model_storage.get_model(model_name)
            if model:
                model_dict = model.to_dict()
                model_dict["search_score"] = similarity
                newly_added_model_details.append(model_dict)
                if verbose:
                    console.print(
                        f"[dim] -> Found relevant new model: {model_name} (Score: {similarity:.2f})[/dim]"
                    )

    if not newly_added_model_details and verbose:
        console.print(
            "[dim] -> Vector store found models, but they were already known or below threshold.[/dim]"
        )

    # Return the list of newly found model *details* (dictionaries)
    return newly_added_model_details


def _search_feedback_logic(
    query: str,
    *,  # Force keyword args for dependencies
    console: Console,
    verbose: bool,
    question_storage: Optional[QuestionStorage],
) -> List[Any]:  # Assuming QuestionStorage returns a list of specific feedback items
    """Core logic for searching past feedback."""
    if verbose:
        console.print(
            f"[bold magenta]🛠️ Executing Tool Logic: _search_feedback_logic(query='{query}') [/bold magenta]"
        )

    if not (
        question_storage
        and hasattr(question_storage, "_get_embedding")
        and hasattr(question_storage, "openai_client")
        and question_storage.openai_client is not None
    ):
        if verbose:
            console.print(
                "[yellow dim] -> Feedback storage not configured or embedding client unavailable.[/yellow dim]"
            )
        return []  # Return empty list

    if verbose:
        console.print("[blue]🔍 Checking for feedback...[/blue]")

    try:
        question_embedding = question_storage._get_embedding(query)
        if not question_embedding:
            if verbose:
                console.print(
                    "[yellow dim] -> Could not generate embedding for feedback search.[/yellow dim]"
                )
            return []  # Return empty list

        relevant_feedback_items = question_storage.find_similar_questions_with_feedback(
            query_embedding=question_embedding,
            limit=3,
            similarity_threshold=0.75,
        )

        if not relevant_feedback_items:
            if verbose:
                console.print("[dim] -> No relevant feedback found.[/dim]")
            return []  # Return empty list

        if verbose:
            console.print(
                f"[dim] -> Found {len(relevant_feedback_items)} relevant feedback item(s).[/dim]"
            )

        # Return the list of feedback items directly
        return relevant_feedback_items

    except Exception as e:
        logger.error(f"Error during feedback search logic: {e}", exc_info=verbose)
        # Return empty list on error, the ToolNode might add an error message to history anyway
        return []


def _finish_workflow_logic(
    final_answer: str,
    *,  # Force keyword args for dependencies
    console: Console,
    verbose: bool,
) -> str:
    """Core logic for finishing the workflow."""
    if verbose:
        console.print(
            f"[bold magenta]🛠️ Executing Tool Logic: _finish_workflow_logic(final_answer='{final_answer[:100]}...')",
        )
    # This function primarily acts as a signal, returning the final answer
    # The actual state update happens in update_state_node based on the tool call
    return final_answer


# --- LangGraph State Definition ---
class AgentState(TypedDict):
    """Represents the state of our agent graph."""

    original_question: str
    messages: List[BaseMessage]  # Conversation history
    accumulated_models: List[Dict[str, Any]]  # Models found so far
    accumulated_model_names: Set[str]  # Names of models found
    search_model_calls: int  # Track number of model search calls
    relevant_feedback: List[
        Any
    ]  # Feedback found (replace Any with specific type if available)
    search_queries_tried: Set[str]  # Track search queries
    final_answer: Optional[str]
    conversation_id: Optional[str]


# Add this class after imports and before the Agent class
class SafeToolExecutor:
    """A wrapper around LangGraph's ToolNode that ensures valid message ordering."""

    def __init__(self, tools, verbose=False, console=None):
        """Initialize with the tools and verbosity settings."""
        self.tools = tools
        self.verbose = verbose
        self.console = console or Console()
        # Create the standard ToolNode for actual execution
        self.tool_node = ToolNode(tools)

    def __call__(self, state):
        """Execute tools while ensuring proper message ordering."""
        if self.verbose:
            self.console.print("[bold cyan]\n--- Safe Tool Executor ---[/bold cyan]")

        # Check if the last message has valid tool_calls
        messages = state.get("messages", [])
        if not messages:
            if self.verbose:
                self.console.print(
                    "[yellow dim]No messages in state, skipping tool execution.[/yellow dim]"
                )
            return state  # No messages to process

        last_message = messages[-1]

        # Only process if last message is an AIMessage with tool_calls
        if not (
            isinstance(last_message, AIMessage)
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            if self.verbose:
                self.console.print(
                    "[yellow dim]Last message doesn't have tool_calls, skipping.[/yellow dim]"
                )
            return state  # Last message doesn't have tool_calls

        # Execute tools using the ToolNode's invoke method instead of calling it directly
        try:
            # Find the tool calls in the last message
            tool_calls = last_message.tool_calls
            tool_results = []

            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id")
                function_name = tool_call.get("name")
                function_args = tool_call.get("args")

                if not function_name or not tool_call_id:
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim]Incomplete tool call: {tool_call}[/yellow dim]"
                        )
                    continue

                # Find the matching tool
                matching_tool = None
                for tool in self.tools:
                    if tool.name == function_name:
                        matching_tool = tool
                        break

                if not matching_tool:
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim]No matching tool found for: {function_name}[/yellow dim]"
                        )
                    continue

                # Parse arguments if needed
                try:
                    if isinstance(function_args, str):
                        import json

                        args = json.loads(function_args)
                    else:
                        args = function_args
                except Exception as e:
                    if self.verbose:
                        self.console.print(
                            f"[red dim]Error parsing arguments: {e}[/red dim]"
                        )
                    args = {}

                # Execute the tool
                try:
                    if self.verbose:
                        self.console.print(
                            f"[dim]Executing tool {function_name} with args: {args}[/dim]"
                        )
                    tool_result = matching_tool.run(args)

                    # Convert result to JSON string before creating ToolMessage
                    # MODIFIED: Import json locally here
                    import json

                    if isinstance(tool_result, (list, dict)):
                        content_str = json.dumps(tool_result)
                    else:
                        content_str = str(tool_result)

                    # Create a ToolMessage with the stringified result
                    tool_message = ToolMessage(
                        content=content_str,  # Use the stringified content
                        name=function_name,
                        tool_call_id=tool_call_id,
                    )
                    tool_results.append(tool_message)

                    if self.verbose:
                        self.console.print(
                            f"[green dim]Tool {function_name} executed successfully.[/green dim]"
                        )
                        # If finish_workflow tool was called, preview the output in verbose mode
                        if function_name == "finish_workflow" and isinstance(
                            tool_result, str
                        ):
                            self.console.print(
                                "[dim]Preview of Markdown-formatted answer:[/dim]"
                            )
                            preview = (
                                tool_result[:200] + "..."
                                if len(tool_result) > 200
                                else tool_result
                            )
                            self.console.print(f"[dim]{preview}[/dim]")

                except Exception as e:
                    if self.verbose:
                        self.console.print(
                            f"[red dim]Error executing tool {function_name}: {e}[/red dim]"
                        )
                    # MODIFIED: Re-raise the exception instead of creating an error message
                    raise e
                    # OLD CODE:
                    # error_msg = f"Error executing {function_name}: {str(e)}"
                    # tool_message = ToolMessage(
                    #     content=error_msg, name=function_name, tool_call_id=tool_call_id
                    # )
                    # tool_results.append(tool_message)

            # Update state with tool results (only if no exceptions occurred)
            if tool_results:
                updated_messages = state["messages"] + tool_results
                updated_state = dict(state)
                updated_state["messages"] = updated_messages

                if self.verbose:
                    self.console.print(
                        f"[green dim]Added {len(tool_results)} tool messages to state.[/green dim]"
                    )

                return updated_state
            else:
                if self.verbose:
                    self.console.print(
                        "[yellow dim]No tool results generated.[/yellow dim]"
                    )
                return state

        except Exception as e:
            if self.verbose:
                self.console.print(f"[red dim]Error in tool execution: {e}[/red dim]")
            # Return state unchanged on error
            return state


class Agent:
    """Agent for interacting with dbt projects using an agentic workflow."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_storage: ModelStorage,
        vector_store: ModelEmbeddingStorage,
        question_storage: Optional[QuestionStorage] = None,
        console: Optional[Console] = None,
        temperature: float = 0.0,
        verbose: bool = False,
        openai_api_key: Optional[str] = None,  # Add API key parameter
        memory: Optional[PostgresSaver] = None,  # Add optional memory using Postgres
    ):
        """Initialize the agent.

        Args:
            llm_client: LLM client for generating text
            model_storage: Storage for dbt models
            vector_store: Vector store for semantic search
            question_storage: Storage for question history (requires openai_api_key for embedding)
            console: Console for interactive prompts
            temperature: Temperature for LLM generation
            verbose: Whether to print verbose output
            openai_api_key: OpenAI API key for question embeddings.
            memory: Optional PgSaver for graph checkpointing using Postgres
        """
        self.llm = llm_client
        self.model_storage = model_storage
        self.vector_store = vector_store
        self.question_storage = question_storage
        self.console = console or Console()
        self.temperature = temperature
        self.verbose = verbose
        self.max_iterations = (
            10  # Max iterations for the graph recursion limit - increased slightly
        )
        self.max_model_searches = 5  # Max calls to search_dbt_models_tool
        # Default to Postgres checkpointer
        # Assumes a function/config helper to get the connection string
        # You might need to adjust how the connection string is obtained
        pg_conn_string = get_config_value("POSTGRES_URI")

        # Create a connection pool with proper parameters as per LangGraph documentation
        # connection_kwargs must be passed as kwargs parameter to ConnectionPool
        connection_kwargs = {
            "autocommit": True,  # Critical for CREATE INDEX CONCURRENTLY
            "prepare_threshold": 0,
        }

        # Create the pool with proper parameters
        pool = ConnectionPool(
            conninfo=pg_conn_string,
            kwargs=connection_kwargs,  # Pass connection_kwargs via kwargs parameter
            max_size=20,
            min_size=5,
        )

        # Instantiate PostgresSaver with the pool
        self.memory = memory or PostgresSaver(conn=pool)

        # Call setup() on the checkpointer instance
        self.memory.setup()

        # Compile the graph
        self.graph_app = self._build_graph()

    # --- LangGraph Graph Construction ---
    def _build_graph(self):
        """Builds the LangGraph StateGraph."""
        workflow = StateGraph(AgentState)

        # --- Create Tool Instances ---
        # Use functools.partial to inject dependencies into the standalone logic functions
        # We pass the *current state's* known model names to the search logic
        # Note: Accessing state within partial might be tricky if state changes dynamically
        # in ways not reflected by simple dependency injection. Passing state directly or
        # handling it purely in update_state_node might be more robust.
        # Let's adjust update_state_node to handle uniqueness instead of passing current_model_names here.

        # Tool for searching dbt models
        search_models_tool_partial = functools.partial(
            _search_dbt_models_logic,
            console=self.console,
            verbose=self.verbose,
            vector_store=self.vector_store,
            model_storage=self.model_storage,
        )
        # Decorate the partial function
        search_dbt_models_tool_decorated = tool(
            "search_dbt_models", args_schema=SearchModelsInput
        )(search_models_tool_partial)
        # Add description (docstring from original logic function is not automatically picked up by partial)
        search_dbt_models_tool_decorated.description = """Searches for relevant dbt models based on the query.
        Use this tool iteratively to find dbt models that can help answer the user's question.
        Provide a concise, targeted query focusing on the *specific information* needed next.
        Analyze the results: if the found models are insufficient, call this tool again with a *refined* query.
        If you have enough information or have reached the search limit, call 'finish_workflow'.
        Do not use this tool if you have already found sufficient models or reached the search limit."""

        # Tool for searching feedback
        search_feedback_tool_partial = functools.partial(
            _search_feedback_logic,
            console=self.console,
            verbose=self.verbose,
            question_storage=self.question_storage,
        )
        search_feedback_tool_decorated = tool(
            "search_past_feedback", args_schema=SearchFeedbackInput
        )(search_feedback_tool_partial)
        search_feedback_tool_decorated.description = """Searches for feedback on previously asked similar questions.
        Use this ONCE at the beginning of the workflow if the user's question might have been asked before.
        Provide the original user question as the query.
        The tool returns a summary of relevant feedback found."""

        # Tool for finishing the workflow
        finish_workflow_tool_partial = functools.partial(
            _finish_workflow_logic,
            console=self.console,
            verbose=self.verbose,
        )
        finish_workflow_tool_decorated = tool(
            "finish_workflow", args_schema=FinishWorkflowInput
        )(finish_workflow_tool_partial)
        finish_workflow_tool_decorated.description = """Concludes the workflow and provides the final answer to the user.
        Use this tool ONLY when you have gathered all necessary information from 'search_dbt_models',
        considered any relevant feedback from 'search_past_feedback',
        and are ready to provide the complete, final answer (including SQL query and explanation).
        Format your answer using Markdown for improved readability:
        - Use headings (# and ##) to organize different sections
        - Put SQL code in code blocks with ```sql and ```
        - Use bullet points (-) for listing models or key points
        - Bold important information with **
        Ensure the answer directly addresses the user's original question."""

        # --- Define Tools List for LangGraph ---
        tools = [
            search_dbt_models_tool_decorated,
            search_feedback_tool_decorated,
            finish_workflow_tool_decorated,
        ]

        # Store tools on instance if needed elsewhere (e.g., for _get_agent_llm)
        self._tools = tools
        # --- End Tool Creation ---

        # Instantiate the custom safe tool executor instead of standard ToolNode
        tool_node = SafeToolExecutor(tools, verbose=self.verbose, console=self.console)

        # Add nodes
        workflow.add_node("agent", self.agent_node)
        # MODIFIED: Use the custom Safe Tool Executor for tool execution
        workflow.add_node("tools", tool_node)
        # ADDED: Add node for custom state updates after tools run
        workflow.add_node("update_state", self.update_state_node)

        # Set entry point
        workflow.set_entry_point("agent")

        # Define conditional edges from agent
        workflow.add_conditional_edges(
            "agent",
            tools_condition,  # Prebuilt condition checks for tool calls
            {
                # If tool calls are present, route to the "tools" node
                "tools": "tools",
                # Otherwise (no tool calls, or specific finish signal), end the graph
                END: END,
            },
        )

        # MODIFIED: Define edges for the new flow: tools -> update_state -> agent
        # workflow.add_edge("tools", "agent") # Old edge
        workflow.add_edge("tools", "update_state")
        workflow.add_edge("update_state", "agent")
        # END MODIFIED

        # Compile the graph with memory
        return workflow.compile(checkpointer=self.memory)

    # --- End LangGraph Graph Construction ---

    # --- DELETED LangGraph Tools ---
    # The tool logic is now outside the class

    # --- End LangGraph Tools ---

    # --- LangGraph Nodes ---

    def _get_agent_llm(self):
        """Helper to get the LLM with tools bound."""
        # Use the tools stored on the instance
        if not hasattr(self, "_tools"):
            raise ValueError(
                "Tools not initialized in _build_graph before calling _get_agent_llm"
            )

        chat_client_instance = self.llm.chat_client

        if hasattr(chat_client_instance, "bind_tools"):
            return chat_client_instance.bind_tools(self._tools)  # Use self._tools
        else:
            logger.warning(
                "LLM chat client does not have 'bind_tools'. Tool calling might not work as expected."
            )
            return chat_client_instance

    def agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Calls the agent LLM to decide the next action or generate the final answer."""
        if self.verbose:
            self.console.print("[bold green]\n--- Calling Agent Model ---[/bold green]")
            self.console.print(f"[dim]Current Messages: {state['messages']}[/dim]")
            self.console.print(
                f"[dim]Found Models: {len(state['accumulated_model_names'])} ({state.get('search_model_calls', 0)} searches used)[/dim]"
            )
            self.console.print(
                f"[dim]Found Feedback: {len(state['relevant_feedback'])}[/dim]"
            )

        # Create a copy of messages and ensure proper ordering for OpenAI
        messages_for_llm = []

        # Ensure we have a valid message sequence for the OpenAI API by reconstructing
        # the messages array with proper tool call and tool message pairings
        pending_tool_calls = (
            {}
        )  # Maps tool_call_id to the message index that contains it

        # First pass: Find tool calls and track them
        for i, msg in enumerate(state["messages"]):
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                for tool_call in msg.tool_calls:
                    tool_call_id = tool_call.get("id")
                    if tool_call_id:
                        pending_tool_calls[tool_call_id] = i

        # Second pass: Construct the properly ordered message array
        i = 0
        while i < len(state["messages"]):
            msg = state["messages"][i]

            # Add the current message
            messages_for_llm.append(msg)

            # If this is an AIMessage with tool_calls, find and add all corresponding tool messages
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                # Collect all matching tool messages for this AI message's tool calls
                tool_message_indices = []
                for tool_call in msg.tool_calls:
                    tool_call_id = tool_call.get("id")
                    if not tool_call_id:
                        continue

                    # Find any ToolMessages that respond to this tool_call_id
                    for j in range(i + 1, len(state["messages"])):
                        response_msg = state["messages"][j]
                        if (
                            isinstance(response_msg, ToolMessage)
                            and response_msg.tool_call_id == tool_call_id
                        ):
                            if j not in tool_message_indices:
                                tool_message_indices.append(j)
                                # Remove this tool_call_id from pending since we found its response
                                pending_tool_calls.pop(tool_call_id, None)

                # Add tool messages in order right after the AI message with tool calls
                for j in sorted(tool_message_indices):
                    messages_for_llm.append(state["messages"][j])

                # Skip all these tool messages in the outer loop since we've added them here
                i = max(tool_message_indices) + 1 if tool_message_indices else i + 1
            else:
                i += 1

        # Check if there are any pending tool calls without responses
        if pending_tool_calls and self.verbose:
            self.console.print(
                f"[yellow dim]Warning: {len(pending_tool_calls)} tool calls have no matching tool message responses.[/yellow dim]"
            )
            # Only send complete pairs to avoid API errors
            # Filter out messages with pending tool calls
            complete_messages = []
            for msg in messages_for_llm:
                if (
                    isinstance(msg, AIMessage)
                    and hasattr(msg, "tool_calls")
                    and msg.tool_calls
                ):
                    # Check if any tool calls in this message are still pending
                    has_pending = False
                    for tool_call in msg.tool_calls:
                        tool_call_id = tool_call.get("id")
                        if tool_call_id in pending_tool_calls:
                            has_pending = True
                            break

                    if not has_pending:
                        complete_messages.append(msg)
                else:
                    complete_messages.append(msg)

            messages_for_llm = complete_messages

        # Prepare messages for the LLM, potentially adding guidance
        current_search_calls = state.get("search_model_calls", 0)
        remaining_searches = self.max_model_searches - current_search_calls

        guidance_message: Optional[BaseMessage] = None
        if remaining_searches > 0:
            guidance_message = SystemMessage(
                content=(
                    f"Guidance: You have used the model search tool {current_search_calls} times. "
                    f"You have {remaining_searches} remaining searches for dbt models. "
                    f"Current models found: {len(state['accumulated_model_names'])}. "
                    f"Original question: '{state['original_question']}'. "
                    f"If the current models are insufficient to answer the question comprehensively, "
                    f"use 'search_dbt_models' again with a *refined, specific* query focusing on the missing information. "
                    f"Otherwise, if you have enough information, use 'finish_workflow' to provide the final answer (SQL query + explanation)."
                )
            )
        else:
            guidance_message = SystemMessage(
                content=(
                    f"Guidance: You have reached the maximum limit of {self.max_model_searches} model searches. "
                    f"You must now synthesize an answer using the models found ({len(state['accumulated_model_names'])} models: {', '.join(state['accumulated_model_names']) or 'None'}) "
                    f"and call 'finish_workflow'. Provide the final SQL query and explanation. Do not call 'search_dbt_models' again."
                )
            )

        # Append the guidance message to the end. This ensures the AI(tool_calls) -> Tool sequence is maintained.
        if guidance_message:
            messages_for_llm.append(guidance_message)

        # If we have zero or just one system message, add the original system message and user question
        # This ensures we have at least a coherent start to the conversation
        if len(messages_for_llm) <= 1 or all(
            isinstance(msg, SystemMessage) for msg in messages_for_llm
        ):
            # Reset messages with the original system message and user question
            messages_for_llm = [
                SystemMessage(
                    content="You are an AI assistant specialized in analyzing dbt projects to answer questions."
                ),
                HumanMessage(content=state["original_question"]),
            ]
            if guidance_message:
                messages_for_llm.append(guidance_message)

        agent_llm = self._get_agent_llm()

        if self.verbose:
            # Log the exact messages being sent
            self.console.print(
                f"[blue dim]Messages being sent to LLM:\n{messages_for_llm}[/blue dim]"
            )

        try:
            response = agent_llm.invoke(messages_for_llm)
        except Exception as e:
            logger.error(f"Error invoking agent LLM: {e}", exc_info=self.verbose)
            # Add the error back into the message history to potentially inform subsequent steps or final output
            error_message = f"LLM invocation failed: {str(e)}"
            return {
                "messages": state["messages"] + [AIMessage(content=error_message)]
            }  # Return an AI message indicating failure

        if self.verbose:
            self.console.print(f"[dim]Agent Response: {response}[/dim]")

        return {"messages": state["messages"] + [response]}

    def update_state_node(self, state: AgentState) -> Dict[str, Any]:
        """Updates AgentState based on the results from the most recent tool call."""
        if self.verbose:
            self.console.print(
                "[bold cyan]\n--- Updating State After Tool Execution ---[/bold cyan]"
            )

        updates: Dict[str, Any] = {}

        # Find the most recent ToolMessage(s) that might have updated state
        messages = state.get("messages", [])
        if not messages:
            if self.verbose:
                self.console.print(
                    "[yellow dim] -> No messages in state, skipping update.[/yellow dim]"
                )
            return updates

        # Find all ToolMessages in the last batch (might be multiple if several tools were called)
        # We only want to process ToolMessages that haven't been processed before
        recent_tool_messages = []

        # Start from the end and work backwards to find the most recent tool messages
        i = len(messages) - 1
        while i >= 0:
            msg = messages[i]
            if isinstance(msg, ToolMessage):
                # Add this tool message to our list to process
                recent_tool_messages.append(msg)
            elif (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                # We've found the AIMessage that called these tools, stop here
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

        # Process each tool message
        for tool_message in recent_tool_messages:
            tool_name = tool_message.name
            tool_content = (
                tool_message.content
            )  # This is the direct return value from the tool logic
            tool_call_id = tool_message.tool_call_id  # Keep for matching if needed

            if self.verbose:
                self.console.print(
                    f"[dim] -> Processing result from tool '{tool_name}' (call_id: {tool_call_id})[/dim]"
                )
                # Be careful logging tool_content if it can be very large
                content_summary = str(tool_content)[:200] + (
                    "..." if len(str(tool_content)) > 200 else ""
                )
                self.console.print(
                    f"[dim] -> Tool content (summary): {content_summary}[/dim]"
                )

            # Ensure the ToolMessage is associated with a valid preceding message with tool_calls
            # Find the corresponding AIMessage with tool_calls matching this tool_call_id
            found_matching_call = False
            for i, msg in enumerate(messages):
                if (
                    isinstance(msg, AIMessage)
                    and hasattr(msg, "tool_calls")
                    and msg.tool_calls
                ):
                    for tool_call in msg.tool_calls:
                        if tool_call.get("id") == tool_call_id:
                            found_matching_call = True
                            break
                if found_matching_call:
                    break

            if not found_matching_call and self.verbose:
                self.console.print(
                    f"[yellow dim]Warning: ToolMessage with id {tool_call_id} has no matching preceding message with tool_calls.[/yellow dim]"
                )
                # Continue processing the tool content, but the warning helps debugging

            # --- Process based on which tool was called ---

            if tool_name == "search_dbt_models":
                # Tool content should be a list of newly found model dictionaries
                newly_found_models = (
                    tool_content if isinstance(tool_content, list) else []
                )

                # Increment search counter regardless of finding new models
                current_calls = state.get("search_model_calls", 0)
                updates["search_model_calls"] = current_calls + 1

                if newly_found_models:
                    current_models = state.get("accumulated_models", [])
                    current_model_names = state.get("accumulated_model_names", set())
                    unique_new_models = []

                    # Double-check uniqueness against current state names
                    for model_detail in newly_found_models:
                        # Ensure it's a dict and has a name before adding
                        if isinstance(model_detail, dict) and "name" in model_detail:
                            model_name = model_detail["name"]
                            if model_name not in current_model_names:
                                unique_new_models.append(model_detail)
                                current_model_names.add(model_name)
                        elif self.verbose:
                            self.console.print(
                                f"[yellow dim]Warning: Invalid item in search_dbt_models result: {model_detail}[/yellow dim]"
                            )

                    if unique_new_models:
                        updates["accumulated_models"] = (
                            current_models + unique_new_models
                        )
                        updates["accumulated_model_names"] = current_model_names
                        if self.verbose:
                            self.console.print(
                                f"[dim] -> Updated state with {len(unique_new_models)} new models. Total models: {len(current_model_names)}. Search calls: {updates['search_model_calls']}"
                            )
                    elif self.verbose:
                        self.console.print(
                            f"[dim] -> search_dbt_models ran, but no *new* unique models were found. Search calls: {updates['search_model_calls']}"
                        )
                elif self.verbose:
                    self.console.print(
                        f"[dim] -> search_dbt_models ran and returned no models. Search calls: {updates['search_model_calls']}"
                    )

            elif tool_name == "search_past_feedback":
                # Tool content should be a list of feedback items
                newly_found_feedback = (
                    tool_content if isinstance(tool_content, list) else []
                )

                if newly_found_feedback:
                    current_feedback = state.get("relevant_feedback", [])
                    # Basic check for duplicates based on a potential ID or hash if available
                    # For now, just append; consider adding more robust duplicate checking if needed
                    updates["relevant_feedback"] = (
                        current_feedback + newly_found_feedback
                    )
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with {len(newly_found_feedback)} new feedback items. Total feedback: {len(updates['relevant_feedback'])}"
                        )
                elif self.verbose:
                    self.console.print(
                        "[dim] -> search_past_feedback ran but found no feedback."
                    )

            elif tool_name == "finish_workflow":
                # Tool content should be the final answer string
                final_answer_from_tool = (
                    tool_content if isinstance(tool_content, str) else None
                )

                if final_answer_from_tool:
                    updates["final_answer"] = final_answer_from_tool
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with final answer signal.[/dim]"
                        )
                        # Preview the Markdown formatting in verbose mode
                        self.console.print(
                            "[dim] -> Markdown preview of final answer:[/dim]"
                        )
                        preview_length = min(300, len(final_answer_from_tool))
                        preview = final_answer_from_tool[:preview_length]
                        if len(final_answer_from_tool) > preview_length:
                            preview += "..."
                        self.console.print(f"[cyan dim]{preview}[/cyan dim]")
                elif self.verbose:
                    self.console.print(
                        f"[yellow dim] -> finish_workflow tool ran but content was not a string: {tool_content}[/yellow dim]"
                    )

            else:
                if self.verbose:
                    self.console.print(
                        f"[yellow dim]Warning: Unrecognized tool name '{tool_name}' in update_state_node.[/yellow dim]"
                    )

        # End of processing all tool messages

        if not updates and self.verbose:
            self.console.print(
                f"[dim] -> No state updates generated from tool messages."
            )

        return updates

    def run_agentic_workflow(self, question: str) -> Dict[str, Any]:
        """Runs the agentic workflow using LangGraph to answer a question."""
        self.console.print(
            f"[bold]🚀 Starting LangGraph workflow for:[/bold] {question}"
        )

        # Generate a unique ID for this conversation thread
        # In a real application, you might use user IDs, session IDs, etc.
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # --- Add Initial System Prompt ---
        initial_system_prompt = """You are an AI assistant specialized in analyzing dbt projects to answer questions.
Your primary goal is to understand the user's question, find the relevant dbt models using the 'search_dbt_models' tool, consider past feedback with 'search_past_feedback' if applicable, and then generate a final answer.
The final answer, provided via the 'finish_workflow' tool, MUST include:
1. A SQL query that directly addresses the user's question based on the models found.
2. A clear explanation of the SQL query, including which models were used and why.
3. A mention of any potential limitations or assumptions made.

Format your answer using Markdown for improved readability:
- Use headings (# and ##) to organize different sections
- Put SQL code in code blocks with ```sql and ```
- Use bullet points (-) for listing models or key points
- Bold important information with **

Use the tools iteratively. Start by searching for models relevant to the core concepts in the question. Refine your search if the initial results are insufficient. Only use 'finish_workflow' when you have enough information to construct the final SQL query and explanation."""
        # --- End Initial System Prompt ---

        # Initial state
        initial_state = AgentState(
            original_question=question,
            # Prepend the system prompt and then add the user question
            messages=[
                SystemMessage(content=initial_system_prompt),
                HumanMessage(content=question),
            ],
            accumulated_models=[],
            accumulated_model_names=set(),
            relevant_feedback=[],
            search_model_calls=0,  # Initialize counter
            search_queries_tried=set(),  # We might not need this if LLM handles redundant calls
            final_answer=None,
            conversation_id=None,  # Will be set after recording
        )

        final_state = None
        try:
            # Invoke the graph
            # Stream events for better visibility (optional but recommended)
            # events = self.graph_app.stream( # Old way
            #     initial_state, config=config, stream_mode="values"
            # )
            # for event in events: # Old way
            #     # Process events if needed (e.g., print updates)
            #     # The final state is the last event yielded
            #     final_state = event
            #     if self.verbose:
            #         # Print the latest state or specific updates
            #         # (Be careful about printing too much, especially messages)
            #         self.console.print(
            #             f"[dim]Graph State Update: Keys={list(final_state.keys())}\[/dim]" # Old, had syntax warning
            #         )
            #         # Check if final answer is set
            #         if final_state.get("final_answer"):
            #             self.console.print(
            #                 "[green dim] -> Final answer received in state.[/green dim]"
            #             )

            # New way: Consume the stream into a list first
            all_states = []
            for state in self.graph_app.stream(
                initial_state, config=config, stream_mode="values"
            ):
                all_states.append(state)
                if self.verbose:
                    # Print the latest state or specific updates
                    self.console.print(
                        f"[dim]Graph State Update: Keys={list(state.keys())}[/dim]"  # Corrected syntax warning
                    )
                    # Check if final answer is set
                    if state.get("final_answer"):
                        self.console.print(
                            "[green dim] -> Final answer received in state.[/green dim]"
                        )

            # Check if any states were produced
            if not all_states:
                raise Exception("Graph execution finished without returning any state.")

            # The final state is the last one in the list
            final_state = all_states[-1]

            if final_state is None:
                raise Exception(
                    "Graph execution finished without returning a final state."
                )

            # Extract results from the final state
            final_answer = final_state.get(
                "final_answer", "Agent did not provide a final answer."
            )
            used_model_names = list(final_state.get("accumulated_model_names", set()))
            conversation_id_from_state = final_state.get(
                "conversation_id"
            )  # If set during graph

            self.console.print("[green]✅ LangGraph workflow finished.[/green]")

            # Pretty print the final answer to the console using Markdown if available
            if final_answer and isinstance(final_answer, str):
                self.console.print("\n[bold blue]📝 Final Answer:[/bold blue]")
                md = Markdown(final_answer)
                self.console.print(md)

            # Record the final question/answer pair (if storage is configured)
            conversation_id_recorded = None
            if self.question_storage:
                try:
                    # Ensure final_answer is a string before recording
                    answer_to_record = (
                        final_answer
                        if isinstance(final_answer, str)
                        else "Agent finished without providing a string answer."
                    )

                    conversation_id_recorded = self.question_storage.record_question(
                        question_text=question,
                        answer_text=answer_to_record,  # Use validated answer
                        model_names=used_model_names,
                        was_useful=None,  # No feedback yet
                        feedback=None,
                        metadata={
                            "agent_type": "langgraph",
                            "thread_id": thread_id,
                            # Add other relevant metadata from final_state if needed
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to record question/answer: {e}")
                    self.console.print(
                        "[yellow]⚠️ Could not record question details to storage.[/yellow]"
                    )

            return {
                "question": question,
                "final_answer": final_answer,
                "used_model_names": used_model_names,
                "conversation_id": conversation_id_recorded
                or conversation_id_from_state,
            }

        except Exception as e:
            logger.error(
                f"Error during LangGraph workflow execution: {str(e)}",
                exc_info=self.verbose,
            )

            # Check if this is an OpenAI message ordering error
            error_str = str(e)
            if "Invalid parameter: messages with role 'tool'" in error_str:
                self.console.print(
                    "[bold red]OpenAI API Message Ordering Error:[/bold red] Invalid message sequence detected."
                )
                self.console.print(
                    "[yellow]This is likely due to a ToolMessage without a preceding message with tool_calls.[/yellow]"
                )
                # Print last few messages if verbose and we have a final state
                if self.verbose and final_state and "messages" in final_state:
                    messages = final_state["messages"]
                    last_n = min(5, len(messages))
                    self.console.print(f"[dim]Last {last_n} messages:[/dim]")
                    for i, msg in enumerate(messages[-last_n:]):
                        msg_type = type(msg).__name__
                        has_tool_calls = hasattr(msg, "tool_calls") and bool(
                            msg.tool_calls
                        )
                        self.console.print(
                            f"[dim]  [{i}] {msg_type}"
                            + (f" (with tool_calls)" if has_tool_calls else "")
                            + f": {str(msg)[:100]}...[/dim]"
                        )
            else:
                self.console.print(
                    f"[bold red]Error during LangGraph workflow:[/bold red] {str(e)}"
                )

            # Try to get partial results if available
            used_models_on_error = (
                list(final_state.get("accumulated_model_names", set()))
                if final_state
                else []
            )
            search_calls_on_error = (
                final_state.get("search_model_calls", 0) if final_state else 0
            )
            error_details = str(e)

            # Log detailed state if verbose
            if self.verbose and final_state:
                self.console.print("[bold red]State at time of error:[/bold red]")
                import pprint

                self.console.print(f"[red dim]{pprint.pformat(final_state)}[/red dim]")
            elif self.verbose:
                self.console.print(
                    "[bold red]No final state available at time of error.[/bold red]"
                )

            return {
                "question": question,
                "final_answer": f"An error occurred during the LangGraph process after {search_calls_on_error} model searches: {error_details}",
                "used_model_names": used_models_on_error,
                "conversation_id": None,
            }

    def generate_documentation(self, model_name: str) -> Dict[str, Any]:
        """Generate documentation for a model.

        Args:
            model_name: Name of the model

        Returns:
            Dict containing the generated documentation
        """
        try:
            logger.info(f"Generating documentation for model: {model_name}")

            model = self.model_storage.get_model(model_name)
            if not model:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found",
                }

            prompt = f"""
            You are an AI assistant specialized in generating documentation for dbt models.
            You will be provided with information about a dbt model, and your task is to generate comprehensive documentation for it.
            
            Here is the information about the model:
            
            Model Name: {model_name}
            
            SQL Code:
            ```sql
            {model.raw_sql}
            ```
            
            Your documentation should include:
            1. A clear and concise description of what the model represents
            2. Descriptions for each column
            3. Information about any important business logic or transformations
            
            For each column, include:
            - What the column represents
            - The data type (if available)
            - Any important business rules or transformations
            
            Please format your response as follows:
            
            # Model Description
            [Your model description here]
            
            # Column Descriptions
            
            ## [Column Name 1]
            [Column 1 description]
            
            ## [Column Name 2]
            [Column 2 description]
            
            ...and so on for each column.
            """

            documentation = self.llm.get_completion(
                prompt=f"Generate documentation for the model {model_name}",
                system_prompt=prompt,
                max_tokens=1500,
            )

            model_description = ""
            column_descriptions = {}

            sections = documentation.split("# ")
            for section in sections:
                if section.startswith("Model Description"):
                    model_description = section.replace("Model Description", "").strip()
                elif section.startswith("Column Descriptions"):
                    column_section = section.replace("Column Descriptions", "").strip()
                    column_subsections = column_section.split("## ")
                    for col_subsection in column_subsections:
                        if col_subsection.strip():
                            col_lines = col_subsection.strip().split("\n", 1)
                            if len(col_lines) >= 2:
                                col_name = col_lines[0].strip()
                                col_desc = col_lines[1].strip()
                                column_descriptions[col_name] = col_desc

            return {
                "model_name": model_name,
                "description": model_description,
                "columns": column_descriptions,
                "raw_documentation": documentation,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error generating documentation for model {model_name}: {e}")
            return {
                "model_name": model_name,
                "error": f"Error generating documentation: {str(e)}",
                "success": False,
            }

    def update_model_documentation(
        self, model_name: str, documentation: Dict[str, Any]
    ) -> bool:
        """Update model documentation in the database.

        Args:
            model_name: Name of the model
            documentation: Documentation to update

        Returns:
            Whether the update was successful
        """
        try:
            model = self.model_storage.get_model(model_name)
            if not model:
                logger.error(f"Model {model_name} not found")
                return False

            model.description = documentation.get("description", model.description)

            if "columns" in documentation:
                for col_name, col_desc in documentation["columns"].items():
                    if col_name in model.columns:
                        model.columns[col_name].description = col_desc
                    else:
                        logger.warning(
                            f"Column {col_name} not found in model {model_name}"
                        )

            success = self.model_storage.update_model(model)
            return success

        except Exception as e:
            logger.error(f"Error updating model documentation: {e}")
            return False

    def interpret_model(
        self, model_name: str, max_verification_iterations: int = 1
    ) -> Dict[str, Any]:
        """Interpret a model and its columns using an agentic workflow.

        This method implements a step-by-step agentic approach to model interpretation:
        1. Read the source code of the model to interpret
        2. Identify upstream models that provide context
        3. Fetch details of the upstream models
        4. Create a draft interpretation
        5. Iteratively verify and refine the interpretation (configurable iterations):
           - Analyze upstream models' source code directly to ensure all columns are identified
           - Extract structured recommendations for columns to add, remove, or modify
           - Refine interpretation based on recommendations until verification passes or iterations complete
        6. Return final interpretation with column recommendations

        Args:
            model_name: Name of the model to interpret.
            max_verification_iterations: Maximum number of verification iterations to run (default: 1)

        Returns:
            Dict containing the interpreted documentation in YAML format and metadata,
            including structured column recommendations and verification iterations info.
        """
        try:
            logger.info(
                f"Starting agentic interpretation workflow for model: {model_name}"
            )

            # Step 1: Read the source code of the model to interpret
            model = self.model_storage.get_model(model_name)
            if not model:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found",
                    "success": False,
                }

            logger.info(f"Retrieved model {model_name} for interpretation")

            # Step 2: Identify upstream models from the model's source code
            logger.info(f"Analyzing upstream dependencies for {model_name}")

            # Get upstream models from the model's metadata
            upstream_model_names = model.all_upstream_models

            if not upstream_model_names:
                logger.warning(
                    f"No upstream models found for {model_name}. Using depends_on list."
                )
                upstream_model_names = model.depends_on

            logger.info(
                f"Found {len(upstream_model_names)} upstream models for {model_name}"
            )

            # Step 3: Fetch details of the upstream models for context
            upstream_models_data = {}
            upstream_info = ""
            for upstream_name in upstream_model_names:
                logger.info(f"Fetching details for upstream model: {upstream_name}")
                upstream_model = self.model_storage.get_model(upstream_name)
                if not upstream_model:
                    logger.warning(
                        f"Upstream model {upstream_name} not found. Skipping."
                    )
                    continue

                upstream_models_data[upstream_name] = upstream_model

                # Add upstream model information to context
                upstream_info += f"\n-- Upstream Model: {upstream_model.name} --\n"
                description = (
                    upstream_model.interpreted_description
                    or upstream_model.description
                    or "No description available."
                )
                upstream_info += f"Description: {description}\n"

                # Add column information from either interpreted columns or YML columns
                if upstream_model.interpreted_columns:
                    upstream_info += "Columns (from LLM interpretation):\n"
                    for (
                        col_name,
                        col_desc,
                    ) in upstream_model.interpreted_columns.items():
                        upstream_info += f"  - {col_name}: {col_desc}\n"
                elif upstream_model.columns:
                    upstream_info += "Columns (from YML):\n"
                    for col_name, col_obj in upstream_model.columns.items():
                        upstream_info += f"  - {col_name}: {col_obj.description or 'No description in YML'}\n"
                else:
                    upstream_info += "Columns: No column information available.\n"

                # Add SQL for context
                upstream_info += f"Raw SQL:\n```sql\n{upstream_model.raw_sql or 'SQL not available'}\n```\n"
                upstream_info += "-- End Upstream Model --\n"

            # Step 4: Create a draft interpretation using the model SQL and upstream info
            logger.info(f"Creating draft interpretation for {model_name}")

            prompt = MODEL_INTERPRETATION_PROMPT.format(
                model_name=model.name,
                model_sql=model.raw_sql,
                upstream_info=upstream_info,
            )

            draft_yaml_documentation = self.llm.get_completion(
                prompt=f"Interpret and generate YAML documentation for the model {model_name} based on its SQL and upstream dependencies",
                system_prompt=prompt,
                max_tokens=4000,
            )

            logger.debug(
                f"Draft interpretation for {model_name}:\n{draft_yaml_documentation}"
            )

            # Extract YAML content from the response
            match = re.search(
                r"```(?:yaml)?\n(.*?)```", draft_yaml_documentation, re.DOTALL
            )
            if match:
                draft_yaml_content = match.group(1).strip()
            else:
                # If no code block found, use the whole response but check for YAML tags
                draft_yaml_content = draft_yaml_documentation.strip()
                # Remove any potential YAML code fence markers at the beginning or end
                if draft_yaml_content.startswith("```yaml"):
                    draft_yaml_content = draft_yaml_content[7:]
                if draft_yaml_content.endswith("```"):
                    draft_yaml_content = draft_yaml_content[:-3]
                draft_yaml_content = draft_yaml_content.strip()

            logger.debug(
                f"Cleaned draft YAML content for {model_name}:\n{draft_yaml_content}"
            )

            # Parse the draft YAML to get column information
            try:
                # Additional safety check to ensure YAML content is clean
                if draft_yaml_content.startswith("```") or draft_yaml_content.endswith(
                    "```"
                ):
                    # Further cleaning if needed
                    lines = draft_yaml_content.strip().splitlines()
                    if lines and (
                        lines[0].startswith("```") or lines[0].startswith("---")
                    ):
                        lines = lines[1:]
                    if lines and (lines[-1] == "```" or lines[-1] == "---"):
                        lines = lines[:-1]
                    draft_yaml_content = "\n".join(lines).strip()

                draft_parsed = yaml.safe_load(draft_yaml_content)
                if draft_parsed is None:
                    logger.warning(
                        f"Draft YAML for {model_name} parsed as None, using empty dict"
                    )
                    draft_parsed = {}

                draft_model_data = draft_parsed.get("models", [{}])[0]
                draft_columns = draft_model_data.get("columns", [])
                if draft_columns is None:
                    draft_columns = []

                draft_column_names = [
                    col.get("name")
                    for col in draft_columns
                    if col and isinstance(col, dict) and "name" in col
                ]

                logger.info(
                    f"Draft interpretation contains {len(draft_column_names)} columns"
                )
            except Exception as e:
                logger.error(f"Error parsing draft YAML: {str(e)}")
                draft_columns = []
                draft_column_names = []

            # Step 5: Verify the draft interpretation against upstream models
            logger.info(
                f"Verifying draft interpretation for {model_name} against upstream models"
            )

            # Initialize results before the loop in case iterations = 0
            final_yaml_content = draft_yaml_content
            verification_result = "Verification skipped (iterations=0)"
            column_recommendations = {
                "columns_to_add": [],
                "columns_to_remove": [],
                "columns_to_modify": [],
            }
            iteration = -1  # To handle iteration + 1 in return value correctly

            # Enhanced verification with multiple iterations
            current_yaml_content = draft_yaml_content
            # final_yaml_content = draft_yaml_content # Already initialized above

            for iteration in range(max_verification_iterations):
                logger.info(
                    f"Starting verification iteration {iteration+1}/{max_verification_iterations}"
                )

                # Parse current YAML to get updated column information for verification prompt
                try:
                    current_parsed = yaml.safe_load(current_yaml_content)
                    if current_parsed is None:
                        logger.warning(
                            f"Current YAML for {model_name} parsed as None, using empty dict"
                        )
                        current_parsed = {}

                    current_model_data = current_parsed.get("models", [{}])[0]
                    current_columns = current_model_data.get("columns", [])
                    if current_columns is None:
                        current_columns = []

                    current_column_names = [
                        col.get("name")
                        for col in current_columns
                        if col and isinstance(col, dict) and "name" in col
                    ]
                    current_column_representation = (
                        ", ".join(current_column_names)
                        if current_column_names
                        else "No columns found"
                    )
                except Exception as e:
                    logger.error(
                        f"Error parsing current YAML in iteration {iteration+1}: {str(e)}"
                    )
                    current_column_representation = "Error parsing YAML"

                # Build a more detailed upstream model information with focus on SQL
                detailed_upstream_info = ""
                for upstream_name in upstream_model_names:
                    upstream_model = upstream_models_data.get(upstream_name)
                    if not upstream_model:
                        continue

                    detailed_upstream_info += (
                        f"\n-- Upstream Model: {upstream_model.name} --\n"
                    )

                    # Emphasize SQL analysis for column extraction
                    detailed_upstream_info += f"Raw SQL of {upstream_model.name}:\n```sql\n{upstream_model.raw_sql or 'SQL not available'}\n```\n"

                    # This model's columns - just as supplementary info
                    if upstream_model.interpreted_columns:
                        detailed_upstream_info += "Previously interpreted columns (for reference, SQL is definitive):\n"
                        for (
                            col_name,
                            col_desc,
                        ) in upstream_model.interpreted_columns.items():
                            detailed_upstream_info += f"  - {col_name}: {col_desc}\n"
                    elif upstream_model.columns:
                        detailed_upstream_info += (
                            "YML-defined columns (for reference, SQL is definitive):\n"
                        )
                        for col_name, col_obj in upstream_model.columns.items():
                            detailed_upstream_info += f"  - {col_name}: {col_obj.description or 'No description in YML'}\n"

                    detailed_upstream_info += "-- End Upstream Model --\n"

                verification_prompt = f"""
                You are validating a dbt model interpretation against its upstream model definitions to ensure it's complete and accurate.
                
                The model being interpreted is: {model_name}
                
                Original SQL of the model:
                ```sql
                {model.raw_sql}
                ```
                
                The current interpretation contains these columns:
                {current_column_representation}
                
                Current YAML content being verified (iteration {iteration+1}/{max_verification_iterations}):
                ```yaml
                {current_yaml_content}
                ```
                
                Here is information about the upstream models:
                {detailed_upstream_info}
                
                YOUR PRIMARY TASK IS TO COMPREHENSIVELY VERIFY THAT EVERY SINGLE COLUMN FROM THIS MODEL'S SQL OUTPUT IS CORRECTLY DOCUMENTED.
                
                Follow these specific steps in your verification:
                
                1. SQL ANALYSIS:
                   - Carefully trace the model's SQL to understand ALL columns in its output
                   - For any SELECT * statements, expand them by examining the source table/CTE's complete column list
                   - For JOINs, include columns from all joined tables that are in the SELECT
                   - For CTEs, carefully trace through each step to identify all columns
                
                2. UPSTREAM COLUMN VALIDATION:
                   - When a model uses SELECT * from an upstream model, carefully examine the SQL of that upstream model
                   - Count the total number of columns in each upstream model referenced with SELECT *
                   - Compare this count with the columns documented in the interpretation
                   - Missing columns in upstream models are the most common error - be extremely thorough
                
                3. COLUMN COUNT CHECK:
                   - Roughly estimate how many columns should appear in the output of this model
                   - Compare this estimate with the number of columns in the interpretation
                   - A significant discrepancy (e.g., interpretation has 5 columns but SQL output should have 60+) indicates missing columns
                
                4. COMPLETENESS CHECK:
                   - The interpretation must include EVERY column that will appear in the model's output
                   - Even if there are 50+ columns, all must be properly documented
                   - Any omission of columns is a critical error
                
                Based on your thorough analysis:
                
                If everything is correct, respond with "VERIFIED: The interpretation is complete and accurate."
                
                If there are issues, provide a structured response with the following format:
                
                VERIFICATION_RESULT:
                [Your general feedback and assessment here]
                [Include a count of total columns expected vs. documented]
                
                COLUMNS_TO_ADD:
                - name: [column_name_1]
                  description: [description]
                  reason: [reason this column should be added, specifically citing where in the SQL it comes from]
                - name: [column_name_2]
                  description: [description]
                  reason: [reason this column should be added, specifically citing where in the SQL it comes from]
                [List ALL missing columns, even if there are dozens]
                
                COLUMNS_TO_REMOVE:
                - name: [column_name_1]
                  reason: [reason this column should be removed, specifically citing evidence from the SQL]
                - name: [column_name_2]
                  reason: [reason this column should be removed, specifically citing evidence from the SQL]
                
                COLUMNS_TO_MODIFY:
                - name: [column_name_1]
                  current_description: [current description]
                  suggested_description: [suggested description]
                  reason: [reason for the change, with specific reference to the SQL]
                - name: [column_name_2]
                  current_description: [current description]
                  suggested_description: [suggested description]
                  reason: [reason for the change, with specific reference to the SQL]
                
                Only include sections that have actual entries (e.g., omit COLUMNS_TO_REMOVE if no columns need to be removed).
                """

                verification_result = self.llm.get_completion(
                    prompt=verification_prompt,
                    system_prompt="You are an AI assistant specialized in verifying dbt model interpretations. Your task is to carefully analyze SQL code to ensure all columns are correctly documented.",
                    max_tokens=4000,
                )

                logger.debug(
                    f"Verification result for {model_name} (iteration {iteration+1}):\n{verification_result}"
                )

                # Parse the verification result to extract recommended column changes
                column_recommendations = {
                    "columns_to_add": [],
                    "columns_to_remove": [],
                    "columns_to_modify": [],
                }

                # Only parse if verification found issues
                if "VERIFIED" not in verification_result:
                    logger.info(
                        f"Verification found issues for {model_name} in iteration {iteration+1}, extracting column recommendations"
                    )

                    # Extract columns to add
                    add_match = re.search(
                        r"COLUMNS_TO_ADD:\s*\n((?:.+\n)+?)(?:(?:COLUMNS_TO_REMOVE|COLUMNS_TO_MODIFY)|\Z)",
                        verification_result,
                        re.DOTALL,
                    )
                    if add_match:
                        add_section = add_match.group(1).strip()
                        # Parse the yaml-like format for columns to add
                        columns_to_add = []
                        current_column = {}
                        for line in add_section.split("\n"):
                            line = line.strip()
                            if line.startswith("- name:"):
                                if current_column and "name" in current_column:
                                    columns_to_add.append(current_column)
                                current_column = {
                                    "name": line.replace("- name:", "").strip()
                                }
                            elif line.startswith("description:") and current_column:
                                current_column["description"] = line.replace(
                                    "description:", ""
                                ).strip()
                            elif line.startswith("reason:") and current_column:
                                current_column["reason"] = line.replace(
                                    "reason:", ""
                                ).strip()
                        if current_column and "name" in current_column:
                            columns_to_add.append(current_column)
                        column_recommendations["columns_to_add"] = columns_to_add

                    # Extract columns to remove
                    remove_match = re.search(
                        r"COLUMNS_TO_REMOVE:\s*\n((?:.+\n)+?)(?:(?:COLUMNS_TO_ADD|COLUMNS_TO_MODIFY)|\Z)",
                        verification_result,
                        re.DOTALL,
                    )
                    if remove_match:
                        remove_section = remove_match.group(1).strip()
                        # Parse the yaml-like format for columns to remove
                        columns_to_remove = []
                        current_column = {}
                        for line in remove_section.split("\n"):
                            line = line.strip()
                            if line.startswith("- name:"):
                                if current_column and "name" in current_column:
                                    columns_to_remove.append(current_column)
                                current_column = {
                                    "name": line.replace("- name:", "").strip()
                                }
                            elif line.startswith("reason:") and current_column:
                                current_column["reason"] = line.replace(
                                    "reason:", ""
                                ).strip()
                        if current_column and "name" in current_column:
                            columns_to_remove.append(current_column)
                        column_recommendations["columns_to_remove"] = columns_to_remove

                    # Extract columns to modify
                    modify_match = re.search(
                        r"COLUMNS_TO_MODIFY:\s*\n((?:.+\n)+?)(?:(?:COLUMNS_TO_ADD|COLUMNS_TO_REMOVE)|\Z)",
                        verification_result,
                        re.DOTALL,
                    )
                    if modify_match:
                        modify_section = modify_match.group(1).strip()
                        # Parse the yaml-like format for columns to modify
                        columns_to_modify = []
                        current_column = {}
                        for line in modify_section.split("\n"):
                            line = line.strip()
                            if line.startswith("- name:"):
                                if current_column and "name" in current_column:
                                    columns_to_modify.append(current_column)
                                current_column = {
                                    "name": line.replace("- name:", "").strip()
                                }
                            elif (
                                line.startswith("current_description:")
                                and current_column
                            ):
                                current_column["current_description"] = line.replace(
                                    "current_description:", ""
                                ).strip()
                            elif (
                                line.startswith("suggested_description:")
                                and current_column
                            ):
                                current_column["suggested_description"] = line.replace(
                                    "suggested_description:", ""
                                ).strip()
                            elif line.startswith("reason:") and current_column:
                                current_column["reason"] = line.replace(
                                    "reason:", ""
                                ).strip()
                        if current_column and "name" in current_column:
                            columns_to_modify.append(current_column)
                        column_recommendations["columns_to_modify"] = columns_to_modify

                    logger.info(
                        f"Iteration {iteration+1}: Extracted column recommendations: {len(column_recommendations['columns_to_add'])} to add, "
                        f"{len(column_recommendations['columns_to_remove'])} to remove, "
                        f"{len(column_recommendations['columns_to_modify'])} to modify"
                    )

                    # If issues were found, refine the interpretation
                    logger.info(
                        f"Refining interpretation for {model_name} based on verification feedback (iteration {iteration+1})"
                    )

                    refinement_prompt = f"""
                    You are refining a dbt model interpretation based on verification feedback.
                    
                    Original model: {model_name}
                    
                    Original SQL:
                    ```sql
                    {model.raw_sql}
                    ```
                    
                    Current interpretation:
                    ```yaml
                    {current_yaml_content}
                    ```
                    
                    Verification feedback from iteration {iteration+1}:
                    {verification_result}
                    
                    Upstream model information:
                    {detailed_upstream_info}
                    
                    YOUR TASK IS TO CREATE A COMPLETE YAML INTERPRETATION THAT INCLUDES ALL COLUMNS FROM THE MODEL.
                    
                    This is absolutely critical:
                    1. ADD ALL MISSING COLUMNS identified in the verification feedback - you must include EVERY column
                    2. Even if there are 50+ columns to add, you must include all of them in your response
                    3. The most common error is not including all columns from upstream models when SELECT * is used
                    4. Be extremely thorough - lack of completeness is a critical issue
                    5. Remove any incorrect columns identified in COLUMNS_TO_REMOVE
                    6. Update descriptions for columns identified in COLUMNS_TO_MODIFY
                    
                    DO NOT OMIT ANY COLUMNS from your response - completeness is the highest priority.
                    
                    Your output should be complete, valid YAML for this model. Include all columns.
                    """

                    refined_yaml_documentation = self.llm.get_completion(
                        prompt=refinement_prompt,
                        system_prompt="You are an AI assistant specialized in refining dbt model interpretations based on SQL analysis.",
                        max_tokens=4000,
                    )

                    logger.debug(
                        f"Refined interpretation for {model_name} (iteration {iteration+1}):\n{refined_yaml_documentation}"
                    )

                    # Extract YAML content from the refined response
                    match = re.search(
                        r"```(?:yaml)?\n(.*?)```", refined_yaml_documentation, re.DOTALL
                    )
                    if match:
                        current_yaml_content = match.group(1).strip()
                    else:
                        # If no code block found, use the whole response but check for YAML tags
                        current_yaml_content = refined_yaml_documentation.strip()
                        # Remove any potential YAML code fence markers at the beginning or end
                        if current_yaml_content.startswith("```yaml"):
                            current_yaml_content = current_yaml_content[7:]
                        if current_yaml_content.endswith("```"):
                            current_yaml_content = current_yaml_content[:-3]
                        current_yaml_content = current_yaml_content.strip()

                    # Additional safety check for YAML content
                    if current_yaml_content.startswith(
                        "```"
                    ) or current_yaml_content.endswith("```"):
                        # Further cleaning if needed
                        lines = current_yaml_content.strip().splitlines()
                        if lines and (
                            lines[0].startswith("```") or lines[0].startswith("---")
                        ):
                            lines = lines[1:]
                        if lines and (lines[-1] == "```" or lines[-1] == "---"):
                            lines = lines[:-1]
                        current_yaml_content = "\n".join(lines).strip()

                    final_yaml_content = current_yaml_content

                    # If we found column issues but are not on last iteration, continue
                    has_column_changes = (
                        len(column_recommendations["columns_to_add"]) > 0
                        or len(column_recommendations["columns_to_remove"]) > 0
                        or len(column_recommendations["columns_to_modify"]) > 0
                    )

                    if (
                        has_column_changes
                        and iteration < max_verification_iterations - 1
                    ):
                        logger.info(
                            f"Moving to next verification iteration to check for additional issues"
                        )
                        continue
                else:
                    # Verification was successful - no issues found
                    logger.info(
                        f"Interpretation for {model_name} verified successfully in iteration {iteration+1}"
                    )
                    final_yaml_content = current_yaml_content
                    break

            # Store the verification history and column recommendations from the final iteration
            final_verification_result = verification_result
            final_column_recommendations = column_recommendations

            # Prepare the result
            result_data = {
                "model_name": model_name,
                "yaml_documentation": final_yaml_content,
                "draft_yaml": draft_yaml_content,
                "verification_result": final_verification_result,
                "column_recommendations": final_column_recommendations,
                "verification_iterations": iteration
                + 1,  # Will be 0 if loop didn't run
                "prompt": prompt,
                "success": True,
            }

            return result_data

        except Exception as e:
            logger.error(
                f"Error in agentic interpretation of model {model_name}: {e}",
                exc_info=True,
            )
            error_result = {
                "model_name": model_name,
                "error": f"Error in agentic interpretation: {str(e)}",
                "success": False,
            }
            return error_result

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
            # Add interpretation_details if needed (e.g., from agentic workflow)
            # model_record.interpretation_details = ...

            session.add(model_record)
            session.commit()
            logger.info(f"Successfully saved interpretation for model {model_name}")

            # Optional: Re-embed the model
            if embed:
                logger.info(
                    f"Re-embedding model {model_name} after saving interpretation"
                )
                # Fetch the updated DBTModel (using the existing storage method)
                updated_model = self.model_storage.get_model(model_name)
                if updated_model:
                    # Generate the text representation for embedding
                    model_text_for_embedding = updated_model.get_text_representation(
                        include_documentation=True
                    )
                    # Use the correct store_model_embedding method
                    self.vector_store.store_model_embedding(
                        model_name=updated_model.name,
                        model_text=model_text_for_embedding,
                        # Optionally add metadata if needed
                        # metadata=updated_model.to_dict() # Example if needed
                    )
                    logger.info(f"Successfully re-embedded model {model_name}")
                else:
                    logger.error(
                        f"Failed to fetch updated model {model_name} for re-embedding"
                    )
                    # Return success=True because saving worked, but warn about embedding
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
