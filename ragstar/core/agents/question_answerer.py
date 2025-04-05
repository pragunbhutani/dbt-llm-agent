"""Agent for answering questions about dbt models."""

import logging
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
from typing_extensions import Annotated
from langgraph.graph.message import add_messages

from rich.console import Console
from rich.markdown import Markdown
from ragstar.storage.question_storage import QuestionStorage

from ragstar.core.llm.client import LLMClient
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


# --- LangGraph State Definition ---
class QuestionAnsweringState(TypedDict):
    """Represents the state of our question answering agent graph."""

    original_question: str
    messages: Annotated[List[BaseMessage], add_messages]
    accumulated_models: List[Dict[str, Any]]  # Models found so far
    accumulated_model_names: Set[str]  # Names of models found
    search_model_calls: int  # Track number of model search calls
    relevant_feedback: List[
        Any
    ]  # Feedback found (replace Any with specific type if available)
    search_queries_tried: Set[str]  # Track search queries
    final_answer: Optional[str]
    conversation_id: Optional[str]


# --- Tool Definitions ---
class QuestionAnswerer:
    """Agent for interacting with dbt projects using an agentic workflow to answer questions."""

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

        # Define tools and then compile the graph
        self._define_tools()
        self.graph_app = self._build_graph()

    def _define_tools(self):
        """Define the tools used by the agent."""

        @tool(args_schema=SearchModelsInput)
        def search_dbt_models(query: str) -> List[Dict[str, Any]]:
            """Searches for relevant dbt models based on the query.
            Use this tool iteratively to find dbt models that can help answer the user's question.
            Provide a concise, targeted query focusing on the *specific information* needed next.
            Analyze the results: if the found models are insufficient, call this tool again with a *refined* query.
            If you have enough information or have reached the search limit, call 'finish_workflow'.
            Do not use this tool if you have already found sufficient models or reached the search limit.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: search_dbt_models(query='{query}')[/bold magenta]"
                )
            self.console.print(
                f"[bold blue]🔍 Searching models relevant to: '{query}'[/bold blue]"
            )

            search_results = self.vector_store.search_models(query=query, n_results=5)
            newly_added_model_details = []

            if not search_results:
                if self.verbose:
                    self.console.print(
                        "[dim] -> No models found by vector store for this query.[/dim]"
                    )
                return []  # Return empty list

            for result in search_results:
                model_name = result["model_name"]
                similarity = result.get("similarity_score", 0)

                if isinstance(similarity, (int, float)) and similarity > 0.3:
                    model = self.model_storage.get_model(model_name)
                    if model:
                        model_dict = model.to_dict()
                        model_dict["search_score"] = similarity
                        newly_added_model_details.append(model_dict)
                        if self.verbose:
                            self.console.print(
                                f"[dim] -> Found relevant new model: {model_name} (Score: {similarity:.2f})[/dim]"
                            )

            if not newly_added_model_details and self.verbose:
                self.console.print(
                    "[dim] -> Vector store found models, but they were already known or below threshold.[/dim]"
                )

            # Return the list of newly found model details (dictionaries)
            return newly_added_model_details

        @tool(args_schema=SearchFeedbackInput)
        def search_past_feedback(query: str) -> List[Any]:
            """Searches for feedback on previously asked similar questions.
            Use this ONCE at the beginning of the workflow if the user's question might have been asked before.
            Provide the original user question as the query.
            The tool returns a summary of relevant feedback found."""
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: search_past_feedback(query='{query}') [/bold magenta]"
                )

            if not (
                self.question_storage
                and hasattr(self.question_storage, "_get_embedding")
                and hasattr(self.question_storage, "openai_client")
                and self.question_storage.openai_client is not None
            ):
                if self.verbose:
                    self.console.print(
                        "[yellow dim] -> Feedback storage not configured or embedding client unavailable.[/yellow dim]"
                    )
                return []  # Return empty list

            if self.verbose:
                self.console.print("[blue]🔍 Checking for feedback...[/blue]")

            try:
                question_embedding = self.question_storage._get_embedding(query)
                if not question_embedding:
                    if self.verbose:
                        self.console.print(
                            "[yellow dim] -> Could not generate embedding for feedback search.[/yellow dim]"
                        )
                    return []  # Return empty list

                relevant_feedback_items = (
                    self.question_storage.find_similar_questions_with_feedback(
                        query_embedding=question_embedding,
                        limit=3,
                        similarity_threshold=0.75,
                    )
                )

                if not relevant_feedback_items:
                    if self.verbose:
                        self.console.print("[dim] -> No relevant feedback found.[/dim]")
                    return []  # Return empty list

                if self.verbose:
                    self.console.print(
                        f"[dim] -> Found {len(relevant_feedback_items)} relevant feedback item(s).[/dim]"
                    )

                # Return the list of feedback items directly
                return relevant_feedback_items

            except Exception as e:
                logger.error(
                    f"Error during feedback search: {e}", exc_info=self.verbose
                )
                # Return empty list on error, the ToolNode might add an error message to history anyway
                return []

        @tool(args_schema=FinishWorkflowInput)
        def finish_workflow(final_answer: str) -> str:
            """Concludes the workflow and provides the final answer to the user.
            Use this tool ONLY when you have gathered all necessary information from 'search_dbt_models',
            considered any relevant feedback from 'search_past_feedback',
            and are ready to provide the complete, final answer (including SQL query and explanation).
            Format your answer using Markdown for improved readability:
            - Use headings (# and ##) to organize different sections
            - Put SQL code in code blocks with ```sql and ```
            - Use bullet points (-) for listing models or key points
            - Bold important information with **
            Ensure the answer directly addresses the user's original question."""
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: finish_workflow(final_answer='{final_answer[:100]}...')",
                )
            # This function primarily acts as a signal, returning the final answer
            return final_answer

        # Store the tools as instance attributes
        self._tools = [search_dbt_models, search_past_feedback, finish_workflow]

    # --- LangGraph Graph Construction ---
    def _build_graph(self):
        """Builds the LangGraph StateGraph."""
        workflow = StateGraph(QuestionAnsweringState)

        # MODIFIED: Use standard ToolNode instead of SafeToolExecutor
        tool_node = ToolNode(self._tools, handle_tool_errors=True)

        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", tool_node)
        # Add node for state updates after tools run
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

        # Define edges for flow: tools -> update_state -> agent
        workflow.add_edge("tools", "update_state")
        workflow.add_edge("update_state", "agent")

        # Compile the graph with memory
        return workflow.compile(checkpointer=self.memory)

    # --- End LangGraph Graph Construction ---

    # --- DELETED LangGraph Tools ---
    # The tool logic is now outside the class

    # --- End LangGraph Tools ---

    # --- LangGraph Nodes ---

    def _get_agent_llm(self):
        """Helper to get the LLM with tools bound."""
        chat_client_instance = self.llm.chat_client

        if hasattr(chat_client_instance, "bind_tools"):
            return chat_client_instance.bind_tools(self._tools)
        else:
            logger.warning(
                "LLM chat client does not have 'bind_tools'. Tool calling might not work as expected."
            )
            return chat_client_instance

    def agent_node(self, state: QuestionAnsweringState) -> Dict[str, Any]:
        """Calls the agent LLM to decide the next action or generate the final answer."""
        # === ADD LOGGING HERE ===
        if self.verbose:
            incoming_messages = state.get("messages", [])
            self.console.print(
                f"[bold yellow]>>> Entering agent_node <<<[/bold yellow]"
            )
            self.console.print(
                f"[dim]Received {len(incoming_messages)} messages in state:[/dim]"
            )
            for i, msg in enumerate(incoming_messages):
                msg_type = type(msg).__name__
                content_preview = (
                    str(msg.content)[:50] + "..."
                    if len(str(msg.content)) > 50
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
                    f"  State[{i}] {msg_type}{tool_calls_info}{tool_id_info}: {content_preview}"
                )
        # === END LOGGING ===

        if self.verbose:
            self.console.print("[bold green]\n--- Calling Agent Model ---[/bold green]")
            self.console.print(
                f"[dim]Current Models: {len(state['accumulated_model_names'])} ({state.get('search_model_calls', 0)} searches used)[/dim]"
            )
            self.console.print(
                f"[dim]Found Feedback: {len(state['relevant_feedback'])}[/dim]"
            )

        # Get the current messages directly from state
        messages = state.get("messages", [])

        # If we don't have any messages (first call), start with the original question
        if not messages:
            messages_for_llm = [
                SystemMessage(
                    content="You are an AI assistant specialized in analyzing dbt projects to answer questions."
                ),
                HumanMessage(content=state["original_question"]),
            ]
        else:
            # Use the messages from state for subsequent calls
            messages_for_llm = list(messages)  # Make a copy

        # Add guidance as a system message (only if not already the last message type)
        current_search_calls = state.get("search_model_calls", 0)
        remaining_searches = self.max_model_searches - current_search_calls

        guidance_content = ""
        if remaining_searches > 0:
            guidance_content = (
                f"Guidance: You have used the model search tool {current_search_calls} times. "
                f"You have {remaining_searches} remaining searches for dbt models. "
                f"Current models found: {len(state['accumulated_model_names'])}. "
                f"Original question: '{state['original_question']}'. "
                f"If the current models are insufficient to answer the question comprehensively, "
                f"use 'search_dbt_models' again with a *refined, specific* query focusing on the missing information. "
                f"Otherwise, if you have enough information, use 'finish_workflow' to provide the final answer (SQL query + explanation)."
            )
        else:
            guidance_content = (
                f"Guidance: You have reached the maximum limit of {self.max_model_searches} model searches. "
                f"You must now synthesize an answer using the models found ({len(state['accumulated_model_names'])} models: "
                f"{', '.join(state['accumulated_model_names']) or 'None'}) "
                f"and call 'finish_workflow'. Provide the final SQL query and explanation. Do not call 'search_dbt_models' again."
            )

        # Add guidance SystemMessage if needed
        if not (messages_for_llm and isinstance(messages_for_llm[-1], HumanMessage)):
            # Avoid adding duplicate guidance if the last message already provides it
            if not (
                messages_for_llm
                and isinstance(messages_for_llm[-1], SystemMessage)
                and messages_for_llm[-1].content.startswith("Guidance:")
            ):
                messages_for_llm.append(SystemMessage(content=guidance_content))

        agent_llm = self._get_agent_llm()

        if self.verbose:
            self.console.print(
                f"[blue dim]Sending {len(messages_for_llm)} messages to LLM with guidance[/blue dim]"
            )
            # Log message structure *before* sending (this includes guidance)
            self.console.print("[dim]Message structure *before* sending to LLM:[/dim]")
            for i, msg in enumerate(messages_for_llm):
                msg_type = type(msg).__name__
                content_preview = (
                    str(msg.content)[:50] + "..."
                    if len(str(msg.content)) > 50
                    else str(msg.content)
                )
                tool_calls_info = ""
                tool_id_info = ""

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_info = (
                        f" [bold]with {len(msg.tool_calls)} tool_calls[/bold]"
                    )

                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    tool_id_info = (
                        f" [bold]responding to tool_call_id: {msg.tool_call_id}[/bold]"
                    )

                self.console.print(
                    f"  [{i}] {msg_type}{tool_calls_info}{tool_id_info}: {content_preview}"
                )

            # Removed custom verification call
            # self._verify_message_structure(messages_for_llm)

        try:
            response = agent_llm.invoke(messages_for_llm)

            if self.verbose:
                self.console.print(
                    f"[dim]Agent Response: {response.content[:100]}...[/dim]"
                )
                if hasattr(response, "tool_calls") and response.tool_calls:
                    self.console.print(f"[dim]Tool calls: {response.tool_calls}[/dim]")

            # Return *only* the new response to be appended by LangGraph
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error invoking agent LLM: {e}", exc_info=self.verbose)
            error_message = AIMessage(content=f"LLM invocation failed: {str(e)}")
            # Return the error message to be appended
            return {"messages": [error_message]}

    def update_state_node(self, state: QuestionAnsweringState) -> Dict[str, Any]:
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

        # Find all recent ToolMessages - the standard ToolNode would have added these
        # We're looking for ToolMessages that haven't been processed yet
        recent_tool_messages = []
        i = len(messages) - 1
        last_assistant_with_tool_calls = None

        # Start from the end and work backwards to find the most recent tool messages
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
            if last_assistant_with_tool_calls:
                self.console.print(
                    f"[dim] -> Found corresponding assistant message with tool_calls (id: {id(last_assistant_with_tool_calls)})[/dim]"
                )
                # Log tool call IDs from the assistant message
                if hasattr(last_assistant_with_tool_calls, "tool_calls"):
                    tool_call_ids = [
                        tc["id"]
                        for tc in last_assistant_with_tool_calls.tool_calls
                        if "id" in tc
                    ]
                    self.console.print(
                        f"[dim] -> Tool call IDs from assistant: {tool_call_ids}[/dim]"
                    )
                    # Check if all tool messages have matching tool_call_ids
                    tool_message_ids = [
                        tm.tool_call_id
                        for tm in recent_tool_messages
                        if hasattr(tm, "tool_call_id")
                    ]
                    self.console.print(
                        f"[dim] -> Tool call IDs from tool messages: {tool_message_ids}[/dim]"
                    )
            else:
                self.console.print(
                    f"[yellow dim] -> WARNING: Could not find corresponding assistant message with tool_calls[/yellow dim]"
                )

        # Process each tool message
        for tool_message in recent_tool_messages:
            tool_name = tool_message.name
            content = (
                tool_message.content
            )  # This might be stringified JSON for some tools

            if self.verbose:
                content_summary = str(content)[:200] + (
                    "..." if len(str(content)) > 200 else ""
                )
                self.console.print(
                    f"[dim] -> Processing result from tool '{tool_name}': {content_summary}[/dim]"
                )

            # --- Process based on which tool was called ---

            if tool_name == "search_dbt_models":
                # Parse the content if it's a stringified list
                try:
                    if isinstance(content, str):
                        import json

                        newly_found_models = json.loads(content)
                    else:
                        newly_found_models = content

                    if not isinstance(newly_found_models, list):
                        newly_found_models = []
                except Exception as e:
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim]Error parsing search_dbt_models result: {e}[/yellow dim]"
                        )
                    newly_found_models = []

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
                                f"[dim] -> Updated state with {len(unique_new_models)} new models. Total: {len(current_model_names)}. Search calls: {updates['search_model_calls']}"
                            )
                    elif self.verbose:
                        self.console.print(
                            f"[dim] -> No new unique models found. Search calls: {updates['search_model_calls']}"
                        )
                elif self.verbose:
                    self.console.print(
                        f"[dim] -> No models returned. Search calls: {updates['search_model_calls']}"
                    )

            elif tool_name == "search_past_feedback":
                # Parse the content if it's a stringified list
                try:
                    if isinstance(content, str):
                        import json

                        newly_found_feedback = json.loads(content)
                    else:
                        newly_found_feedback = content

                    if not isinstance(newly_found_feedback, list):
                        newly_found_feedback = []
                except Exception as e:
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim]Error parsing feedback result: {e}[/yellow dim]"
                        )
                    newly_found_feedback = []

                if newly_found_feedback:
                    current_feedback = state.get("relevant_feedback", [])
                    updates["relevant_feedback"] = (
                        current_feedback + newly_found_feedback
                    )
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with {len(newly_found_feedback)} new feedback items."
                        )
                elif self.verbose:
                    self.console.print("[dim] -> No feedback found.")

            elif tool_name == "finish_workflow":
                # Tool content should be the final answer string
                if content:
                    updates["final_answer"] = content
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with final answer.[/dim]"
                        )
                        # Preview the Markdown formatting in verbose mode
                        self.console.print(
                            "[dim] -> Markdown preview of final answer:[/dim]"
                        )
                        preview_length = min(300, len(content))
                        preview = content[:preview_length]
                        if len(content) > preview_length:
                            preview += "..."
                        self.console.print(f"[cyan dim]{preview}[/cyan dim]")
                elif self.verbose:
                    self.console.print(
                        f"[yellow dim] -> finish_workflow tool ran but returned empty content[/yellow dim]"
                    )

            else:
                if self.verbose:
                    self.console.print(
                        f"[yellow dim]Warning: Unrecognized tool name '{tool_name}'[/yellow dim]"
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
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        if self.verbose:
            self.console.print(f"[dim]Generated thread ID: {thread_id}[/dim]")
            self.console.print("[dim]Initializing LangGraph workflow with memory[/dim]")

        # Initial system prompt - important for guiding the LLM
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

        # Initial state
        initial_state = QuestionAnsweringState(
            original_question=question,
            messages=[
                SystemMessage(content=initial_system_prompt),
                HumanMessage(content=question),
            ],
            accumulated_models=[],
            accumulated_model_names=set(),
            relevant_feedback=[],
            search_model_calls=0,
            search_queries_tried=set(),
            final_answer=None,
            conversation_id=None,
        )

        if self.verbose:
            self.console.print(
                "[dim]Initial state prepared with system prompt and question[/dim]"
            )

        try:
            # Get the final state by executing the graph
            final_state = self.graph_app.invoke(initial_state, config=config)

            # Extract results
            final_answer = final_state.get(
                "final_answer", "Agent did not provide a final answer."
            )
            used_model_names = list(final_state.get("accumulated_model_names", set()))
            message_count = len(final_state.get("messages", []))

            self.console.print("[green]✅ LangGraph workflow finished.[/green]")

            if self.verbose:
                self.console.print(
                    f"[dim]Final state contains {message_count} messages[/dim]"
                )
                self.console.print(
                    f"[dim]Used {len(used_model_names)} models: {', '.join(used_model_names)}[/dim]"
                )

            # Pretty print the final answer using Markdown
            if final_answer and isinstance(final_answer, str):
                self.console.print("\n[bold blue]📝 Final Answer:[/bold blue]")
                md = Markdown(final_answer)
                self.console.print(md)

            # Record the final question/answer pair if storage is configured
            conversation_id = None
            if self.question_storage:
                try:
                    answer_to_record = (
                        final_answer
                        if isinstance(final_answer, str)
                        else "No answer provided."
                    )

                    conversation_id = self.question_storage.record_question(
                        question_text=question,
                        answer_text=answer_to_record,
                        model_names=used_model_names,
                        was_useful=None,  # No feedback yet
                        feedback=None,
                        metadata={
                            "agent_type": "langgraph_question_answerer",
                            "thread_id": thread_id,
                        },
                    )

                    if self.verbose:
                        self.console.print(
                            f"[dim]Recorded conversation with ID: {conversation_id}[/dim]"
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
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.error(
                f"Error during LangGraph workflow execution: {str(e)}",
                exc_info=self.verbose,
            )
            self.console.print(
                f"[bold red]Error during LangGraph workflow:[/bold red] {str(e)}"
            )

            return {
                "question": question,
                "final_answer": f"An error occurred during the workflow: {str(e)}",
                "used_model_names": [],
                "conversation_id": None,
            }
