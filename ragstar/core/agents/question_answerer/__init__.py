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

from ragstar.core.llm.client import LLMClient, TokenUsageLogger
from ragstar.storage.model_storage import ModelStorage
from ragstar.storage.model_embedding_storage import ModelEmbeddingStorage
from ragstar.core.models import DBTModel, ModelTable
from ragstar.utils.cli_utils import get_config_value

# --- NEW: Import prompt creation functions ---
from .prompts import create_system_prompt, create_guidance_message

from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


# --- Tool Input Schemas ---
class SearchModelsInput(BaseModel):
    query: str = Field(
        description="Concise, targeted query focusing on specific information needed to find relevant dbt models."
    )


# --- NEW: Input schema for fetching specific models ---
class FetchModelsInput(BaseModel):
    model_names: List[str] = Field(
        description="A list of specific dbt model names to retrieve details for."
    )


# --- END NEW ---


class SearchFeedbackInput(BaseModel):
    query: str = Field(
        description="The original user question to search for similar past questions and feedback."
    )


# --- NEW: Input schema for feedback content search ---
class SearchFeedbackContentInput(BaseModel):
    query: str = Field(
        description="Specific query about a concept, definition, or clarification to search within the text content of past feedback entries."
    )


# --- NEW: Input schema for organizational context search ---
class SearchOrganizationalContextInput(BaseModel):
    query: str = Field(
        description="Query based on the current question to find relevant definitions or context in past *original* user messages."
    )


# --- RE-ADD: FinishWorkflowInput schema ---
class FinishWorkflowInput(BaseModel):
    final_answer: str = Field(
        description="The comprehensive final answer text containing the SQL query, explanations (as comments), and any footnotes. Follow the SQL style guide provided in the prompt."
    )


# --- LangGraph State Definition ---
class QuestionAnsweringState(TypedDict):
    """Represents the state of our question answering agent graph."""

    original_question: str
    messages: Annotated[List[BaseMessage], add_messages]
    accumulated_models: List[Dict[str, Any]]  # Models found so far
    accumulated_model_names: Set[str]  # Names of models found
    vector_search_calls: int  # Track number of *vector* search calls
    relevant_feedback: Dict[str, List[Any]]  # Keys: 'by_question', 'by_content'
    # --- RE-ADD: final_answer field ---
    final_answer: Optional[str]
    conversation_id: Optional[str]
    # --- NEW: Add thread_context field ---
    thread_context: Optional[List[Dict[str, Any]]]  # Raw history from Slack
    # --- NEW: Add similar original messages field ---
    similar_original_messages: Optional[
        List[Dict[str, Any]]
    ]  # Store results from context search


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
        # --- NEW: Rename max search limit variable ---
        self.max_vector_searches = 5  # Max calls to model_similarity_search tool
        # --- END NEW ---
        # --- NEW: Store usable model names and descriptions ---
        self.all_models_summary: List[Dict[str, str]] = []
        try:
            # Get names of models marked as usable from the vector store
            usable_model_names = self.vector_store.get_answerable_model_names()

            # Fetch details for only these usable models from the main storage
            for model_name in usable_model_names:
                model = self.model_storage.get_model(model_name)
                if model:
                    self.all_models_summary.append(
                        {
                            "name": model.name,
                            "description": model.description
                            or "No description available.",
                        }
                    )
                else:
                    logger.warning(
                        f"Model '{model_name}' marked as usable in vector store but not found in main model storage."
                    )

            if self.verbose:
                self.console.print(
                    f"[dim]Loaded summary for {len(self.all_models_summary)} usable models.[/dim]"
                )

        except Exception as e:
            logger.error(
                f"Failed to load usable model summaries: {e}", exc_info=self.verbose
            )
            if self.verbose:
                self.console.print(
                    f"[red]Error loading usable model summaries: {e}[/red]"
                )
        # --- END NEW ---

        # Default to Postgres checkpointer
        # Assumes a function/config helper to get the connection string
        # You might need to adjust how the connection string is obtained
        pg_conn_string = get_config_value("database_url")

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

        # --- NEW: Tool to fetch specific model details ---
        @tool(args_schema=FetchModelsInput)
        def fetch_model_details(model_names: List[str]) -> List[Dict[str, Any]]:
            """Retrieves the full details for a specified list of dbt model names.
            Use this tool when you have identified potential models from the provided list of all models and need their schemas or other detailed information.
            Provide the exact names of the models you need details for.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: fetch_model_details(model_names={model_names})[/bold magenta]"
                )
            self.console.print(
                f"[bold blue]🔍 Fetching details for models: {model_names}[/bold blue]"
            )

            fetched_model_details = []
            for name in model_names:
                model = self.model_storage.get_model(name)
                if model:
                    model_dict = model.to_dict()
                    # Optionally add a marker to distinguish how it was fetched?
                    model_dict["fetch_method"] = "direct"
                    fetched_model_details.append(model_dict)
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Successfully fetched details for: {name}[/dim]"
                        )
                elif self.verbose:
                    self.console.print(
                        f"[yellow dim] -> Model not found in storage: {name}[/yellow dim]"
                    )

            if not fetched_model_details and self.verbose:
                self.console.print(
                    "[dim] -> No details found for the requested model names.[/dim]"
                )

            return fetched_model_details

        # --- END NEW ---

        # --- RENAMED: search_dbt_models -> model_similarity_search ---
        # --- MODIFIED: Schema, Description ---
        @tool(
            args_schema=SearchModelsInput
        )  # Keep SearchModelsInput for now, query is fine
        def model_similarity_search(query: str) -> List[Dict[str, Any]]:
            """Searches for relevant dbt models using vector similarity based on the provided query.
            Use this tool as a fallback if you cannot identify relevant models from the initial list provided,
            or if you need to find models related to a specific concept or calculation not obvious from model names/descriptions.
            Provide a concise, targeted query focusing on the *specific information* needed.
            Analyze the results. If you still need more information, consider refining the query.
            Do not use this tool if you have already found sufficient models or reached the search limit.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: model_similarity_search(query='{query}')[/bold magenta]"
                )
            self.console.print(
                f"[bold blue]🔍 Performing vector search for models relevant to: '{query}'[/bold blue]"
            )
            # --- END RENAMED/MODIFIED ---

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
                                f"[dim] -> Found relevant model via vector search: {model_name} (Score: {similarity:.2f})[/dim]"
                            )
                        # --- NEW: Add fetch_method marker ---
                        model_dict["fetch_method"] = "vector_search"
                        # --- END NEW ---

            if not newly_added_model_details and self.verbose:
                self.console.print(
                    "[dim] -> Vector store found models, but they were already known or below threshold.[/dim]"
                )

            # Return the list of newly found model details (dictionaries)
            return newly_added_model_details

        @tool(args_schema=SearchFeedbackInput)
        def search_past_feedback(query: str) -> List[Any]:
            """Searches for feedback on previously asked questions whose core topic is *similar* to the provided query.
            NOTE: Initial relevant feedback based on the original question is already provided in the state.
            Use this tool ONLY if you need to search for feedback related to a *different* or more *specific* topic that emerged during the workflow.
            Provide a concise query reflecting the new information need.
            The tool returns a list of relevant past question/feedback pairs."""
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: search_past_feedback(query='{query}') [/bold magenta]"
                )
                # ADDED: Log the query being used for feedback search
                self.console.print(
                    f"[dim]  -> Using query for feedback search: '{query}'[/dim]"
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
                self.console.print(
                    "[blue]🔍 Checking for feedback related to similar questions...[/blue]"
                )

            try:
                question_embedding = self.question_storage._get_embedding(query)
                if not question_embedding:
                    if self.verbose:
                        self.console.print(
                            "[yellow dim] -> Could not generate embedding for feedback search.[/yellow dim]"
                        )
                    return []  # Return empty list

                # ADDED: Log the similarity threshold
                similarity_threshold = 0.30  # Define threshold variable for logging
                if self.verbose:
                    self.console.print(
                        f"[dim]  -> Using similarity threshold: {similarity_threshold}[/dim]"
                    )

                relevant_feedback_items = (
                    self.question_storage.find_similar_questions_with_feedback(
                        query_embedding=question_embedding,
                        limit=3,
                        similarity_threshold=similarity_threshold,
                    )
                )

                if not relevant_feedback_items:
                    if self.verbose:
                        # MODIFIED: More explicit log for no results
                        self.console.print(
                            f"[dim] -> No similar questions found meeting threshold {similarity_threshold}.[/dim]"
                        )
                    return []  # Return empty list

                if self.verbose:
                    # MODIFIED: Log details of found items
                    self.console.print(
                        f"[dim] -> Found {len(relevant_feedback_items)} potentially relevant feedback item(s) via question similarity:[/dim]"
                    )
                    for i, item in enumerate(relevant_feedback_items):
                        q_text = getattr(item, "question_text", "N/A")
                        item_id = getattr(item, "id", "N/A")
                        # Note: Similarity score might not be directly available on the returned object here, depends on storage implementation.
                        # We'll just log the ID and question text for now.
                        self.console.print(
                            f"[dim]    [{i+1}] ID: {item_id}, Question: '{q_text[:50]}...'[/dim]"
                        )

                # Return the list of feedback items directly
                return relevant_feedback_items

            except Exception as e:
                logger.error(
                    f"Error during feedback search: {e}", exc_info=self.verbose
                )
                # ADDED: Log the error in verbose mode
                if self.verbose:
                    self.console.print(
                        f"[red dim] -> Error during feedback search: {e}[/red dim]"
                    )
                # Return empty list on error, the ToolNode might add an error message to history anyway
                return []

        # --- NEW: Tool to search feedback content ---
        @tool(args_schema=SearchFeedbackContentInput)
        def search_feedback_content(query: str) -> List[Any]:
            """Searches the *text content* of past feedback entries for specific information.
            NOTE: Initial relevant feedback content based on the original question is already provided in the state.
            Use this tool ONLY if you need clarification on a *different* or more *specific* term, concept, or definition
            that might have been mentioned in past feedback, beyond what was initially retrieved.
            Provide a targeted query for the specific information needed (e.g., 'definition of mandate', 'how to calculate watch time').
            The tool returns a list of past question/feedback pairs where the feedback content matched your query.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: search_feedback_content(query='{query}') [/bold magenta]"
                )
                # ADDED: Log the query being used for content search
                self.console.print(
                    f"[dim]  -> Using query for feedback *content* search: '{query}'[/dim]"
                )

            if not self.question_storage:
                if self.verbose:
                    self.console.print(
                        "[yellow dim] -> Feedback storage not configured.[/yellow dim]"
                    )
                return []

            if self.verbose:
                self.console.print(
                    f"[blue]🔍 Searching feedback content for: '{query}'...[/blue]"
                )

            try:
                # Use the new storage method
                # ADDED: Log the similarity threshold
                similarity_threshold = 0.6  # Define threshold variable for logging
                if self.verbose:
                    self.console.print(
                        f"[dim]  -> Using content similarity threshold: {similarity_threshold}[/dim]"
                    )

                relevant_feedback_items = (
                    self.question_storage.find_similar_feedback_content(
                        query=query,
                        limit=3,
                        similarity_threshold=similarity_threshold,
                    )
                )

                if not relevant_feedback_items:
                    if self.verbose:
                        # MODIFIED: More explicit log for no results
                        self.console.print(
                            f"[dim] -> No feedback content found matching query '{query}' with threshold {similarity_threshold}.[/dim]"
                        )
                    return []

                if self.verbose:
                    # MODIFIED: Log details of found items
                    self.console.print(
                        f"[dim] -> Found {len(relevant_feedback_items)} relevant feedback item(s) via content search:[/dim]"
                    )
                    for i, item in enumerate(relevant_feedback_items):
                        q_text = getattr(item, "question_text", "N/A")
                        f_text = getattr(item, "feedback", "N/A")
                        item_id = getattr(item, "id", "N/A")
                        # Note: Similarity score might not be directly available on the returned object here, depends on storage implementation.
                        self.console.print(
                            f"[dim]    [{i+1}] ID: {item_id}, Question: '{q_text[:50]}...', Feedback Snippet: '{f_text[:50]}...'[/dim]"
                        )

                # Return the list of feedback items (Question domain objects)
                return relevant_feedback_items

            except Exception as e:
                logger.error(
                    f"Error during feedback content search: {e}", exc_info=self.verbose
                )
                # ADDED: Log the error in verbose mode
                if self.verbose:
                    self.console.print(
                        f"[red dim] -> Error during feedback content search: {e}[/red dim]"
                    )
                return []

        # --- NEW: Tool to search organizational context ---
        @tool(args_schema=SearchOrganizationalContextInput)
        def search_organizational_context(query: str) -> List[Dict[str, Any]]:
            """Searches past *original user questions* for relevant context, definitions, or explanations.
            NOTE: Initial relevant context based on the original question is already provided in the state ('similar_original_messages').
            Use this tool ONLY if you need context related to a *different* or more *specific* term or concept than initially retrieved.
            Provide a query based on the specific term or concept you need context for (e.g., 'what is mandate ID?', 'definition of active user').
            Returns a list of past interactions (question/answer pairs) where the original question text was similar to your query.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: search_organizational_context(query='{query}') [/bold magenta]"
                )
                self.console.print(
                    f"[dim]  -> Using query for organizational context search: '{query}'[/dim]"
                )

            if not (
                self.question_storage
                and hasattr(self.question_storage, "_get_embedding")
                and hasattr(self.question_storage, "openai_client")
                and self.question_storage.openai_client is not None
            ):
                if self.verbose:
                    self.console.print(
                        "[yellow dim] -> Question storage not configured or embedding client unavailable for context search.[/yellow dim]"
                    )
                return []

            if self.verbose:
                self.console.print(
                    f"[blue]🔍 Searching past original questions for context related to: '{query}'...[/blue]"
                )

            try:
                query_embedding = self.question_storage._get_embedding(query)
                if not query_embedding:
                    if self.verbose:
                        self.console.print(
                            "[yellow dim] -> Could not generate embedding for context search query.[/yellow dim]"
                        )
                    return []

                # Use the new storage method
                similarity_threshold = 0.7  # Threshold for original message similarity
                if self.verbose:
                    self.console.print(
                        f"[dim]  -> Using similarity threshold: {similarity_threshold}[/dim]"
                    )

                relevant_interactions = (
                    self.question_storage.find_similar_original_messages(
                        query_embedding=query_embedding,
                        limit=3,
                        similarity_threshold=similarity_threshold,
                    )
                )

                if not relevant_interactions:
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> No relevant context found in past original questions matching query '{query}' with threshold {similarity_threshold}.[/dim]"
                        )
                    return []

                # Convert Question domain objects to dictionaries for state storage
                results_for_state = [q.to_dict() for q in relevant_interactions]

                if self.verbose:
                    self.console.print(
                        f"[dim] -> Found {len(results_for_state)} relevant interaction(s) via original message context search:[/dim]"
                    )
                    for i, item_dict in enumerate(results_for_state):
                        orig_q_text = item_dict.get("original_message_text", "N/A")
                        item_id = item_dict.get("id", "N/A")
                        self.console.print(
                            f"[dim]    [{i+1}] ID: {item_id}, Original Question: '{orig_q_text[:60]}...'[/dim]"
                        )

                return results_for_state

            except Exception as e:
                logger.error(
                    f"Error during organizational context search: {e}",
                    exc_info=self.verbose,
                )
                if self.verbose:
                    self.console.print(
                        f"[red dim] -> Error during context search: {e}[/red dim]"
                    )
                return []

        # --- RE-ADD: finish_workflow tool function ---
        @tool(args_schema=FinishWorkflowInput)
        def finish_workflow(final_answer: str) -> str:
            """Concludes the workflow and provides the final answer text to the user.
            Use this tool ONLY when you have gathered all necessary information from 'search_dbt_models',
            considered any relevant feedback from 'search_past_feedback' and 'search_feedback_content',
            and are ready to provide the complete, final answer text.
            The final answer MUST contain:
            1. A SQL query based strictly on retrieved models (following style guide).
            2. Explanations as comments within the SQL query or in a 'Footnotes' section.
            3. A 'Footnotes' section *after* the SQL block if needed for limitations or clarifying questions (no SQL comments for these).

            **CRITICAL: GROUNDING & ACCURACY:**
            - **DO NOT HALLUCINATE:** Base the SQL query and explanation strictly on the columns and relationships present in the retrieved dbt models. Never invent table or column names.
            - **HANDLE INCOMPLETENESS:** If models are insufficient, generate the best possible answer using *only* available information. Clearly state limitations in the SQL comments or 'Footnotes'.

            Ensure the final answer directly addresses the user's original question, acknowledging any limitations.
            Do NOT format the answer using Slack mrkdwn.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: finish_workflow(final_answer='{final_answer[:100]}...')",
                )
            # This function primarily acts as a signal, returning the final answer text
            return final_answer

        # Store the tools as instance attributes - ADD fetch_model_details, UPDATE search_dbt_models -> model_similarity_search
        self._tools = [
            fetch_model_details,  # NEW
            model_similarity_search,  # RENAMED
            search_past_feedback,
            search_feedback_content,
            search_organizational_context,  # NEW
            finish_workflow,
        ]

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
            tools_condition,  # Check if agent wants to call a tool
            {
                # If tool calls are present, route to the "tools" node
                "tools": "tools",
                # If no tool calls, end the graph (agent decided to finish)
                END: END,
            },
        )

        # Define edge for flow: tools -> update_state
        workflow.add_edge("tools", "update_state")

        # --- NEW: Conditional edge after updating state ---
        workflow.add_conditional_edges(
            "update_state",
            self.route_after_update,  # Check if final answer is set
            {
                "agent": "agent",  # If no final answer, loop back to agent
                END: END,  # If final answer is set, end the graph
            },
        )
        # --- REMOVED: Direct edge from update_state to agent ---
        # workflow.add_edge("update_state", "agent")

        # Compile the graph with memory
        # Use default recursion limit unless overridden
        return workflow.compile(checkpointer=self.memory)

    # --- End LangGraph Graph Construction ---

    # --- NEW: Conditional routing function after state update ---
    def route_after_update(self, state: QuestionAnsweringState) -> str:
        """Determines routing after the update_state node.
        Ends if final answer is present, otherwise goes back to agent.
        """
        if state.get("final_answer"):
            if self.verbose:
                self.console.print(
                    "[dim] -> Final answer set in state after update, ending workflow.[/dim]"
                )
            return END
        else:
            if self.verbose:
                self.console.print(
                    "[dim] -> No final answer set after update, looping back to agent.[/dim]"
                )
            return "agent"

    # --- End LangGraph Tools ---

    # --- LangGraph Nodes ---

    def _get_agent_llm(self):
        """Helper to get the LLM with tools bound."""
        chat_client_instance = self.llm.chat_client
        # Restore the use of bind_tools
        if hasattr(chat_client_instance, "bind_tools"):
            return chat_client_instance.bind_tools(self._tools)
        else:
            logger.warning(
                "LLM chat client does not have 'bind_tools'. Tool calling might not work as expected."
            )
            return chat_client_instance

    def agent_node(self, state: QuestionAnsweringState) -> Dict[str, Any]:
        """The main node that calls the LLM to decide the next action or generate the final answer.

        Args:
            state: The current state of the graph.

        Returns:
            A dictionary with updates to the state's messages list.
        """
        # --- NEW: Early exit if final answer is already set (belt-and-suspenders) ---
        # This shouldn't strictly be necessary with the new routing, but adds robustness.
        if state.get("final_answer"):
            if self.verbose:
                self.console.print(
                    "[bold yellow]>>> Entering agent_node (Workflow Finished - Early Exit) <<<[/bold yellow]"
                )
                self.console.print(
                    "[dim] -> Final answer already exists in state. Skipping LLM call and ending workflow.[/dim]"
                )
            # Return nothing to add to messages, let the END state be reached by routing
            return {"messages": []}
        # --- END NEW ---

        if self.verbose:
            self.console.print(
                "[bold yellow]\n>>> Entering agent_node <<<[/bold yellow]"
            )
            self.console.print(
                f"[dim]Received {len(state.get('messages', []))} messages in state:[/dim]"
            )
            for i, msg in enumerate(state.get("messages", [])):
                msg_type = type(msg).__name__
                content_preview = repr(msg.content)[:70] + (
                    "..." if len(repr(msg.content)) > 70 else ""
                )
                # Check for tool calls if it's an AIMessage
                tool_call_info = ""
                # --- MODIFIED: Log tool call details ---
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        tool_call_count = len(msg.tool_calls)
                        self.console.print(
                            f"[dim]  State[{i}] {msg_type} (requesting {tool_call_count} tool call{'s' if tool_call_count > 1 else ''}): {content_preview}[/dim]"
                        )
                        for tc in msg.tool_calls:
                            # Safely get name and args
                            tc_name = tc.get("name", "N/A")
                            tc_args = tc.get("args", {})
                            self.console.print(
                                f"[dim]    - Tool Call: {tc_name}, Args: {tc_args}[/dim]"
                            )
                    else:
                        self.console.print(
                            f"[dim]  State[{i}] {msg_type}: {content_preview}[/dim]"
                        )
                # --- END MODIFIED ---
                elif isinstance(msg, ToolMessage):
                    tool_call_info = f" for tool_call_id: {msg.tool_call_id[:8]}..."
                    self.console.print(
                        f"[dim]  State[{i}] {msg_type}{tool_call_info}: {content_preview}[/dim]"
                    )
                else:  # Handle other message types like HumanMessage, SystemMessage
                    self.console.print(
                        f"[dim]  State[{i}] {msg_type}: {content_preview}[/dim]"
                    )

        messages = state["messages"]
        original_question = state["original_question"]
        accumulated_models = state.get("accumulated_models", [])
        search_model_calls = state.get("vector_search_calls", 0)
        relevant_feedback_by_question = state.get("relevant_feedback", {}).get(
            "by_question", []
        )
        relevant_feedback_by_content = state.get("relevant_feedback", {}).get(
            "by_content", []
        )
        # --- NEW: Get similar original messages ---
        similar_original_messages = state.get("similar_original_messages", [])

        # --- RE-ADD: Check for final_answer ---
        final_answer = state.get("final_answer")
        # --- NEW: Get thread_context from state ---
        thread_context = state.get("thread_context")

        # --- System Prompt Setup ---
        # --- MODIFIED: Call prompt creation function (removed sql_style_example) --- #
        system_prompt = create_system_prompt(
            all_models_summary=self.all_models_summary,
            relevant_feedback_by_question=relevant_feedback_by_question,
            relevant_feedback_by_content=relevant_feedback_by_content,
            similar_original_messages=similar_original_messages,
            accumulated_models=accumulated_models,
            search_model_calls=search_model_calls,
            max_vector_searches=self.max_vector_searches,
        )
        # --- END MODIFIED Prompt ---

        # --- Guidance Logic ---
        # --- MODIFIED: Call guidance creation function ---
        guidance = create_guidance_message(
            search_model_calls=search_model_calls,
            max_vector_searches=self.max_vector_searches,
            accumulated_models=accumulated_models,
        )
        # --- END MODIFIED GUIDANCE ---

        # --- Combine system prompt and messages ---
        messages_for_llm: List[
            Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]
        ] = [SystemMessage(content=system_prompt)]

        # --- NEW: Add thread context as a separate message for clarity? ---
        # Or rely on it being in the system prompt description?
        # Let's add it explicitly after the system prompt.
        if thread_context:
            # Simple string representation for now
            context_str = "\n".join(
                [
                    f"{msg.get('user', 'Unknown')}: {msg.get('text', '')}"
                    for msg in thread_context
                ]
            )
            messages_for_llm.append(
                SystemMessage(
                    content=f"**Slack Thread Context:**\n```\n{context_str}\n```"
                )
            )
        # --- END NEW ---

        # Add original question if messages list is empty (first turn)
        # Note: original_question might now be slightly redundant if context has it,
        # but keeping it helps anchor the primary request.
        if not messages:
            messages_for_llm.append(HumanMessage(content=original_question))
        else:
            # Add existing message history (tool calls/results within QA workflow)
            messages_for_llm.extend(messages)

        # Add guidance as a final SystemMessage if needed
        # --- MODIFIED: Use the generated guidance string ---
        if guidance:
            # Add guidance if the last message isn't already the *exact same* guidance
            last_message = messages_for_llm[-1] if messages_for_llm else None
            if not (
                isinstance(last_message, SystemMessage)
                and last_message.content == guidance
            ):
                messages_for_llm.append(SystemMessage(content=guidance))
        # --- END MODIFIED ---

        if self.verbose:
            self.console.print("\n[blue]--- Calling Agent Model ---[/blue]")
            self.console.print(
                f"[dim]Current Models: {len(accumulated_models)} ({search_model_calls} searches used)[/dim]"
            )
            self.console.print(
                f"[dim]Found Feedback Items: {len(relevant_feedback_by_question)} (by question), {len(relevant_feedback_by_content)} (by content)[/dim]"
            )
            self.console.print(
                f"[dim]Found Org Context Items: {len(similar_original_messages)}"
            )
            self.console.print(
                f"[dim]Sending {len(messages_for_llm)} messages to LLM (including feedback and guidance)[/dim]"
            )
            # Optionally print the structure being sent
            self.console.print("[dim]Message structure *before* sending to LLM:[/dim]")
            for i, msg in enumerate(messages_for_llm):
                msg_type = type(msg).__name__
                content_preview = repr(msg.content)[:70] + (
                    "..." if len(repr(msg.content)) > 70 else ""
                )
                tool_call_info = ""
                # --- MODIFIED: Log tool call details before sending ---
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        tool_call_count = len(msg.tool_calls)
                        self.console.print(
                            f"[dim]  [{i}] {msg_type} (requesting {tool_call_count} tool call{'s' if tool_call_count > 1 else ''}): {content_preview}[/dim]"
                        )
                        for tc in msg.tool_calls:
                            # Safely get name and args
                            tc_name = tc.get("name", "N/A")
                            tc_args = tc.get("args", {})
                            self.console.print(
                                f"[dim]      - Tool Call: {tc_name}, Args: {tc_args}[/dim]"  # Indent further
                            )
                    else:
                        self.console.print(
                            f"[dim]  [{i}] {msg_type}: {content_preview}[/dim]"
                        )
                # --- END MODIFIED ---
                elif isinstance(msg, ToolMessage):
                    tool_call_info = f" for tool_call_id: {msg.tool_call_id[:8]}..."
                    self.console.print(
                        f"[dim]  [{i}] {msg_type}{tool_call_info}: {content_preview}[/dim]"
                    )
                else:  # Handle other message types like HumanMessage, SystemMessage
                    self.console.print(
                        f"[dim]  [{i}] {msg_type}: {content_preview}[/dim]"
                    )

        # --- MODIFIED: LLM Invocation with Callback (Tools are now bound) --- #
        agent_llm = self._get_agent_llm()  # Gets the client with tools bound
        token_logger = TokenUsageLogger()  # Instantiate the callback
        config = {
            "callbacks": [token_logger],
            "run_name": "QuestionAnswererAgentNode",  # Optional: Add a run name for tracing
        }
        try:
            # Remove tools=self._tools from invoke, as they are bound via _get_agent_llm
            response = agent_llm.invoke(messages_for_llm, config=config)
            if self.verbose:
                self.console.print(
                    f"[green]Agent Response:[/green] {response.content[:100]}..."
                )
                if response.tool_calls:
                    self.console.print(
                        f"[cyan]Tool calls:[/cyan] {response.tool_calls}"
                    )
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error invoking agent LLM: {e}", exc_info=self.verbose)
            error_message = AIMessage(content=f"LLM invocation failed: {str(e)}")
            return {"messages": [error_message]}
        # --- END MODIFIED --- #

    def update_state_node(self, state: QuestionAnsweringState) -> Dict[str, Any]:
        """Updates the state based on the results of the most recent tool calls.

        Processes ToolMessages, updates accumulated models, feedback, context, and search counts.

        Args:
            state: The current graph state.

        Returns:
            A dictionary containing updates to the state.
        """
        if self.verbose:
            self.console.print("\n[cyan]--- Updating State ---[/cyan]")

        updates: Dict[str, Any] = {}
        messages = state["messages"]
        accumulated_models = list(
            state.get("accumulated_models", [])
        )  # Get a mutable copy
        accumulated_model_names = set(
            state.get("accumulated_model_names", set())
        )  # Get a mutable copy
        search_model_calls = state.get("vector_search_calls", 0)
        # MODIFIED: Ensure relevant_feedback is initialized correctly
        relevant_feedback = state.get(
            "relevant_feedback", {}
        ).copy()  # Get mutable copy
        if "by_question" not in relevant_feedback:
            relevant_feedback["by_question"] = []
        if "by_content" not in relevant_feedback:
            relevant_feedback["by_content"] = []
        # --- NEW: Initialize context storage ---
        similar_original_messages = list(state.get("similar_original_messages", []))

        # --- Process Tool Messages ---
        # Find the most recent AIMessage with tool calls
        last_ai_message = None
        last_ai_message_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage) and messages[i].tool_calls:
                last_ai_message = messages[i]
                last_ai_message_index = i
                break

        if not last_ai_message:
            if self.verbose:
                self.console.print(
                    "[dim] -> No AIMessage with tool calls found to process.[/dim]"
                )
            return updates  # No tool results to process

        # Get all ToolMessages *after* the last AIMessage with tool calls
        tool_messages = []
        for i in range(last_ai_message_index + 1, len(messages)):
            if isinstance(messages[i], ToolMessage):
                tool_messages.append(messages[i])

        if not tool_messages:
            if self.verbose:
                self.console.print(
                    "[dim] -> No ToolMessages found after the last AIMessage with tool calls.[/dim]"
                )
            return updates

        if self.verbose:
            self.console.print(
                f"[dim] -> Processing {len(tool_messages)} recent tool messages[/dim]"
            )
            self.console.print(
                f"[dim] -> Found corresponding assistant message with tool_calls (id: {id(last_ai_message)})[/dim]"
            )
            self.console.print(
                f"[dim] -> Tool call IDs from assistant: {[tc['id'] for tc in last_ai_message.tool_calls]}[/dim]"
            )
            self.console.print(
                f"[dim] -> Tool call IDs from tool messages: {[tm.tool_call_id for tm in tool_messages]}[/dim]"
            )

        for tool_message in tool_messages:
            tool_name = tool_message.name
            content = (
                tool_message.content
            )  # Content is stringified result from ToolNode
            if self.verbose:
                content_summary = str(content)[:100] + (
                    "..." if len(str(content)) > 100 else ""
                )
                self.console.print(
                    f"[dim] -> Processing result from tool '{tool_name}': {content_summary}[/dim]"
                )

            try:
                # Attempt to parse content if it looks like JSON, otherwise use as string
                parsed_content = content
                if isinstance(content, str):
                    try:
                        import json

                        parsed_content = json.loads(content)
                    except json.JSONDecodeError:
                        pass  # Keep content as string if not valid JSON

                # --- Handle specific tool results ---
                if tool_name == "fetch_model_details":
                    search_model_calls += 1  # Increment search count
                    # Result should be a list of model dictionaries
                    if isinstance(parsed_content, list):
                        newly_added_count = 0
                        for model_detail in parsed_content:
                            if (
                                isinstance(model_detail, dict)
                                and "name" in model_detail
                            ):
                                model_name = model_detail["name"]
                                if model_name not in accumulated_model_names:
                                    accumulated_models.append(model_detail)
                                    accumulated_model_names.add(model_name)
                                    newly_added_count += 1
                        if self.verbose:
                            if newly_added_count > 0:
                                self.console.print(
                                    f"[dim] -> Updated state with {newly_added_count} new models. Total: {len(accumulated_models)}. Search calls: {search_model_calls}[/dim]"
                                )
                            else:
                                self.console.print(
                                    f"[dim] -> No new models added from this search. Total: {len(accumulated_models)}. Search calls: {search_model_calls}[/dim]"
                                )
                    else:
                        if self.verbose:
                            self.console.print(
                                f"[yellow dim]Warning: fetch_model_details returned non-list content: {type(parsed_content)}[/yellow dim]"
                            )

                elif tool_name == "model_similarity_search":
                    search_model_calls += 1  # Increment search count
                    # Result should be a list of model dictionaries
                    if isinstance(parsed_content, list):
                        newly_added_count = 0
                        for model_detail in parsed_content:
                            if (
                                isinstance(model_detail, dict)
                                and "name" in model_detail
                            ):
                                model_name = model_detail["name"]
                                if model_name not in accumulated_model_names:
                                    accumulated_models.append(model_detail)
                                    accumulated_model_names.add(model_name)
                                    newly_added_count += 1
                        if self.verbose:
                            if newly_added_count > 0:
                                self.console.print(
                                    f"[dim] -> Updated state with {newly_added_count} new models. Total: {len(accumulated_models)}. Search calls: {search_model_calls}[/dim]"
                                )
                            else:
                                self.console.print(
                                    f"[dim] -> No new models added from this search. Total: {len(accumulated_models)}. Search calls: {search_model_calls}[/dim]"
                                )
                    else:
                        if self.verbose:
                            self.console.print(
                                f"[yellow dim]Warning: model_similarity_search returned non-list content: {type(parsed_content)}[/yellow dim]"
                            )

                elif tool_name == "search_past_feedback":
                    # Result should be a list of feedback items (or empty)
                    if isinstance(parsed_content, list) and parsed_content:
                        relevant_feedback["by_question"] = (
                            parsed_content  # Replace or add?
                        )
                        if self.verbose:
                            self.console.print(
                                f"[dim] -> Updated state with {len(parsed_content)} feedback items found via similar questions.[/dim]"
                            )
                    elif self.verbose:
                        self.console.print(
                            "[dim] -> No feedback found via similar questions."
                        )

                # --- NEW: Handle feedback content search results ---
                elif tool_name == "search_feedback_content":
                    # Result should be a list of feedback items (or empty)
                    if isinstance(parsed_content, list) and parsed_content:
                        # Append results - might get duplicates if called multiple times?
                        # Or replace? Let's replace for now, assuming one content search is enough.
                        relevant_feedback["by_content"] = parsed_content
                        if self.verbose:
                            self.console.print(
                                f"[dim] -> Updated state with {len(parsed_content)} feedback items found via content search.[/dim]"
                            )
                    elif self.verbose:
                        self.console.print(
                            "[dim] -> No feedback found via content search."
                        )

                # --- NEW: Handle organizational context search results ---
                elif tool_name == "search_organizational_context":
                    # Result should be a list of Question dicts (or empty)
                    if isinstance(parsed_content, list):
                        # Append results? Or replace? Let's replace for simplicity.
                        similar_original_messages = parsed_content
                        if self.verbose:
                            self.console.print(
                                f"[dim] -> Updated state with {len(parsed_content)} items from organizational context search.[/dim]"
                            )
                    elif self.verbose:
                        self.console.print(
                            "[dim] -> No context found via original message search."
                        )
                # --- END NEW ---

                # --- RE-ADD: Handle finish_workflow ---
                elif tool_name == "finish_workflow":
                    # Content is the final answer string itself
                    updates["final_answer"] = parsed_content
                    if self.verbose:
                        self.console.print(
                            "[dim] -> Updated state with final answer.[/dim]"
                        )

                else:
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim]Warning: Unrecognized tool name '{tool_name}' in update_state_node[/yellow dim]"
                        )

            except Exception as e:
                logger.error(
                    f"Error processing tool ({tool_name}) result in update_state_node: {e}",
                    exc_info=self.verbose,
                )
                if self.verbose:
                    self.console.print(f"[red]Error processing tool result: {e}[/red]")
                # Decide if we should set an error state here?
                # updates["error_message"] = f"Internal error processing tool {tool_name}: {e}"

        # --- Update overall state fields ---
        updates["accumulated_models"] = accumulated_models
        updates["accumulated_model_names"] = accumulated_model_names
        updates["vector_search_calls"] = search_model_calls
        updates["relevant_feedback"] = relevant_feedback
        # --- NEW: Add context results to updates ---
        updates["similar_original_messages"] = similar_original_messages

        if self.verbose:
            if not updates:
                self.console.print(
                    "[dim] -> No state updates generated from tool messages.[/dim]"
                )
            # else:
            #     self.console.print(f"[dim] -> Generated updates: {updates.keys()}[/dim]") # Avoid printing large values

        return updates

    def run_agentic_workflow(
        self, question: str, thread_context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Runs the agentic workflow to answer a question.

        Args:
            question: The user's question compiled by the SlackResponder.
            thread_context: Optional list of message dicts from the Slack thread.

        Returns:
            A dictionary containing the final answer, context used, or an error message.
            Keys:
                - final_answer (Optional[str]): The generated SQL and explanation.
                - searched_models (Optional[List[Dict]]): Models considered.
                - relevant_feedback (Optional[Dict]): Feedback considered.
                - similar_original_messages (Optional[List[Dict]]): Context items considered.
                - error (Optional[str]): Error message if the workflow failed.
        """
        if self.verbose:
            self.console.print(
                f"[bold blue]🚀 Starting LangGraph workflow for: {question}[/bold blue]"
            )
            if thread_context:
                self.console.print(
                    f"[dim] -> Received thread context with {len(thread_context)} messages.[/dim]"
                )

        # Generate a unique thread ID for this execution
        thread_id = str(uuid.uuid4())
        if self.verbose:
            self.console.print(f"[dim]Generated thread ID: {thread_id}[/dim]")
            self.console.print("[dim]Initializing LangGraph workflow with memory[/dim]")

        config = {"configurable": {"thread_id": thread_id}}

        # --- BEGIN MODIFICATION: Pre-fetch context and feedback ---
        initial_similar_messages = []
        initial_feedback_by_question = []
        initial_feedback_by_content = []

        if self.question_storage:
            if self.verbose:
                self.console.print("[dim]Pre-fetching context and feedback...[/dim]")
            try:
                # Fetch similar original messages
                query_embedding = self.question_storage._get_embedding(question)
                if query_embedding:
                    # TODO: Make thresholds configurable?
                    initial_similar_messages = (
                        self.question_storage.find_similar_original_messages(
                            query_embedding=query_embedding,
                            limit=3,
                            similarity_threshold=0.7,
                        )
                        or []
                    )  # Ensure list
                    # Convert Question domain objects to dictionaries for state storage
                    initial_similar_messages = [
                        q.to_dict() for q in initial_similar_messages
                    ]
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Found {len(initial_similar_messages)} potentially relevant past original messages.[/dim]"
                        )

                    # Fetch similar questions with feedback
                    initial_feedback_by_question = (
                        self.question_storage.find_similar_questions_with_feedback(
                            query_embedding=query_embedding,
                            limit=3,
                            similarity_threshold=0.30,
                        )
                        or []
                    )  # Ensure list
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Found {len(initial_feedback_by_question)} potentially relevant past feedback items (by question).[/dim]"
                        )
                else:
                    if self.verbose:
                        self.console.print(
                            "[yellow dim] -> Could not generate embedding for initial context/feedback search.[/yellow dim]"
                        )

                # Fetch feedback by content similarity
                # TODO: Make threshold configurable?
                initial_feedback_by_content = (
                    self.question_storage.find_similar_feedback_content(
                        query=question, limit=3, similarity_threshold=0.6
                    )
                    or []
                )  # Ensure list
                if self.verbose:
                    self.console.print(
                        f"[dim] -> Found {len(initial_feedback_by_content)} potentially relevant past feedback items (by content).[/dim]"
                    )

            except Exception as e:
                logger.error(
                    f"Error pre-fetching context/feedback: {e}", exc_info=self.verbose
                )
                if self.verbose:
                    self.console.print(
                        f"[red dim] -> Error pre-fetching context/feedback: {e}[/red dim]"
                    )
        elif self.verbose:
            self.console.print(
                "[yellow dim] -> Question storage not available, skipping context/feedback pre-fetch.[/yellow dim]"
            )

        # --- END MODIFICATION ---

        # --- MODIFIED: Define the initial state including new field and pre-fetched data --- #
        initial_state = QuestionAnsweringState(
            original_question=question,  # This is the compiled question from SlackResponder
            messages=[],
            accumulated_models=[],
            accumulated_model_names=set(),
            vector_search_calls=0,
            # --- MODIFIED: Populate feedback from pre-fetch ---
            relevant_feedback={
                "by_question": initial_feedback_by_question,
                "by_content": initial_feedback_by_content,
            },
            final_answer=None,
            thread_context=thread_context,
            conversation_id=None,
            # --- MODIFIED: Populate messages from pre-fetch ---
            similar_original_messages=initial_similar_messages,
        )

        if self.verbose:
            self.console.print(
                "[dim]Initial state prepared with pre-fetched context/feedback.[/dim]"
            )

        try:
            # --- Invoke the graph --- #
            final_state_result = self.graph_app.invoke(initial_state, config=config)

            if final_state_result is None:
                logger.error(
                    "Graph execution finished but failed to retrieve final state."
                )
                return {
                    "error": "Graph execution failed to return a final state.",
                    "final_answer": None,
                    "searched_models": [],
                    "relevant_feedback": {},
                    "similar_original_messages": [],
                }

            # Extract results from the final state dictionary
            final_answer = final_state_result.get("final_answer")
            searched_models = final_state_result.get("accumulated_models", [])
            relevant_feedback_items = final_state_result.get("relevant_feedback", {})
            # --- NEW: Extract context results ---
            similar_messages_items = final_state_result.get(
                "similar_original_messages", []
            )

            if final_answer:
                if self.verbose:
                    self.console.print(
                        "[green]✅ Workflow finished successfully with final answer.[/green]"
                    )

                return {
                    "final_answer": final_answer,
                    "searched_models": searched_models,
                    "relevant_feedback": relevant_feedback_items,
                    "similar_original_messages": similar_messages_items,
                    "error": None,
                }
            else:
                # Check if there was an error message in the final state messages
                final_messages = final_state_result.get("messages", [])
                last_message_content = (
                    final_messages[-1].content if final_messages else ""
                )
                error_msg = "Workflow finished without a final answer."
                if "LLM invocation failed" in last_message_content:
                    error_msg = (
                        f"Workflow failed due to LLM error: {last_message_content}"
                    )
                elif "GraphRecursionError" in last_message_content:
                    # This case might not happen if the graph errors out before returning state
                    error_msg = f"Workflow failed due to recursion limit: {last_message_content}"

                logger.warning(
                    f"Workflow finished without a final answer. State: {final_state_result}"
                )
                return {
                    "error": error_msg,
                    "final_answer": None,
                    "searched_models": searched_models,
                    "relevant_feedback": relevant_feedback_items,
                    "similar_original_messages": similar_messages_items,
                }

        except Exception as e:
            logger.error(
                f"Error during LangGraph workflow execution: {str(e)}",
                exc_info=self.verbose,
            )
            # ADDED: Log the error in verbose mode
            if self.verbose:
                self.console.print(
                    f"[bold red]Error during LangGraph workflow:[/bold red] {str(e)}"
                )
            return {
                "error": f"An error occurred during the QuestionAnswerer workflow: {str(e)}",
                "final_answer": None,
                "searched_models": (
                    initial_state.get("accumulated_models", []) if initial_state else []
                ),
                "relevant_feedback": (
                    initial_state.get("relevant_feedback", {}) if initial_state else {}
                ),
                "similar_original_messages": (
                    initial_state.get("similar_original_messages", [])
                    if initial_state
                    else []
                ),
            }
