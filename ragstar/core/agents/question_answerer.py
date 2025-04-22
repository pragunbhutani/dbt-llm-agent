"""Agent for answering questions about dbt models."""

import logging
from typing import Dict, List, Any, Optional, Union, Set, TypedDict
import uuid
import pathlib

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


# --- NEW: Input schema for feedback content search ---
class SearchFeedbackContentInput(BaseModel):
    query: str = Field(
        description="Specific query about a concept, definition, or clarification to search within the text content of past feedback entries."
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
    search_model_calls: int  # Track number of model search calls
    relevant_feedback: Dict[str, List[Any]]  # Keys: 'by_question', 'by_content'
    search_queries_tried: Set[str]  # Track search queries
    # --- RE-ADD: final_answer field ---
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

        # --- ADDED: Load SQL style example ---
        try:
            # Construct path relative to the current file
            style_file_path = pathlib.Path(__file__).parent / "sql_style_example.sql"
            if style_file_path.is_file():
                self.sql_style_example = style_file_path.read_text()
                if self.verbose:
                    self.console.print(
                        f"[dim]Loaded SQL style example from: {style_file_path}[/dim]"
                    )
            else:
                self.sql_style_example = None
                logger.warning(
                    f"SQL style example file not found at: {style_file_path}"
                )
                if self.verbose:
                    self.console.print(
                        f"[yellow]Warning: SQL style example file not found at: {style_file_path}[/yellow]"
                    )
        except Exception as e:
            self.sql_style_example = None
            logger.error(f"Error loading SQL style example: {e}", exc_info=self.verbose)
            if self.verbose:
                self.console.print(f"[red]Error loading SQL style example: {e}[/red]")
        # --- END ADDED ---

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
            If you have enough information or have reached the search limit, prepare to generate the structured output.
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
            """Searches for feedback on previously asked questions whose core topic is *similar* to the provided query.
            Use this ONCE at the beginning of the workflow if relevant feedback might exist.
            Provide a concise query reflecting the core information need (e.g., the same refined query used for model search).
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
            Use this tool *during* the workflow if you need clarification on a specific term, concept, or definition
            that might have been mentioned in past feedback, even if related to a different original question.
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

        # Store the tools as instance attributes - RE-ADD finish_workflow
        self._tools = [
            search_dbt_models,
            search_past_feedback,
            search_feedback_content,
            finish_workflow,  # Re-added
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
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    tool_call_info = f" with {len(msg.tool_calls)} tool_calls"
                elif isinstance(msg, ToolMessage):
                    tool_call_info = f" for tool_call_id: {msg.tool_call_id[:8]}..."
                self.console.print(
                    f"[dim]  State[{i}] {msg_type}{tool_call_info}: {content_preview}[/dim]"
                )

        messages = state["messages"]
        original_question = state["original_question"]
        accumulated_models = state.get("accumulated_models", [])
        search_model_calls = state.get("search_model_calls", 0)
        relevant_feedback_by_question = state.get("relevant_feedback", {}).get(
            "by_question", []
        )
        relevant_feedback_by_content = state.get("relevant_feedback", {}).get(
            "by_content", []
        )
        # --- RE-ADD: Check for final_answer ---
        final_answer = state.get("final_answer")

        # --- System Prompt Setup ---
        system_prompt = f"""You are an AI assistant specialized in analyzing dbt projects and generating SQL queries based ONLY on provided dbt model context.

        **Overall Goal:** Answer the user's question by:
        1. Understanding the question and identifying necessary information.
        2. Iteratively searching for relevant dbt models using `search_dbt_models`.
        3. Searching for relevant past feedback using `search_past_feedback` (for similar questions) and `search_feedback_content` (for specific concepts).
        4. Synthesizing information from models and feedback.
        5. Generating a final, grounded SQL query and explanation using the `finish_workflow` tool.

        **CRITICAL RULE:** Your final action in this workflow MUST be a call to the `finish_workflow` tool. Do NOT output plain text or ask clarification questions as your final response. If you cannot fully answer the question with the available information, you MUST still call `finish_workflow` and explain the limitations in the 'Footnotes' section of the `final_answer`.

        **Tool Usage Strategy:**
        - `search_dbt_models`: Use iteratively (max {self.max_model_searches} times). Refine your query based on previous results. **IMPORTANT: If the user's question (check the initial HumanMessage and subsequent messages) mentions specific table names (e.g., 'fct_frontend_events') or column names (e.g., 'app_id'), prioritize using these exact names in your `query` argument for `search_dbt_models` to ensure the relevant models are found. Do not rely solely on generic semantic search terms if specific identifiers are available.** Stop searching if you have enough models or hit the limit.
        - `search_past_feedback`: Use ONCE near the beginning if similar questions might exist.
        - `search_feedback_content`: Use if you need clarification on specific terms/concepts found in models or the question, *before* generating the final answer.
        - `finish_workflow`: Use ONLY when ready to provide the complete, final answer. Ensure the SQL is based *strictly* on the retrieved models. This MUST be your final action.

        **SQL Generation Rules (CRITICAL):**
        - **Grounding:** ONLY use tables, columns, and relationships explicitly present in the `accumulated_models` provided in the state. DO NOT HALLUCINATE table or column names.
        - **Completeness:** If the retrieved models are insufficient to fully answer the question, generate the best possible SQL using *only* the available information. Clearly state limitations in SQL comments or the 'Footnotes' section of the `final_answer` when calling `finish_workflow`.
        - **Style:** Follow the provided SQL style guide example.
        - **No Slack Mrkdwn:** Do not use Slack formatting in the `final_answer` for `finish_workflow`.

        **Current State:**
        - Models Found: {len(accumulated_models)}
        - Model Search Calls Used: {search_model_calls} / {self.max_model_searches}
        - Feedback Found (Similar Questions): {len(relevant_feedback_by_question)}
        - Feedback Found (Content Search): {len(relevant_feedback_by_content)}
        """

        # --- ADDED: Include SQL style example in prompt if available ---
        if self.sql_style_example:
            system_prompt += f"""

**SQL Style Guide Example:**
```sql
{self.sql_style_example}
```"""
        else:
            system_prompt += """

**SQL Style Guide:** (Example not loaded) Please ensure SQL is well-commented and uses CTEs for clarity."""
        # --- END ADDED ---

        # --- Guidance Logic ---
        guidance_items = []

        # Check if max model searches reached
        if search_model_calls >= self.max_model_searches:
            guidance_items.append(
                f"You have reached the maximum ({self.max_model_searches}) model searches."
            )
            if not accumulated_models:
                guidance_items.append(
                    "No relevant models were found. You MUST now call `finish_workflow` and explain in the 'Footnotes' that you cannot answer the question due to missing model context."
                )
            else:
                guidance_items.append(
                    "You MUST now use the `finish_workflow` tool to generate the final answer based *only* on the models found so far, noting any limitations."
                )
        else:
            guidance_items.append(
                f"You have used the model search tool {search_model_calls} times (max {self.max_model_searches})."
            )
            if not accumulated_models:
                guidance_items.append(
                    "No models found yet. Your next step should likely be to call `search_dbt_models` with a relevant query, or `search_past_feedback` if appropriate."
                )
            else:
                # --- MODIFIED GUIDANCE ---
                guidance_items.append(
                    f"Analyze the models found: {[m['name'] for m in accumulated_models]}. You should now attempt to answer the original question using ONLY these models by calling `finish_workflow`. Explain any limitations clearly in the 'Footnotes' section of the `final_answer`. Only if these models are clearly insufficient *and* you can formulate a *specific, different* query for missing information should you call `search_dbt_models` again (if under the limit {self.max_model_searches}). Do not ask the user for clarification; finish the workflow."
                )
                # --- END MODIFIED GUIDANCE ---

        # Combine system prompt and messages
        messages_for_llm: List[
            Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]
        ] = [SystemMessage(content=system_prompt)]
        # Add original question if messages list is empty (first turn)
        if not messages:
            messages_for_llm.append(HumanMessage(content=original_question))
        else:
            # Add existing message history
            messages_for_llm.extend(messages)

        # Add guidance as a final SystemMessage if needed
        if guidance_items:
            guidance = "Guidance: " + " ".join(guidance_items)
            # Add guidance if the last message isn't already the *exact same* guidance
            last_message = messages_for_llm[-1] if messages_for_llm else None
            if not (
                isinstance(last_message, SystemMessage)
                and last_message.content == guidance
            ):
                messages_for_llm.append(SystemMessage(content=guidance))

        if self.verbose:
            self.console.print("\n[blue]--- Calling Agent Model ---[/blue]")
            self.console.print(
                f"[dim]Current Models: {len(accumulated_models)} ({search_model_calls} searches used)[/dim]"
            )
            self.console.print(
                f"[dim]Found Feedback Items: {len(relevant_feedback_by_question)} (by question), {len(relevant_feedback_by_content)} (by content)[/dim]"
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
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    tool_call_info = f" with {len(msg.tool_calls)} tool_calls"
                elif isinstance(msg, ToolMessage):
                    tool_call_info = f" for tool_call_id: {msg.tool_call_id[:8]}..."
                self.console.print(
                    f"[dim]  [{i}] {msg_type}{tool_call_info}: {content_preview}[/dim]"
                )

        # Invoke the agent LLM
        agent_llm = self._get_agent_llm()
        try:
            response = agent_llm.invoke(messages_for_llm)
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
            # Return an error message to be added to the state
            error_message = AIMessage(
                content=f"LLM invocation failed: {str(e)}"
            )  # Changed to AIMessage
            return {"messages": [error_message]}

    def update_state_node(self, state: QuestionAnsweringState) -> Dict[str, Any]:
        """Updates the state based on the results of the most recent tool calls.

        Processes ToolMessages, updates accumulated models, feedback, and search counts.

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
        search_model_calls = state.get("search_model_calls", 0)
        search_queries_tried = set(state.get("search_queries_tried", set()))
        # MODIFIED: Ensure relevant_feedback is initialized correctly
        relevant_feedback = state.get(
            "relevant_feedback", {}
        ).copy()  # Get mutable copy
        if "by_question" not in relevant_feedback:
            relevant_feedback["by_question"] = []
        if "by_content" not in relevant_feedback:
            relevant_feedback["by_content"] = []

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
                if tool_name == "search_dbt_models":
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
                                f"[yellow dim]Warning: search_dbt_models returned non-list content: {type(parsed_content)}[/yellow dim]"
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
        updates["search_model_calls"] = search_model_calls
        updates["search_queries_tried"] = search_queries_tried
        updates["relevant_feedback"] = relevant_feedback

        if self.verbose:
            if not updates:
                self.console.print(
                    "[dim] -> No state updates generated from tool messages.[/dim]"
                )
            # else:
            #     self.console.print(f"[dim] -> Generated updates: {updates.keys()}[/dim]") # Avoid printing large values

        return updates

    def run_agentic_workflow(self, question: str) -> Dict[str, Any]:
        """Runs the agentic workflow to answer a question.

        Args:
            question: The user's question.

        Returns:
            A dictionary containing the final answer or an error message.
            Keys:
                - final_answer (Optional[str]): The generated SQL and explanation.
                - searched_models (Optional[List[Dict]]): Models considered.
                - relevant_feedback (Optional[Dict]): Feedback considered.
                - error (Optional[str]): Error message if the workflow failed.
        """
        if self.verbose:
            self.console.print(
                f"[bold blue]🚀 Starting LangGraph workflow for: {question}[/bold blue]"
            )

        # Generate a unique thread ID for this execution
        thread_id = str(uuid.uuid4())
        if self.verbose:
            self.console.print(f"[dim]Generated thread ID: {thread_id}[/dim]")
            self.console.print("[dim]Initializing LangGraph workflow with memory[/dim]")

        config = {"configurable": {"thread_id": thread_id}}

        # Define the initial state
        initial_state = QuestionAnsweringState(
            original_question=question,
            messages=[],
            accumulated_models=[],
            accumulated_model_names=set(),
            search_model_calls=0,
            relevant_feedback={"by_question": [], "by_content": []},
            search_queries_tried=set(),
            final_answer=None,  # Ensure final_answer starts as None
        )

        if self.verbose:
            self.console.print(
                "[dim]Initial state prepared with system prompt and question[/dim]"
            )

        try:
            # --- Invoke the graph ---
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
                }

            # Extract results from the final state dictionary
            final_answer = final_state_result.get("final_answer")
            searched_models = final_state_result.get("accumulated_models", [])
            relevant_feedback_items = final_state_result.get("relevant_feedback", {})

            if final_answer:
                if self.verbose:
                    self.console.print(
                        "[green]✅ Workflow finished successfully with final answer.[/green]"
                    )
                return {
                    "final_answer": final_answer,
                    "searched_models": searched_models,
                    "relevant_feedback": relevant_feedback_items,
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
                    state.get("accumulated_models", []) if "state" in locals() else []
                ),  # Try to return models if state exists
                "relevant_feedback": (
                    state.get("relevant_feedback", {}) if "state" in locals() else {}
                ),
            }
