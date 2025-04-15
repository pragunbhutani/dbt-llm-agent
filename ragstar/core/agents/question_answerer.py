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


# --- END NEW SCHEMA ---


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
    # --- MODIFIED: Use a dict to store feedback by source ---
    # relevant_feedback: List[Any]
    relevant_feedback: Dict[str, List[Any]]  # Keys: 'by_question', 'by_content'
    # --- END MODIFIED ---
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

        # --- END NEW TOOL ---

        @tool(args_schema=FinishWorkflowInput)
        def finish_workflow(final_answer: str) -> str:
            """Concludes the workflow and provides the final answer to the user.
            Use this tool ONLY when you have gathered all necessary information from 'search_dbt_models',
            considered any relevant feedback from 'search_past_feedback' and 'search_feedback_content',
            and are ready to provide the complete, final answer (including SQL query and explanation).

            **CRITICAL: GROUNDING & ACCURACY:**
            - **DO NOT HALLUCINATE:** You MUST strictly base the SQL query and explanation on the columns and relationships present in the dbt models retrieved via 'search_dbt_models'. Never invent table or column names.
            - **HANDLE INCOMPLETENESS:** If the retrieved models are insufficient to fully answer the question, generate the best possible answer using *only* the available information. Clearly state in the explanation that the answer may be incomplete and specify what information (e.g., specific metrics, columns, relationships between models) was missing or could not be confirmed.
            - **CLARIFYING QUESTIONS (Optional Footnotes):** If needed, include clarifying questions in a 'Footnotes' section *after* the SQL code block (as normal text, not as SQL comments). For example:
              "Footnotes:\n1. Could you clarify if 'metric X' refers to column Y or Z?". This helps continue the conversation in Slack. Do NOT include footnotes or clarifying questions as SQL comments; they must be outside the code block.
            - **EXPLANATIONS FOR NON-OBVIOUS LOGIC:** Place explanations for non-obvious logic or limitations as comments *above* the relevant CTE or section in the SQL query. If it is not possible to place the explanation meaningfully in the SQL, include it in the 'Footnotes' section after the query. Do NOT use inline comments at the end of SQL lines, and do NOT add explanations after the SQL block except in the 'Footnotes' section.

            **CRITICAL: Format your answer strictly using Slack's `mrkdwn` syntax:**
            - Use *bold text* for emphasis or section names. Surround text with asterisks: *your text*
            - Use _italic text_ for minor emphasis. Surround text with underscores: _your text_
            - Use ~strikethrough text~. Surround text with tildes: ~your text~
            - Use `inline code` for model names, field names, or short code snippets. Surround text with backticks: `your text`
            - Use code blocks for SQL queries or multi-line code. Surround text with triple backticks: ```your code``` (Do NOT specify a language like ```sql).
            - Use >blockquote for quoting text. Add > before the text: >your text
            - Use bulleted lists starting with * or - followed by a space: * your text
            - Use numbered lists starting with 1. followed by a space: 1. your text

            **DO NOT use Markdown headings (# or ##). Slack does not support them.**
            Ensure the final answer directly addresses the user's original question, acknowledging any limitations found.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: finish_workflow(final_answer='{final_answer[:100]}...')",
                )
            # This function primarily acts as a signal, returning the final answer
            return final_answer

        # Store the tools as instance attributes - ADD THE NEW TOOL
        self._tools = [
            search_dbt_models,
            search_past_feedback,
            search_feedback_content,
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
        # === ADDED: Check if workflow is already finished ===
        if state.get("final_answer"):
            if self.verbose:
                self.console.print(
                    "[bold yellow]>>> Entering agent_node (Workflow Finished) <<<[/bold yellow]"
                )
                self.console.print(
                    "[dim] -> Final answer already exists in state. Skipping LLM call and ending workflow.[/dim]"
                )
            # Return empty messages to signal completion (tools_condition will route to END)
            return {"messages": []}
        # === END ADDED CHECK ===

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
            # Log feedback count from both sources
            feedback_by_q = state.get("relevant_feedback", {}).get("by_question", [])
            feedback_by_c = state.get("relevant_feedback", {}).get("by_content", [])
            self.console.print(
                f"[dim]Found Feedback Items: {len(feedback_by_q)} (by question), {len(feedback_by_c)} (by content)[/dim]"
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

        # --- START: Inject Feedback Content (from both sources) ---
        feedback_for_prompt = []
        relevant_feedback_state = state.get("relevant_feedback", {})
        feedback_by_question = relevant_feedback_state.get("by_question", [])
        feedback_by_content = relevant_feedback_state.get("by_content", [])

        all_feedback_items = feedback_by_question + feedback_by_content
        processed_ids = set()  # Avoid duplicate feedback if found by both methods

        if all_feedback_items:
            feedback_texts = []
            for item in all_feedback_items:
                # Assuming item is a Question domain object now
                item_id = item.id
                if item_id in processed_ids:
                    continue
                processed_ids.add(item_id)

                q_text = getattr(item, "question_text", "N/A")
                f_text = getattr(item, "feedback", "N/A")
                f_useful = getattr(item, "was_useful", None)

                # Only include if there is actual feedback text
                if f_text and f_text != "N/A":
                    feedback_texts.append(
                        f"- Past Question ID {item_id}: '{q_text}'\\n  - Feedback (Useful: {f_useful}): '{f_text}'"
                    )

            if feedback_texts:
                feedback_summary = (
                    "\\n\\nRelevant Past Feedback Found (from similar questions and/or content search):\\n"
                    + "\\n".join(feedback_texts)
                )
                messages_for_llm.append(SystemMessage(content=feedback_summary))
                if self.verbose:
                    self.console.print(
                        f"[dim blue]Injecting {len(feedback_texts)} unique feedback items into context.[/dim blue]"
                    )
        # --- END: Inject Feedback Content ---

        # --- MODIFIED: Update Guidance Prompt ---
        current_search_calls = state.get("search_model_calls", 0)
        remaining_searches = self.max_model_searches - current_search_calls
        feedback_searched = (
            state.get("relevant_feedback", {}).get("by_question") is not None
            or state.get("relevant_feedback", {}).get("by_content") is not None
        )

        guidance_pieces = [
            f"Guidance: You have used the model search tool {current_search_calls} times.",
            f"You have {remaining_searches} remaining searches for dbt models.",
            f"Current models found: {len(state['accumulated_model_names'])}.",
            f"Original question: '{state['original_question']}'.",
        ]

        # Check if search_past_feedback has been attempted (even if it returned nothing)
        # We infer this by checking if the specific key exists in the feedback dict
        past_feedback_key_exists = "by_question" in state.get("relevant_feedback", {})

        if not past_feedback_key_exists:
            guidance_pieces.append(
                "Reminder: You MUST call 'search_past_feedback' first with a concise query relevant to the original question before searching for models."
            )
        else:
            guidance_pieces.append(
                "You have already checked for feedback related to similar past questions."
            )
            # Add guidance for content search only *after* initial feedback search is done
            guidance_pieces.append(
                "If you need clarification on specific terms or concepts mentioned *within* feedback, use 'search_feedback_content' to query past feedback text directly."
            )

        if remaining_searches > 0:
            guidance_pieces.append(
                f"When calling 'search_past_feedback', use a concise query reflecting the core information need, potentially similar to your model search query. "  # Keep this for reference even if reminded above
            )
            guidance_pieces.append(
                f"If the current models (and feedback) are insufficient to answer the question comprehensively, "
                f"use 'search_dbt_models' again with a *refined, specific* query focusing on the missing information. "
            )
            guidance_pieces.append(
                f"Otherwise, if you have enough information, use 'finish_workflow' to provide the final answer (SQL query + explanation)."
            )
        else:
            guidance_pieces.append(
                f"You have reached the maximum limit of {self.max_model_searches} model searches. "
                f"You must now synthesize an answer using the models found ({len(state['accumulated_model_names'])} models: "
                f"{', '.join(state['accumulated_model_names']) or 'None'}) "
                f"and considering any provided feedback. Call 'finish_workflow'. Provide the final SQL query and explanation. Do not call 'search_dbt_models' or 'search_feedback_content' again."
            )

        guidance_content = " ".join(guidance_pieces)
        # --- END MODIFIED GUIDANCE ---

        # Add guidance SystemMessage if needed
        if not (messages_for_llm and isinstance(messages_for_llm[-1], HumanMessage)):
            # Avoid adding duplicate guidance if the last message already provides it
            if not (
                messages_for_llm
                and isinstance(messages_for_llm[-1], SystemMessage)
                and messages_for_llm[-1].content.startswith("Guidance:")
            ):
                # Make sure we don't add guidance if the feedback message was just added
                # (Check if the last message contains the feedback summary)
                last_msg_content = (
                    messages_for_llm[-1].content if messages_for_llm else ""
                )
                if (
                    not relevant_feedback_state
                    or "Relevant Past Feedback Found:" not in last_msg_content
                ):
                    messages_for_llm.append(SystemMessage(content=guidance_content))

        agent_llm = self._get_agent_llm()

        if self.verbose:
            self.console.print(
                f"[blue dim]Sending {len(messages_for_llm)} messages to LLM (including feedback and guidance)[/blue dim]"
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
            content = tool_message.content

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
                # Parse result (list of Question domain objects)
                try:
                    # Content should already be a list of Question objects from the tool
                    newly_found_feedback = content if isinstance(content, list) else []
                    # Further validation could be added here if needed
                except Exception as e:
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim]Error interpreting search_past_feedback result: {e}[/yellow dim]"
                        )
                    newly_found_feedback = []

                if newly_found_feedback:
                    # --- MODIFIED: Store feedback under 'by_question' key ---
                    current_feedback_dict = state.get("relevant_feedback", {})
                    existing_by_question = current_feedback_dict.get("by_question", [])
                    # Add new items, could add uniqueness check later if needed
                    updated_by_question = existing_by_question + newly_found_feedback
                    current_feedback_dict["by_question"] = updated_by_question
                    updates["relevant_feedback"] = current_feedback_dict
                    # --- END MODIFIED ---
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with {len(newly_found_feedback)} new feedback items (from similar questions)."
                        )
                elif self.verbose:
                    self.console.print(
                        "[dim] -> No feedback found via similar questions."
                    )

            # --- NEW: Handle results from search_feedback_content ---
            elif tool_name == "search_feedback_content":
                try:
                    newly_found_feedback = content if isinstance(content, list) else []
                except Exception as e:
                    if self.verbose:
                        self.console.print(
                            f"[yellow dim]Error interpreting search_feedback_content result: {e}[/yellow dim]"
                        )
                    newly_found_feedback = []

                if newly_found_feedback:
                    current_feedback_dict = state.get("relevant_feedback", {})
                    existing_by_content = current_feedback_dict.get("by_content", [])
                    updated_by_content = existing_by_content + newly_found_feedback
                    current_feedback_dict["by_content"] = updated_by_content
                    updates["relevant_feedback"] = current_feedback_dict
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with {len(newly_found_feedback)} new feedback items (from content search)."
                        )
                elif self.verbose:
                    self.console.print("[dim] -> No feedback found via content search.")
            # --- END NEW HANDLER ---

            elif tool_name == "finish_workflow":
                # Tool content should be the final answer string
                if content:
                    updates["final_answer"] = content
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with final answer.[/dim]"
                        )
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

        # --- MODIFIED: Update initial system prompt ---
        # --- ADDED: Dynamically create SQL style section ---
        sql_style_section = ""
        if self.sql_style_example:
            # Escape backticks within the example SQL for f-string compatibility inside the final prompt's code block
            escaped_sql_example = self.sql_style_example.replace("`", "\`")
            sql_style_section = f"""
**SQL STYLE GUIDE:**
Format the SQL query in your final answer *exactly* like this example, paying close attention to indentation, spacing, CTE structure, and comments:
```sql
{escaped_sql_example}
```"""
        else:
            sql_style_section = "\n**SQL STYLE GUIDE:**\nAim for readable SQL with consistent indentation and spacing. Use CTEs where appropriate."
        # --- END ADDED ---

        initial_system_prompt = f"""You are an AI assistant specialized in analyzing dbt projects to answer questions.
Your primary goal is to understand the user's question, find the relevant dbt models using the 'search_dbt_models' tool, consider past feedback, and then generate a final answer.

**Workflow Steps:**
1.  **Analyze the Question:** Understand the core concepts.
2.  **Check Past Feedback (Mandatory First Step):** Use 'search_past_feedback' *immediately* with a concise query reflecting the core information need (similar to what you'd use for model search) to see if this question or similar ones have feedback.
3.  **Search for Models:** Use 'search_dbt_models' iteratively to find relevant dbt models. Refine your search query based on findings.
4.  **Search Feedback Content (Optional):** If you need clarification on specific terms or concepts mentioned *within* feedback text later, use 'search_feedback_content'.
5.  **Synthesize Answer:** Once sufficient models and feedback are gathered, use 'finish_workflow' to provide the final answer.

The final answer, provided via the 'finish_workflow' tool, MUST include:
1. A SQL query that directly addresses the user's question based *strictly* on the models and columns found. **Follow the SQL STYLE GUIDE below.**
2. Only include simple, concise explanations as comments *above* the relevant CTE or section in the SQL query, and only where necessary to clarify non-obvious logic or limitations. If you cannot place the explanation meaningfully in the SQL, include it in the 'Footnotes' section after the query. Do not use inline comments at the end of SQL lines, and do not add any explanation text after the SQL block except in the 'Footnotes' section.
3. If there are limitations or missing information, briefly note this as a comment above the relevant section in the SQL query, or in the 'Footnotes' section if not possible.
4. If you have clarifying questions or footnotes, include them *after* the SQL code block as normal text in a 'Footnotes' section (Slack mrkdwn). Do NOT include footnotes or clarifying questions as SQL comments.

{sql_style_section}

**CRITICAL RULES FOR ANSWER GENERATION:**
- **NO HALLUCINATION:** Absolutely do *not* invent database names, schema names, table names, **column names**, or relationships that are not explicitly present in the information retrieved from the dbt models. Stick strictly to the provided schema details (table names, column names, types). If database/schema information is missing for a model, use *only* the model (table) name in the query (e.g., `FROM my_model` instead of `FROM my_db.my_schema.my_model`). Do not "assume" columns exist; only use columns explicitly listed for the relevant models.
- **ACKNOWLEDGE LIMITATIONS:** If you cannot fully answer the question due to missing information (e.g., required columns not found in retrieved models, ambiguity in the question), clearly state this limitation as a comment above the relevant section in the SQL query, or in the 'Footnotes' section if not possible.
- **SQL COMMENTS ONLY:** All explanations must be in the form of simple, concise comments *above* the relevant CTE or section in the SQL query, and only where necessary to clarify non-obvious logic or limitations. If you cannot place the explanation meaningfully in the SQL, include it in the 'Footnotes' section after the query. Do not use inline comments at the end of SQL lines, and do not add any explanation or summary text after the SQL block, except for a 'Footnotes' section as described below.
- **FOOTNOTES OUTSIDE SQL:** If you have clarifying questions or footnotes, include them *after* the SQL code block as normal text in a 'Footnotes' section (Slack mrkdwn). Do NOT include footnotes or clarifying questions as SQL comments.

**CRITICAL SLACK MARKDOWN (mrkdwn) FORMATTING FOR FINAL ANSWER:**
Your final answer *must* strictly adhere to Slack's `mrkdwn` format:
- Use *bold text* for emphasis or section names. Surround text with asterisks: *your text*
- Use _italic text_ for minor emphasis. Surround text with underscores: _your text_
- Use ~strikethrough text~. Surround text with tildes: ~your text~
- Use `inline code` for model names, field names, or short code snippets. Surround text with backticks: `your text`
- Use code blocks for SQL queries or multi-line code. Surround text with triple backticks: ```your code``` (Do NOT specify a language like ```sql).
- Use >blockquote for quoting text. Add > before the text: >your text
- Use bulleted lists starting with * or - followed by a space: * your text
- Use numbered lists starting with 1. followed by a space: 1. your text

Reference: https://slack.com/intl/en-in/help/articles/202288908-Format-your-messages#markup

Remember to call 'search_past_feedback' first!"""
        # --- END MODIFIED PROMPT ---

        # Initial state - MODIFIED: Initialize relevant_feedback as a dict
        initial_state = QuestionAnsweringState(
            original_question=question,
            messages=[
                SystemMessage(content=initial_system_prompt),
                HumanMessage(content=question),
            ],
            accumulated_models=[],
            accumulated_model_names=set(),
            # relevant_feedback=[],
            relevant_feedback={},  # Initialize as empty dict
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
