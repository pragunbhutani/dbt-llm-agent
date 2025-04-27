import logging
import uuid
import tiktoken
from typing import Dict, List, Any, Optional, Set, TypedDict
from typing_extensions import Annotated

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
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from rich.console import Console

# RAGstar Imports
from ragstar.core.agents import QuestionAnswerer
from ragstar.core.llm.client import LLMClient, TokenUsageLogger
from ragstar.utils.cli_utils import get_config_value
from ragstar.storage.model_storage import ModelStorage
from ragstar.storage.model_embedding_storage import ModelEmbeddingStorage
from ragstar.storage.question_storage import QuestionStorage

# --- NEW: Import prompt creation functions ---
from .prompts import (
    create_initial_system_prompt,
    create_verification_system_prompt,
    create_guidance_message,
)

# --- END NEW ---

# Import Slack SDK components
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)


# --- Tool Input Schemas ---
class FetchThreadInput(BaseModel):
    channel_id: str = Field(description="The ID of the Slack channel.")
    thread_ts: str = Field(
        description="The timestamp of the parent message in the thread."
    )


class AskQuestionAnswererInput(BaseModel):
    question: str = Field(
        description="The formulated question (potentially with context) to send to the QuestionAnswerer agent."
    )
    thread_context: Optional[List[Dict[str, Any]]] = Field(
        description="A list of message dictionaries representing the relevant Slack thread history.",
        default=None,
    )


# --- REMOVED: RespondToThreadInput Schema ---
# class RespondToThreadInput(BaseModel):
#     channel_id: str = Field(description="The ID of the Slack channel to respond in.")
#     thread_ts: str = Field(
#         description="The timestamp of the parent message in the thread to respond to."
#     )
#     answer_text: str = Field(
#         description="The final answer text to post in the Slack thread."
#     )


# --- NEW: Input schema for acknowledgement message ---
class AcknowledgeQuestionInput(BaseModel):
    channel_id: str = Field(description="The ID of the Slack channel.")
    thread_ts: str = Field(
        description="The timestamp of the parent message in the thread to acknowledge."
    )
    acknowledgement_text: str = Field(
        description="A brief, friendly message acknowledging the user's question and summarizing your understanding of it before proceeding."
    )


# --- NEW: Input schema for final response with snippet ---
class PostFinalResponseInput(BaseModel):
    channel_id: str = Field(description="The ID of the Slack channel to respond in.")
    thread_ts: str = Field(
        description="The timestamp of the parent message in the thread to respond to."
    )
    message_text: str = Field(
        description="The user-facing message text introducing the SQL query snippet."
    )
    sql_query: str = Field(
        description="The final, verified SQL query content to be uploaded as a snippet."
    )
    optional_notes: Optional[str] = Field(
        description="Optional notes or footnotes to include in the message after the snippet reference.",
        default=None,
    )


# --- NEW: Input schema for simple text response ---
class PostTextResponseInput(BaseModel):
    channel_id: str = Field(description="The ID of the Slack channel to respond in.")
    thread_ts: str = Field(
        description="The timestamp of the parent message in the thread to respond to."
    )
    message_text: str = Field(
        description="The plain text message to post in the Slack thread (e.g., explaining a verification failure or asking for clarification)."
    )


# --- END NEW SCHEMA ---


# --- LangGraph State Definition ---
class SlackResponderState(TypedDict):
    """Represents the state of the Slack responding agent."""

    original_question: str
    channel_id: str
    thread_ts: str
    messages: Annotated[List[BaseMessage], add_messages]
    thread_history: Optional[List[Dict[str, Any]]]  # Store fetched thread messages
    # contextual_question: Optional[str]  # Question formulated with context
    acknowledgement_sent: Optional[bool] = (
        None  # Flag to indicate if acknowledgement was sent
    )
    # --- REMOVED: qa_structured_result field ---
    # qa_structured_result: Optional[Dict[str, Any]] # Stores the dict from QuestionAnswererResult
    # --- NEW: Fields for QA text answer and context ---
    qa_final_answer: Optional[str]  # The final text answer from QA
    qa_models: Optional[List[Dict[str, Any]]]  # Models QA used
    qa_feedback: Optional[List[Dict[str, Any]]]  # Feedback QA used
    error_message: Optional[str] = (
        None  # Store error messages, e.g., from failed tool calls
    )
    response_sent: Optional[bool] = None  # Flag to indicate if final response was sent
    response_message_ts: Optional[str] = (
        None  # Timestamp of the final message posted by this agent
    )
    response_file_message_ts: Optional[str] = None  # <<< RENAME state field
    qa_similar_original_messages: Optional[
        List[Dict[str, Any]]
    ]  # Context QA used (from QA state)


# --- Tool Definitions ---
class SlackResponder:
    """Agent for handling Slack interactions and delegating to QuestionAnswerer."""

    def __init__(
        self,
        llm_client: LLMClient,
        # Dependencies needed to create QuestionAnswerer
        model_storage: ModelStorage,
        vector_store: ModelEmbeddingStorage,
        question_storage: QuestionStorage,
        # slack_client: Any, # TODO: Pass configured Slack client
        slack_client: AsyncWebClient,
        memory: Optional[BaseCheckpointSaver] = None,
        console: Optional[Console] = None,
        verbose: bool = False,
    ):
        """Initialize the SlackResponder agent.

        Args:
            llm_client: LLM client for generating text.
            question_answerer: An instance of the QuestionAnswerer agent.
            # Dependencies for QuestionAnswerer
            model_storage: Storage for dbt models.
            vector_store: Vector store for semantic search.
            question_storage: Storage for question history.
            slack_client: An initialized AsyncWebClient instance from slack_sdk.
            memory: Optional LangGraph checkpoint saver.
            console: Console for output.
            verbose: Whether to print verbose output.
        """
        self.llm = llm_client
        # Initialize QuestionAnswerer internally
        self.question_answerer = QuestionAnswerer(
            llm_client=llm_client,
            model_storage=model_storage,
            vector_store=vector_store,
            question_storage=question_storage,
            console=console,  # Pass console if needed
            verbose=verbose,  # Pass verbose flag if needed
            # Ensure QuestionAnswerer does not require components SlackResponder doesn't have
        )
        # self.slack_client = slack_client # TODO
        self.slack_client = slack_client
        # Initialize default memory if None is provided and Postgres is configured
        if memory is None:
            pg_conn_string = get_config_value("database_url", None)
            if pg_conn_string:
                try:
                    connection_kwargs = {
                        "autocommit": True,
                        "prepare_threshold": 0,
                    }
                    pool = ConnectionPool(
                        conninfo=pg_conn_string,
                        kwargs=connection_kwargs,  # Pass via kwargs parameter
                        max_size=20,  # Sensible defaults for pool size
                        min_size=5,
                    )
                    self.memory = PostgresSaver(conn=pool)
                    self.memory.setup()  # Setup tables if needed
                    if verbose:
                        (console or Console()).print(
                            "[dim]Initialized default PostgresSaver for SlackResponder state.[/dim]"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize default PostgresSaver for SlackResponder: {e}. State will not be persisted.",
                        exc_info=verbose,
                    )
                    self.memory = None  # Ensure memory is None if init fails
            else:
                if verbose:
                    (console or Console()).print(
                        "[dim yellow]DATABASE_URL not set. SlackResponder state will not be persisted.[/dim yellow]"
                    )
                self.memory = None  # No connection string, no memory
        else:
            self.memory = memory  # Use provided memory

        self.console = console or Console()
        self.verbose = verbose
        self.question_storage = question_storage  # Store question storage for recording

        self._define_tools()
        self.graph_app = self._build_graph()

    def _define_tools(self):
        """Define the tools used by the agent."""

        # --- Tool: Fetch Slack Thread History ---
        @tool(args_schema=FetchThreadInput)
        def fetch_slack_thread(channel_id: str, thread_ts: str) -> List[Dict[str, Any]]:
            """Fetches the message history from a specific Slack thread.
            Use this tool ONCE at the beginning to understand the context of the user's question within the Slack thread.
            Returns a list of messages on success, or a dictionary containing an 'error' key on failure.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: fetch_slack_thread(channel_id='{channel_id}', thread_ts='{thread_ts}')[/bold magenta]"
                )
            try:
                import asyncio

                async def _fetch():
                    return await self.slack_client.conversations_replies(
                        channel=channel_id,
                        ts=thread_ts,
                        limit=20,  # Limit history fetched
                    )

                result = asyncio.run(_fetch())

                if result["ok"]:
                    messages = result.get("messages", [])
                    history = [
                        {
                            "user": msg.get("user"),
                            "text": msg.get("text"),
                            "ts": msg.get("ts"),
                        }
                        for msg in messages
                    ]
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Fetched {len(history)} messages from thread.[/dim]"
                        )
                    return history
                else:
                    error_msg = (
                        f"Slack API error (conversations.replies): {result['error']}"
                    )
                    logger.error(error_msg)
                    return {
                        "error": error_msg,
                        "details": {"slack_error": result["error"]},
                    }
            except SlackApiError as e:
                error_msg = f"Slack API Error fetching thread: {e.response['error']}"
                logger.error(error_msg, exc_info=self.verbose)
                return {
                    "error": error_msg,
                    "details": {"slack_error": e.response["error"]},
                }
            except Exception as e:
                error_msg = f"Unexpected error fetching thread: {e}"
                logger.error(error_msg, exc_info=self.verbose)
                return {"error": error_msg, "details": {"exception": str(e)}}

        # --- NEW Tool: Acknowledge Question ---
        @tool(args_schema=AcknowledgeQuestionInput)
        def acknowledge_question(
            channel_id: str, thread_ts: str, acknowledgement_text: str
        ) -> Dict[str, Any]:
            """Posts a brief acknowledgement message to the Slack thread *before* getting the final answer.
            Use this tool AFTER fetching the thread history and formulating your understanding of the user's request.
            The message should confirm receipt and briefly state your understanding of the question.
            Example: "Got it. Just to confirm, you're asking about [rephrased question]? I'll look into this and get back to you."
            Returns a dictionary indicating success or failure.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: acknowledge_question(channel_id='{channel_id}', thread_ts='{thread_ts}', text='{acknowledgement_text[:50]}...')[/bold magenta]"
                )
            try:
                import asyncio

                async def _post_ack():
                    # Keep acknowledgement simple text for now
                    return await self.slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=acknowledgement_text,
                    )

                result = asyncio.run(_post_ack())
                if result["ok"]:
                    if self.verbose:
                        self.console.print(
                            f"[green] -> Successfully posted acknowledgement to {channel_id}/{thread_ts}[/green]"
                        )
                    return {"success": True, "message": "Acknowledgement sent."}
                else:
                    error_msg = (
                        f"Slack API error (chat.postMessage for ack): {result['error']}"
                    )
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "details": {"slack_error": result["error"]},
                    }
            except SlackApiError as e:
                error_msg = (
                    f"Slack API Error posting acknowledgement: {e.response['error']}"
                )
                logger.error(error_msg, exc_info=self.verbose)
                return {
                    "success": False,
                    "error": error_msg,
                    "details": {"slack_error": e.response["error"]},
                }
            except Exception as e:
                error_msg = f"Unexpected error posting acknowledgement: {e}"
                logger.error(error_msg, exc_info=self.verbose)
                return {
                    "success": False,
                    "error": error_msg,
                    "details": {"exception": str(e)},
                }

        # --- END NEW Tool ---

        # --- NEW Tool: Post Simple Text Response ---
        @tool(args_schema=PostTextResponseInput)
        def post_text_response(
            channel_id: str, thread_ts: str, message_text: str
        ) -> Dict[str, Any]:
            """Posts a simple plain text message to the specified Slack thread.
            Use this tool for messages that do NOT include a SQL snippet, such as:
            - Explaining why SQL verification failed after receiving the answer from QuestionAnswerer.
            - Informing the user if QuestionAnswerer couldn't generate an answer.
            - Asking for clarification if the initial request is ambiguous (after attempting to understand context).
            Returns a dictionary indicating success or failure.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: post_text_response(channel_id='{channel_id}', thread_ts='{thread_ts}', text='{message_text[:50]}...')[/bold magenta]"
                )

            # --- ADD FOOTNOTE --- #
            footnote = "\n\n_💡 React with 👍 or 👎 to this message to provide feedback. Mention me again to continue the conversation._"
            full_message_text = message_text + footnote
            # --- END ADD FOOTNOTE --- #

            try:
                import asyncio

                async def _post_text():
                    return await self.slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=full_message_text,  # Use modified text
                    )

                result = asyncio.run(_post_text())
                if result["ok"]:
                    posted_message_ts = result.get("ts")
                    if self.verbose:
                        self.console.print(
                            f"[green] -> Successfully posted simple text response to {channel_id}/{thread_ts} (ts: {posted_message_ts})[/green]"
                        )
                    return {
                        "success": True,
                        "message": "Text response sent.",
                        "message_ts": posted_message_ts,  # Return the timestamp
                    }
                else:
                    error_msg = f"Slack API error (chat.postMessage for text response): {result['error']}"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "details": {"slack_error": result["error"]},
                    }
            except SlackApiError as e:
                error_msg = (
                    f"Slack API Error posting text response: {e.response['error']}"
                )
                logger.error(error_msg, exc_info=self.verbose)
                return {
                    "success": False,
                    "error": error_msg,
                    "details": {"slack_error": e.response["error"]},
                }
            except Exception as e:
                error_msg = f"Unexpected error posting text response: {e}"
                logger.error(error_msg, exc_info=self.verbose)
                return {
                    "success": False,
                    "error": error_msg,
                    "details": {"exception": str(e)},
                }

        # --- END NEW TOOL ---

        # --- Tool: Ask Question Answerer ---
        @tool(args_schema=AskQuestionAnswererInput)
        def ask_question_answerer(
            question: str, thread_context: Optional[List[Dict[str, Any]]] = None
        ) -> Dict[str, Any]:
            """Sends a question and optional thread context to the specialized QuestionAnswerer agent.
            Use this tool AFTER analyzing the Slack thread context AND sending an acknowledgement message.
            Formulate the 'question' input carefully, incorporating thread context if it helps clarify the user's intent.
            Provide the fetched 'thread_context' (list of message dicts) to give the QuestionAnswerer more background.
            Returns a dictionary containing the final answer text, models searched, and feedback considered on success,
            or a dictionary containing an 'error' key on failure.
            """
            if self.verbose:
                context_summary = (
                    f"{len(thread_context)} messages" if thread_context else "None"
                )
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: ask_question_answerer(question='{question[:100]}...', context={context_summary})[/bold magenta]"
                )
            self.console.print(
                f"[blue]❓ Passing question to QuestionAnswerer: '{question[:100]}...'[/blue]"
            )
            try:
                # --- MODIFIED: Pass thread_context to the workflow runner ---
                qa_result_dict = self.question_answerer.run_agentic_workflow(
                    question=question, thread_context=thread_context
                )

                # Check if the result indicates an error from QA workflow
                if qa_result_dict and qa_result_dict.get("error"):
                    error_content = qa_result_dict["error"]
                    logger.error(
                        f"QuestionAnswerer workflow returned an error: {error_content}"
                    )
                    return {
                        "success": False,
                        "error": f"QuestionAnswerer failed: {error_content}",
                        "final_answer": qa_result_dict.get("final_answer"),
                        "searched_models": qa_result_dict.get("searched_models", []),
                        "relevant_feedback": qa_result_dict.get(
                            "relevant_feedback", []
                        ),
                        "similar_original_messages": qa_result_dict.get(
                            "similar_original_messages", []
                        ),
                    }
                elif qa_result_dict and "final_answer" in qa_result_dict:
                    # Return success structure with the answer text and context
                    return {
                        "success": True,
                        "final_answer": qa_result_dict.get("final_answer"),
                        "searched_models": qa_result_dict.get("searched_models", []),
                        "relevant_feedback": qa_result_dict.get(
                            "relevant_feedback", []
                        ),
                        "similar_original_messages": qa_result_dict.get(
                            "similar_original_messages", []
                        ),
                        "error": None,
                    }
                else:
                    # Handle unexpected empty or malformed result
                    error_msg = "QuestionAnswerer workflow returned an unexpected result format."
                    logger.error(f"{error_msg} Result: {qa_result_dict}")
                    return {"success": False, "error": error_msg}
            except Exception as e:
                error_msg = f"Error running QuestionAnswerer workflow: {e}"
                logger.error(error_msg, exc_info=self.verbose)
                return {
                    "success": False,
                    "error": error_msg,
                }

        # --- NEW Tool: Post Final Response with Snippet ---
        @tool(args_schema=PostFinalResponseInput)
        def post_final_response_with_snippet(
            channel_id: str,
            thread_ts: str,
            message_text: str,
            sql_query: str,
            optional_notes: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Posts the final response message and uploads the SQL query as a text snippet to the specified Slack thread.
            Use this tool ONLY after verifying the SQL query from QuestionAnswerer and composing the accompanying message text and notes.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: post_final_response_with_snippet(channel_id='{channel_id}', thread_ts='{thread_ts}', message='{message_text[:50]}...', notes='{str(optional_notes)[:50]}...')",
                )

            # --- ADD FOOTNOTE --- #
            footnote = "\n\n_💡 React with 👍 or 👎 to this message or the snippet to provide feedback. Mention me again to continue the conversation._"
            full_message_body = message_text
            if optional_notes:
                full_message_body += f"\n\n*Notes:*\n{optional_notes}"
            # Append footnote to the text message itself.
            # Note: We could try adding it to the file's initial_comment, but it might not render well or be as visible.
            full_message_body += footnote
            # --- END ADD FOOTNOTE --- #

            try:
                import asyncio

                async def _upload_and_post():
                    # 1. Post the main message.
                    post_result = await self.slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=full_message_body,  # Use text with footnote
                        # No complex blocks needed here now
                        unfurl_links=False,
                        unfurl_media=False,
                    )

                    if not post_result or not post_result["ok"]:
                        error_msg = f"Slack API error (chat.postMessage for final response): {post_result.get('error', 'Unknown post error')}"
                        logger.error(error_msg)
                        return {
                            "success": False,
                            "error": error_msg,
                            "details": {"slack_error": post_result.get("error")},
                        }

                    message_ts = post_result.get("ts")  # Get the message timestamp

                    # 2. Upload the SQL query as a snippet *after* the message
                    # Use files_upload_v2 which returns more details
                    upload_result = await self.slack_client.files_upload_v2(
                        channel=channel_id,
                        content=sql_query,
                        title="Generated SQL Query",
                        filename="generated_query.sql",
                        thread_ts=thread_ts,  # Keep in the same thread
                        # initial_comment="Here is the generated SQL query:", # Optional initial comment for the file
                    )

                    if not upload_result or not upload_result.get("ok"):
                        error_msg = f"Message posted (ts:{message_ts}), but Slack API error uploading snippet (files.upload_v2): {upload_result.get('error', 'Unknown upload error')}"
                        logger.error(error_msg)
                        # Return failure, but include the message_ts for potential partial logging
                        return {
                            "success": False,
                            "error": error_msg,
                            "message_ts": message_ts,  # Include ts even on failure
                            "details": {"slack_error": upload_result.get("error")},
                        }

                    # --- EXTRACT FILE'S MESSAGE TIMESTAMP --- #
                    file_message_ts = None
                    uploaded_file_info = upload_result.get("file")
                    if uploaded_file_info:
                        # The `shares` dict contains info about where the file is shared.
                        # We expect it to be shared in the specified channel/thread.
                        shares = uploaded_file_info.get("shares", {})
                        # Look through public shares first (adjust if using private channels differently)
                        public_shares = shares.get("public", {})
                        if channel_id in public_shares:
                            # Each share is a list of message objects where it was shared
                            share_messages = public_shares[channel_id]
                            if share_messages:
                                # Find the share matching our thread_ts
                                for share in share_messages:
                                    if share.get("thread_ts") == thread_ts or (
                                        not share.get("thread_ts") and not thread_ts
                                    ):
                                        # Found the message where this file was posted in our thread
                                        file_message_ts = share.get("ts")
                                        break  # Stop looking once found
                        if not file_message_ts:
                            logger.warning(
                                f"Could not find matching share message ts for file {uploaded_file_info.get('id')} in channel {channel_id} thread {thread_ts}. Shares: {shares}"
                            )
                    else:
                        logger.warning(
                            f"files_upload_v2 response missing 'file' object. Response: {upload_result}"
                        )
                    # --- END EXTRACT FILE'S MESSAGE TIMESTAMP --- #

                    # Return success with both message timestamps
                    return {
                        "success": True,
                        "message": "Successfully posted response and uploaded snippet.",
                        "message_ts": message_ts,  # TS of the text message
                        "file_message_ts": file_message_ts,  # TS of the file upload message
                    }

                result = asyncio.run(_upload_and_post())
                return result  # Return the success/failure dict

            except SlackApiError as e:
                error_msg = f"Slack API Error posting final response/snippet: {e.response['error']}"
                logger.error(error_msg, exc_info=self.verbose)
                return {
                    "success": False,
                    "error": error_msg,
                    "details": {"slack_error": e.response["error"]},
                }
            except Exception as e:
                error_msg = f"Unexpected error posting final response/snippet: {e}"
                logger.error(error_msg, exc_info=self.verbose)
                return {
                    "success": False,
                    "error": error_msg,
                    "details": {"exception": str(e)},
                }

        # Store the tools as instance attributes
        self._tools = [
            fetch_slack_thread,
            acknowledge_question,
            post_text_response,  # Added new tool
            ask_question_answerer,
            post_final_response_with_snippet,
        ]

    # --- LangGraph Graph Construction ---
    def _build_graph(self):
        """Builds the LangGraph StateGraph for the SlackResponder."""
        workflow = StateGraph(SlackResponderState)

        tool_node = ToolNode(self._tools, handle_tool_errors=True)

        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node(
            "update_state", self.update_state_node
        )  # Node to process tool results
        workflow.add_node(
            "error_responder", self.error_responder_node
        )  # Node to handle errors and respond
        workflow.add_node(
            "check_if_response_sent",
            self.check_if_response_sent,  # Node to check if final response was sent
        )
        # --- NEW NODE: Record Interaction --- #
        workflow.add_node("record_interaction", self.record_interaction_node)

        # Set entry point
        workflow.set_entry_point("agent")

        # Define conditional edges from agent
        workflow.add_conditional_edges(
            "agent",
            tools_condition,  # Checks for tool calls in the last AIMessage
            {
                "tools": "tools",  # If tools called, go to tools node
                END: END,  # Otherwise, end the graph (e.g., if agent decides no tool needed)
            },
        )

        # Define edge for flow: tools -> update_state
        workflow.add_edge("tools", "update_state")

        # Conditional edge after update_state (Check for errors first)
        workflow.add_conditional_edges(
            "update_state",
            self.should_continue_or_handle_error,  # Check if there was an error in the tool/update
            {
                # If no error, proceed to check if the final response was sent
                "check_if_response_sent": "check_if_response_sent",  # Route to the check node if no error
                # If error, go to error handler
                "error_responder": "error_responder",
            },
        )

        # Conditional edge after the check_if_response_sent node updates the state
        workflow.add_conditional_edges(
            "check_if_response_sent",  # Origin is the node that just ran
            self.route_after_response_check,  # Use the routing function
            {
                "record_interaction": "record_interaction",  # If response sent, record it
                "agent": "agent",  # If response not sent, loop back to agent
            },
        )

        # Error handler leads to END
        workflow.add_edge("error_responder", END)

        # --- NEW: Edge from record_interaction to END --- #
        workflow.add_edge("record_interaction", END)
        # --- END NEW --- #

        # Compile the graph with optional memory
        return workflow.compile(checkpointer=self.memory)

    def _get_agent_llm(self):
        """Helper to get the LLM with tools bound."""
        chat_client_instance = self.llm.chat_client
        # Restore the use of bind_tools
        if hasattr(chat_client_instance, "bind_tools"):
            return chat_client_instance.bind_tools(self._tools)
        else:
            logger.warning(
                "LLM chat client does not have 'bind_tools'. Tool calling might not work."
            )
            return chat_client_instance

    # --- LangGraph Nodes ---
    def agent_node(self, state: SlackResponderState) -> Dict[str, Any]:
        """The main node that calls the LLM to decide the next action."""
        # Check if the workflow is already complete (response sent)
        if state.get("response_sent"):
            if self.verbose:
                self.console.print(
                    "[bold yellow]>>> Entering agent_node (Response Already Sent) <<<[/bold yellow]"
                )
                self.console.print(
                    "[dim] -> Response flag is true. Ending workflow immediately.[/dim]"
                )
            return {"messages": []}

        if self.verbose:
            self.console.print(
                "[bold yellow]\n>>> Entering SlackResponder agent_node <<<[/bold yellow]"
            )
            self.console.print(
                f"[dim]State: Channel={state['channel_id']}, Thread={state['thread_ts']}[/dim]"
            )
            self.console.print(
                f"[dim]Messages so far: {len(state.get('messages', []))}[/dim]"
            )
            self.console.print(
                f"[dim]Thread history fetched: {'Yes' if state.get('thread_history') else 'No'}[/dim]"
            )
            self.console.print(
                f"[dim]Acknowledgement sent: {'Yes' if state.get('acknowledgement_sent') else 'No'}[/dim]"
            )
            # --- NEW: Log current messages in state --- #
            self.console.print(
                f"[dim]Current messages in state ({len(state.get('messages', []))}):[/dim]"
            )
            for i, msg in enumerate(state.get("messages", [])):
                msg_type = type(msg).__name__
                content_preview = repr(msg.content)[:70] + (
                    "..." if len(repr(msg.content)) > 70 else ""
                )
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        tool_call_count = len(msg.tool_calls)
                        self.console.print(
                            f"[dim]  State[{i}] {msg_type} (requesting {tool_call_count} tool call{'s' if tool_call_count > 1 else ''}): {content_preview}[/dim]"
                        )
                        for tc in msg.tool_calls:
                            tc_name = tc.get("name", "N/A")
                            tc_args = tc.get("args", {})
                            self.console.print(
                                f"[dim]    - Tool Call: {tc_name}, Args: {tc_args}[/dim]"
                            )
                    else:
                        self.console.print(
                            f"[dim]  State[{i}] {msg_type}: {content_preview}[/dim]"
                        )
                elif isinstance(msg, ToolMessage):
                    tool_call_info = f" for tool_call_id: {msg.tool_call_id[:8]}..."
                    self.console.print(
                        f"[dim]  State[{i}] {msg_type}{tool_call_info}: {content_preview}[/dim]"
                    )
                else:  # Handle other message types
                    self.console.print(
                        f"[dim]  State[{i}] {msg_type}: {content_preview}[/dim]"
                    )
            # --- END NEW Logging Block --- #
            # --- END MODIFIED Logging ---

        messages = state.get("messages", [])
        # --- NEW: Get QA context from state ---
        qa_final_answer_text = state.get("qa_final_answer")
        qa_models_used = state.get("qa_models")
        qa_feedback_used = state.get("qa_feedback")
        thread_history = state.get("thread_history")

        # --- Determine the next step based on state ---
        if not messages:
            # --- MODIFIED: Call prompt function ---
            system_prompt = create_initial_system_prompt()
            # --- END MODIFIED ---
            initial_human_message = f'New question received from Slack:\\nChannel ID: {state["channel_id"]}\\nThread TS: {state["thread_ts"]}\\nQuestion: "{state["original_question"]}"\\n\\nPlease fetch the thread history first.'
            messages_for_llm = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=initial_human_message),
            ]
        # --- NEW: Logic block for handling QA result ---
        elif (
            qa_final_answer_text
            and qa_models_used is not None
            and qa_feedback_used is not None
        ):
            # QA agent has returned its result, now verify and post
            if self.verbose:
                self.console.print(
                    "[dim] -> QA result received. Preparing verification and final posting prompt.[/dim]"
                )

            # --- MODIFIED: Call prompt function ---
            verification_system_prompt = create_verification_system_prompt(
                original_question=state["original_question"],
                thread_history=thread_history,
                qa_final_answer_text=qa_final_answer_text,
                qa_models_used=qa_models_used,
                qa_feedback_used=qa_feedback_used,  # Pass feedback
                channel_id=state["channel_id"],
                thread_ts=state["thread_ts"],
            )
            # --- END MODIFIED ---

            # --- MODIFIED: Send only the verification prompt for the final step ---
            # The verification prompt contains the QA answer and original context needed.
            # Avoids sending previous tool messages which can cause structure errors (400 Bad Request).
            messages_for_llm = [SystemMessage(content=verification_system_prompt)]
            # --- END MODIFIED ---

            if self.verbose:
                self.console.print(
                    "[dim] -> Constructed prompt for SQL verification and final posting.[/dim]"
                )

        # --- END NEW LOGIC BLOCK ---
        else:
            # --- MODIFIED: Call guidance creation function ---
            guidance = create_guidance_message(
                thread_history=state.get("thread_history"),
                acknowledgement_sent=state.get("acknowledgement_sent"),
                qa_final_answer_text=qa_final_answer_text,
            )
            # --- END MODIFIED ---

            messages_for_llm = list(messages)  # Make a copy
            # Add guidance if the last message isn't already guidance and guidance exists
            if guidance and not (
                messages_for_llm
                and isinstance(messages_for_llm[-1], SystemMessage)
                and messages_for_llm[-1].content
                == guidance  # Check against the generated guidance
            ):
                messages_for_llm.append(SystemMessage(content=guidance))

        # --- MODIFIED: Token Count Logging Before Request ---
        try:
            # Assuming o4-mini uses cl100k_base encoding like gpt-4/gpt-3.5
            # Use the model name configured in LLMClient if available and different
            encoding = tiktoken.get_encoding("cl100k_base")

            # Estimate tokens based on message content length (simple method)
            # A more accurate method would replicate OpenAI's exact chat format rules
            num_tokens = 0
            messages_for_token_count = []
            for message in messages_for_llm:
                # Basic token counting per message content
                if isinstance(message.content, str):
                    num_tokens += len(encoding.encode(message.content))
                    messages_for_token_count.append(
                        {"role": message.type, "content": message.content}
                    )  # For potential future more accurate counting
                elif isinstance(
                    message.content, list
                ):  # Handle list content (e.g. vision models, though not used here)
                    for item in message.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            num_tokens += len(encoding.encode(item.get("text", "")))
                # Add buffer for message roles, etc.
                num_tokens += 5  # Rough estimate per message overhead

            logger.info(
                f"Estimated tokens to be sent to LLM: {num_tokens}",
                extra={"token_count": num_tokens},
            )
            if self.verbose:
                self.console.print(
                    f"[cyan dim] -> Estimated prompt tokens: {num_tokens}[/cyan dim]"
                )

        except Exception as tk_err:
            logger.warning(
                f"Could not estimate token count before LLM call: {tk_err}",
                exc_info=self.verbose,
            )
        # --- END ADDED ---

        # --- MODIFIED: LLM Invocation with Callback (Tools are now bound) --- #
        agent_llm = self._get_agent_llm()  # Gets the client with tools bound
        token_logger = TokenUsageLogger()  # Instantiate the callback
        config = {
            "callbacks": [token_logger],
            "run_name": "SlackResponderAgentNode",  # Optional: Add a run name for tracing
        }

        if self.verbose:
            # ... (logging before call) ...
            pass

        try:
            # Remove tools=self._tools from invoke, as they are bound via _get_agent_llm
            response = agent_llm.invoke(messages_for_llm, config=config)
            if self.verbose:
                # ... (logging after call) ...
                pass
            # Clear error message state if LLM runs successfully
            return {"messages": [response], "error_message": None}
        except Exception as e:
            logger.error(f"Error invoking agent LLM: {e}", exc_info=self.verbose)
            error_message = AIMessage(content=f"LLM invocation failed: {str(e)}")
            # Set error message state if LLM fails
            return {
                "messages": [error_message],
                "error_message": f"LLM invocation failed: {str(e)}",
            }
        # --- END MODIFIED --- #

    def update_state_node(self, state: SlackResponderState) -> Dict[str, Any]:
        """Updates the state based on the results of the most recent tool call.
        Sets 'error_message' if a tool fails, otherwise clears it and updates relevant state.
        """
        if self.verbose:
            self.console.print(
                "[bold cyan]\n--- Updating SlackResponder State After Tool Execution ---[/bold cyan]"
            )

        updates: Dict[str, Any] = {"error_message": None}  # Default clear error
        messages = state.get("messages", [])
        # Find the most recent ToolMessage, not necessarily the very last message
        last_tool_message = None
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break

        if not last_tool_message:
            if self.verbose:
                self.console.print(
                    "[yellow dim] -> No ToolMessage found in history, skipping state update.[/yellow dim]"
                )
            return updates  # No tool result to process

        tool_name = last_tool_message.name
        content = (
            last_tool_message.content
        )  # Content is usually stringified result from ToolNode

        if self.verbose:
            content_summary = str(content)[:200] + (
                "..." if len(str(content)) > 200 else ""
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
                    # Keep content as string if not valid JSON
                    pass

            # Default to clearing error unless tool explicitly fails
            updates["error_message"] = None

            if tool_name == "fetch_slack_thread":
                # Result should be list (success) or dict with 'error' (failure)
                if isinstance(parsed_content, dict) and "error" in parsed_content:
                    updates["error_message"] = parsed_content.get(
                        "error", "Unknown error fetching thread"
                    )
                    updates["thread_history"] = (
                        None  # Ensure history is cleared on error
                    )
                    if self.verbose:
                        self.console.print(
                            f"[yellow] -> Setting error state from fetch_slack_thread: {updates['error_message']}[/yellow]"
                        )
                elif isinstance(parsed_content, list):
                    updates["thread_history"] = parsed_content
                    updates["error_message"] = None  # Clear error on success
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with {len(parsed_content)} messages from thread history.[/dim]"
                        )
                else:
                    # Unexpected format
                    error_msg = f"fetch_slack_thread tool returned unexpected type: {type(parsed_content)}"
                    logger.warning(error_msg)
                    updates["error_message"] = error_msg
                    updates["thread_history"] = None

            # --- NEW: Handle acknowledge_question result ---
            elif tool_name == "acknowledge_question":
                if isinstance(parsed_content, dict):
                    if parsed_content.get("success"):
                        updates["acknowledgement_sent"] = True
                        # updates["error_message"] = None # Already cleared by default
                        if self.verbose:
                            self.console.print(
                                f"[dim] -> Updated state: acknowledgement_sent = True.[/dim]"
                            )
                    else:
                        # Tool reported failure
                        updates["error_message"] = parsed_content.get(
                            "error",
                            "acknowledge_question failed without specific error.",
                        )
                        updates["acknowledgement_sent"] = (
                            False  # Explicitly set false on error
                        )
                        if self.verbose:
                            self.console.print(
                                f"[yellow] -> Setting error state from acknowledge_question: {updates['error_message']}[/yellow]"
                            )
                else:
                    error_msg = f"acknowledge_question tool returned unexpected content: {parsed_content}"
                    logger.warning(error_msg)
                    updates["error_message"] = error_msg
                    updates["acknowledgement_sent"] = False
            # --- END NEW HANDLER ---

            elif tool_name == "ask_question_answerer":
                # Expecting a dict potentially containing 'error', 'final_answer', 'models', 'feedback'
                if isinstance(parsed_content, dict):
                    # --- MODIFIED: Check for truthiness of the 'error' key instead of 'is not None' ---\n                    # Check if the result indicates an error from QA workflow\n                    if qa_result_dict and qa_result_dict.get(\"error\"): # Check if error key exists and is truthy\n                        error_content = qa_result_dict['error'] # Get the actual error content\n                        logger.error(\n                            f\"QuestionAnswerer workflow returned an error: {error_content}\"\n                        )\n                        return {\n                            \"success\": False, # Keep this structure for update_state_node compatibility for now\n                            \"error\": f\"QuestionAnswerer failed: {error_content}\", # Use the actual error\n                            \"final_answer\": qa_result_dict.get(\"final_answer\"),\n                            \"models\": qa_result_dict.get(\"searched_models\", []),\n                            \"feedback\": qa_result_dict.get(\"relevant_feedback\", []),\n                        }\n                    elif qa_result_dict and \"final_answer\" in qa_result_dict:\n                        # Return success structure with the answer text and context\n                        return {\n                            \"success\": True, # Keep this structure\n                            \"final_answer\": qa_result_dict.get(\"final_answer\"),\n                            \"models\": qa_result_dict.get(\"searched_models\", []),\n                            \"feedback\": qa_result_dict.get(\"relevant_feedback\", []),\n                            \"error\": None # Explicitly add error: None on success\n                        }\n                    else:\n                        # Handle unexpected empty or malformed result\n                    error_msg = "QuestionAnswerer workflow returned an unexpected result format."\n                    logger.error(f"{error_msg} Result: {qa_result_dict}")
                    qa_error = parsed_content.get("error")
                    if not qa_error:  # Success if error is None or empty string
                        # --- Store text answer and context ---
                        updates["qa_final_answer"] = parsed_content.get("final_answer")
                        updates["qa_models"] = parsed_content.get("searched_models", [])
                        updates["qa_feedback"] = parsed_content.get(
                            "relevant_feedback", {}
                        )
                        updates["qa_similar_original_messages"] = parsed_content.get(
                            "similar_original_messages", []
                        )
                        updates["error_message"] = None
                        if self.verbose:
                            self.console.print(
                                f"[dim] -> Updated state with final answer and context from QuestionAnswerer (Success detected: error is None).[/dim]"
                            )
                            self.console.print(
                                f"[dim]   -> Answer Text: {str(updates.get('qa_final_answer'))[:100]}..."
                            )  # Use .get for safety
                            self.console.print(
                                f"[dim]   -> Models Used Count: {len(updates.get('qa_models', []))}"
                            )
                            self.console.print(
                                f"[dim]   -> Feedback Items Count: {len(updates.get('qa_feedback', []))}"
                            )
                        # Validate that we received the final answer text
                        if not updates.get("qa_final_answer"):
                            error_msg = "ask_question_answerer tool succeeded (no error reported) but returned missing final_answer text."
                            logger.warning(error_msg)
                            updates["error_message"] = (
                                error_msg  # Set SlackResponder error state
                            )
                            # Clear potentially partial data if answer is missing despite success
                            updates["qa_models"] = None
                            updates["qa_feedback"] = None
                            updates["qa_similar_original_messages"] = None
                        # --- END MODIFIED Success Block ---
                    else:
                        # --- Failure Case (error key has a value) ---
                        updates["error_message"] = (
                            f"QuestionAnswerer failed: {qa_error}"  # Use the actual error
                        )
                        # Store partial results if available from the error response
                        updates["qa_final_answer"] = parsed_content.get("final_answer")
                        updates["qa_models"] = parsed_content.get(
                            "searched_models"
                        )  # Allow None
                        updates["qa_feedback"] = parsed_content.get(
                            "relevant_feedback"
                        )  # Allow None
                        updates["qa_similar_original_messages"] = parsed_content.get(
                            "similar_original_messages"
                        )
                        if self.verbose:
                            self.console.print(
                                f"[yellow] -> Setting error state from ask_question_answerer: {updates['error_message']}[/yellow]"
                            )
                else:
                    # Handle unexpected content type
                    error_msg = f"ask_question_answerer tool returned unexpected content type: {type(parsed_content).__name__}"
                    logger.warning(f"{error_msg}. Content: {parsed_content}")
                    updates["error_message"] = error_msg
                    updates["qa_final_answer"] = None
                    updates["qa_models"] = None
                    updates["qa_feedback"] = None
                    updates["qa_similar_original_messages"] = None

            # --- MODIFIED: Handle new final response tool ---
            elif tool_name == "post_final_response_with_snippet":
                # Expecting dict with "success": True/False
                if isinstance(parsed_content, dict):
                    if parsed_content.get("success"):
                        updates["response_message_ts"] = parsed_content.get(
                            "message_ts"
                        )
                        updates["response_file_message_ts"] = parsed_content.get(
                            "file_message_ts"
                        )  # <<< Store file_message_ts
                        updates["error_message"] = None
                        if self.verbose:
                            self.console.print(
                                f"[dim] -> Received success result from post_final_response_with_snippet: {parsed_content.get('message')}[/dim]"
                            )
                    else:
                        # Tool reported failure
                        updates["error_message"] = parsed_content.get(
                            "error",
                            "post_final_response_with_snippet failed without specific error.",
                        )
                        updates["response_sent"] = (
                            False  # Ensure response_sent is False on error
                        )
                        if self.verbose:
                            self.console.print(
                                f"[yellow] -> Setting error state from post_final_response_with_snippet: {updates['error_message']}[/yellow]"
                            )
                        # Keep message_ts if available, clear file message ts on failure
                        updates["response_message_ts"] = parsed_content.get(
                            "message_ts"
                        )  # Keep message_ts if available
                        updates["response_file_message_ts"] = (
                            None  # <<< Clear file message ts on failure
                        )
                else:
                    error_msg = f"post_final_response_with_snippet tool returned unexpected content format: {type(parsed_content).__name__}"
                    logger.warning(f"{error_msg}. Content: {parsed_content}")
                    updates["error_message"] = error_msg
                    updates["response_sent"] = False
            # --- END NEW HANDLER ---

            # --- NEW: Handle simple text response tool ---
            elif tool_name == "post_text_response":
                # Expecting dict with "success": True/False
                if isinstance(parsed_content, dict):
                    if parsed_content.get("success"):
                        updates["response_message_ts"] = parsed_content.get(
                            "message_ts"
                        )
                        updates["error_message"] = None
                        if self.verbose:
                            self.console.print(
                                f"[dim] -> Received success result from post_text_response: {parsed_content.get('message')}[/dim]"
                            )
                    else:
                        # Tool reported failure
                        updates["error_message"] = parsed_content.get(
                            "error",
                            "post_text_response failed without specific error.",
                        )
                        updates["response_sent"] = False
                        if self.verbose:
                            self.console.print(
                                f"[yellow] -> Setting error state from post_text_response: {updates['error_message']}[/yellow]"
                            )
                else:
                    error_msg = f"post_text_response tool returned unexpected content format: {type(parsed_content).__name__}"
                    logger.warning(f"{error_msg}. Content: {parsed_content}")
                    updates["error_message"] = error_msg
                    updates["response_sent"] = False
            # --- END NEW HANDLER ---

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
            # Set error state if processing fails
            updates["error_message"] = (
                f"Internal error processing tool {tool_name}: {e}"
            )

        return updates

    # --- Conditional Edge Logic ---
    def should_continue_or_handle_error(self, state: SlackResponderState) -> str:
        """Determines the next step after state update based on error presence."""
        if state.get("error_message"):
            if self.verbose:
                self.console.print(
                    f"[yellow] -> Error detected ('{state['error_message']}'), routing to error_responder.[/yellow]"
                )
            return "error_responder"
        else:
            if self.verbose:
                self.console.print(
                    "[green] -> No error detected, routing to check_if_response_sent.[/green]"
                )
            return "check_if_response_sent"

    # --- Error Handling Node ---
    def error_responder_node(self, state: SlackResponderState) -> Dict[str, Any]:
        """Handles errors by posting a message to Slack and terminating."""
        error_msg = state.get("error_message", "An unknown error occurred.")
        channel_id = state["channel_id"]
        thread_ts = state["thread_ts"]

        if self.verbose:
            self.console.print(
                f"[bold red]\n>>> Entering error_responder_node <<<[/bold red]"
            )
            self.console.print(f"[red]Error to report: {error_msg}[/red]")

        # Format a user-friendly message
        user_message = f"Sorry, I encountered an error and cannot proceed: {error_msg}"
        # Add specific advice for common errors like missing scope
        if "missing_scope" in error_msg and "channels:history" in error_msg:
            user_message += "\n\nIt looks like I don't have the necessary permissions (scope: 'channels:history') to read messages in this channel. Please check the app's configuration."
        elif "missing_scope" in error_msg:
            user_message += "\n\nIt seems I'm missing some required permissions. Please check the app's configuration."

        self.console.print(
            f"[red]Attempting to post error message to {channel_id}/{thread_ts}...[/red]"
        )

        try:
            import asyncio

            async def _post_error():
                return await self.slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=user_message,
                )

            result = asyncio.run(_post_error())

            if result["ok"]:
                self.console.print(
                    f"[green] -> Successfully posted error message to thread {channel_id}/{thread_ts}[/green]"
                )
            else:
                error_details = result.get("error", "Unknown Slack posting error")
                log_msg = f"Failed to post error message to Slack thread {channel_id}/{thread_ts}: {error_details}"
                logger.error(log_msg)
                self.console.print(f"[bold red] -> {log_msg}[/bold red]")

        except Exception as e:
            log_msg = f"Unexpected exception posting error message to Slack: {e}"
            logger.error(log_msg, exc_info=self.verbose)
            self.console.print(f"[bold red] -> {log_msg}[/bold red]")

        # This node signifies the end of the workflow path due to an error.
        # It doesn't return state updates, the graph transitions to END from here.
        return {}

    # --- New Conditional Logic for Finishing ---
    def check_if_response_sent(self, state: SlackResponderState) -> Dict[str, Any]:
        """Checks if the last action was successfully sending the response to Slack and updates the state.
        This node modifies the 'response_sent' flag in the state.
        """
        messages = state.get("messages", [])
        state_has_qa_answer = state.get("qa_final_answer") is not None
        last_tool_message: Optional[ToolMessage] = None
        # Find the last ToolMessage and the AIMessage that preceded it
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], ToolMessage):
                last_tool_message = messages[i]
                break

        # Define the names of tools that signify a final response
        final_response_tool_names = [
            "post_final_response_with_snippet",
            "post_text_response",  # Also consider the simple text response tool
        ]

        # Check if the last tool was a final response tool
        is_final_response_tool = (
            last_tool_message and last_tool_message.name in final_response_tool_names
        )

        # Determine if workflow should end
        if state_has_qa_answer and is_final_response_tool:
            # If the QA answer is set and the last tool was a final response tool,
            # and we reached here (meaning should_continue_or_handle_error found no error in the tool execution),
            # we need to parse the tool's output content to confirm *it* succeeded.
            try:
                parsed_content = last_tool_message.content
                if isinstance(parsed_content, str):
                    import json

                    parsed_content = json.loads(parsed_content)
                if isinstance(parsed_content, dict) and parsed_content.get("success"):
                    if self.verbose:
                        self.console.print(
                            f"[green] -> Final response successfully sent to Slack via snippet. Ending workflow.[/green]"
                        )
                    return {"response_sent": True}  # Update state flag
                else:
                    if self.verbose:
                        self.console.print(
                            f"[yellow] -> Final response tool {last_tool_message.name} ran but reported failure or content parsing failed. Not ending workflow yet.[/yellow]"
                        )
                    # Don't set response_sent to True if the tool itself failed internally or content parsing failed
                    return {"response_sent": False}  # Explicitly set to False
            except Exception as e:
                logger.warning(
                    f"Could not parse success status from post_final_response_with_snippet tool message content: {e}"
                )
                return {"response_sent": False}  # Explicitly set to False
        else:
            # Condition not met: QA answer wasn't set OR last tool wasn't a final response tool.
            if self.verbose:
                tool_name = getattr(last_tool_message, "name", "N/A")
                reason = (
                    "QA answer not yet set"
                    if not state_has_qa_answer
                    else f"last tool ({tool_name}) was not a final response tool"
                )
                self.console.print(
                    f"[dim] -> Workflow continuing because {reason}.[/dim]"
                )
            # No update needed if response wasn't sent
            return {"response_sent": False}  # Explicitly set to False

    def route_after_response_check(self, state: SlackResponderState) -> str:
        """Routes to record_interaction if response was sent, otherwise back to agent."""
        if state.get("response_sent"):
            # Response was sent successfully, route to record the interaction
            return "record_interaction"
        else:
            # Response not sent (or previous step wasn't respond_to_slack), continue
            return "agent"

    # --- NEW NODE: Record Interaction --- #
    def record_interaction_node(self, state: SlackResponderState) -> Dict[str, Any]:
        """Calls QuestionStorage.record_question with all collected information."""
        if self.verbose:
            self.console.print(
                "[bold blue]\n>>> Entering record_interaction_node <<<[/bold blue]"
            )

        if not self.question_storage:
            if self.verbose:
                self.console.print(
                    "[yellow dim] -> QuestionStorage not configured, skipping recording.[/yellow dim]"
                )
            return {}  # End workflow here

        try:
            # Gather all the necessary data from the state
            original_question = state.get("original_question")  # Use .get for safety
            original_message_ts = state["thread_ts"]
            response_message_ts = state.get("response_message_ts")
            response_file_message_ts = state.get(
                "response_file_message_ts"
            )  # <<< Get file_message_ts from state
            # Ensure question_text_for_db has a fallback even if original_question is None
            question_text_for_db = original_question or ""
            answer_text = state.get("qa_final_answer")
            models_used = state.get("qa_models", [])
            feedback_considered = state.get("qa_feedback", {})
            context_considered = state.get("qa_similar_original_messages", [])
            thread_history = state.get("thread_history")

            if not response_message_ts:
                logger.warning(
                    "Cannot record interaction: response_message_ts is missing from state."
                )
                if self.verbose:
                    self.console.print(
                        "[red] -> Cannot record: response_message_ts missing.[/red]"
                    )
                return {}  # Or potentially raise an error/route differently

            # Extract model names
            model_names = [
                m.get("name")
                for m in models_used
                if isinstance(m, dict) and m.get("name")
            ]

            # Prepare metadata
            metadata = {
                "feedback_considered_by_qa": feedback_considered,  # Store structure from QA
                "context_considered_by_qa": context_considered,  # Store structure from QA
                "slack_thread_history": thread_history,  # Can be large
                "channel_id": state["channel_id"],
                # Add any other relevant metadata
            }

            if self.verbose:
                # Safely log potentially None values before slicing
                original_q_safe = str(original_question or "")[:60]
                q_text_db_safe = str(question_text_for_db or "")[:60]
                answer_text_safe = str(answer_text or "")[:60]

                self.console.print(f"[dim] -> Recording interaction:")
                self.console.print(
                    f"[dim]    Original Question TS: {original_message_ts}"
                )
                self.console.print(f"[dim]    Response TS: {response_message_ts}")
                self.console.print(
                    f"[dim]    Response File ID: {response_file_message_ts}"
                )  # <<< Log file_message_ts
                self.console.print(f"[dim]    Original Text: {original_q_safe}...")
                self.console.print(
                    f"[dim]    Question Text for DB: {q_text_db_safe}..."
                )
                self.console.print(f"[dim]    Answer Text: {answer_text_safe}...")
                self.console.print(f"[dim]    Models Used: {model_names}")

            # Call the storage method (question_text_for_db should now be a string)
            question_id = self.question_storage.record_question(
                question_text=question_text_for_db,
                original_message_text=original_question
                or "",  # Ensure original_message_text is also string
                original_message_ts=original_message_ts,
                response_message_ts=response_message_ts,
                response_file_message_ts=response_file_message_ts,  # <<< Pass file_message_ts
                answer_text=answer_text or "",  # Ensure answer_text is also string
                model_names=model_names,
                metadata=metadata,
                # was_useful and feedback are set later via user interaction
            )

            if self.verbose:
                self.console.print(
                    f"[green] -> Successfully recorded interaction with ID: {question_id}[/green]"
                )

        except Exception as e:
            logger.error(
                f"Failed to record interaction to QuestionStorage: {e}",
                exc_info=self.verbose,
            )
            if self.verbose:
                self.console.print(f"[red] -> Error recording interaction: {e}[/red]")
            # Decide how to handle recording errors - potentially just log and end?

        return {}  # This node leads to END

    # --- END NEW NODE --- #

    # --- Workflow Execution ---
    def run_slack_workflow(
        self, question: str, channel_id: str, thread_ts: str
    ) -> Dict[str, Any]:
        """Runs the Slack responder workflow for a given question and thread."""
        self.console.print(
            f"[bold]🚀 Starting SlackResponder workflow for thread {channel_id}/{thread_ts}[/bold]"
        )
        self.console.print(f"[dim]Question: {question}[/dim]")

        # Generate a unique ID for this workflow instance if using memory
        # This assumes memory might be shared or needs unique identifiers per run
        # If memory is instance-specific or thread_id is handled differently, adjust this.
        workflow_id = str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": workflow_id}
        }  # Use workflow_id for LangGraph memory

        # --- MODIFIED: Update initial state ---
        initial_state = SlackResponderState(
            original_question=question,
            channel_id=channel_id,
            thread_ts=thread_ts,
            messages=[],
            thread_history=None,
            acknowledgement_sent=None,
            qa_final_answer=None,
            qa_models=None,
            qa_feedback=None,
            qa_similar_original_messages=None,  # Init new field
            error_message=None,
            response_message_ts=None,  # Init new field
            response_file_message_ts=None,  # <<< Init renamed field
            response_sent=None,
        )
        # --- END MODIFIED ---

        if self.verbose:
            self.console.print(
                f"[dim]Generated workflow ID for config: {workflow_id}[/dim]"
            )
            self.console.print("[dim]Initial state prepared.[/dim]")

        try:
            # Execute the graph
            # invoke() runs the graph until completion and returns the final state.
            final_state = self.graph_app.invoke(initial_state, config=config)

            if final_state is None:
                logger.error(
                    "Graph execution finished but failed to retrieve final state."
                )
                self.console.print(
                    "[bold red]Error: Could not retrieve final state after workflow execution.[/bold red]"
                )
                return {
                    "success": False,
                    "channel_id": channel_id,
                    "thread_ts": thread_ts,
                    "error": "Failed to retrieve final state",
                }

            self.console.print("[green]✅ SlackResponder workflow finished.[/green]")

            # Use the state dict directly from invoke result
            current_state_values = final_state

            final_error = current_state_values.get("error_message")
            # --- MODIFIED: Check qa_structured_result presence ---
            final_answer_provided = (
                current_state_values.get("qa_final_answer") is not None
            )
            # --- END MODIFIED ---

            if self.verbose:
                if final_error:
                    self.console.print(
                        f"[dim]Workflow ended with error: {final_error}[/dim]"
                    )

            # Return results (actual Slack posting happens via tool or error node)
            return {
                "success": final_error is None,  # Success means no error at the end
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "final_answer_provided": final_answer_provided,  # Now indicates if structured result was generated
                "error": final_error,  # Include final error message if any
            }

        except Exception as e:
            logger.error(
                f"Error during SlackResponder workflow execution: {str(e)}",
                exc_info=self.verbose,
            )
            self.console.print(
                f"[bold red]Error during SlackResponder workflow:[/bold red] {str(e)}"
            )
            return {
                "success": False,
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "error": str(e),
            }
