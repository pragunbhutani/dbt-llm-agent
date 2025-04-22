import logging
import uuid
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

# RAGstar Imports (assuming QuestionAnswerer is accessible)
from ragstar.core.llm.client import LLMClient
from ragstar.utils.cli_utils import get_config_value

# Import components needed to initialize QuestionAnswerer
from ragstar.storage.model_storage import ModelStorage
from ragstar.storage.model_embedding_storage import ModelEmbeddingStorage
from ragstar.storage.question_storage import QuestionStorage
from ragstar.core.agents.question_answerer import (
    QuestionAnswerer,
)

# Need to potentially create a slack integration module
# from ragstar.integrations.slack import get_thread_history, post_message_to_thread

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
    contextual_question: Optional[str]  # Question formulated with context
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
            try:
                import asyncio

                async def _post_text():
                    return await self.slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=message_text,  # Simple text message
                    )

                result = asyncio.run(_post_text())
                if result["ok"]:
                    if self.verbose:
                        self.console.print(
                            f"[green] -> Successfully posted simple text response to {channel_id}/{thread_ts}[/green]"
                        )
                    return {"success": True, "message": "Text response sent."}
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
                        "models": qa_result_dict.get("searched_models", []),
                        "feedback": qa_result_dict.get("relevant_feedback", []),
                    }
                elif qa_result_dict and "final_answer" in qa_result_dict:
                    # Return success structure with the answer text and context
                    return {
                        "success": True,
                        "final_answer": qa_result_dict.get("final_answer"),
                        "models": qa_result_dict.get("searched_models", []),
                        "feedback": qa_result_dict.get("relevant_feedback", []),
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
            try:
                import asyncio

                async def _upload_and_post():
                    # 1. Post the main message referencing the snippet-to-be-uploaded
                    full_message = message_text
                    if optional_notes:
                        full_message += f"\n\n*Notes:*\n{optional_notes}"

                    post_result = await self.slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=full_message,
                        blocks=[
                            {
                                "type": "section",
                                "text": {"type": "mrkdwn", "text": full_message},
                            }
                        ],
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

                    # 2. Upload the SQL query as a snippet *after* the message
                    upload_result = await self.slack_client.files_upload_v2(
                        channel=channel_id,
                        content=sql_query,
                        # filetype="sql", # Removed unsupported parameter
                        title="Generated SQL Query",
                        filename="generated_query.sql",
                        # initial_comment=f"SQL query for thread {thread_ts}", # Removed as comment is in message above
                        thread_ts=thread_ts,  # Keep in the same thread
                    )

                    # files_upload_v2 returns a different structure, check 'ok' directly
                    if not upload_result or not upload_result.get(
                        "ok"
                    ):  # Check 'ok' in the response dict
                        # Note: The message was already posted successfully at this point.
                        # We might want to indicate partial success or post a follow-up error.
                        # For now, report the upload failure.
                        error_msg = f"Message posted, but Slack API error uploading snippet (files.upload_v2): {upload_result.get('error', 'Unknown upload error')}"
                        logger.error(error_msg)
                        # Return failure despite message success, as snippet failed.
                        return {
                            "success": False,
                            "error": error_msg,
                            "details": {"slack_error": upload_result.get("error")},
                        }

                    # If both message post and snippet upload succeed
                    return {
                        "success": True,
                        "message": "Successfully posted response and uploaded snippet.",
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

        # --- END NEW TOOL ---

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
                "agent": "agent",  # If response not sent, loop back to agent
                END: END,  # If response was sent, end the graph
            },
        )

        # Error handler leads to END
        workflow.add_edge("error_responder", END)

        # Compile the graph with optional memory
        return workflow.compile(checkpointer=self.memory)

    def _get_agent_llm(self):
        """Helper to get the LLM with tools bound."""
        chat_client_instance = self.llm.chat_client
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
            # --- MODIFIED Logging for QA Result ---
            self.console.print(
                f"[dim]QA final answer received: {'Yes' if state.get('qa_final_answer') else 'No'}[/dim]"
            )
            # --- END MODIFIED Logging ---

        messages = state.get("messages", [])
        # --- NEW: Get QA context from state ---
        qa_final_answer_text = state.get("qa_final_answer")
        qa_models_used = state.get("qa_models")
        qa_feedback_used = state.get("qa_feedback")
        thread_history = state.get("thread_history")

        # --- Determine the next step based on state ---
        if not messages:
            # --- MODIFIED: Updated Initial System Prompt ---
            system_prompt = """You are an AI assistant integrated with Slack. Your primary role is to manage the workflow for answering user questions asked in Slack threads.

**Overall Workflow:**
1.  A user asks a question in a Slack thread.
2.  **Fetch Context:** Use `fetch_slack_thread` to get history.
3.  **Acknowledge:** Use `acknowledge_question` to confirm receipt and understanding.
4.  **Delegate to QA Agent:** Use `ask_question_answerer` with the formulated question to get a detailed text answer (containing SQL, explanations, notes) and context (models/feedback used) from a specialized agent.
5.  **Process QA Result:** Once the QA agent responds, you (the SlackResponder) will receive its text answer, list of models used, and list of feedback considered.
6.  **Verify & Prepare Final Response:** You will then analyze the QA agent's text answer, verify the SQL query within it against the provided context (models, feedback, thread history), extract notes, compose a user-facing message, and prepare to post it.
7.  **Post Final Response:** Use `post_final_response_with_snippet` to upload the verified SQL as a snippet and post the final message.

**Your Current Task is determined by the state:**
- If thread history is missing, call `fetch_slack_thread`.
- If acknowledgement hasn't been sent, analyze history and call `acknowledge_question`.
- If acknowledgement is sent but QA answer is missing, call `ask_question_answerer`.
- If QA answer and context are available, perform step 6 (Verify & Prepare) and then step 7 (Post Final Response).

**Responsibility:** Manage this sequence. Call tools appropriately. When processing the QA result, ensure the verification step occurs before posting.
"""
            # --- END MODIFIED Prompt ---
            initial_human_message = f'New question received from Slack:\nChannel ID: {state["channel_id"]}\nThread TS: {state["thread_ts"]}\nQuestion: "{state["original_question"]}"\n\nPlease fetch the thread history first.'
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

            # Construct a specific prompt for verification and final tool call
            verification_system_prompt = f"""Your task is to process the final answer received from the QuestionAnswerer agent, verify its SQL query, and format the final response for Slack.

**Context Provided:**
1.  **Original User Question:** {state['original_question']}
2.  **Slack Thread History:** {thread_history}
3.  **QuestionAnswerer (QA) Final Answer Text:**
    ```
    {qa_final_answer_text}
    ```
4.  **Models Used by QA:** {qa_models_used}
5.  **Feedback Considered by QA:** {qa_feedback_used}

**Your Steps:**
1.  **Extract SQL:** Identify and extract the complete SQL query from the 'QA Final Answer Text'. If no SQL query is present, note this.
2.  **Extract Notes:** Identify and extract any footnotes or explanations intended for the user from the 'QA Final Answer Text' (usually in a 'Footnotes:' section after the SQL).
3.  **Verify SQL (if extracted):**
    *   **CRITICAL GROUNDING CHECK:** Does the extracted SQL query use ONLY tables, columns, and relationships explicitly mentioned or clearly derivable from the schemas listed in the 'Models Used by QA' context? Check carefully for any hallucinated table or column names. **If the query is NOT grounded (uses hallucinated elements), verification FAILS.**
    *   **LOGICAL COMPLETENESS CHECK:** Does the SQL query logically attempt to answer the 'Original User Question' given the context? Acknowledge that the query might have limitations noted in the 'Footnotes' or if ideal models weren't available. **Verification PASSES if the SQL is *grounded*, even if it doesn't fully answer the question.** Your analysis of its limitations should be added to the `optional_notes`.
    *   Is the SQL syntax likely correct (basic check)?
4.  **Compose Message Text:** Write a brief, friendly introductory message for the Slack post (e.g., "Here's the SQL query based on the available data models:" or "Here's the SQL query generated based on the information provided. Please note the following limitations:").
5.  **Call Final Tool:**
    *   **If SQL was extracted AND verification passes (SQL is grounded, even if logically incomplete):** Call the `post_final_response_with_snippet` tool. Provide:
        *   `channel_id`: {state['channel_id']}
        *   `thread_ts`: {state['thread_ts']}
        *   `message_text`: Your composed introductory message.
        *   `sql_query`: The verified (grounded) SQL query you extracted.
        *   `optional_notes`: Combine the notes/footnotes you extracted from the QA answer AND any limitations you identified during the 'LOGICAL COMPLETENESS CHECK'.
    *   **If verification fails (SQL is ungrounded) OR no SQL was extracted from the QA answer:** Call the `post_text_response` tool with:
        *   `channel_id`: {state['channel_id']}
        *   `thread_ts`: {state['thread_ts']}
        *   `message_text`: Your message explaining the verification failure (e.g., "I received a response, but the SQL query references tables/columns not found in our models, so I cannot share it." or "The QuestionAnswerer could not generate a valid SQL query based on the available data models to answer your request."). **Do NOT include the problematic SQL in this message.**
"""
            # Use the most recent messages plus the new system prompt
            # Taking last ~4 messages for context + the system prompt should be enough
            relevant_history = messages[-4:] if len(messages) > 4 else messages
            messages_for_llm = relevant_history + [
                SystemMessage(content=verification_system_prompt)
            ]

            if self.verbose:
                self.console.print(
                    "[dim] -> Constructed prompt for SQL verification and final posting.[/dim]"
                )

        # --- END NEW LOGIC BLOCK ---
        else:
            # Standard guidance logic (fetch history, acknowledge, ask QA)
            guidance_items = []
            if not state.get("thread_history"):
                guidance_items.append("You MUST use 'fetch_slack_thread' now.")
            elif not state.get("acknowledgement_sent"):
                guidance_items.append(
                    "Analyze the thread history and original question. Formulate a brief acknowledgement message summarizing your understanding and use the 'acknowledge_question' tool now."
                )
            # --- MODIFIED: Check for qa_final_answer instead of qa_result ---
            elif not qa_final_answer_text:  # Check if QA answer text is missing
                # --- MODIFIED: New guidance for compiling question for QA, now including context --- #
                guidance_items.append(
                    "You have sent the acknowledgement. Now, review the original question and the full thread history."
                )
                guidance_items.append(
                    "Compile the core information request from the conversation into the 'question' argument. Correct spelling/grammar."
                )
                guidance_items.append(
                    "Simplify phrasing if necessary, but *preserve the original meaning and all key details mentioned by the user*."
                )
                guidance_items.append(
                    "You MUST also provide the full `thread_history` from the state in the `thread_context` argument when calling the `ask_question_answerer` tool."  # Emphasize passing context
                )
                # --- END MODIFIED --- #
            # This else block should now be unreachable if the new block above works correctly
            # else:
            #     guidance_items.append("Error: State indicates QA answer received, but processing block was skipped.")
            # --- END MODIFIED ---

            guidance = "Guidance: " + " ".join(guidance_items)
            messages_for_llm = list(messages)  # Make a copy
            # Add guidance if the last message isn't already guidance
            if not (
                messages_for_llm
                and isinstance(messages_for_llm[-1], SystemMessage)
                and messages_for_llm[-1].content.startswith("Guidance:")
            ):
                # --- NEW: Inject thread history into the context for the LLM's decision ---
                # We need the LLM making the *tool call* to know about the history
                # so it can correctly *populate* the thread_context argument.
                history_str = (
                    f"\\n\\n**Available Thread History:**\\n{thread_history}\\n"
                    if thread_history
                    else "\\n\\n**Thread History:** Not available or not fetched yet.\\n"
                )
                messages_for_llm.append(SystemMessage(content=history_str + guidance))
            # --- END NEW --- #

        # --- LLM Invocation (Common for all paths) ---
        agent_llm = self._get_agent_llm()
        if self.verbose:
            self.console.print(
                f"[blue dim]Sending {len(messages_for_llm)} messages to LLM...[/blue dim]"
            )

        try:
            response = agent_llm.invoke(messages_for_llm)
            if self.verbose:
                self.console.print(
                    f"[dim]LLM Response: {response.content[:100]}...[/dim]"
                )
                if hasattr(response, "tool_calls") and response.tool_calls:
                    self.console.print(f"[dim]Tool calls: {response.tool_calls}[/dim]")
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
                        updates["qa_models"] = parsed_content.get("models", [])
                        updates["qa_feedback"] = parsed_content.get("feedback", [])
                        updates["error_message"] = (
                            None  # Explicitly clear SlackResponder error state
                        )
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
                        # --- END MODIFIED Success Block ---
                    else:
                        # --- Failure Case (error key has a value) ---
                        updates["error_message"] = (
                            f"QuestionAnswerer failed: {qa_error}"  # Use the actual error
                        )
                        # Store partial results if available from the error response
                        updates["qa_final_answer"] = parsed_content.get("final_answer")
                        updates["qa_models"] = parsed_content.get(
                            "models"
                        )  # Allow None
                        updates["qa_feedback"] = parsed_content.get(
                            "feedback"
                        )  # Allow None
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

            # --- MODIFIED: Handle new final response tool ---
            elif tool_name == "post_final_response_with_snippet":
                # Expecting dict with "success": True/False
                if isinstance(parsed_content, dict):
                    if parsed_content.get("success"):
                        # The check_if_response_sent node will set the flag.
                        updates["error_message"] = None  # Clear error on success
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
                            False  # Ensure flag is false on error
                        )
                        if self.verbose:
                            self.console.print(
                                f"[yellow] -> Setting error state from post_final_response_with_snippet: {updates['error_message']}[/yellow]"
                            )
                else:
                    error_msg = f"post_final_response_with_snippet tool returned unexpected content format: {type(parsed_content).__name__}"
                    logger.warning(f"{error_msg}. Content: {parsed_content}")
                    updates["error_message"] = error_msg
                    updates["response_sent"] = False  # Ensure flag is false on error

            # --- NEW: Handle simple text response tool ---
            elif tool_name == "post_text_response":
                # Expecting dict with "success": True/False
                if isinstance(parsed_content, dict):
                    if parsed_content.get("success"):
                        # This tool is also used for final responses (like verification failures).
                        # The check_if_response_sent node will handle setting the flag based on context.
                        updates["error_message"] = None  # Clear error on success
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
                        updates["response_sent"] = (
                            False  # Ensure flag is false on error
                        )
                        if self.verbose:
                            self.console.print(
                                f"[yellow] -> Setting error state from post_text_response: {updates['error_message']}[/yellow]"
                            )
                else:
                    error_msg = f"post_text_response tool returned unexpected content format: {type(parsed_content).__name__}"
                    logger.warning(f"{error_msg}. Content: {parsed_content}")
                    updates["error_message"] = error_msg
                    updates["response_sent"] = False  # Ensure flag is false on error
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
        """Routes to END if response was sent, otherwise back to agent."""
        if state.get("response_sent"):
            # Response was sent successfully, end the workflow
            return END
        else:
            # Response not sent (or previous step wasn't respond_to_slack), continue
            return "agent"

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
            contextual_question=None,
            acknowledgement_sent=None,  # Initialize new state field
            qa_final_answer=None,  # Initialize new state field
            qa_models=None,  # Initialize new state field
            qa_feedback=None,  # Initialize new state field
            error_message=None,  # Ensure error starts as None
            response_sent=None,  # Ensure response_sent starts as None
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
