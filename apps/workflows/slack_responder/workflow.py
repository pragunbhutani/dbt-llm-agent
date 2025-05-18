# core/agents/slack_responder/workflow.py
import logging
import uuid
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Set, TypedDict
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
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from psycopg_pool import ConnectionPool

# Django & Project Imports
from django.conf import settings
from django.utils import timezone  # For recording interaction time
from asgiref.sync import sync_to_async  # For DB operations in async context

# Import our models and the other agent
from apps.workflows.models import Question
from apps.workflows.question_answerer import QuestionAnswererAgent

# Import prompts
from .prompts import (
    create_slack_responder_system_prompt,
)  # Assuming prompts are defined here

# Import Slack SDK components if needed directly (though Bolt client is preferred)
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)


# --- Tool Input Schemas ---
class FetchThreadInput(BaseModel):
    pass  # Indicates no arguments are expected from the LLM


class AskQuestionAnswererInput(BaseModel):
    question: str = Field(
        description="The formulated question to send to the QuestionAnswerer agent."
    )
    thread_context: Optional[List[Dict[str, Any]]] = Field(default=None)


class AcknowledgeQuestionInput(BaseModel):
    acknowledgement_text: str


class PostFinalResponseInput(BaseModel):
    message_text: str
    sql_query: str
    optional_notes: Optional[str] = None


class PostTextResponseInput(BaseModel):
    message_text: str


# --- LangGraph State Definition (Copied from original) ---
class SlackResponderState(TypedDict):
    original_question: str
    channel_id: str
    thread_ts: str
    user_id: str  # Added user_id for context/recording
    messages: Annotated[List[BaseMessage], add_messages]
    thread_history: Optional[List[Dict[str, Any]]]
    acknowledgement_sent: Optional[bool]
    qa_final_answer: Optional[str]
    qa_models: Optional[List[Dict[str, Any]]]  # Keep track of models used by QA
    error_message: Optional[str]
    response_sent: Optional[bool]
    response_message_ts: Optional[str]
    response_file_message_ts: Optional[str]
    recorded_question_id: Optional[int]  # To link feedback later


# --- Slack Responder Agent ---
class SlackResponderAgent:
    """Orchestrates Slack interaction and calls QuestionAnswererAgent."""

    def __init__(
        self,
        slack_client: AsyncWebClient,  # Pass the client from Bolt
        memory: Optional[BaseCheckpointSaver] = None,
    ):
        self.slack_client = slack_client
        # Set verbosity based on Django settings
        self.verbose = settings.RAGSTAR_LOG_LEVEL == "DEBUG"
        if self.verbose:
            logger.info(
                f"SlackResponderAgent initialized with verbose=True (LogLevel: {settings.RAGSTAR_LOG_LEVEL})"
            )

        # Instantiate the QuestionAnswerer agent internally
        # Note: QA agent now uses Django ORM/services, no direct DB deps needed here
        self.question_answerer = QuestionAnswererAgent(
            # verbose=self.verbose, # QA agent will also use settings
            memory=memory,
            data_warehouse_type=getattr(settings, "DATA_WAREHOUSE_TYPE", None),
        )  # Pass memory if QA should use same checkpoint
        self.llm = (
            self.question_answerer.llm
        )  # Use the LLM from the QA agent for consistency
        self.memory = (
            memory  # Use the same memory as QA agent for this orchestration layer
        )

        # For storing current request's context
        self.current_channel_id: Optional[str] = None
        self.current_thread_ts: Optional[str] = None

        self._define_tools()
        self.graph_app = self._build_graph()

    # --- Tool Definitions ---
    def _define_tools(self):

        @tool(args_schema=FetchThreadInput)
        async def fetch_slack_thread(
            # No arguments needed in signature now
        ) -> List[Dict[str, Any]]:
            """Fetches the message history from the current Slack thread in context."""
            if self.verbose:
                logger.info(f"\nTool: fetch_slack_thread (using agent context for IDs)")

            if not self.current_channel_id or not self.current_thread_ts:
                error_msg = "Channel ID or Thread TS not set in agent context for fetch_slack_thread"
                logger.error(error_msg)
                # Return an error structure that ToolNode can hopefully relay
                # Or, if this tool is critical and shouldn't proceed, could raise an exception
                return [
                    {"error": error_msg, "user": "system", "text": error_msg, "ts": "0"}
                ]

            try:
                result = await self.slack_client.conversations_replies(
                    channel=self.current_channel_id, ts=self.current_thread_ts, limit=20
                )
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
                        logger.info(f"Fetched {len(history)} messages.")
                    return history
                else:
                    error_msg = f"Slack API error (replies): {result['error']}"
                    logger.error(error_msg)
                    return {"error": error_msg}
            except Exception as e:
                error_msg = f"Error fetching thread: {e}"
                logger.exception(error_msg)
                return {"error": error_msg}

        @tool(args_schema=AcknowledgeQuestionInput)
        async def acknowledge_question(acknowledgement_text: str) -> Dict[str, Any]:
            """Posts a brief acknowledgement message to the Slack thread."""
            if self.verbose:
                logger.info(f"\nTool: acknowledge_question(text=...)")
            if not self.current_channel_id or not self.current_thread_ts:
                error_msg = "Channel ID or Thread TS not set in agent context for acknowledge_question"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            try:
                result = await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=acknowledgement_text,
                )
                if result["ok"]:
                    if self.verbose:
                        logger.info("Posted acknowledgement.")
                    return {"success": True, "message_ts": result.get("ts")}
                else:
                    error_msg = f"Slack API error (ack post): {result['error']}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg}
            except Exception as e:
                error_msg = f"Error posting acknowledgement: {e}"
                logger.exception(error_msg)
                return {"success": False, "error": error_msg}

        @tool(args_schema=AskQuestionAnswererInput)
        async def ask_question_answerer(
            question: str, thread_context: Optional[List[Dict[str, Any]]] = None
        ) -> Dict[str, Any]:
            """Sends a question to the specialized QuestionAnswerer agent."""
            if self.verbose:
                logger.info(
                    f"Tool: ask_question_answerer(question='{question[:50]}...')"
                )
            try:
                # Now await the async QA workflow
                qa_final_state = await self.question_answerer.run_agentic_workflow(
                    question=question,
                    thread_context=thread_context,
                    # conversation_id can be managed by QA agent itself if needed
                )
                # Extract the required fields from the QA agent's *returned dictionary*
                # The key for the answer in the returned dict is "answer"
                final_answer = qa_final_state.get("answer")
                models_used = qa_final_state.get("models_used", [])

                # The warning condition should check if final_answer is still None or "Could not determine..."
                if (
                    final_answer is None
                    or final_answer == "Could not determine a final answer."
                ):
                    logger.warning(
                        f"QuestionAnswerer did not produce a valid answer for question: '{question[:50]}...'. QA response: {qa_final_state}"
                    )
                    return {
                        "answer": "Could not determine a final answer.",
                        "models_used": models_used,
                        "error": qa_final_state.get("error")
                        or "QA agent did not produce a valid answer.",
                    }

                # Return the structured result
                return {
                    "answer": final_answer,
                    "models_used": models_used,
                }
            except Exception as e:
                error_msg = f"Error calling QuestionAnswerer: {e}"
                logger.exception(error_msg)
                return {"error": error_msg}

        @tool(args_schema=PostFinalResponseInput)
        async def post_final_response_with_snippet(
            message_text: str,
            sql_query: str,
            optional_notes: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Posts the final answer message and uploads the SQL as a snippet."""
            if self.verbose:
                logger.info(f"\nTool: post_final_response_with_snippet(sql=...)")
            if not self.current_channel_id or not self.current_thread_ts:
                error_msg = "Channel ID or Thread TS not set in agent context for post_final_response_with_snippet"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            try:
                # 1. Upload Snippet
                upload_result = await self.slack_client.files_upload_v2(
                    channel=self.current_channel_id,
                    content=sql_query,
                    filename="query.sql",
                    initial_comment=message_text,  # Main text goes here
                    thread_ts=self.current_thread_ts,
                    request_file_info=False,  # Don't need full file info back
                )
                file_upload_ts = None
                if upload_result["ok"] and upload_result.get(
                    "files"
                ):  # Check for files key
                    # Get timestamp of the file upload message
                    file_upload_messages = upload_result.get("files", [])
                    if file_upload_messages:
                        # Assuming the first message is the one in the thread
                        file_upload_ts = file_upload_messages[0].get("ts")
                    if not file_upload_ts:  # Fallback if ts not directly in files
                        file_upload_ts = upload_result.get(
                            "ts"
                        )  # Might be top-level ts
                    logger.info(
                        f"Successfully uploaded snippet to {self.current_channel_id}/{self.current_thread_ts}. FileMsgTS: {file_upload_ts}"
                    )
                else:
                    error_msg = f"Slack API error (files.upload): {upload_result.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    # Try posting text only as fallback?
                    return {"success": False, "error": error_msg}

                # 2. Post Optional Notes (if any) - as separate message for clarity
                notes_message_ts = None
                if optional_notes:
                    notes_text = f"*Notes:*\n{optional_notes}\n\n_💡 React with 👍 or 👎 to the SQL snippet message above to provide feedback._"
                    notes_result = await self.slack_client.chat_postMessage(
                        channel=self.current_channel_id,
                        thread_ts=self.current_thread_ts,
                        text=notes_text,
                    )
                    if notes_result["ok"]:
                        notes_message_ts = notes_result.get("ts")
                        logger.info("Successfully posted optional notes.")
                    else:
                        logger.warning(
                            f"Failed to post optional notes: {notes_result.get('error')}"
                        )
                else:
                    # Add feedback prompt to main message if no separate notes
                    feedback_prompt = "\n\n_💡 React with 👍 or 👎 to the SQL snippet message above to provide feedback._"
                    await self.slack_client.chat_postMessage(
                        channel=self.current_channel_id,
                        thread_ts=self.current_thread_ts,
                        text=feedback_prompt,
                    )

                return {
                    "success": True,
                    "message": "Final response and snippet posted.",
                    "response_message_ts": notes_message_ts,  # TS of the text message (notes or just feedback prompt)
                    "response_file_message_ts": file_upload_ts,  # TS of the message containing the file
                }

            except Exception as e:
                error_msg = f"Error posting final response/snippet: {e}"
                logger.exception(error_msg)
                return {"success": False, "error": error_msg}

        @tool(args_schema=PostTextResponseInput)
        async def post_text_response(message_text: str) -> Dict[str, Any]:
            """Posts a simple plain text message to the specified Slack thread."""
            if self.verbose:
                logger.info(f"\nTool: post_text_response(text=...)")
            if not self.current_channel_id or not self.current_thread_ts:
                error_msg = "Channel ID or Thread TS not set in agent context for post_text_response"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            footnote = (
                "\n\n_💡 React with 👍 or 👎 to this message to provide feedback._"
            )
            full_message_text = message_text + footnote
            try:
                result = await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=full_message_text,
                )
                if result["ok"]:
                    ts = result.get("ts")
                    if self.verbose:
                        logger.info(f"Posted simple text response. TS: {ts}")
                    return {"success": True, "message_ts": ts}
                else:
                    error_msg = f"Slack API error (text post): {result['error']}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg}
            except Exception as e:
                error_msg = f"Error posting text response: {e}"
                logger.exception(error_msg)
                return {"success": False, "error": error_msg}

        # TODO: Add record_interaction tool? Or handle recording within update_state?
        # @tool(...)
        # def record_interaction(...)

        self._tools = [
            fetch_slack_thread,
            acknowledge_question,
            ask_question_answerer,
            post_final_response_with_snippet,
            post_text_response,
        ]

    # --- Graph Construction (Simplified Flow) ---
    def _build_graph(self):
        workflow = StateGraph(SlackResponderState)

        tool_node = ToolNode(self._tools, handle_tool_errors=True)

        # Nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("update_state", self.update_state_node)
        workflow.add_node("record_interaction", self.record_interaction_node)

        # Edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,  # Route based on whether agent requested tools
            {
                "tools": "tools",  # New route: If tools, go directly to tools
                END: "record_interaction",
            },
        )
        workflow.add_edge("tools", "update_state")
        # After updating state based on tool results, decide if we need agent again or can record
        workflow.add_conditional_edges(
            "update_state",
            self.check_if_response_sent,  # Custom routing function
            {"agent": "agent", "record_interaction": "record_interaction"},
        )
        # After recording, end the flow
        workflow.add_edge("record_interaction", END)

        # Compile with memory if available
        compile_kwargs = {}
        if self.memory:
            compile_kwargs["checkpointer"] = self.memory
        # Ensure graph nodes are async compatible if run with ainvoke
        return workflow.compile(**compile_kwargs)

    # --- Routing Logic ---
    def check_if_response_sent(self, state: SlackResponderState) -> str:
        """Routes to agent if no response sent yet, otherwise records interaction."""
        if state.get("response_sent"):
            if self.verbose:
                logger.info("Response sent, routing to record_interaction.")
            return "record_interaction"
        else:
            if self.verbose:
                logger.info("Response not sent, routing back to agent.")
            return "agent"

    # --- Agent Node ---
    def agent_node(self, state: SlackResponderState) -> Dict[str, Any]:
        """Main logic node for Slack Responder LLM call."""
        if self.verbose:
            logger.info("\n>>> Entering SlackResponder agent_node")

        # Prepare context for the LLM
        current_history = state["messages"]
        system_prompt_content = create_slack_responder_system_prompt(
            original_question=state["original_question"],
            thread_history=state.get("thread_history"),
            qa_final_answer=state.get("qa_final_answer"),
            qa_models=state.get("qa_models"),
            acknowledgement_sent=state.get("acknowledgement_sent"),
            error_message=state.get("error_message"),
        )
        if self.verbose:
            logger.info(
                f"--- SlackResponder System Prompt Content ---\n{system_prompt_content}\n--- End System Prompt ---"
            )

        messages_to_send_to_llm: List[BaseMessage]
        # Store the human message if it's the first turn, to add to state later
        first_turn_human_message: Optional[HumanMessage] = None

        if not current_history:  # This is the first turn for the agent
            human_message = HumanMessage(content=state["original_question"])
            first_turn_human_message = human_message  # Capture for adding to state
            messages_to_send_to_llm = [
                SystemMessage(content=system_prompt_content),
                human_message,
            ]
        else:  # Subsequent turns, messages list already contains the history
            messages_to_send_to_llm = [
                SystemMessage(content=system_prompt_content)
            ] + current_history

        if self.verbose:
            logger.info(
                f"Calling SlackResponder LLM with {len(messages_to_send_to_llm)} messages."
            )
            # Detailed log of messages being sent
            log_messages = []
            for msg in messages_to_send_to_llm:
                if isinstance(msg, SystemMessage):
                    log_messages.append(f"System: {msg.content[:200]}...")
                elif isinstance(msg, HumanMessage):
                    log_messages.append(f"Human: {msg.content[:200]}...")
                elif isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        tool_calls_summary = ", ".join(
                            [
                                f"{tc['name']}(args={str(tc['args'])[:50]}...)"
                                for tc in msg.tool_calls
                            ]
                        )
                        log_messages.append(
                            f"AI: ToolCalls({tool_calls_summary}) Content: {msg.content[:100]}..."
                        )
                    else:
                        log_messages.append(f"AI: {msg.content[:200]}...")
                elif isinstance(msg, ToolMessage):
                    log_messages.append(
                        f"Tool (id={msg.tool_call_id}): {msg.content[:200]}..."
                    )
                else:
                    log_messages.append(f"UnknownMessage: {str(msg)[:200]}...")
            logger.info(
                f"Messages for LLM (SlackResponder):\n" + "\n".join(log_messages)
            )

        if not self.llm:
            error_ai_message = AIMessage(content="Error: LLM client not available.")
            if first_turn_human_message:
                return {"messages": [first_turn_human_message, error_ai_message]}
            return {"messages": [error_ai_message]}

        # Bind tools and invoke
        agent_llm_with_tools = self.llm.bind_tools(self._tools)
        config = {"run_name": "SlackResponderAgentNode"}

        newly_added_messages_to_state: List[BaseMessage]

        try:
            response = agent_llm_with_tools.invoke(
                messages_to_send_to_llm, config=config
            )

            # --- Add thread_context to ask_question_answerer tool call args if present ---
            if response.tool_calls:
                for call in response.tool_calls:
                    if call.get("name") == "ask_question_answerer":
                        # Ensure args exist and add thread_history from state if available
                        call["args"] = call.get("args", {})
                        if state.get("thread_history"):
                            call["args"]["thread_context"] = state["thread_history"]
            # --- End modification ---

            if self.verbose:
                logger.info(f"SlackResponder Agent response: {response}")

            if first_turn_human_message:
                newly_added_messages_to_state = [first_turn_human_message, response]
            else:
                newly_added_messages_to_state = [response]
            return {"messages": newly_added_messages_to_state}

        except Exception as e:
            logger.exception(f"Error invoking SlackResponder LLM: {e}")
            error_ai_message = AIMessage(content=f"LLM Error: {e}")
            if first_turn_human_message:
                newly_added_messages_to_state = [
                    first_turn_human_message,
                    error_ai_message,
                ]
            else:
                newly_added_messages_to_state = [error_ai_message]
            return {"messages": newly_added_messages_to_state}

    # --- State Update Node ---
    def update_state_node(self, state: SlackResponderState) -> Dict[str, Any]:
        """Processes tool results and updates the SlackResponderState."""
        if self.verbose:
            logger.info("--- Updating SlackResponder State ---")
        updates: Dict[str, Any] = {}
        state_changed = False  # Initialize state_changed
        tool_processed = False  # Initialize tool_processed
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if not isinstance(last_message, ToolMessage):
            if self.verbose:
                logger.info("No tool message to process for state update.")
            return updates  # No tool was called or tool failed before returning message

        tool_content = last_message.content
        if isinstance(tool_content, str) and tool_content.startswith("{"):
            try:
                tool_content = json.loads(tool_content)
            except json.JSONDecodeError:
                pass

        if self.verbose:
            logger.info(f"Processing ToolMessage: {last_message.name}")

        # Update state based on tool call
        if last_message.name == "fetch_slack_thread":
            if isinstance(tool_content, list):
                updates["thread_history"] = tool_content
                state_changed = True  # History was updated
            elif isinstance(tool_content, dict) and tool_content.get("error"):
                updates["error_message"] = (
                    f"Failed to fetch thread: {tool_content['error']}"
                )
                logger.error(f"Tool fetch_slack_thread failed: {tool_content['error']}")
            tool_processed = True

        elif last_message.name == "acknowledge_question":
            if isinstance(tool_content, dict) and tool_content.get("success"):
                updates["acknowledgement_sent"] = True
                state_changed = True  # Acknowledgement status changed
            else:
                # Handle potential failure - check type before accessing keys
                error_detail = "Unknown acknowledge failure"
                if isinstance(tool_content, dict):
                    error_detail = tool_content.get("error", "Unknown error in dict")
                elif isinstance(tool_content, str):
                    error_detail = tool_content  # Log the string error
                else:
                    error_detail = f"Unexpected content type: {type(tool_content)}"

                logger.error(f"Tool acknowledge_question failed: {error_detail}")
                # Optionally set error message in state
                # updates['error_message'] = f"Failed to acknowledge: {error_detail}"
            tool_processed = True

        elif last_message.name == "ask_question_answerer":
            if isinstance(tool_content, dict):
                # Check for error from QA agent first
                if tool_content.get("error"):
                    error_msg = tool_content.get("error", "QA agent failed")
                    updates["error_message"] = error_msg
                    logger.error(
                        f"Tool ask_question_answerer returned error: {error_msg}"
                    )
                else:
                    # Success - store answer and models used
                    updates["qa_final_answer"] = tool_content.get(
                        "answer"
                    )  # Changed key from final_answer
                    updates["qa_models"] = tool_content.get(
                        "models_used", []
                    )  # Get models_used list
                    state_changed = True  # Mark state as changed
                    if self.verbose:
                        logger.info(
                            f"Received QA answer: {str(updates['qa_final_answer'])[:100]}..."
                        )
                    if self.verbose:
                        logger.info(
                            f"QA agent used models: {[m.get('name') for m in updates['qa_models'] if m]}"
                        )
            else:
                logger.error(
                    f"Unexpected result format from ask_question_answerer: {type(tool_content)}"
                )
                updates["error_message"] = "Internal error processing QA result."
            tool_processed = True  # Mark as processed even if error occurred

        elif last_message.name == "post_final_response_with_snippet":
            if isinstance(tool_content, dict) and tool_content.get("success"):
                updates["response_sent"] = True
                updates["response_message_ts"] = tool_content.get("response_message_ts")
                updates["response_file_message_ts"] = tool_content.get(
                    "response_file_message_ts"
                )
                state_changed = True  # Response status changed
            else:
                error_msg = tool_content.get("error", "Failed to post final response")
                updates["error_message"] = (
                    error_msg  # Store error to potentially end gracefully
                )
                logger.error(
                    f"Tool post_final_response_with_snippet failed: {error_msg}"
                )
            tool_processed = True  # Mark as processed even if error occurred

        elif last_message.name == "post_text_response":
            if isinstance(tool_content, dict) and tool_content.get("success"):
                updates["response_sent"] = True  # Mark response as sent
                updates["response_message_ts"] = tool_content.get("message_ts")
                updates["response_file_message_ts"] = None  # No file for text response
                state_changed = True  # Response status changed
            else:
                error_msg = tool_content.get("error", "Failed to post text response")
                updates["error_message"] = error_msg
                logger.error(f"Tool post_text_response failed: {error_msg}")
            tool_processed = True  # Mark as processed even if error occurred

        # Add other tool result processing here
        # Ensure tool_processed = True is set for any other tool handled

        if not tool_processed:
            logger.warning(
                f"Tool '{last_message.name}' result was not processed in update_state_node."
            )

        if self.verbose and updates:
            logger.info(f"SlackResponder state updates: {list(updates.keys())}")

        # --- Prepare updates dictionary ---
        # This section seems redundant now as updates are built directly.
        # If state_changed check was intended for something else, it needs clarification.
        # Removing the potentially problematic conditional block for now.
        # if state_changed:
        # Check individual fields against original state before adding to updates
        # ... (other state field checks) ...
        # if updates.get("qa_final_answer") != state.get("qa_final_answer"):
        #     pass  # Already in updates if changed
        # if updates.get("qa_models") != state.get("qa_models"):
        #     pass  # Already in updates if changed
        # ... (rest of the state update checks) ...

        return updates

    # --- Interaction Recording Node ---
    @sync_to_async  # Make the DB operation async compatible
    def _record_interaction_db(self, state: SlackResponderState):
        """Helper function to perform the DB save operation."""
        # Fetch thread_ts early for reliable logging in except block
        thread_ts = state.get("thread_ts", "unknown_thread")
        try:
            # Extract necessary data from state
            question_text = state["original_question"]
            answer_text = state.get("qa_final_answer")

            # Explicitly handle None for qa_models, defaulting to empty list
            models_used_list = state.get("qa_models")
            if models_used_list is None:
                models_used_list = []

            # Get model names used from the list of dicts stored in the state
            model_names = [
                m.get("name")
                for m in models_used_list
                if isinstance(m, dict) and m.get("name")
            ]

            response_ts = state.get("response_message_ts")
            response_file_ts = state.get("response_file_message_ts")
            channel_id = state.get("channel_id")
            user_id = state.get("user_id")

            # Create Question record
            question_record = Question.objects.create(
                question_text=question_text,
                answer_text=answer_text,
                original_message_text=question_text,
                original_message_ts=thread_ts,  # Use variable fetched earlier
                response_message_ts=response_ts,
                response_file_message_ts=response_file_ts,
                question_metadata={
                    "channel_id": channel_id,
                    "user_id": user_id,
                    "slack_thread_ts": thread_ts,  # Use variable fetched earlier
                    "agent_type": "SlackResponder",
                    "created_at_iso": timezone.now().isoformat(),
                    "qa_models_used": model_names,  # Store the list of model names used
                },
            )
            logger.info(
                f"Recorded interaction for thread {thread_ts} as Question ID {question_record.pk} (Models: {model_names})"
            )
            return question_record.pk
        except Exception as e:
            # thread_ts is now available here
            logger.error(
                f"Failed to record interaction for thread {thread_ts}: {e}",
                exc_info=True,
            )
            return None

    async def record_interaction_node(
        self, state: SlackResponderState
    ) -> Dict[str, Any]:
        """Records the question and answer interaction to the database."""
        if self.verbose:
            logger.info("--- Recording Interaction ---")

        # Don't record if no response was actually sent
        if not state.get("response_sent"):
            logger.warning(
                f"Skipping recording for thread {state.get('thread_ts')} as no response was marked as sent."
            )
            return {}

        question_id = await self._record_interaction_db(state)

        if question_id:
            return {"recorded_question_id": question_id}
        else:
            return {"error_message": "Failed to record interaction in database."}

    # --- Main Workflow Runner ---
    async def run_slack_workflow(
        self, question: str, channel_id: str, thread_ts: str, user_id: str
    ) -> None:
        """Runs the full Slack interaction workflow asynchronously."""
        if self.verbose:
            logger.info(
                f"Starting SlackResponder workflow for question in {channel_id}/{thread_ts}"
            )

        # Set current context for this run
        self.current_channel_id = channel_id
        self.current_thread_ts = thread_ts

        conversation_id = (
            f"slack-{channel_id}-{thread_ts}"  # Unique ID for checkpointing
        )
        config = {"configurable": {"thread_id": conversation_id}}

        # Initial state
        initial_state = SlackResponderState(
            original_question=question,
            channel_id=channel_id,
            thread_ts=thread_ts,
            user_id=user_id,
            messages=[],
            thread_history=None,
            acknowledgement_sent=None,
            qa_final_answer=None,
            qa_models=None,
            error_message=None,
            response_sent=None,
            response_message_ts=None,
            response_file_message_ts=None,
            recorded_question_id=None,
        )

        try:
            if self.verbose:
                logger.info(f"Invoking SlackResponder graph for {conversation_id}")

            # Use asynchronous invocation
            final_state = await self.graph_app.ainvoke(initial_state, config=config)

            if self.verbose:
                logger.info(f"SlackResponder graph finished for {conversation_id}.")
                # Optional: Log final state details if needed
                # logger.debug(f"Final state for {conversation_id}: {final_state}")

            # Check for errors in the final state, although ideally handled by error nodes
            if final_state and final_state.get("error_message"):
                logger.error(
                    f"Workflow for {conversation_id} completed with error state: {final_state['error_message']}"
                )
                # Graph's error handling nodes should manage sending messages.

            # Clear context after run
            self.current_channel_id = None
            self.current_thread_ts = None

        except Exception as e:
            logger.exception(
                f"Critical error running SlackResponder graph for {conversation_id}: {e}"
            )
            # Clear context in case of error too
            self.current_channel_id = None
            self.current_thread_ts = None
            # Try to send a final error message directly via Slack client
            try:
                # Need to use the async client stored in self.slack_client
                await self.slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=f"Sorry <@{user_id}>, a critical internal error occurred while processing your request.",
                )
            except Exception as post_err:
                logger.error(
                    f"Failed to send final critical error message to Slack for {conversation_id}: {post_err}"
                )

        # The function returns None, results are posted via tools within the graph
