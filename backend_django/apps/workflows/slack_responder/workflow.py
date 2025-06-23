# core/agents/slack_responder/workflow.py
import logging
import uuid
import json
import asyncio
import re  # Import re for regex
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
from apps.workflows.models import (
    Conversation,
    ConversationPart,
    ConversationStatus,
    ConversationTrigger,
)
from apps.workflows.question_answerer import QuestionAnswererAgent
from apps.workflows.sql_verifier.workflow import SQLVerifierWorkflow
from apps.workflows.services import ConversationLogger
from apps.accounts.models import OrganisationSettings

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


class VerifySQLInput(BaseModel):
    sql_query: str = Field(description="SQL query to verify")
    models_info: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Model information for context"
    )


class AcknowledgeQuestionInput(BaseModel):
    acknowledgement_text: str


class PostFinalResponseInput(BaseModel):
    message_text: str
    sql_query: str
    optional_notes: Optional[str] = None


class PostTextResponseInput(BaseModel):
    message_text: str


# --- LangGraph State Definition ---
class SlackResponderState(TypedDict):
    original_question: str
    channel_id: str
    thread_ts: str
    user_id: str
    messages: Annotated[List[BaseMessage], add_messages]
    thread_history: Optional[List[Dict[str, Any]]]
    acknowledgement_sent: Optional[bool]
    qa_final_answer: Optional[str]
    qa_sql_query: Optional[str]
    qa_models: Optional[List[Dict[str, Any]]]

    # SQL Verification fields
    sql_is_verified: Optional[bool]
    verified_sql_query: Optional[str]
    sql_verification_error: Optional[str]
    sql_verification_explanation: Optional[str]

    error_message: Optional[str]
    response_sent: Optional[bool]

    # Conversation tracking
    conversation_id: Optional[int]


# --- Slack Responder Agent ---
class SlackResponderAgent:
    """Orchestrates Slack interaction and coordinates with other agents."""

    def __init__(
        self,
        org_settings: OrganisationSettings,
        slack_client: AsyncWebClient,
        memory: Optional[BaseCheckpointSaver] = None,
    ):
        self.org_settings = org_settings
        self.slack_client = slack_client
        self.memory = memory

        # Set verbosity based on Django settings
        self.verbose = settings.RAGSTAR_LOG_LEVEL == "DEBUG"
        if self.verbose:
            logger.info(f"SlackResponderAgent initialized with verbose=True")

        # Initialize other agents with proper parameters
        self.question_answerer = QuestionAnswererAgent(
            org_settings=org_settings,
            memory=memory,
            data_warehouse_type=getattr(settings, "DATA_WAREHOUSE_TYPE", None),
        )

        self.sql_verifier = SQLVerifierWorkflow(
            org_settings=org_settings,
            memory=None,  # SQL verifier can have its own memory
            max_debug_loops=3,
        )

        # Use the LLM from the QA agent for consistency
        self.llm = self.question_answerer.llm

        # For storing current request's context
        self.current_channel_id: Optional[str] = None
        self.current_thread_ts: Optional[str] = None

        # Conversation logging
        self.conversation_logger: Optional[ConversationLogger] = None
        self.conversation: Optional[Conversation] = None

        self._define_tools()
        self.graph_app = self._build_graph()

    # --- Tool Definitions ---
    def _define_tools(self):
        """Define tools for the SlackResponder agent."""

        @tool(args_schema=FetchThreadInput)
        async def fetch_slack_thread() -> List[Dict[str, Any]]:
            """Fetches the message history from the current Slack thread."""
            if self.verbose:
                logger.info("Tool: fetch_slack_thread")

            if not self.current_channel_id or not self.current_thread_ts:
                error_msg = "Channel ID or Thread TS not available"
                logger.error(error_msg)
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
                        logger.info(f"Fetched {len(history)} messages from thread")
                    return history
                else:
                    error_msg = f"Slack API error: {result['error']}"
                    logger.error(error_msg)
                    return [{"error": error_msg}]
            except Exception as e:
                error_msg = f"Error fetching thread: {e}"
                logger.exception(error_msg)
                return [{"error": error_msg}]

        @tool(args_schema=AcknowledgeQuestionInput)
        async def acknowledge_question(acknowledgement_text: str) -> Dict[str, Any]:
            """Acknowledges the user's question in Slack."""
            if self.verbose:
                logger.info(
                    f"Tool: acknowledge_question with text: {acknowledgement_text[:50]}..."
                )

            if not self.current_channel_id or not self.current_thread_ts:
                return {
                    "success": False,
                    "error": "Channel/Thread context not available",
                }

            try:
                # Log the acknowledgment action
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_agent_response)(
                        content=acknowledgement_text,
                        metadata={"action": "acknowledgment", "channel": "slack"},
                    )

                await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=acknowledgement_text,
                )
                if self.verbose:
                    logger.info("Acknowledgement sent successfully")
                return {"success": True}
            except Exception as e:
                logger.exception(f"Error sending acknowledgement: {e}")
                return {"success": False, "error": str(e)}

        @tool(args_schema=AskQuestionAnswererInput)
        async def ask_question_answerer(
            question: str, thread_context: Optional[List[Dict[str, Any]]] = None
        ) -> Dict[str, Any]:
            """Asks the Question Answerer agent to analyze the question."""
            if self.verbose:
                logger.info(
                    f"Tool: ask_question_answerer for question: {question[:50]}..."
                )

            try:
                # Log the delegation to question answerer
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_system_message)(
                        content=f"Delegating to Question Answerer: {question}",
                        metadata={
                            "action": "delegate_to_qa",
                            "has_thread_context": thread_context is not None,
                            "thread_context_length": (
                                len(thread_context) if thread_context else 0
                            ),
                        },
                    )

                # Use the existing question answerer workflow
                result = await self.question_answerer.run_agentic_workflow(
                    question=question,
                    thread_context=thread_context,
                    conversation_id=f"slack-{self.current_channel_id}-{self.current_thread_ts}",
                )

                # The QA workflow now returns keys: 'answer' and 'models_used'
                qa_answer: Optional[str] = result.get("answer")
                sql_query: Optional[str] = result.get("sql_query")
                models_used: List[Dict[str, Any]] = result.get("models_used", [])

                # Log the QA response
                if self.conversation_logger:
                    model_names = [
                        m.get("name") for m in models_used if isinstance(m, dict)
                    ]

                    await sync_to_async(self.conversation_logger.log_system_message)(
                        content=f"Question Answerer completed. Models used: {', '.join(model_names)}",
                        metadata={
                            "action": "qa_completion",
                            "models_used": model_names,
                            "has_answer": bool(qa_answer),
                        },
                    )

                if self.verbose:
                    logger.info("Question Answerer completed successfully")

                return {
                    "success": True,
                    "answer": qa_answer,
                    "sql_query": sql_query,
                    "models_used": models_used,
                }
            except Exception as e:
                error_msg = f"Error in question analysis: {e}"
                logger.exception(error_msg)

                # Log the error
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_error)(
                        content=error_msg,
                        error_details={
                            "component": "question_answerer",
                            "error": str(e),
                        },
                    )

                return {"success": False, "error": error_msg}

        @tool(args_schema=VerifySQLInput)
        async def verify_sql_query(
            sql_query: str, models_info: Optional[List[Dict[str, Any]]] = None
        ) -> Dict[str, Any]:
            """Verifies a SQL query using the SQL Verifier."""
            if self.verbose:
                logger.info(f"Tool: verify_sql_query for SQL: {sql_query[:50]}...")

            try:
                # Log the SQL verification action
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_system_message)(
                        content=f"Starting SQL verification for query: {sql_query[:100]}...",
                        metadata={
                            "action": "sql_verification_start",
                            "sql_length": len(sql_query),
                            "has_models_info": models_info is not None,
                        },
                    )

                result = await self.sql_verifier.run(
                    sql_query=sql_query,
                    warehouse_type=getattr(
                        settings, "DATA_WAREHOUSE_TYPE", "snowflake"
                    ),
                    max_debug_attempts=3,
                    dbt_models_info=models_info,
                    conversation_id=f"verify-{self.current_channel_id}-{self.current_thread_ts}",
                )

                # Log verification result
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_system_message)(
                        content=f"SQL verification completed. Valid: {result.get('is_valid')}",
                        metadata={
                            "action": "sql_verification_complete",
                            "is_valid": result.get("is_valid"),
                            "has_error": bool(result.get("execution_error")),
                        },
                    )

                if self.verbose:
                    logger.info(f"SQL verification completed: {result.get('is_valid')}")

                return {
                    "success": True,
                    "is_valid": result.get("is_valid"),
                    "verified_sql": result.get("corrected_sql_query"),
                    "error": result.get("execution_error"),
                    "explanation": result.get("debug_explanation"),
                }
            except Exception as e:
                error_msg = f"Error in SQL verification: {e}"
                logger.exception(error_msg)

                # Log the error
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_error)(
                        content=error_msg,
                        error_details={"component": "sql_verifier", "error": str(e)},
                    )

                return {"success": False, "error": error_msg}

        @tool(args_schema=PostFinalResponseInput)
        async def post_final_response_with_snippet(
            message_text: str,
            sql_query: str,
            optional_notes: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Posts the final response with SQL snippet to Slack."""
            if self.verbose:
                logger.info("Tool: post_final_response_with_snippet")

            if not self.current_channel_id or not self.current_thread_ts:
                return {
                    "success": False,
                    "error": "Channel/Thread context not available",
                }

            try:
                # Create SQL file content
                sql_content = f"-- Generated SQL Query\n-- Question: {message_text[:100]}...\n\n{sql_query}"
                if optional_notes:
                    sql_content += f"\n\n-- Notes:\n-- {optional_notes}"

                # Log the final response
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_agent_response)(
                        content=f"{message_text}\n\nSQL Query:\n```sql\n{sql_query}\n```",
                        metadata={
                            "action": "final_response_with_sql",
                            "has_sql": True,
                            "sql_length": len(sql_query),
                            "has_notes": bool(optional_notes),
                            "channel": "slack",
                        },
                    )

                # Post message with file
                await self.slack_client.files_upload_v2(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    initial_comment=message_text,
                    content=sql_content,
                    filename="query.sql",
                    title="SQL Query",
                )

                if self.verbose:
                    logger.info("Final response with SQL snippet posted successfully")
                return {"success": True, "response_sent": True}
            except Exception as e:
                logger.exception(f"Error posting final response: {e}")
                return {"success": False, "error": str(e)}

        @tool(args_schema=PostTextResponseInput)
        async def post_text_response(message_text: str) -> Dict[str, Any]:
            """Posts a text response to Slack."""
            if self.verbose:
                logger.info(f"Tool: post_text_response: {message_text[:50]}...")

            if not self.current_channel_id or not self.current_thread_ts:
                return {
                    "success": False,
                    "error": "Channel/Thread context not available",
                }

            try:
                # Log the text response
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_agent_response)(
                        content=message_text,
                        metadata={"action": "text_response", "channel": "slack"},
                    )

                await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=message_text,
                )

                if self.verbose:
                    logger.info("Text response posted successfully")
                return {"success": True, "response_sent": True}
            except Exception as e:
                logger.exception(f"Error posting text response: {e}")
                return {"success": False, "error": str(e)}

        # Store tools for graph building
        self._tools = [
            fetch_slack_thread,
            acknowledge_question,
            ask_question_answerer,
            verify_sql_query,
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
        workflow.add_node("handle_direct_response", self.handle_direct_response_node)
        workflow.add_node("record_interaction", self.record_interaction_node)
        workflow.add_node("error_handler", self.error_handler_node)

        # Edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: "handle_direct_response",
            },
        )
        workflow.add_edge("tools", "update_state")
        workflow.add_conditional_edges(
            "update_state",
            self.route_after_update_state,
            {
                "agent": "agent",
                "record_interaction": "record_interaction",
                "error_handler": "error_handler",
                END: END,
            },
        )
        workflow.add_edge("handle_direct_response", "record_interaction")
        workflow.add_edge("record_interaction", END)
        workflow.add_edge("error_handler", END)

        # Compile with memory if available
        compile_kwargs = {}
        if self.memory:
            compile_kwargs["checkpointer"] = self.memory
        return workflow.compile(**compile_kwargs)

    # --- Routing Logic ---
    def route_after_update_state(self, state: SlackResponderState) -> str:
        """Routes after the update_state node."""
        if self.verbose:
            logger.info("Routing after update_state_node...")

        # If a response has been sent, go to record and end
        if state.get("response_sent"):
            if self.verbose:
                logger.info("Response sent, routing to record_interaction")
            return "record_interaction"

        # If there's an error, handle it
        if state.get("error_message"):
            logger.warning(f"Error in state: {state['error_message']}")
            return "error_handler"

        # Otherwise, go back to agent for next decision
        if self.verbose:
            logger.info("Routing back to agent for next decision")
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
            qa_sql_query=state.get("qa_sql_query"),
            qa_models=state.get("qa_models"),
            sql_is_verified=state.get("sql_is_verified"),
            verified_sql_query=state.get("verified_sql_query"),
            sql_verification_error=state.get("sql_verification_error"),
            sql_verification_explanation=state.get("sql_verification_explanation"),
            acknowledgement_sent=state.get("acknowledgement_sent"),
            error_message=state.get("error_message"),
        )

        if self.verbose:
            logger.info(
                f"--- SlackResponder System Prompt ---\n{system_prompt_content}\n--- End Prompt ---"
            )

        # Prepare messages for LLM
        first_turn_human_message: Optional[HumanMessage] = None

        if not current_history:  # First turn
            human_message = HumanMessage(content=state["original_question"])
            first_turn_human_message = human_message
            messages_to_send_to_llm = [
                SystemMessage(content=system_prompt_content),
                human_message,
            ]
        else:  # Subsequent turns
            messages_to_send_to_llm = [
                SystemMessage(content=system_prompt_content)
            ] + current_history

        if not self.llm:
            error_ai_message = AIMessage(content="Error: LLM client not available.")
            if first_turn_human_message:
                return {"messages": [first_turn_human_message, error_ai_message]}
            return {"messages": [error_ai_message]}

        # Bind tools and invoke
        agent_llm_with_tools = self.llm.bind_tools(self._tools)
        config = {"run_name": "SlackResponderAgentNode"}

        try:
            response = agent_llm_with_tools.invoke(
                messages_to_send_to_llm, config=config
            )

            # Add thread_context to ask_question_answerer tool call args if present
            if response.tool_calls:
                for call in response.tool_calls:
                    if call.get("name") == "ask_question_answerer":
                        call["args"] = call.get("args", {})
                        if state.get("thread_history"):
                            call["args"]["thread_context"] = state["thread_history"]

            if self.verbose:
                logger.info(f"SlackResponder Agent response: {response}")

            if first_turn_human_message:
                return {"messages": [first_turn_human_message, response]}
            else:
                return {"messages": [response]}

        except Exception as e:
            logger.exception(f"Error invoking SlackResponder LLM: {e}")
            error_ai_message = AIMessage(content=f"LLM Error: {e}")
            if first_turn_human_message:
                return {"messages": [first_turn_human_message, error_ai_message]}
            else:
                return {"messages": [error_ai_message]}

    # --- State Update Node ---
    def update_state_node(self, state: SlackResponderState) -> Dict[str, Any]:
        """Processes tool results and updates the SlackResponderState."""
        if self.verbose:
            logger.info("--- Updating SlackResponder State ---")

        updates: Dict[str, Any] = {}
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if not isinstance(last_message, ToolMessage):
            if self.verbose:
                logger.info("No tool message to process for state update.")
            return updates

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
            elif isinstance(tool_content, dict) and tool_content.get("error"):
                updates["error_message"] = (
                    f"Failed to fetch thread: {tool_content['error']}"
                )
                logger.error(f"Tool fetch_slack_thread failed: {tool_content['error']}")

        elif last_message.name == "acknowledge_question":
            if isinstance(tool_content, dict) and tool_content.get("success"):
                updates["acknowledgement_sent"] = True
            else:
                error_detail = (
                    tool_content.get("error", "Unknown error")
                    if isinstance(tool_content, dict)
                    else str(tool_content)
                )
                logger.error(f"Tool acknowledge_question failed: {error_detail}")

        elif last_message.name == "ask_question_answerer":
            if isinstance(tool_content, dict):
                if tool_content.get("error"):
                    updates["error_message"] = (
                        f"Error in question analysis: {tool_content.get('error')}"
                    )
                    logger.error(
                        f"Question Answerer failed: {tool_content.get('error')}"
                    )
                elif tool_content.get("success"):
                    # Store the answer and extract SQL if present
                    qa_answer = tool_content.get("answer")
                    updates["qa_final_answer"] = qa_answer
                    updates["qa_models"] = tool_content.get("models_used", [])
                    # Store sql_query directly if provided
                    if "sql_query" in tool_content and tool_content["sql_query"]:
                        updates["qa_sql_query"] = tool_content["sql_query"]

        elif last_message.name == "verify_sql_query":
            if isinstance(tool_content, dict):
                if tool_content.get("error"):
                    updates["error_message"] = (
                        f"SQL verification error: {tool_content.get('error')}"
                    )
                    logger.error(
                        f"SQL verification failed: {tool_content.get('error')}"
                    )
                elif tool_content.get("success"):
                    updates["sql_is_verified"] = tool_content.get("is_valid", False)
                    updates["verified_sql_query"] = tool_content.get("verified_sql")
                    updates["sql_verification_error"] = tool_content.get("error")
                    updates["sql_verification_explanation"] = tool_content.get(
                        "explanation"
                    )

                    if self.verbose:
                        logger.info(
                            f"SQL verification completed: {updates['sql_is_verified']}"
                        )

        elif last_message.name in [
            "post_final_response_with_snippet",
            "post_text_response",
        ]:
            if isinstance(tool_content, dict):
                if tool_content.get("success"):
                    updates["response_sent"] = True
                    if self.verbose:
                        logger.info(
                            f"Response posted successfully via {last_message.name}"
                        )
                else:
                    error_detail = tool_content.get("error", "Unknown error")
                    updates["error_message"] = (
                        f"Failed to post response: {error_detail}"
                    )
                    logger.error(f"Tool {last_message.name} failed: {error_detail}")

        return updates

    # --- Handle Direct Response Node ---
    async def handle_direct_response_node(
        self, state: SlackResponderState
    ) -> Dict[str, Any]:
        """Handles direct text responses from the agent when no tools are called."""
        if self.verbose:
            logger.info("--- Handling Direct Response ---")

        messages = state["messages"]
        last_message = messages[-1] if messages else None

        # Check if we have a direct AI response without tool calls
        if (
            isinstance(last_message, AIMessage)
            and last_message.content
            and not last_message.tool_calls
        ):
            response_text = last_message.content.strip()

            # Avoid posting duplicate acknowledgements
            if state.get("acknowledgement_sent") and not state.get("qa_final_answer"):
                if self.verbose:
                    logger.info(
                        "Skipping direct AI response because acknowledgement was already sent and no analysis has been done yet"
                    )
                return {}

            if self.verbose:
                logger.info(
                    f"Found direct AI response, posting to Slack: {response_text[:100]}..."
                )

            try:
                # Log the response
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_agent_response)(
                        content=response_text,
                        metadata={"action": "direct_text_response", "channel": "slack"},
                    )

                # Post the response to Slack
                await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=response_text,
                )

                if self.verbose:
                    logger.info("Direct response posted successfully")

                return {"response_sent": True}

            except Exception as e:
                logger.exception(f"Error posting direct response: {e}")
                return {"error_message": f"Failed to post response: {str(e)}"}

        else:
            if self.verbose:
                logger.info("No direct response to handle")
            return {}

    # --- Interaction Recording Node ---
    async def record_interaction_node(
        self, state: SlackResponderState
    ) -> Dict[str, Any]:
        """Records the conversation completion using ConversationLogger."""
        if self.verbose:
            logger.info("--- Recording Conversation Completion ---")

        # Don't record if no response was actually sent
        if not state.get("response_sent"):
            logger.warning(
                f"Skipping recording for thread {state.get('thread_ts')} as no response was marked as sent."
            )
            return {}

        try:
            # Update conversation status to completed
            if self.conversation_logger:
                await sync_to_async(self.conversation_logger.set_conversation_status)(
                    ConversationStatus.COMPLETED, completed_at=timezone.now()
                )

                # Log final summary if we have results
                if state.get("qa_final_answer"):
                    await sync_to_async(self.conversation_logger.log_system_message)(
                        f"Workflow completed successfully. Response sent: {state.get('response_sent')}"
                    )

                if self.verbose:
                    logger.info(
                        f"Conversation {self.conversation.id} marked as completed"
                    )

                return {"conversation_completed": True}
            else:
                logger.warning("No conversation logger available to record completion")
                return {"error_message": "Conversation logging not available"}

        except Exception as e:
            logger.exception(f"Failed to record conversation completion: {e}")
            return {"error_message": f"Failed to record interaction: {str(e)}"}

    @sync_to_async
    def _get_or_create_conversation(
        self, question: str, channel_id: str, thread_ts: str, user_id: str
    ) -> Conversation:
        """Get or create a conversation for this Slack thread."""
        try:
            # Try to get existing conversation
            conversation = Conversation.objects.get(
                organisation=self.org_settings.organisation,
                external_id=thread_ts,
                channel_id=channel_id,
                status__in=[ConversationStatus.ACTIVE, ConversationStatus.COMPLETED],
            )

            if self.verbose:
                logger.info(
                    f"Found existing conversation {conversation.id} for thread {thread_ts}"
                )

        except Conversation.DoesNotExist:
            # Get enabled integrations from the OrganisationIntegration model
            from apps.integrations.models import OrganisationIntegration

            enabled_integrations = list(
                OrganisationIntegration.objects.filter(
                    organisation=self.org_settings.organisation, is_enabled=True
                ).values_list("integration_key", flat=True)
            )

            # Create new conversation
            conversation = Conversation.objects.create(
                organisation=self.org_settings.organisation,
                external_id=thread_ts,
                channel="slack",
                channel_type="slack",
                channel_id=channel_id,
                user_id=user_id,
                user_external_id=user_id,
                status=ConversationStatus.ACTIVE,
                trigger=ConversationTrigger.SLACK_MENTION,
                initial_question=question,
                llm_provider=getattr(self.org_settings, "llm_anthropic_api_key", None)
                and "anthropic"
                or "openai",
                enabled_integrations=enabled_integrations,
            )

            if self.verbose:
                logger.info(
                    f"Created new conversation {conversation.id} for thread {thread_ts}"
                )

        return conversation

    # --- Main Workflow Runner ---
    async def run_slack_workflow(
        self, question: str, channel_id: str, thread_ts: str, user_id: str
    ) -> Dict[str, Any]:
        """Runs the full Slack interaction workflow asynchronously."""
        if self.verbose:
            logger.info(
                f"Starting SlackResponder workflow for question in {channel_id}/{thread_ts}"
            )

        # Set current context for this run
        self.current_channel_id = channel_id
        self.current_thread_ts = thread_ts

        # Initialize conversation_id early for error handling
        conversation_id = f"slack-{channel_id}-{thread_ts}"

        try:
            # Get or create conversation
            self.conversation = await self._get_or_create_conversation(
                question, channel_id, thread_ts, user_id
            )

            # Initialize conversation logger
            self.conversation_logger = ConversationLogger(self.conversation)

            # Log the initial user message (wrapped in sync_to_async)
            await sync_to_async(self.conversation_logger.log_user_message)(
                content=question,
                metadata={
                    "channel_id": channel_id,
                    "thread_ts": thread_ts,
                    "user_id": user_id,
                },
            )

            if self.verbose:
                logger.info(
                    f"Initialized conversation logging for conversation {self.conversation.id}"
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
                qa_sql_query=None,
                qa_models=None,
                sql_is_verified=None,
                verified_sql_query=None,
                sql_verification_error=None,
                sql_verification_explanation=None,
                error_message=None,
                response_sent=None,
                conversation_id=self.conversation.id,
            )

            if self.verbose:
                logger.info(f"Invoking SlackResponder graph for {conversation_id}")

            # Use asynchronous invocation
            final_state = await self.graph_app.ainvoke(initial_state, config=config)

            if self.verbose:
                logger.info(f"SlackResponder graph finished for {conversation_id}")

            # Check for errors in the final state
            if final_state and final_state.get("error_message"):
                logger.error(
                    f"Workflow for {conversation_id} completed with error: {final_state['error_message']}"
                )

                # Log error in conversation (wrapped in sync_to_async)
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_error)(
                        content=f"Workflow error: {final_state['error_message']}"
                    )
                    await sync_to_async(
                        self.conversation_logger.set_conversation_status
                    )(ConversationStatus.ERROR)

            # Clear context after run
            self.current_channel_id = None
            self.current_thread_ts = None

            return {
                "success": True,
                "final_state": final_state,
                "conversation_id": self.conversation.id,
            }

        except Exception as e:
            logger.exception(
                f"Critical error running SlackResponder graph for {conversation_id}: {e}"
            )

            # Log error in conversation if logger is available (wrapped in sync_to_async)
            if self.conversation_logger:
                try:
                    await sync_to_async(self.conversation_logger.log_error)(
                        content=f"Critical workflow error: {str(e)}"
                    )
                    await sync_to_async(
                        self.conversation_logger.set_conversation_status
                    )(ConversationStatus.ERROR)
                except Exception as log_error:
                    logger.error(f"Failed to log error to conversation: {log_error}")

            # Clear context on error
            self.current_channel_id = None
            self.current_thread_ts = None

            return {"success": False, "error": str(e)}

    # --- Error Handling Node ---
    async def error_handler_node(self, state: SlackResponderState) -> Dict[str, Any]:
        """Handles errors gracefully by sending user-friendly messages."""
        if self.verbose:
            logger.info(">>> Entering error_handler_node")

        error_message = state.get("error_message", "An unexpected issue occurred")

        # Create user-friendly error message
        user_message = "I'm having trouble processing your request right now. "

        if "thread" in error_message.lower():
            user_message += "I couldn't access the conversation history. Please try rephrasing your question."
        elif (
            "question analysis" in error_message.lower()
            or "models" in error_message.lower()
        ):
            user_message += "I need more information about your data models to answer that question. Could you provide more details?"
        elif "sql" in error_message.lower():
            user_message += "I'm having trouble with the data analysis part. Let me try a different approach."
        else:
            user_message += "Please try again in a moment, or contact support if the issue persists."

        # Send user-friendly error message
        try:
            if self.current_channel_id and self.current_thread_ts:
                await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=user_message,
                )
                if self.verbose:
                    logger.info("Sent user-friendly error message")
        except Exception as e:
            logger.exception(f"Failed to send error message: {e}")

        return {"response_sent": True, "error_handled": True}
