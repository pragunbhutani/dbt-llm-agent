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

# Shared schema for SQL verification results
from apps.workflows.schemas import SQLVerificationResponse
from apps.workflows.schemas import QAResponse
from pydantic import ValidationError
from apps.workflows.utils import ensure_contract
from apps.workflows.utils import format_for_slack, rich_text_to_blocks

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


class PostAnalysisWithUnverifiedSQLInput(BaseModel):
    message_text: str
    sql_query: str
    verification_error: Optional[str] = None
    models_used: Optional[List[Dict[str, Any]]] = None


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
    qa_notes: Optional[List[str]]

    # SQL Verification fields
    sql_is_verified: Optional[bool]
    verified_sql_query: Optional[str]
    sql_verification_error: Optional[str]
    sql_verification_explanation: Optional[str]
    sql_style_violations: Optional[List[str]]

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
                    text=format_for_slack(acknowledgement_text),
                    blocks=rich_text_to_blocks(acknowledgement_text),
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

                # Validate against the canonical schema for safety
                qa_obj = ensure_contract(
                    QAResponse,
                    result,
                    component="slack_responder.ask_question_answerer",
                    strict_mode=False,
                )
                qa_answer: Optional[str] = qa_obj.answer
                sql_query: Optional[str] = qa_obj.sql_query
                models_used: List[Dict[str, Any]] = qa_obj.models_used

                # If no explicit sql_query returned, attempt to extract from fenced code block in answer
                if sql_query is None and isinstance(qa_answer, str):
                    import re

                    sql_match = re.search(
                        r"```sql\s*([\s\S]+?)```", qa_answer, re.IGNORECASE
                    )
                    if sql_match:
                        sql_query = sql_match.group(1).strip()

                # Log the QA response
                if self.conversation_logger:
                    model_names = [
                        m.get("name") for m in models_used if isinstance(m, dict)
                    ]

                    # Only include "Models used:" if there are actually models
                    if model_names:
                        content = f"Question Answerer completed. Models used: {', '.join(model_names)}"
                    else:
                        content = "Question Answerer completed."

                    await sync_to_async(self.conversation_logger.log_system_message)(
                        content=content,
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

                raw_result = await self.sql_verifier.run(
                    sql_query=sql_query,
                    warehouse_type=getattr(
                        settings, "DATA_WAREHOUSE_TYPE", "snowflake"
                    ),
                    max_debug_attempts=3,
                    dbt_models_info=models_info,
                    conversation_id=f"verify-{self.current_channel_id}-{self.current_thread_ts}",
                )

                # Cast into shared contract model for reliable access
                result = SQLVerificationResponse(**raw_result)

                # Log verification result
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_system_message)(
                        content=f"SQL verification completed. Valid: {result.is_valid}",
                        metadata={
                            "action": "sql_verification_complete",
                            "is_valid": result.is_valid,
                            "has_error": bool(result.error),
                        },
                    )

                if self.verbose:
                    logger.info(f"SQL verification completed: {result.is_valid}")

                if result.is_valid:
                    return {
                        "success": True,
                        "is_valid": True,
                        "verified_sql": result.verified_sql,
                        "error": result.error,
                        "explanation": result.explanation,
                    }
                else:
                    error_msg = result.error
                    logger.warning(f"SQL verification failed: {error_msg}")

                    # Provide non-fatal fallback so downstream logic can still share the query.
                    return {
                        "success": True,
                        "is_valid": False,
                        "verified_sql": result.verified_sql or sql_query,
                        "error": error_msg,
                        "explanation": result.explanation,
                    }
            except Exception as e:
                error_msg = f"Error in SQL verification: {e}"
                # Early graceful fallback so the workflow continues even if verification itself
                # crashes. We send back the unverified SQL.
                return {
                    "success": True,
                    "is_valid": False,
                    "verified_sql": sql_query,
                    "error": error_msg,
                    "explanation": None,
                }
                logger.exception(error_msg)

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
                # Prepare a clean summary for the SQL file header – avoid dumping raw JSON
                if message_text.lstrip().startswith("{"):
                    summary = "Generated SQL for the user's question"
                else:
                    summary = _collapse_spaces(message_text)[:120]

                sql_content = f"-- Generated SQL Query\n-- {summary}\n\n{sql_query}"
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

                # 1️⃣ Send the rich Block Kit message first so Slack renders it correctly
                await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=format_for_slack(message_text),
                    blocks=rich_text_to_blocks(message_text),
                )

                # 2️⃣ Upload the SQL file with a minimal, plain-text comment (Block Kit not supported here)
                await self.slack_client.files_upload_v2(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    initial_comment="SQL query attached.",
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
                    text=format_for_slack(message_text),
                    blocks=rich_text_to_blocks(message_text),
                )

                if self.verbose:
                    logger.info("Text response posted successfully")
                return {"success": True, "response_sent": True}
            except Exception as e:
                logger.exception(f"Error posting text response: {e}")
                return {"success": False, "error": str(e)}

        @tool(args_schema=PostAnalysisWithUnverifiedSQLInput)
        async def post_analysis_with_unverified_sql(
            message_text: str,
            sql_query: str,
            verification_error: Optional[str] = None,
            models_used: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Any]:
            """Posts analysis results with unverified SQL when verification fails but we still want to provide value."""
            if self.verbose:
                logger.info("Tool: post_analysis_with_unverified_sql")

            if not self.current_channel_id or not self.current_thread_ts:
                return {
                    "success": False,
                    "error": "Channel/Thread context not available",
                }

            try:
                raw_error = verification_error
                friendly_error = SlackResponderAgent._friendly_verification_error(
                    raw_error
                )

                # Build analysis message with minimal duplication
                response_parts = [message_text]

                if models_used:
                    model_names = [
                        m.get("name", "Unknown")
                        for m in models_used
                        if isinstance(m, dict)
                    ]
                    if model_names:
                        response_parts.append(
                            f"\n📊 *Analysis based on models:* {', '.join(model_names)}"
                        )

                response_parts.append(
                    f"\n⚠️ *Note:* I couldn't run this query automatically because {friendly_error}. The SQL is attached below — please review before using."
                )

                combined_message = "".join(response_parts)

                # SQL file content with concise header
                sql_content = f"""-- Generated SQL Query (UNVERIFIED)
-- Reason: {friendly_error}
-- Review carefully before use.

{sql_query}"""

                # Log the fallback response
                if self.conversation_logger:
                    await sync_to_async(self.conversation_logger.log_agent_response)(
                        content=f"{combined_message}\n\nUnverified SQL Query:\n```sql\n{sql_query}\n```",
                        metadata={
                            "action": "fallback_response_with_unverified_sql",
                            "verification_failed": True,
                            "verification_error": verification_error,
                            "has_sql": True,
                            "sql_length": len(sql_query),
                            "models_used": [
                                m.get("name")
                                for m in (models_used or [])
                                if isinstance(m, dict)
                            ],
                            "channel": "slack",
                        },
                    )

                # 1️⃣ Post the analysis text with full Block Kit formatting
                await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=format_for_slack(combined_message),
                    blocks=rich_text_to_blocks(combined_message),
                )

                # 2️⃣ Upload the SQL file with a concise plain-text comment
                await self.slack_client.files_upload_v2(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    initial_comment="SQL query attached (unverified).",
                    content=sql_content,
                    filename="unverified_query.sql",
                    title="SQL Query (Unverified)",
                )

                if self.verbose:
                    logger.info(
                        "Fallback response with unverified SQL posted successfully"
                    )
                return {"success": True, "response_sent": True}
            except Exception as e:
                logger.exception(f"Error posting fallback response: {e}")
                return {"success": False, "error": str(e)}

        # Store tools for graph building
        self._tools = [
            acknowledge_question,
            ask_question_answerer,
            verify_sql_query,
            post_final_response_with_snippet,
            post_text_response,
            post_analysis_with_unverified_sql,
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
        workflow.add_node(
            "auto_post_unverified_sql", self.auto_post_unverified_sql_node
        )

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
                "auto_post_unverified_sql": "auto_post_unverified_sql",
                END: END,
            },
        )
        workflow.add_edge("handle_direct_response", "record_interaction")
        workflow.add_edge("record_interaction", END)
        workflow.add_edge("error_handler", END)
        workflow.add_edge("auto_post_unverified_sql", "record_interaction")

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

        # NEW: Verification failed but we still have SQL – auto-post it as file.
        if (
            state.get("sql_is_verified") is False
            and state.get("qa_sql_query")
            and not state.get("response_sent")
        ):
            if self.verbose:
                logger.info(
                    "SQL verification failed – routing to auto_post_unverified_sql node"
                )
            return "auto_post_unverified_sql"

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
            qa_notes=state.get("qa_notes"),
            sql_is_verified=state.get("sql_is_verified"),
            verified_sql_query=state.get("verified_sql_query"),
            sql_verification_error=state.get("sql_verification_error"),
            sql_verification_explanation=state.get("sql_verification_explanation"),
            sql_style_violations=state.get("sql_style_violations"),
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

            # --- NEW: capture prompt & completion tokens ---
            if getattr(self, "conversation_logger", None):
                try:
                    usage_meta = (
                        getattr(response, "additional_kwargs", {}).get("usage", {})
                        or getattr(response, "response_metadata", {}).get("usage", {})
                        or getattr(response, "usage_metadata", {})
                    )

                    prompt_tokens = int(
                        usage_meta.get("prompt_tokens")
                        or usage_meta.get("input_tokens")
                        or 0
                    )
                    completion_tokens = int(
                        usage_meta.get("completion_tokens")
                        or usage_meta.get("output_tokens")
                        or 0
                    )

                    if prompt_tokens:
                        self.conversation_logger.log_agent_response(
                            content="[LLM Prompt – SlackResponder]",
                            tokens_used=prompt_tokens,
                            metadata={"type": "llm_input"},
                        )

                    self.conversation_logger.log_agent_response(
                        content=(
                            response.content
                            if hasattr(response, "content")
                            else "[LLM Response]"
                        ),
                        tokens_used=completion_tokens,
                        metadata={"type": "llm_output"},
                    )
                except Exception as log_err:
                    logger.warning(
                        f"SlackResponder: failed to record LLM token usage: {log_err}"
                    )

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
        if last_message.name == "acknowledge_question":
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
                    updates["qa_notes"] = tool_content.get("notes", [])
                    # Store sql_query directly if provided
                    if "sql_query" in tool_content and tool_content["sql_query"]:
                        updates["qa_sql_query"] = tool_content["sql_query"]

        elif last_message.name == "verify_sql_query":
            if isinstance(tool_content, dict):
                # Treat verification errors as non-fatal so we can still provide maximum value.
                # We only mark the workflow as errored if the verifier call itself failed (success == False).
                if tool_content.get("success"):
                    updates["sql_is_verified"] = tool_content.get("is_valid", False)
                    updates["verified_sql_query"] = tool_content.get("verified_sql")
                    updates["sql_verification_error"] = tool_content.get("error")
                    updates["sql_verification_explanation"] = tool_content.get(
                        "explanation"
                    )
                    updates["sql_style_violations"] = tool_content.get(
                        "style_violations", []
                    )
                    if self.verbose:
                        logger.info(
                            f"SQL verification completed: {updates['sql_is_verified']}"
                        )
                else:
                    # The verifier tool invocation itself failed; propagate as an error.
                    updates["error_message"] = (
                        f"SQL verification failed to run: {tool_content.get('error', 'Unknown error')}"
                    )
                    logger.error(
                        f"SQL verification tool invocation failed: {tool_content.get('error')}"
                    )

        elif last_message.name in [
            "post_final_response_with_snippet",
            "post_text_response",
            "post_analysis_with_unverified_sql",
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
            # Ensure we have a clean string to post. Some LLMs return `content` as a
            # list of strings instead of a single string. Join list items if needed
            # and then strip surrounding whitespace.
            content = last_message.content
            if isinstance(content, list):
                response_text = "\n".join(
                    [str(part) for part in content if part is not None]
                ).strip()
            else:
                response_text = str(content).strip()

            # Avoid posting duplicate acknowledgements
            if state.get("acknowledgement_sent") and not state.get("qa_final_answer"):
                if self.verbose:
                    logger.info(
                        "Skipping direct AI response because acknowledgement was already sent and no analysis has been done yet"
                    )
                return {}

            # -------------------------------------------------------------
            # NEW: If the direct response contains a ```sql code block, we
            # bypass a plain chat_postMessage and instead upload the SQL as
            # a file so Slack renders it properly. This is a final guardrail
            # in case the agent forgot to call post_analysis_with_unverified_sql.
            # -------------------------------------------------------------
            sql_block_match = re.search(
                r"```sql\s*([\s\S]+?)```", response_text, re.IGNORECASE
            )
            if sql_block_match:
                sql_query = sql_block_match.group(1).strip()

                # Remove the code block from the text to craft a cleaner comment
                initial_comment = re.sub(
                    r"```sql[\s\S]+?```", "", response_text, flags=re.IGNORECASE
                ).strip()
                if not initial_comment:
                    initial_comment = "Here's the SQL query I generated (unverified)."

                try:
                    # Log to conversation
                    if self.conversation_logger:
                        await sync_to_async(
                            self.conversation_logger.log_agent_response
                        )(
                            content=f"{initial_comment}\n\nUnverified SQL uploaded as file.",
                            metadata={
                                "action": "fallback_sql_file_upload",
                                "channel": "slack",
                            },
                        )

                    await self.slack_client.files_upload_v2(
                        channel=self.current_channel_id,
                        thread_ts=self.current_thread_ts,
                        initial_comment=format_for_slack(initial_comment),
                        content=sql_query,
                        filename="query.sql",
                        title="SQL Query",
                    )

                    if self.verbose:
                        logger.info(
                            "Fallback file upload posted instead of raw code block"
                        )

                    return {"response_sent": True}
                except Exception as e:
                    logger.exception(f"Error uploading SQL file fallback: {e}")
                    # fall through to normal posting below

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
                    text=format_for_slack(response_text),
                    blocks=rich_text_to_blocks(response_text),
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

    # -------------------------------------------------------------
    # NEW NODE: auto_post_unverified_sql_node
    # -------------------------------------------------------------
    async def auto_post_unverified_sql_node(
        self, state: SlackResponderState
    ) -> Dict[str, Any]:
        """Automatically posts analysis + unverified SQL as a file when the agent failed to do so."""

        if self.verbose:
            logger.info(">>> Entering auto_post_unverified_sql_node")

        sql_query: str = state.get("qa_sql_query", "") or ""
        if not sql_query:
            return {"error_message": "No SQL query available for auto post."}

        raw_error = state.get("sql_verification_error")
        friendly_error = SlackResponderAgent._friendly_verification_error(raw_error)

        # Build analysis text (will be posted as a normal message)
        analysis_parts: List[str] = []

        qa_answer = state.get("qa_final_answer")
        if qa_answer:
            # Remove any SQL fenced blocks so we don't duplicate the query text
            stripped_answer = re.sub(
                r"```sql[\s\S]+?```", "", qa_answer, flags=re.IGNORECASE
            ).strip()
            if stripped_answer:
                analysis_parts.append("\n\n" + stripped_answer)

        analysis_parts.append(
            f"\n\n⚠️ *Note:* I couldn't run this query automatically because {friendly_error}. The SQL is attached below — please review before using."
        )

        qa_models = state.get("qa_models") or []
        if qa_models:
            model_names = [
                m.get("name", "Unknown") for m in qa_models if isinstance(m, dict)
            ]
            if model_names:
                analysis_parts.append(
                    f"\n\n📊 *Analysis based on models:* {', '.join(model_names)}"
                )

        analysis_text = "".join(analysis_parts)

        sql_content = f"""-- Generated SQL Query (UNVERIFIED)
-- Reason: {friendly_error}
-- Review carefully before use.

{sql_query}
"""

        try:
            # 1) post analysis text
            await self.slack_client.chat_postMessage(
                channel=self.current_channel_id,
                thread_ts=self.current_thread_ts,
                text=format_for_slack(_collapse_spaces(analysis_text)[:3000]),
                blocks=rich_text_to_blocks(analysis_text),
            )

            # 2) upload SQL file with a concise comment to avoid duplication
            await self.slack_client.files_upload_v2(
                channel=self.current_channel_id,
                thread_ts=self.current_thread_ts,
                initial_comment="SQL query attached (unverified).",
                content=sql_content,
                filename="unverified_query.sql",
                title="SQL Query (Unverified)",
            )

            if self.conversation_logger:
                await sync_to_async(self.conversation_logger.log_agent_response)(
                    content=analysis_text + "\n\nUnverified SQL uploaded as file.",
                    metadata={
                        "action": "auto_post_unverified_sql",
                        "verification_failed": True,
                        "channel": "slack",
                    },
                )

            if self.verbose:
                logger.info(
                    "auto_post_unverified_sql_node: Analysis + file posted successfully"
                )

            return {"response_sent": True}
        except Exception as e:
            logger.exception(
                f"auto_post_unverified_sql_node: failed delivering messages: {e}"
            )
            # Attempt fallback: upload trimmed SQL as code block
            try:
                fallback_text = (
                    analysis_text
                    + "\n\n```sql\n"
                    + sql_query[:3000]
                    + ("\n...```" if len(sql_query) > 3000 else "\n```")
                )
                await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=format_for_slack(fallback_text),
                    blocks=rich_text_to_blocks(fallback_text),
                )
                return {"response_sent": True}
            except Exception as ee:
                logger.exception(
                    f"auto_post_unverified_sql_node: secondary fallback failed: {ee}"
                )
                return {"error_message": str(ee)}

    @sync_to_async
    def _get_or_create_conversation(
        self, question: str, channel_id: str, thread_ts: str, user_id: str
    ) -> Conversation:
        """Get or create a conversation for this Slack thread."""
        try:
            # Try to get existing conversation using get_or_create to avoid duplicates
            conversation, created = Conversation.objects.get_or_create(
                organisation=self.org_settings.organisation,
                external_id=thread_ts,
                defaults={
                    "channel": "slack",
                    "channel_type": "slack",
                    "channel_id": channel_id,
                    "user_id": user_id,
                    "user_external_id": user_id,
                    "status": ConversationStatus.ACTIVE,
                    "trigger": ConversationTrigger.SLACK_MENTION,
                    "initial_question": question,
                    "llm_provider": self.org_settings.llm_chat_provider,
                    "llm_chat_model": self.org_settings.llm_chat_model,
                    "enabled_integrations": self._get_enabled_integrations(),
                },
            )

            if created:
                if self.verbose:
                    logger.info(
                        f"Created new conversation {conversation.id} for thread {thread_ts}"
                    )
            else:
                if self.verbose:
                    logger.info(
                        f"Found existing conversation {conversation.id} for thread {thread_ts}"
                    )

        except Conversation.MultipleObjectsReturned:
            # Handle case where duplicates still exist - get the first one
            logger.warning(
                f"Multiple conversations found for thread {thread_ts}, using the first one"
            )
            conversation = Conversation.objects.filter(
                organisation=self.org_settings.organisation,
                external_id=thread_ts,
            ).first()

        except Exception as e:
            logger.error(f"Error getting/creating conversation: {e}")
            # Fallback: try to get any existing conversation for this thread
            conversation = Conversation.objects.filter(
                organisation=self.org_settings.organisation,
                external_id=thread_ts,
            ).first()

            if not conversation:
                # Fallback: create new conversation if none exists
                enabled_integrations = self._get_enabled_integrations()
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
                    llm_provider=self.org_settings.llm_chat_provider,
                    llm_chat_model=self.org_settings.llm_chat_model,
                    enabled_integrations=enabled_integrations,
                )
                if self.verbose:
                    logger.info(
                        f"Created fallback conversation {conversation.id} for thread {thread_ts}"
                    )

        return conversation

    def _get_enabled_integrations(self):
        """Get list of enabled integrations for this organization."""
        from apps.integrations.models import OrganisationIntegration

        return list(
            OrganisationIntegration.objects.filter(
                organisation=self.org_settings.organisation, is_enabled=True
            ).values_list("integration_key", flat=True)
        )

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

        # --- NEW: Pre-fetch complete Slack thread history (incl. files) ---
        thread_history: List[Dict[str, Any]] | None = None
        try:
            logger.info(
                f"Fetching Slack thread history for channel={channel_id}, thread_ts={thread_ts}"
            )
            history_resp = await self.slack_client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=200,  # Fetch up to 200 messages (Slack max for one call)
            )
            if history_resp.get("ok"):
                thread_history = [
                    {
                        "user": m.get("user"),
                        "text": m.get("text"),
                        "ts": m.get("ts"),
                        # Preserve any files metadata so downstream agents can act on them
                        "files": m.get("files", []),
                    }
                    for m in history_resp.get("messages", [])
                ]
                logger.info(
                    f"Fetched {len(thread_history)} messages for thread {thread_ts}"
                )
            else:
                logger.warning(
                    f"Slack conversations_replies returned error: {history_resp.get('error')}"
                )
        except Exception as e:
            logger.exception(f"Failed to fetch Slack thread history: {e}")
        # --- END NEW ---

        # --- Fetch Slack user display name for better UX ---
        user_display_name: str | None = None
        user_locale: str | None = None
        try:
            # Attempt to fetch user info from Slack to get their real/display name
            if self.slack_client and user_id:
                logger.info(f"Fetching Slack user info for user_id: {user_id}")
                user_info_resp = await self.slack_client.users_info(user=user_id)
                logger.info(f"Slack users_info response: {user_info_resp}")
                if user_info_resp.get("ok"):
                    profile = user_info_resp["user"].get("profile", {})
                    logger.info(f"User profile data: {profile}")
                    user_display_name = (
                        profile.get("real_name")
                        or profile.get("display_name")
                        or user_info_resp["user"].get("name")
                    )
                    user_locale = user_info_resp["user"].get("locale")
                    logger.info(
                        f"Successfully fetched user display name: {user_display_name}"
                    )
                else:
                    logger.warning(
                        f"Slack users_info API returned not ok: {user_info_resp}"
                    )
            else:
                logger.warning(
                    f"Slack client not available or user_id empty. slack_client: {self.slack_client}, user_id: {user_id}"
                )
        except SlackApiError as e:
            logger.warning(
                f"Could not fetch Slack user info for {user_id}: {e.response['error']}"
            )
        except Exception as e:
            logger.warning(
                f"Unexpected error fetching Slack user info for {user_id}: {e}"
            )

        # Initialize conversation_id early for error handling
        conversation_id = f"slack-{channel_id}-{thread_ts}"

        try:
            # Get or create conversation (initially stores user_id as both id & external_id)
            self.conversation = await self._get_or_create_conversation(
                question, channel_id, thread_ts, user_id
            )

            # If we successfully fetched the user's display name, update the conversation
            logger.info(
                f"Checking if user_external_id needs update. user_display_name: '{user_display_name}', current user_external_id: '{self.conversation.user_external_id}'"
            )
            if (
                user_display_name
                and user_display_name != self.conversation.user_external_id
            ):
                logger.info(
                    f"Updating conversation user_external_id from '{self.conversation.user_external_id}' to '{user_display_name}'"
                )
                self.conversation.user_external_id = user_display_name
                await sync_to_async(self.conversation.save)(
                    update_fields=["user_external_id"]
                )
                logger.info(
                    f"Successfully updated conversation user_external_id to: {user_display_name}"
                )
            else:
                logger.info(
                    f"Not updating user_external_id. user_display_name: {user_display_name}, current user_external_id: {self.conversation.user_external_id}"
                )

            # Initialize conversation logger
            self.conversation_logger = ConversationLogger(self.conversation)

            # Pass the logger down to child workflows so they can log tool usage
            self.question_answerer.conversation_logger = self.conversation_logger
            self.sql_verifier.conversation_logger = self.conversation_logger

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

            # Save user context for prompt generation
            self._user_display_name = user_display_name
            self._user_locale = user_locale

            config = {"configurable": {"thread_id": conversation_id}}

            # Initial state
            initial_state = SlackResponderState(
                original_question=question,
                channel_id=channel_id,
                thread_ts=thread_ts,
                user_id=user_id,
                messages=[],
                thread_history=thread_history,  # Pass the pre-fetched history
                acknowledgement_sent=None,
                qa_final_answer=None,
                qa_sql_query=None,
                qa_models=None,
                qa_notes=None,
                sql_is_verified=None,
                verified_sql_query=None,
                sql_verification_error=None,
                sql_verification_explanation=None,
                sql_style_violations=None,
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
        """Handles errors gracefully by sending user-friendly messages, with fallback value when possible."""
        if self.verbose:
            logger.info(">>> Entering error_handler_node")

        error_message = state.get("error_message", "An unexpected issue occurred")

        # Check if we have partial results we can still provide value with
        qa_answer = state.get("qa_final_answer")
        qa_sql = state.get("qa_sql_query")
        qa_models = state.get("qa_models")
        sql_verification_error = state.get("sql_verification_error")

        # Try to provide maximum value even in error scenarios
        if qa_answer and qa_sql and "sql" in error_message.lower():
            # We have analysis and SQL but verification failed - provide unverified SQL
            try:
                fallback_message = f"I was able to analyze your question and generate a response, though I couldn't fully verify the SQL query. Here's what I found:\n\n{qa_answer}"

                await self.slack_client.files_upload_v2(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    initial_comment=format_for_slack(
                        fallback_message
                        + f"\n\n*Note:* I couldn't run this query automatically because {SlackResponderAgent._friendly_verification_error(sql_verification_error)}. Please review before using."
                    ),
                    content=f"""-- Generated SQL Query (UNVERIFIED)
-- Reason: {SlackResponderAgent._friendly_verification_error(sql_verification_error)}
-- Review carefully before use.

{qa_sql}""",
                    filename="unverified_query.sql",
                    title="SQL Query (Unverified)",
                )

                if self.verbose:
                    logger.info(
                        "Sent fallback response with unverified SQL from error handler"
                    )
                return {"response_sent": True, "error_handled": True}
            except Exception as e:
                logger.exception(f"Failed to send fallback response: {e}")
                # Fall through to generic error message

        elif qa_answer and not qa_sql:
            # We have analysis but no SQL - just send the analysis
            try:
                fallback_message = (
                    f"I was able to analyze your question:\n\n{qa_answer}"
                )
                if qa_models:
                    model_names = [
                        m.get("name", "Unknown")
                        for m in qa_models
                        if isinstance(m, dict)
                    ]
                    if model_names:
                        fallback_message += (
                            f"\n\n📊 *Based on models:* {', '.join(model_names)}"
                        )

                await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=format_for_slack(fallback_message),
                    blocks=rich_text_to_blocks(fallback_message),
                )

                if self.verbose:
                    logger.info(
                        "Sent fallback response with analysis from error handler"
                    )
                return {"response_sent": True, "error_handled": True}
            except Exception as e:
                logger.exception(f"Failed to send analysis fallback: {e}")
                # Fall through to generic error message

        # Create user-friendly error message as last resort
        user_message = "I'm having trouble processing your request right now. "

        if "thread" in error_message.lower():
            user_message += "I couldn't access the conversation history. Please try rephrasing your question."
        elif (
            "question analysis" in error_message.lower()
            or "models" in error_message.lower()
        ):
            user_message += "I need more information about your data models to answer that question. Could you provide more details?"
        elif "sql" in error_message.lower():
            user_message += "I'm having trouble with the data analysis part, but I'll keep trying to help you find insights."
        else:
            user_message += "Please try again in a moment, or contact support if the issue persists."

        # Send user-friendly error message
        try:
            if self.current_channel_id and self.current_thread_ts:
                await self.slack_client.chat_postMessage(
                    channel=self.current_channel_id,
                    thread_ts=self.current_thread_ts,
                    text=format_for_slack(user_message),
                    blocks=rich_text_to_blocks(user_message),
                )
                if self.verbose:
                    logger.info("Sent user-friendly error message")
        except Exception as e:
            logger.exception(f"Failed to send error message: {e}")

        return {"response_sent": True, "error_handled": True}

    # --- Helper: map raw verifier errors to concise, user-friendly messages ---

    @staticmethod
    def _friendly_verification_error(error: Optional[str]) -> str:
        """Return a concise, user-friendly reason why verification failed.

        We try to detect common failure patterns (missing warehouse credentials,
        authentication failures, dialect/unsupported warehouse, etc.) and map
        them to a short explanation.  If we cannot recognise the pattern we
        return the original error string so the user still gets some context.
        """
        if not error:
            return "warehouse connection is not configured"

        lowered = error.lower()

        # Missing configuration / integration disabled
        if (
            "not configured" in lowered
            or "no credentials" in lowered
            or "credentials not available" in lowered
        ):
            return "warehouse connection is not configured"

        # Authentication / password issues
        if any(
            word in lowered
            for word in [
                "authentication",
                "auth failed",
                "password",
                "invalid credentials",
            ]
        ):
            return "warehouse credentials appear to be incorrect"

        # Network / connectivity
        if any(
            word in lowered
            for word in [
                "could not connect",
                "connection refused",
                "network",
                "timeout",
            ]
        ):
            return "unable to connect to the warehouse"

        # Generic Snowflake error pattern
        if "snowflake" in lowered:
            return "unable to connect to Snowflake"

        return error.strip()
