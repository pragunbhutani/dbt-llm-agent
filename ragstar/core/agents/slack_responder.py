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
from rich.console import Console

# RAGstar Imports (assuming QuestionAnswerer is accessible)
from ragstar.core.llm.client import LLMClient

# Import components needed to initialize QuestionAnswerer
from ragstar.storage.model_storage import ModelStorage
from ragstar.storage.model_embedding_storage import ModelEmbeddingStorage
from ragstar.storage.question_storage import QuestionStorage
from ragstar.core.agents.question_answerer import (
    QuestionAnswerer,
)  # Adjust import if needed

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


class RespondToThreadInput(BaseModel):
    channel_id: str = Field(description="The ID of the Slack channel to respond in.")
    thread_ts: str = Field(
        description="The timestamp of the parent message in the thread to respond to."
    )
    answer_text: str = Field(
        description="The final answer text to post in the Slack thread."
    )


# --- LangGraph State Definition ---
class SlackResponderState(TypedDict):
    """Represents the state of the Slack responding agent."""

    original_question: str
    channel_id: str
    thread_ts: str
    messages: Annotated[List[BaseMessage], add_messages]
    thread_history: Optional[List[Dict[str, Any]]]  # Store fetched thread messages
    contextual_question: Optional[str]  # Question formulated with context
    final_answer: Optional[str]  # Answer received from QuestionAnswerer


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
        self.memory = memory
        self.console = console or Console()
        self.verbose = verbose

        self._define_tools()
        self.graph_app = self._build_graph()

    def _define_tools(self):
        """Define the tools used by the agent."""

        @tool(args_schema=FetchThreadInput)
        def fetch_slack_thread(channel_id: str, thread_ts: str) -> List[Dict[str, Any]]:
            """Fetches the message history from a specific Slack thread.
            Use this tool ONCE at the beginning to understand the context of the user's question within the Slack thread.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: fetch_slack_thread(channel_id='{channel_id}', thread_ts='{thread_ts}')[/bold magenta]"
                )
            # TODO: Implement actual Slack API call using self.slack_client
            # Example:
            # history = get_thread_history(self.slack_client, channel_id, thread_ts)
            # return history
            # logger.warning(
            #     "fetch_slack_thread tool not fully implemented (needs Slack client). Returning placeholder."
            # )
            # return [
            #     {"user": "user1", "text": "Placeholder message 1", "ts": thread_ts},
            #     {"user": "user2", "text": "Placeholder message 2"},
            # ]  # Placeholder
            try:
                # Note: Slack SDK methods are typically async when using AsyncWebClient
                # LangGraph tools ideally should be synchronous, or the graph needs to handle async tool execution.
                # For simplicity here, we'll call the async method from a sync context, which isn't ideal.
                # A better approach involves using an async LangGraph or running the sync tool in an event loop.
                # This example uses `asyncio.run` for simplicity, but consider alternatives in production.
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
                    # Extract relevant info, e.g., user, text, ts
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
                    return [{"error": error_msg}]
            except SlackApiError as e:
                error_msg = f"Slack API Error fetching thread: {e.response['error']}"
                logger.error(error_msg, exc_info=self.verbose)
                return [{"error": error_msg}]
            except Exception as e:
                error_msg = f"Unexpected error fetching thread: {e}"
                logger.error(error_msg, exc_info=self.verbose)
                return [{"error": error_msg}]

        @tool(args_schema=AskQuestionAnswererInput)
        def ask_question_answerer(question: str) -> Dict[str, Any]:
            """Sends a question to the specialized QuestionAnswerer agent to get an answer based on dbt models.
            Use this tool after analyzing the Slack thread context (if necessary).
            Formulate the 'question' input carefully, incorporating thread context if it helps clarify the user's intent.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: ask_question_answerer(question='{question[:100]}...')[/bold magenta]"
                )
            self.console.print(
                f"[blue]❓ Passing question to QuestionAnswerer: '{question[:100]}...'[/blue]"
            )
            # Run the QuestionAnswerer workflow
            # Note: QuestionAnswerer runs its own graph, potentially with its own memory.
            # We might need a way to link these workflows or handle potential blocking.
            # For now, run it synchronously. Consider async execution later.
            try:
                qa_result = self.question_answerer.run_agentic_workflow(question)
                # Extract the relevant part (the final answer)
                return {
                    "final_answer": qa_result.get(
                        "final_answer", "QuestionAnswerer did not provide an answer."
                    )
                }
            except Exception as e:
                logger.error(
                    f"Error running QuestionAnswerer workflow: {e}",
                    exc_info=self.verbose,
                )
                return {"final_answer": f"Error contacting QuestionAnswerer: {e}"}

        @tool(args_schema=RespondToThreadInput)
        def respond_to_slack_thread(
            channel_id: str, thread_ts: str, answer_text: str
        ) -> str:
            """Posts a message back to the specified Slack thread.
            Use this tool ONLY when you have the final answer from the QuestionAnswerer agent and are ready to respond to the user.
            """
            if self.verbose:
                self.console.print(
                    f"[bold magenta]🛠️ Executing Tool: respond_to_slack_thread(channel_id='{channel_id}', thread_ts='{thread_ts}', answer_text='{answer_text[:100]}...')[/bold magenta]"
                )
            # TODO: Implement actual Slack API call using self.slack_client
            # Example:
            # response = post_message_to_thread(self.slack_client, channel_id, thread_ts, answer_text)
            # return "Successfully posted message." if response.get("ok") else f"Failed to post message: {response.get('error')}"
            # logger.warning(
            #     "respond_to_slack_thread tool not fully implemented (needs Slack client). Simulating success."
            # )
            # self.console.print(
            #     f"[green]💬 Simulated posting to Slack thread {channel_id}/{thread_ts}[/green]"
            # )
            # return "Successfully posted message (simulated)."  # Placeholder
            try:
                import asyncio

                # Similar async issue as above
                async def _post():
                    return await self.slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,  # Important: reply in thread
                        text=answer_text,
                        # Optional: Use blocks for richer formatting
                        # blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": answer_text}}]
                    )

                result = asyncio.run(_post())

                if result["ok"]:
                    if self.verbose:
                        self.console.print(
                            f"[green] -> Successfully posted message to thread {channel_id}/{thread_ts}[/green]"
                        )
                    return "Successfully posted message to Slack."
                else:
                    error_msg = f"Slack API error (chat.postMessage): {result['error']}"
                    logger.error(error_msg)
                    return error_msg  # Return error string
            except SlackApiError as e:
                error_msg = f"Slack API Error posting message: {e.response['error']}"
                logger.error(error_msg, exc_info=self.verbose)
                return error_msg  # Return error string
            except Exception as e:
                error_msg = f"Unexpected error posting message: {e}"
                logger.error(error_msg, exc_info=self.verbose)
                return error_msg  # Return error string

        self._tools = [
            fetch_slack_thread,
            ask_question_answerer,
            respond_to_slack_thread,
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

        # Set entry point
        workflow.set_entry_point("agent")

        # Define conditional edges from agent
        workflow.add_conditional_edges(
            "agent",
            tools_condition,  # Checks for tool calls in the last AIMessage
            {
                "tools": "tools",  # If tools called, go to tools node
                END: END,  # Otherwise, end the graph
            },
        )

        # Define edges for flow: tools -> update_state -> agent
        workflow.add_edge("tools", "update_state")
        workflow.add_edge("update_state", "agent")

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
                f"[dim]Final answer ready: {'Yes' if state.get('final_answer') else 'No'}[/dim]"
            )

        messages = state.get("messages", [])

        # Initial prompt construction
        if not messages:
            system_prompt = """You are an AI assistant integrated with Slack. Your goal is to understand user questions asked in Slack threads and get answers using a specialized 'QuestionAnswerer' agent.

Workflow:
1. A user asks a question in a Slack thread. You receive the question, channel ID, and thread timestamp.
2. Use the 'fetch_slack_thread' tool ONCE to get the recent message history for context.
3. Analyze the original question AND the thread history. If the history provides important context (e.g., clarifying a previous point, referring to an earlier topic), formulate a combined question. Otherwise, just use the original question.
4. Use the 'ask_question_answerer' tool with the formulated question. This tool will invoke another agent to find the answer based on dbt models.
5. Once you receive the final answer from 'ask_question_answerer', use the 'respond_to_slack_thread' tool to post this answer back into the original Slack thread.
6. Your final step should always be calling 'respond_to_slack_thread'. Do not try to chat further after getting the answer.
"""
            # Use single quotes for f-string, double for dict keys
            initial_human_message = f'New question received from Slack:\\nChannel ID: {state["channel_id"]}\\nThread TS: {state["thread_ts"]}\\nQuestion: "{state["original_question"]}"\\n\\nPlease fetch the thread history to understand the context.'
            messages_for_llm = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=initial_human_message),
            ]
        else:
            # Add guidance based on current state for subsequent calls
            guidance_items = []
            if not state.get("thread_history"):
                guidance_items.append("You should use 'fetch_slack_thread' now.")
            elif not state.get("final_answer"):
                guidance_items.append(
                    "Analyze the thread history and the original question. Formulate the question for the 'ask_question_answerer' tool."
                )
            else:
                guidance_items.append(
                    f"You have received the final answer. Use 'respond_to_slack_thread' to post it back to channel {state['channel_id']} in thread {state['thread_ts']}."
                )

            guidance = "Guidance: " + " ".join(guidance_items)
            messages_for_llm = list(messages)  # Make a copy
            # Add guidance if the last message isn't already guidance
            if not (
                messages_for_llm
                and isinstance(messages_for_llm[-1], SystemMessage)
                and messages_for_llm[-1].content.startswith("Guidance:")
            ):
                messages_for_llm.append(SystemMessage(content=guidance))

        agent_llm = self._get_agent_llm()
        if self.verbose:
            self.console.print(
                f"[blue dim]Sending {len(messages_for_llm)} messages to LLM...[/blue dim]"
            )
            # Optional: Log message structure here if needed for debugging

        try:
            response = agent_llm.invoke(messages_for_llm)
            if self.verbose:
                self.console.print(
                    f"[dim]LLM Response: {response.content[:100]}...[/dim]"
                )
                if hasattr(response, "tool_calls") and response.tool_calls:
                    self.console.print(f"[dim]Tool calls: {response.tool_calls}[/dim]")
            # The graph's add_messages will append this response to the state's 'messages' list
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error invoking agent LLM: {e}", exc_info=self.verbose)
            error_message = AIMessage(content=f"LLM invocation failed: {str(e)}")
            return {"messages": [error_message]}

    def update_state_node(self, state: SlackResponderState) -> Dict[str, Any]:
        """Updates the state based on the results of the most recent tool call."""
        if self.verbose:
            self.console.print(
                "[bold cyan]\n--- Updating SlackResponder State After Tool Execution ---[/bold cyan]"
            )

        updates: Dict[str, Any] = {}
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

            if tool_name == "fetch_slack_thread":
                if isinstance(parsed_content, list):
                    updates["thread_history"] = parsed_content
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with {len(parsed_content)} messages from thread history.[/dim]"
                        )
                else:
                    logger.warning(
                        f"fetch_slack_thread tool returned unexpected type: {type(parsed_content)}"
                    )

            elif tool_name == "ask_question_answerer":
                # Expecting a dict with 'final_answer' key from the tool wrapper
                if (
                    isinstance(parsed_content, dict)
                    and "final_answer" in parsed_content
                ):
                    updates["final_answer"] = parsed_content["final_answer"]
                    if self.verbose:
                        self.console.print(
                            f"[dim] -> Updated state with final answer from QuestionAnswerer.[/dim]"
                        )
                else:
                    logger.warning(
                        f"ask_question_answerer tool returned unexpected content: {parsed_content}"
                    )
                    # Store the raw response as the answer if structure is wrong
                    updates["final_answer"] = str(parsed_content)

            elif tool_name == "respond_to_slack_thread":
                # This tool doesn't typically update state, it's a final action.
                # We just log its success/failure based on its return string.
                if self.verbose:
                    self.console.print(
                        f"[dim] -> Received result from respond_to_slack_thread: {content}[/dim]"
                    )
                # No state updates needed here, the workflow should end after this.

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

        return updates

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

        initial_state = SlackResponderState(
            original_question=question,
            channel_id=channel_id,
            thread_ts=thread_ts,
            messages=[],
            thread_history=None,
            contextual_question=None,
            final_answer=None,
        )

        if self.verbose:
            self.console.print(
                f"[dim]Generated workflow ID for config: {workflow_id}[/dim]"
            )
            self.console.print("[dim]Initial state prepared.[/dim]")

        try:
            # Execute the graph
            # We iterate through the stream to ensure all steps execute
            final_state = None
            for event in self.graph_app.stream(initial_state, config=config):
                # Optionally process intermediate events if needed
                # The last event will contain the final state
                for key, value in event.items():
                    if self.verbose:
                        self.console.print(
                            f"[grey]Graph Event: Node='{key}', State Changes='{list(value.keys())}'[/grey]"
                        )
                    if key == "__end__":  # Check if it's the final state event
                        final_state = value
                        break  # Found the final state
                if final_state:
                    break

            if final_state is None:
                # Fallback if stream ends without explicit __end__ key (shouldn't happen with proper graph)
                # This might happen if invoke is used instead of stream/astream
                logger.warning(
                    "Graph execution finished but final state marker '__end__' not found in last event."
                )
                # Try invoking to get final state directly as a fallback
                final_state = self.graph_app.invoke(initial_state, config=config)

            self.console.print("[green]✅ SlackResponder workflow finished.[/green]")

            if self.verbose:
                final_msg_count = len(final_state.get("messages", []))
                self.console.print(
                    f"[dim]Final state contains {final_msg_count} messages.[/dim]"
                )
                self.console.print(
                    f"[dim]Final Answer provided to Slack: {'Yes' if final_state.get('final_answer') else 'No (or workflow ended before response)'}[/dim]"
                )

            # Return results (actual Slack posting happens via tool)
            return {
                "success": True,  # Indicates workflow completed without Python errors
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "final_answer_provided": final_state.get("final_answer") is not None,
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
