import logging
import uuid
import json
from typing import Dict, List, Any, Optional, Union, Set, TypedDict

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
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from psycopg_pool import AsyncConnectionPool

# Django & Project Imports
from django.conf import settings
from django.db.models import Q
from pgvector.django import CosineDistance

# Update model imports
from apps.knowledge_base.models import Model
from apps.embeddings.models import ModelEmbedding
from apps.workflows.models import Question

# Update service imports
from apps.llm_providers.services import (
    default_embedding_service,
    default_chat_service,
)

# Import DRF serializers from new locations
from apps.knowledge_base.serializers import ModelSerializer
from apps.workflows.serializers import QuestionSerializer

# Import prompts from the same directory
from .prompts import create_system_prompt, create_guidance_message

# Import sync_to_async correctly
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)


# --- Tool Input Schemas ---
# (Copied from previous agents.py)
class SearchModelsInput(BaseModel):
    query: str = Field(
        description="Concise, targeted query focusing on specific information needed to find relevant dbt models."
    )


class FetchModelsInput(BaseModel):
    model_names: List[str] = Field(
        description="A list of specific dbt model names to retrieve details for."
    )


class SearchFeedbackInput(BaseModel):
    query: str = Field(
        description="The original user question to search for similar past questions and feedback."
    )


class SearchFeedbackContentInput(BaseModel):
    query: str = Field(
        description="Specific query about a concept, definition, or clarification to search within the text content of past feedback entries."
    )


class SearchOrganizationalContextInput(BaseModel):
    query: str = Field(
        description="Query based on the current question to find relevant definitions or context in past *original* user messages."
    )


class FinishWorkflowInput(BaseModel):
    final_answer: str = Field(
        description="The comprehensive final answer text containing the SQL query, explanations (as comments), and any footnotes. Follow the SQL style guide provided in the prompt."
    )


# --- LangGraph State Definition ---
# (Copied from previous agents.py)
class QuestionAnsweringState(TypedDict):
    original_question: str
    messages: Annotated[List[BaseMessage], add_messages]
    accumulated_models: List[Dict[str, Any]]
    accumulated_model_names: Set[str]
    vector_search_calls: int
    relevant_feedback: Dict[str, List[Any]]
    final_answer: Optional[str]
    models_snapshot_for_final_answer: Optional[List[Dict[str, Any]]]
    conversation_id: Optional[str]
    thread_context: Optional[List[Dict[str, Any]]]
    similar_original_messages: Optional[List[Dict[str, Any]]]


# --- Refactored QuestionAnswerer Agent ---
class QuestionAnswererAgent:
    """Agent for answering questions about dbt models using Django ORM and Services."""

    def __init__(
        self,
        temperature: float = 0.0,
        verbose: bool = False,
        memory: Optional[AsyncPostgresSaver] = None,
    ):
        self.embedding_service = default_embedding_service
        self.chat_service = default_chat_service
        self.llm = self.chat_service.get_client()
        self.temperature = temperature
        self.verbose = verbose
        self.max_iterations = 10
        self.max_vector_searches = 5

        # Initialize summary as empty, load lazily
        self.all_models_summary: List[Dict[str, str]] = []
        self._models_loaded = False  # Flag to track loading

        # Initialize memory
        self.memory = memory
        if self.memory is None:
            # Async initialization needed for AsyncPostgresSaver
            # This part needs careful handling as __init__ is sync
            # For now, assume sync setup is okay, but ainvoke requires async saver
            # Ideally, initialization should happen async or pass an async pool
            # Let's stick to the structure but use AsyncPostgresSaver
            try:
                db_settings = settings.DATABASES["default"]
                pg_conn_string = f"postgresql://{db_settings['USER']}:{db_settings['PASSWORD']}@{db_settings['HOST']}:{db_settings['PORT']}/{db_settings['NAME']}"

                if pg_conn_string:
                    connection_kwargs = {
                        "autocommit": True,
                        "prepare_threshold": 0,
                    }
                    # NOTE: AsyncPostgresSaver typically needs an async connection pool
                    # Using a sync ConnectionPool here might cause issues later
                    # A better approach might be to pass an async pool/connection during agent init
                    conn_pool = AsyncConnectionPool(
                        conninfo=pg_conn_string,
                        kwargs=connection_kwargs,
                        max_size=20,
                        min_size=5,
                    )
                    # Use AsyncPostgresSaver, but it needs an async connection
                    # This initialization might need refactoring if issues persist
                    self.memory = AsyncPostgresSaver(conn=conn_pool)
                    if self.verbose:
                        logger.info(
                            "Initialized AsyncPostgresSaver checkpointer (using async pool)."
                        )
                else:
                    logger.warning(
                        "DB Connection Pool not available (DATABASE_URL missing?). Agent state will not be persisted."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to initialize AsyncPostgresSaver: {e}",
                    exc_info=self.verbose,
                )
        self._define_tools()
        # Compile graph
        compile_kwargs = {}
        if self.memory:
            # Pass the async checkpointer instance
            compile_kwargs["checkpointer"] = self.memory
        self.graph_app = self._build_graph().compile(**compile_kwargs)

    # --- Added Helper for Lazy Loading Model Summary ---
    @sync_to_async
    def _load_model_summary_db(self):
        """Performs the synchronous DB query for model summaries."""
        summary = []
        try:
            usable_model_embeddings = ModelEmbedding.objects.filter(
                can_be_used_for_answers=True
            ).values_list("model_name", flat=True)
            usable_models = Model.objects.filter(name__in=list(usable_model_embeddings))
            for model in usable_models:
                summary.append(
                    {
                        "name": model.name,
                        "description": model.interpreted_description
                        or model.yml_description
                        or "No description available.",
                    }
                )
            logger.info(
                f"Successfully loaded summary for {len(summary)} usable models from DB."
            )
            return summary
        except Exception as e:
            logger.error(
                f"Failed to load usable model summaries from DB: {e}",
                exc_info=self.verbose,
            )
            return []  # Return empty list on error

    async def _get_or_load_model_summary(self) -> List[Dict[str, str]]:
        """Lazily loads model summary using sync_to_async if not already loaded."""
        if not self._models_loaded:
            if self.verbose:
                logger.info("Model summary not loaded, attempting lazy load...")
            self.all_models_summary = await self._load_model_summary_db()
            self._models_loaded = True  # Mark as loaded (even if empty due to error)
        return self.all_models_summary

    # --- End Helper ---

    def _define_tools(self):
        # (Tool definitions copied and adapted from previous agents.py - verified in that step)
        @tool(args_schema=FetchModelsInput)
        async def fetch_model_details(model_names: List[str]) -> List[Dict[str, Any]]:
            """Fetches detailed information for specific dbt models by name using Django ORM."""
            if self.verbose:
                # Add newline for spacing
                logger.info(f"\nTool: fetch_model_details(names={model_names})")

            @sync_to_async
            def _fetch():
                models = Model.objects.filter(name__in=model_names)
                serializer = ModelSerializer(models, many=True)
                return serializer.data

            try:
                results = await _fetch()
                if self.verbose:
                    logger.info(f"Fetched details for {len(results)} models.")
                return results
            except Exception as e:
                logger.exception("Error fetching model details")
                return []  # Return empty list on error

        @tool(args_schema=SearchModelsInput)
        async def model_similarity_search(query: str) -> List[Dict[str, Any]]:
            """Searches for relevant dbt models using vector similarity via Django ORM and pgvector."""
            if self.verbose:
                # Add newline for spacing
                logger.info(f"\nTool: model_similarity_search(query='{query}')")

            @sync_to_async
            def _search():
                query_embedding = self.embedding_service.get_embedding(query)
                if not query_embedding:
                    return []
                n_results = 5
                similarity_threshold = 0.3
                embeddings = (
                    ModelEmbedding.objects.filter(can_be_used_for_answers=True)
                    .annotate(distance=CosineDistance("embedding", query_embedding))
                    .filter(distance__lt=(1.0 - similarity_threshold))
                    .order_by("distance")[:n_results]
                )
                found_model_names = [emb.model_name for emb in embeddings]
                distances_map = {emb.model_name: emb.distance for emb in embeddings}
                models = Model.objects.filter(name__in=found_model_names)
                serializer = ModelSerializer(models, many=True)
                results = serializer.data
                for model_dict in results:
                    distance = distances_map.get(model_dict["name"])
                    if distance is not None:
                        model_dict["search_score"] = 1.0 - distance
                    model_dict["fetch_method"] = "vector_search"
                return results

            try:
                results = await _search()
                if self.verbose:
                    logger.info(f"Found {len(results)} models via vector search.")
                return results
            except Exception as e:
                logger.exception("Error during model similarity search")
                return []  # Return empty list on error

        @tool(args_schema=SearchFeedbackInput)
        async def search_past_feedback(query: str) -> List[Dict[str, Any]]:
            """Searches past questions based on similarity using Django ORM and pgvector."""
            if self.verbose:
                # Add newline for spacing
                logger.info(f"\nTool: search_past_feedback(query='{query}')")

            @sync_to_async
            def _search():
                query_embedding = self.embedding_service.get_embedding(query)
                if not query_embedding:
                    return []
                n_results = 3
                similarity_threshold = 0.30
                similar_questions = (
                    Question.objects.exclude(question_embedding__isnull=True)
                    .annotate(
                        distance=CosineDistance("question_embedding", query_embedding)
                    )
                    .filter(
                        Q(feedback__isnull=False) | Q(was_useful__isnull=False),
                        distance__lt=(1.0 - similarity_threshold),
                    )
                    .order_by("distance")[:n_results]
                )
                serializer = QuestionSerializer(similar_questions, many=True)
                return serializer.data

            try:
                results = await _search()
                if self.verbose:
                    logger.info(
                        f"Found {len(results)} feedback items via question similarity."
                    )
                return results
            except Exception as e:
                logger.exception("Error searching past feedback by question")
                return []  # Return empty list on error

        @tool(args_schema=SearchFeedbackContentInput)
        async def search_feedback_content(query: str) -> List[Dict[str, Any]]:
            """Searches feedback text content using Django ORM and pgvector."""
            if self.verbose:
                # Add newline for spacing
                logger.info(f"\nTool: search_feedback_content(query='{query}')")

            @sync_to_async
            def _search():
                query_embedding = self.embedding_service.get_embedding(query)
                if not query_embedding:
                    return []
                n_results = 3
                similarity_threshold = 0.6
                similar_questions = (
                    Question.objects.filter(
                        feedback__isnull=False, feedback_embedding__isnull=False
                    )
                    .annotate(
                        distance=CosineDistance("feedback_embedding", query_embedding)
                    )
                    .filter(distance__lt=(1.0 - similarity_threshold))
                    .order_by("distance")[:n_results]
                )
                serializer = QuestionSerializer(similar_questions, many=True)
                return serializer.data

            try:
                results = await _search()
                if self.verbose:
                    logger.info(
                        f"Found {len(results)} feedback items via content search."
                    )
                return results
            except Exception as e:
                logger.exception("Error searching feedback content")
                return []  # Return empty list on error

        @tool(args_schema=SearchOrganizationalContextInput)
        async def search_organizational_context(query: str) -> List[Dict[str, Any]]:
            """Searches past original questions using Django ORM and pgvector."""
            if self.verbose:
                # Add newline for spacing
                logger.info(f"\nTool: search_organizational_context(query='{query}')")

            @sync_to_async
            def _search():
                query_embedding = self.embedding_service.get_embedding(query)
                if not query_embedding:
                    return []
                n_results = 3
                similarity_threshold = 0.7
                similar_questions = (
                    Question.objects.filter(
                        original_message_text__isnull=False,
                        original_message_embedding__isnull=False,
                    )
                    .annotate(
                        distance=CosineDistance(
                            "original_message_embedding", query_embedding
                        )
                    )
                    .filter(distance__lt=(1.0 - similarity_threshold))
                    .order_by("distance")[:n_results]
                )
                serializer = QuestionSerializer(similar_questions, many=True)
                return serializer.data

            try:
                results = await _search()
                if self.verbose:
                    logger.info(
                        f"Found {len(results)} context items via original message search."
                    )
                return results
            except Exception as e:
                logger.exception("Error searching organizational context")
                return []  # Return empty list on error

        @tool(args_schema=FinishWorkflowInput)
        async def finish_workflow(final_answer: str) -> str:
            """Concludes the workflow and provides the final answer text to the user."""
            if self.verbose:
                # Add newline for spacing
                logger.info(
                    f"\nTool: finish_workflow(final_answer='{final_answer[:100]}...')"
                )
            # This tool needs to be async to be called by ainvoke/astream
            # but doesn't perform any async operations itself.
            return final_answer

        self._tools = [
            fetch_model_details,
            model_similarity_search,
            search_past_feedback,
            search_feedback_content,
            search_organizational_context,
            finish_workflow,
        ]

    def _build_graph(self):
        # (Graph building logic copied from previous agents.py - verified)
        workflow = StateGraph(QuestionAnsweringState)
        tool_node = ToolNode(self._tools, handle_tool_errors=True)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("update_state", self.update_state_node)
        workflow.add_node("finalize_direct_answer", self.finalize_direct_answer_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", END: "finalize_direct_answer"}
        )
        workflow.add_edge("tools", "update_state")
        workflow.add_edge("finalize_direct_answer", END)
        compile_kwargs = {}
        if self.memory:
            compile_kwargs["checkpointer"] = self.memory
        return workflow.compile(**compile_kwargs)

    def route_after_update(self, state: QuestionAnsweringState) -> str:
        # (Routing logic copied from previous agents.py - verified)
        if state.get("final_answer"):
            return END
        else:
            return "agent"

    # --- Nodes to be refactored ---
    async def agent_node(self, state: QuestionAnsweringState) -> Dict[str, Any]:
        """The main node that calls the LLM to decide the next action or generate the final answer."""
        if self.verbose:
            # Add separators
            logger.info("\n--- Entering QuestionAnswerer Agent Node ---")

        # --- Lazily load model summary ---
        current_model_summary = await self._get_or_load_model_summary()
        # --- End Lazy loading ---

        # Check for final answer early exit
        if state.get("final_answer"):
            return {"messages": []}  # Already finished

        # ... (rest of existing agent_node logic, using current_model_summary)
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
        similar_original_messages = state.get("similar_original_messages", [])
        thread_context = state.get("thread_context")

        # System Prompt Setup
        system_prompt = create_system_prompt(
            all_models_summary=current_model_summary,  # Use loaded summary
            relevant_feedback_by_question=relevant_feedback_by_question,
            relevant_feedback_by_content=relevant_feedback_by_content,
            similar_original_messages=similar_original_messages,
            accumulated_models=accumulated_models,
            search_model_calls=search_model_calls,
            max_vector_searches=self.max_vector_searches,
        )

        # Guidance Logic
        guidance = create_guidance_message(
            search_model_calls=search_model_calls,
            max_vector_searches=self.max_vector_searches,
            accumulated_models=accumulated_models,
        )

        # Prepare messages for LLM
        messages_for_llm = [SystemMessage(content=system_prompt)]
        if thread_context:
            # Add thread context as human message (simplified)
            context_str = "\n".join(
                [f"{m.get('user')}: {m.get('text')}" for m in thread_context]
            )
            messages_for_llm.append(
                HumanMessage(content=f"Slack Thread Context:\n{context_str}")
            )
        messages_for_llm.extend(messages)
        messages_for_llm.append(HumanMessage(content=guidance))

        # Bind tools and invoke LLM
        if self.verbose:
            # Add separators
            logger.info("\n--- Calling QuestionAnswerer LLM ---")

        agent_llm_with_tools = self.llm.bind_tools(self._tools)
        config = {"run_name": "QuestionAnswererAgentNode"}

        try:
            response = await agent_llm_with_tools.ainvoke(
                messages_for_llm, config=config
            )
            if self.verbose:
                # Add separators and format response
                response_str = json.dumps(
                    response.dict(), indent=2
                )  # Pretty print response
                logger.info(
                    f"\n--- QuestionAnswerer LLM Response ---\n{response_str}\n-------------------------------------"
                )
            return {"messages": [response]}
        except Exception as e:
            logger.exception("Error invoking QuestionAnswerer LLM")
            return {"messages": [AIMessage(content=f"LLM Error: {e}")]}

    async def update_state_node(self, state: QuestionAnsweringState) -> Dict[str, Any]:
        """Updates the state based on the results of the most recent tool calls."""
        if self.verbose:
            # Add separators
            logger.info("\n--- Updating QuestionAnswerer State ---")
        # Ensure this node is async if called by ainvoke
        updates: Dict[str, Any] = {}
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if not isinstance(last_message, ToolMessage):
            return updates

        tool_content = last_message.content
        # Safely parse JSON if needed
        if isinstance(tool_content, str) and tool_content.startswith("{"):
            try:
                tool_content = json.loads(tool_content)
            except json.JSONDecodeError:
                pass

        tool_name = last_message.name
        if self.verbose:
            # Add separators
            logger.info(f"\n--- Processing ToolMessage: {tool_name} ---")

        # Handle state updates based on tool results
        # (Existing logic for processing tool results, ensure it doesn't have sync DB calls)
        # If DB calls are needed here (e.g., verifying models), use sync_to_async
        # --- (Example adaptation for fetch_model_details, others similar) ---
        if tool_name == "fetch_model_details":
            if isinstance(tool_content, list):
                # ... (existing logic to add models to state) ...
                accumulated_models = list(state.get("accumulated_models", []))
                accumulated_model_names = set(
                    state.get("accumulated_model_names", set())
                )
                new_models_added = 0
                for model_dict in tool_content:
                    if (
                        isinstance(model_dict, dict)
                        and model_dict.get("name") not in accumulated_model_names
                    ):
                        accumulated_models.append(model_dict)
                        accumulated_model_names.add(model_dict["name"])
                        new_models_added += 1
                if new_models_added > 0:
                    updates["accumulated_models"] = accumulated_models
                    updates["accumulated_model_names"] = accumulated_model_names
                    if self.verbose:
                        logger.info(f"Added {new_models_added} model details to state.")
        elif tool_name == "model_similarity_search":
            # ... (existing logic, assuming tool returns list of dicts) ...
            updates["vector_search_calls"] = state.get("vector_search_calls", 0) + 1
            # ... add models similar to fetch_model_details ...
        elif tool_name == "search_past_feedback":
            if isinstance(tool_content, list):
                relevant_feedback = state.get("relevant_feedback", {}).copy()
                relevant_feedback["by_question"] = tool_content  # Update feedback
                updates["relevant_feedback"] = relevant_feedback
                if self.verbose:
                    logger.info(
                        f"Updated feedback by question: {len(tool_content)} items"
                    )
        elif tool_name == "search_feedback_content":
            if isinstance(tool_content, list):
                relevant_feedback = state.get("relevant_feedback", {}).copy()
                relevant_feedback["by_content"] = tool_content  # Update feedback
                updates["relevant_feedback"] = relevant_feedback
                if self.verbose:
                    logger.info(
                        f"Updated feedback by content: {len(tool_content)} items"
                    )
        elif tool_name == "search_organizational_context":
            if isinstance(tool_content, list):
                updates["similar_original_messages"] = tool_content
                if self.verbose:
                    logger.info(
                        f"Updated similar messages context: {len(tool_content)} items"
                    )
        elif tool_name == "finish_workflow":
            if isinstance(tool_content, str):
                updates["final_answer"] = tool_content
                # Explicitly capture the current accumulated_models when final_answer is set
                updates["models_snapshot_for_final_answer"] = list(
                    state.get("accumulated_models", [])
                )
                if self.verbose:
                    # Add separators
                    logger.info("\n--- Final Answer Received via Tool ---")
            else:
                logger.warning(
                    f"Finish tool returned non-string content: {type(tool_content)}"
                )
                updates["final_answer"] = "Error: Failed to finalize answer format."

        if self.verbose and updates:
            # Add separators
            updates_str = json.dumps(list(updates.keys()), indent=2)
            logger.info(
                f"\n--- QuestionAnswerer State Updates ---\n{updates_str}\n--------------------------------------"
            )

        return updates

    async def finalize_direct_answer_node(
        self, state: QuestionAnsweringState
    ) -> Dict[str, Any]:
        """
        Checks if the last AIMessage is a direct answer (no tool calls, content present, finish_reason='stop')
        and sets final_answer in the state if so.
        This node is typically reached if the agent decides to answer directly without tools.
        """
        updates = {}
        # Check if final_answer is already set (e.g., by a prior forced finish, though unlikely on this path)
        if state.get("final_answer"):
            return updates  # Already finalized

        last_message = state["messages"][-1] if state.get("messages") else None

        if isinstance(last_message, AIMessage) and last_message.content:
            # Condition for being routed here from 'agent' via 'tools_condition' is that 'tool_calls' is empty.
            # We double-check content and that finish_reason is 'stop'.
            is_direct_stop_answer = (
                not last_message.tool_calls
                and hasattr(last_message, "response_metadata")
                and isinstance(last_message.response_metadata, dict)
                and last_message.response_metadata.get("finish_reason") == "stop"
            )

            if is_direct_stop_answer:
                updates["final_answer"] = last_message.content
                if self.verbose:
                    logger.info(
                        "QuestionAnswerer graph: Finalizing direct answer from LLM (finish_reason='stop')."
                    )
            elif not last_message.tool_calls and self.verbose:
                # This case means 'tools_condition' sent us here, but it wasn't a clean 'stop' with content.
                # For example, LLM might have just stopped without content, or finish_reason was length etc.
                logger.warning(
                    "QuestionAnswerer graph: Agent stopped without tool calls, "
                    "but last message was not a clear direct answer (content or finish_reason mismatch)."
                )
        return updates

    # --- Routing Logic ---
    async def should_continue(self, state: QuestionAnsweringState) -> str:
        """Determines whether to continue iteration or finish."""
        # Ensure this node is async if called by ainvoke
        messages = state["messages"]
        if state.get("final_answer"):
            return "__end__"
        if (
            len(messages) > self.max_iterations * 2
        ):  # Account for AIMessage + ToolMessage per loop
            logger.warning("Max iterations reached, ending workflow.")
            return "__end__"
        if state.get("vector_search_calls", 0) >= self.max_vector_searches:
            logger.warning("Max vector searches reached.")
            # Allow agent one more turn to synthesize answer
            # or maybe force finish? For now, let it try to finish.

        # Check if the last message was an AIMessage with a call to 'finish_workflow'
        last_message = messages[-1] if messages else None
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            if any(
                call.get("name") == "finish_workflow"
                for call in last_message.tool_calls
            ):
                # Let the tool run, state update will set final_answer, then end
                return "tools"

        return "agent"

    # --- Graph Construction ---
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(QuestionAnsweringState)
        tool_node = ToolNode(self._tools, handle_tool_errors=True)

        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("update_state", self.update_state_node)
        workflow.add_node("finalize_direct_answer", self.finalize_direct_answer_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: "finalize_direct_answer"},
        )
        workflow.add_edge("tools", "update_state")
        workflow.add_edge("finalize_direct_answer", END)

        workflow.add_conditional_edges(
            "update_state",
            self.should_continue,  # Use the routing function
            {"agent": "agent", "__end__": END},
        )
        return workflow

    async def run_agentic_workflow(
        self,
        question: str,
        thread_context: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the agentic workflow asynchronously to answer a question."""
        if not self.llm:
            return {
                "error": "LLM Client not initialized.",
                "answer": None,
                "models_used": [],
            }
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": conversation_id}}

        initial_state = QuestionAnsweringState(
            original_question=question,
            messages=[],
            accumulated_models=[],
            accumulated_model_names=set(),
            vector_search_calls=0,
            relevant_feedback={},
            final_answer=None,
            models_snapshot_for_final_answer=None,
            conversation_id=conversation_id,
            thread_context=thread_context,
            similar_original_messages=[],
        )

        # --- Pre-fetch context/feedback asynchronously ---
        try:
            # Use sync_to_async for DB calls during pre-fetch
            @sync_to_async
            def _get_embedding(text):
                return self.embedding_service.get_embedding(text)

            @sync_to_async
            def _fetch_similar_messages(q_embedding):
                similar_msgs_q = (
                    Question.objects.filter(
                        original_message_text__isnull=False,
                        original_message_embedding__isnull=False,
                    )
                    .annotate(
                        distance=CosineDistance(
                            "original_message_embedding", q_embedding
                        )
                    )
                    .filter(distance__lt=0.3)
                    .order_by("distance")[:3]
                )
                return QuestionSerializer(similar_msgs_q, many=True).data

            @sync_to_async
            def _fetch_feedback_by_question(q_embedding):
                similar_feedback_q = (
                    Question.objects.exclude(question_embedding__isnull=True)
                    .annotate(
                        distance=CosineDistance("question_embedding", q_embedding)
                    )
                    .filter(
                        Q(feedback__isnull=False) | Q(was_useful__isnull=False),
                        distance__lt=0.7,
                    )
                    .order_by("distance")[:3]
                )
                return QuestionSerializer(similar_feedback_q, many=True).data

            @sync_to_async
            def _fetch_feedback_by_content(q_embedding):
                similar_feedback_content_q = (
                    Question.objects.filter(
                        feedback__isnull=False, feedback_embedding__isnull=False
                    )
                    .annotate(
                        distance=CosineDistance("feedback_embedding", q_embedding)
                    )
                    .filter(distance__lt=0.4)
                    .order_by("distance")[:3]
                )
                return QuestionSerializer(similar_feedback_content_q, many=True).data

            q_embedding = await _get_embedding(question)
            if q_embedding:
                initial_state["similar_original_messages"] = (
                    await _fetch_similar_messages(q_embedding)
                )
                initial_state["relevant_feedback"]["by_question"] = (
                    await _fetch_feedback_by_question(q_embedding)
                )
                # We need embedding for content search too
                initial_state["relevant_feedback"]["by_content"] = (
                    await _fetch_feedback_by_content(q_embedding)
                )

        except Exception as e:
            logger.error(
                f"Error pre-fetching initial context/feedback: {e}", exc_info=True
            )

        final_state = None
        try:
            initial_state["messages"] = [HumanMessage(content=question)]
            # Use ainvoke for the async graph execution
            final_state = await self.graph_app.ainvoke(initial_state, config=config)

        except Exception as e:
            logger.error(f"Error running agent graph: {e}", exc_info=True)
            return {
                "error": f"Agent execution failed: {e}",
                "answer": None,
                "models_used": [],
            }

        # Prepare result dictionary (same as sync version)
        result = {
            "answer": "Could not determine a final answer.",
            "models_used": (
                final_state.get("models_snapshot_for_final_answer", [])
                if final_state
                and final_state.get("models_snapshot_for_final_answer") is not None
                else (final_state.get("accumulated_models", []) if final_state else [])
            ),
            "warning": None,
            "error": None,
        }

        if final_state:
            if final_state.get(
                "final_answer"
            ):  # This should now be set by the graph if an answer was found
                result["answer"] = final_state["final_answer"]
                # If final_answer is set, there should be no warning about premature end
                result["warning"] = (
                    None  # Clear any previous warning if answer is now found
                )
            else:
                # If after graph execution, final_answer is STILL not set, then it's a genuine issue.
                result["warning"] = (
                    "Workflow ended, but no final_answer was set in the state by tools or direct response."
                )
                # Ensure default answer is used if final_answer is None
                result["answer"] = "Could not determine a final answer."

            # Preserve any error message that might have been set in final_state by the graph
            # (e.g., by ToolNode error handling or other logic)
            if final_state.get(
                "error_message"
            ):  # Assuming an 'error_message' key might be used
                result["error"] = final_state["error_message"]
            elif result.get(
                "answer", ""
            ) == "Could not determine a final answer." and not result.get("warning"):
                # If we have the default answer but no specific warning or error from graph state, set a generic warning.
                # This case might be hit if graph ends without setting final_answer and without explicit error.
                result["warning"] = (
                    "Workflow ended without a definitive answer or specific error."
                )

        else:  # final_state is None
            result["error"] = "Agent execution failed to produce a final state."
            result["answer"] = (
                "Could not determine a final answer due to agent failure."
            )

        return result
