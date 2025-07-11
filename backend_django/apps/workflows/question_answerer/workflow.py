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
from apps.llm_providers.services import ChatService, EmbeddingService
from apps.accounts.models import OrganisationSettings

# Import DRF serializers from new locations
from apps.knowledge_base.serializers import ModelSerializer
from apps.workflows.serializers import QuestionSerializer

# Import prompts from the same directory
from .prompts import create_system_prompt, create_guidance_message

# Import sync_to_async correctly
from asgiref.sync import sync_to_async

# Import QAResponse for contract validation
from apps.workflows.schemas import QAResponse
from apps.workflows.services import ConversationLogger

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
    answer: str = Field(..., description="The user-facing answer text.")
    sql_query: Optional[str] = Field(
        default=None,
        description="A SQL query that supports the answer, if applicable.",
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
    sql_query: Optional[str]
    conversation_id: Optional[str]
    thread_context: Optional[List[Dict[str, Any]]]
    similar_original_messages: Optional[List[Dict[str, Any]]]


# --- Refactored QuestionAnswerer Agent ---
class QuestionAnswererAgent:
    """Agent for answering questions about dbt models using Django ORM and Services."""

    def __init__(
        self,
        org_settings: OrganisationSettings,
        memory: Optional[AsyncPostgresSaver] = None,
        data_warehouse_type: Optional[str] = None,
    ):
        self.embedding_service = EmbeddingService(org_settings=org_settings)
        self.chat_service = ChatService(org_settings=org_settings)
        self.llm = self.chat_service.get_client()
        self.temperature = 0.0
        self.data_warehouse_type = data_warehouse_type
        # Set verbosity based on Django settings
        self.verbose = settings.RAGSTAR_LOG_LEVEL == "DEBUG"
        if self.verbose:
            logger.info(
                f"QuestionAnswererAgent initialized with verbose=True (LogLevel: {settings.RAGSTAR_LOG_LEVEL})"
            )

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
                    self.conn_pool = AsyncConnectionPool(
                        conninfo=pg_conn_string,
                        kwargs=connection_kwargs,
                        max_size=20,
                        min_size=5,
                        open=False,
                    )
                    # Use AsyncPostgresSaver, but it needs an async connection
                    # This initialization might need refactoring if issues persist
                    try:
                        self.memory = AsyncPostgresSaver(conn=self.conn_pool)
                        if self.verbose:
                            logger.info(
                                "Initialized AsyncPostgresSaver checkpointer (using async pool)."
                            )
                    except RuntimeError as e:
                        if "no running event loop" in str(e):
                            logger.info(
                                "AsyncPostgresSaver requires running event loop, deferring initialization until needed"
                            )
                            self.memory = None  # Will be initialized later when needed
                        else:
                            raise
                else:
                    self.conn_pool = None
                    logger.warning(
                        "DB Connection Pool not available (DATABASE_URL missing?). Agent state will not be persisted."
                    )
            except Exception as e:
                self.conn_pool = None
                logger.error(
                    f"Failed to initialize AsyncPostgresSaver: {e}",
                    exc_info=self.verbose,
                )

        # Optional conversation logger (propagated by higher-level orchestrator)
        self.conversation_logger: Optional[ConversationLogger] = None

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
            # Get usable model embeddings with their related models
            usable_model_embeddings = ModelEmbedding.objects.filter(
                can_be_used_for_answers=True
            ).select_related("model")

            # Extract model names from the embeddings
            usable_model_names = [
                emb.model.name for emb in usable_model_embeddings if emb.model
            ]
            usable_models = Model.objects.filter(name__in=usable_model_names)

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
            """Fetches detailed information for specified dbt models."""
            if self.verbose:
                # Add newline for spacing
                logger.info(f"\nTool: fetch_model_details(names={model_names})")

            # Log the tool call
            if getattr(self, "conversation_logger", None):
                await sync_to_async(self.conversation_logger.log_tool_call)(
                    tool_name="fetch_model_details",
                    tool_input={"model_names": model_names},
                )

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
            """Performs a semantic search for models similar to the query."""
            if self.verbose:
                # Add newline for spacing
                logger.info(f"\nTool: model_similarity_search(query='{query}')")

            # Log the tool call
            if getattr(self, "conversation_logger", None):
                await sync_to_async(self.conversation_logger.log_tool_call)(
                    tool_name="model_similarity_search",
                    tool_input={"query": query},
                )

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
                found_model_names = [emb.model.name for emb in embeddings]
                distances_map = {emb.model.name: emb.distance for emb in embeddings}
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
            """Searches past feedback linked to similar questions."""
            if self.verbose:
                # Add newline for spacing
                logger.info(f"\nTool: search_past_feedback(query='{query}')")

            # Log the tool call
            if getattr(self, "conversation_logger", None):
                await sync_to_async(self.conversation_logger.log_tool_call)(
                    tool_name="search_past_feedback",
                    tool_input={"query": query},
                )

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
            """Searches the content of past feedback entries for relevant information."""
            if self.verbose:
                # Add newline for spacing
                logger.info(f"\nTool: search_feedback_content(query='{query}')")

            # Log the tool call
            if getattr(self, "conversation_logger", None):
                await sync_to_async(self.conversation_logger.log_tool_call)(
                    tool_name="search_feedback_content",
                    tool_input={"query": query},
                )

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
            """Searches organizational context based on the current question."""
            if self.verbose:
                # Add newline for spacing
                logger.info(f"\nTool: search_organizational_context(query='{query}')")

            # Log the tool call
            if getattr(self, "conversation_logger", None):
                await sync_to_async(self.conversation_logger.log_tool_call)(
                    tool_name="search_organizational_context",
                    tool_input={"query": query},
                )

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
        async def finish_workflow(
            answer: str, sql_query: Optional[str] = None
        ) -> QAResponse:
            """Concludes the workflow and returns both the answer text and an optional SQL query."""
            if self.verbose:
                logger.info(
                    f"\nTool: finish_workflow(answer='{answer[:80]}...', sql_present={bool(sql_query)})"
                )

            # Use the canonical QAResponse model for strong typing at the boundary
            response_obj = QAResponse(answer=answer, sql_query=sql_query)
            # Return as a plain dict for JSON serialization while preserving the static type annotation
            return response_obj.model_dump()

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
            # Log the actual accumulated_models from the input state more visibly
            logger.info(
                f"Agent Node: Received state with accumulated_models: {_create_models_summary(state.get('accumulated_models', []))}"
            )

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
            data_warehouse_type=self.data_warehouse_type,
        )

        # Log the accumulated_models before creating guidance message
        if self.verbose:
            logger.info(
                f"Agent Node: accumulated_models variable before guidance: {_create_models_summary(accumulated_models)}"
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
            # Log the system prompt and guidance message parts
            logger.info(f"QA System Prompt: {system_prompt[:500]}...")  # Log a preview
            if thread_context:
                context_str_log = "\n".join(
                    [
                        f"{m.get('user')}: {m.get('text')[:100]}..."
                        for m in thread_context[-3:]
                    ]
                )  # Preview last 3
                logger.info(f"QA Slack Thread Context (last 3):\n{context_str_log}")
            logger.info(f"QA Guidance Message: {guidance[:500]}...")

            # Detailed log of messages being sent to QA LLM
            log_messages_qa = []
            for msg_idx, msg in enumerate(messages_for_llm):
                if isinstance(msg, SystemMessage):
                    log_messages_qa.append(
                        f"  {msg_idx}. System: {msg.content[:200]}..."
                    )
                elif isinstance(msg, HumanMessage):
                    log_messages_qa.append(
                        f"  {msg_idx}. Human: {msg.content[:200]}..."
                    )
                elif isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        tool_calls_summary = ", ".join(
                            [
                                f"{tc['name']}(args={str(tc['args'])[:50]}...)"
                                for tc in msg.tool_calls
                            ]
                        )
                        log_messages_qa.append(
                            f"  {msg_idx}. AI: ToolCalls({tool_calls_summary}) Content: {msg.content[:100]}..."
                        )
                    else:
                        log_messages_qa.append(
                            f"  {msg_idx}. AI: {msg.content[:200]}..."
                        )
                elif isinstance(msg, ToolMessage):
                    log_messages_qa.append(
                        f"  {msg_idx}. Tool (id={msg.tool_call_id}): {msg.content[:200]}..."
                    )
                else:
                    log_messages_qa.append(
                        f"  {msg_idx}. UnknownMessage: {str(msg)[:200]}..."
                    )
            logger.info(
                f"Messages for LLM (QuestionAnswerer) - Total {len(messages_for_llm)}:\n"
                + "\n".join(log_messages_qa)
            )

        agent_llm_with_tools = self.llm.bind_tools(self._tools)
        config = {"run_name": "QuestionAnswererAgentNode"}

        try:
            response = await agent_llm_with_tools.ainvoke(
                messages_for_llm, config=config
            )

            # --- NEW: Log token usage for this LLM call ---
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

                    # Log LLM input (prompt)
                    if prompt_tokens:
                        await sync_to_async(
                            self.conversation_logger.log_agent_response
                        )(
                            content="[LLM Prompt]",  # We avoid logging the full prompt to DB
                            tokens_used=prompt_tokens,
                            metadata={"type": "llm_input"},
                        )

                    # Log LLM output (response)
                    await sync_to_async(self.conversation_logger.log_agent_response)(
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
                        f"Failed to record LLM token usage in conversation log: {log_err}"
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
            logger.info(
                f"Update State Node: Received state with accumulated_models: {_create_models_summary(state.get('accumulated_models', []))}"
            )
        # Ensure this node is async if called by ainvoke
        updates: Dict[str, Any] = {}
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if not isinstance(last_message, ToolMessage):
            return updates

        tool_content = last_message.content
        # Safely parse JSON if needed (handle JSON objects and arrays)
        if isinstance(tool_content, str) and tool_content.strip().startswith(
            ("{", "[")
        ):
            try:
                tool_content = json.loads(tool_content)
            except json.JSONDecodeError:
                # Fallback: try ast.literal_eval which can handle single quotes, etc.
                try:
                    import ast

                    tool_content = ast.literal_eval(tool_content)
                except (ValueError, SyntaxError):
                    # leave as string if still unparseable
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
            updates["vector_search_calls"] = state.get("vector_search_calls", 0) + 1
            if isinstance(tool_content, list):
                # Ensure we make copies from the state to avoid modifying it directly before update
                current_accumulated_models = list(state.get("accumulated_models", []))
                current_model_names = set(state.get("accumulated_model_names", set()))
                newly_added_count = 0
                for model_data in tool_content:
                    if (
                        isinstance(model_data, dict)
                        and model_data.get("name") not in current_model_names
                    ):
                        current_accumulated_models.append(model_data)
                        current_model_names.add(model_data["name"])
                        newly_added_count += 1
                if newly_added_count > 0:
                    updates["accumulated_models"] = current_accumulated_models
                    updates["accumulated_model_names"] = current_model_names
                    if self.verbose:
                        logger.info(
                            f"Added {newly_added_count} models from similarity search to state."
                        )
            elif self.verbose:
                logger.warning(
                    f"Tool 'model_similarity_search' did not return a list: {type(tool_content)}"
                )
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
            # Expecting dict with keys 'answer' and optional 'sql_query'
            if isinstance(tool_content, dict):
                updates["final_answer"] = tool_content.get("answer")
                updates["sql_query"] = tool_content.get("sql_query")

                updates["models_snapshot_for_final_answer"] = list(
                    state.get("accumulated_models", [])
                )

                if self.verbose:
                    logger.info("\n--- Final Answer Received via Tool ---")
            else:
                logger.warning(
                    f"Finish tool returned unexpected content type: {type(tool_content)}"
                )

        if self.verbose and updates:
            # Add separators
            updates_str = json.dumps(list(updates.keys()), indent=2)
            logger.info(
                f"\n--- QuestionAnswerer State Updates (keys) ---\n{updates_str}\n--------------------------------------"
            )
            # Log the actual content of updates for accumulated_models
            if "accumulated_models" in updates:
                logger.info(
                    f"Update State Node: Returning updates with accumulated_models: {_create_models_summary(updates['accumulated_models'])}"
                )
            else:
                logger.info(
                    "Update State Node: Returning updates WITHOUT accumulated_models."
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
            if self.verbose:
                logger.info(
                    "finalize_direct_answer_node: Final answer already exists in state."
                )
            return updates  # Already finalized

        last_message = state["messages"][-1] if state.get("messages") else None

        if self.verbose:
            logger.info(
                f"finalize_direct_answer_node: last_message type: {type(last_message)}"
            )
            if isinstance(last_message, AIMessage):
                logger.info(
                    f"finalize_direct_answer_node: last_message.content: {last_message.content}"
                )
                logger.info(
                    f"finalize_direct_answer_node: last_message.tool_calls: {last_message.tool_calls}"
                )
                logger.info(
                    f"finalize_direct_answer_node: hasattr response_metadata: {hasattr(last_message, 'response_metadata')}"
                )
                if hasattr(last_message, "response_metadata"):
                    logger.info(
                        f"finalize_direct_answer_node: isinstance response_metadata dict: {isinstance(last_message.response_metadata, dict)}"
                    )
                    if isinstance(last_message.response_metadata, dict):
                        logger.info(
                            f"finalize_direct_answer_node: response_metadata.get('finish_reason'): {last_message.response_metadata.get('finish_reason')}"
                        )

        if isinstance(last_message, AIMessage) and last_message.content:
            # Condition for being routed here from 'agent' via 'tools_condition' is that 'tool_calls' is empty.
            # We double-check content and that finish_reason is 'stop'.
            has_response_metadata = hasattr(last_message, "response_metadata")
            is_response_metadata_dict = (
                isinstance(last_message.response_metadata, dict)
                if has_response_metadata
                else False
            )
            actual_finish_reason = (
                last_message.response_metadata.get("finish_reason")
                if is_response_metadata_dict
                else None
            )

            is_direct_stop_answer = (
                not last_message.tool_calls  # Expect: True (empty list)
                and has_response_metadata  # Expect: True
                and is_response_metadata_dict  # Expect: True
                and isinstance(actual_finish_reason, str)
                and actual_finish_reason.lower() == "stop"
            )

            if self.verbose:
                logger.info(
                    f"finalize_direct_answer_node: Calculated is_direct_stop_answer: {is_direct_stop_answer}"
                )
                logger.info(
                    f"  - not last_message.tool_calls: {not last_message.tool_calls}"
                )
                logger.info(f"  - has_response_metadata: {has_response_metadata}")
                logger.info(
                    f"  - is_response_metadata_dict: {is_response_metadata_dict}"
                )
                logger.info(
                    f"  - actual_finish_reason == 'stop': {actual_finish_reason == 'stop'} (actual: {actual_finish_reason})"
                )

            if is_direct_stop_answer:
                answer_content = last_message.content
                if isinstance(answer_content, list):
                    # Ensure all elements are strings before joining
                    processed_content = []
                    for item in answer_content:
                        if isinstance(item, str):
                            processed_content.append(item)
                        elif (
                            isinstance(item, dict) and "text" in item
                        ):  # Handle potential dict content like Anthropic
                            processed_content.append(item["text"])
                        else:
                            processed_content.append(
                                str(item)
                            )  # Fallback to string conversion
                    updates["final_answer"] = "\\n".join(processed_content)
                    if self.verbose:
                        logger.info(
                            f"finalize_direct_answer_node: Joined list content for final_answer. Preview: {updates['final_answer'][:200]}..."
                        )
                else:
                    updates["final_answer"] = str(
                        answer_content
                    )  # Ensure it's a string
                    if self.verbose:
                        logger.info(
                            f"finalize_direct_answer_node: Set string content for final_answer. Preview: {updates['final_answer'][:200]}..."
                        )

                # Also capture the models used for this direct answer
                updates["models_snapshot_for_final_answer"] = list(
                    state.get("accumulated_models", [])
                )
                if self.verbose:
                    logger.info(
                        "QuestionAnswerer graph: Finalizing direct answer from LLM (finish_reason='stop'). Models captured."
                    )
            elif not last_message.tool_calls and self.verbose:
                # This case means 'tools_condition' sent us here, but it wasn't a clean 'stop' with content.
                # For example, LLM might have just stopped without content, or finish_reason was length etc.
                logger.warning(
                    "QuestionAnswerer graph: Agent stopped without tool calls, "
                    "but last message was not a clear direct answer (content or finish_reason mismatch). "
                    f"Content present: {bool(last_message.content)}. Tool calls: {last_message.tool_calls}. Finish reason: {actual_finish_reason}"
                )
        elif self.verbose:
            logger.info(
                "finalize_direct_answer_node: last_message was not an AIMessage with content."
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
            sql_query=None,
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

            # --- Ensure connection pool is open before invoking graph ---
            if self.conn_pool and self.conn_pool.closed:
                if self.verbose:
                    logger.info(
                        f"Connection pool '{self.conn_pool.name}' is closed. Attempting to open."
                    )
                try:
                    await self.conn_pool.open(wait=True)
                    if self.verbose:
                        logger.info(
                            f"Connection pool '{self.conn_pool.name}' opened successfully."
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to open connection pool '{self.conn_pool.name}': {e}",
                        exc_info=True,
                    )
                    # Depending on desired behavior, might re-raise or return error
                    return {
                        "error": f"Failed to open DB connection pool: {e}",
                        "answer": None,
                        "models_used": [],
                    }
            # --- End pool opening ---

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
            "sql_query": None,
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
            if final_state.get("final_answer"):
                result["answer"] = final_state["final_answer"]
                result["sql_query"] = final_state.get("sql_query")
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


def _create_models_summary(models: List[Dict[str, Any]]) -> str:
    """Create a concise summary of models for logging purposes."""
    if not models:
        return "No models"

    model_names = [model.get("name", "Unknown") for model in models]
    return f"{len(models)} models: {', '.join(model_names)}"
