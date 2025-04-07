import logging
import os
from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
import asyncio  # For running sync code in async context

# RAGstar Imports - Adjust paths as necessary
from ragstar.core.llm.client import LLMClient
from ragstar.storage.model_storage import ModelStorage
from ragstar.storage.model_embedding_storage import ModelEmbeddingStorage
from ragstar.storage.question_storage import QuestionStorage
from ragstar.core.agents.slack_responder import SlackResponder
from ragstar.utils.cli_utils import get_config_value, setup_logging

# Import Slack Bolt components
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- FastAPI App Setup ---
app = FastAPI(
    title="RAGstar Slack Integration API",
    description="API endpoint to handle questions from Slack.",
)

# --- Global Variables / Resource Management ---
# Initialize necessary components (consider a dependency injection framework for larger apps)
# These might be initialized once at startup
llm_client = None
model_storage = None
vector_store = None
question_storage = None
slack_responder = None
slack_client = None


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    global llm_client, model_storage, vector_store, question_storage
    global slack_responder, slack_client

    logger.info("Initializing RAGstar components for API...")
    try:
        openai_api_key = get_config_value("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables or config."
            )

        # LLM Client
        llm_client = LLMClient(api_key=openai_api_key)

        # Storage Components
        db_uri = get_config_value("POSTGRES_URI")
        if not db_uri:
            raise ValueError(
                "POSTGRES_URI not found in environment variables or config."
            )
        model_storage = ModelStorage(database_uri=db_uri)
        vector_store = ModelEmbeddingStorage(
            database_uri=db_uri, embedder_api_key=openai_api_key
        )
        question_storage = QuestionStorage(
            database_uri=db_uri, openai_api_key=openai_api_key
        )

        # Initialize the shared AsyncWebClient (needed by SlackResponder)
        slack_client = AsyncWebClient(
            token=slack_bot_token
        )  # Use token fetched globally

        # Slack Responder Agent (Now initializes its own QuestionAnswerer)
        slack_responder = SlackResponder(
            llm_client=llm_client,
            # Pass dependencies needed by SlackResponder to create QuestionAnswerer
            model_storage=model_storage,
            vector_store=vector_store,
            question_storage=question_storage,
            slack_client=slack_client,
            verbose=True,  # Or configure based on environment
        )

        # Attach handler (defined outside) to the globally defined bolt_app
        async def handle_app_mention(event, say, ack):
            await ack()  # Acknowledge event within 3 seconds
            logger.info(f"Received app_mention event: {event}")

            channel_id = event.get("channel")
            # Use event["thread_ts"] if present (message is in thread), else use event["ts"]
            thread_ts = event.get("thread_ts", event.get("ts"))
            user_question = event.get("text", "").strip()
            user_id = event.get("user")

            # Refine question extraction (remove bot mention)
            bot_user_id = None
            try:
                # Fetch bot's user ID using the client provided by Bolt handler
                auth_test_res = await slack_client.auth_test()
                bot_user_id = auth_test_res.get("user_id")
                if bot_user_id:
                    user_question = user_question.replace(
                        f"<@{bot_user_id}>", ""
                    ).strip()
                else:
                    logger.warning("Could not determine bot user ID from auth.test")
            except SlackApiError as e:
                logger.error(f"Slack API error getting bot user ID: {e}")
            except Exception as e:
                logger.error(
                    f"Unexpected error getting bot user ID: {e}"
                )  # Catch broader exceptions

            if not user_question:
                logger.warning("Received mention without question text.")
                try:
                    # Use say utility provided by Bolt
                    await say(
                        text="It looks like you mentioned me, but didn't ask a question. Please include your question after the mention.",
                        thread_ts=thread_ts,
                    )
                except Exception as e:
                    logger.error(f"Failed to send 'ask question' message to Slack: {e}")
                return

            # Check if the responder agent is ready (initialized in startup)
            if not slack_responder:
                logger.error("SlackResponder not initialized. Cannot process mention.")
                try:
                    await say(
                        text="Sorry, I encountered an internal error and cannot process your request right now. Please try again later.",
                        thread_ts=thread_ts,
                    )
                except Exception as e:
                    logger.error(f"Failed to send error message to Slack: {e}")
                return

            logger.info(
                f"Triggering SlackResponder workflow for mention in {channel_id}/{thread_ts}"
            )
            # Run the potentially blocking workflow in a separate thread using asyncio.to_thread
            # This prevents blocking Bolt's async event loop
            try:
                # Pass necessary arguments to the workflow method
                await asyncio.to_thread(
                    slack_responder.run_slack_workflow,
                    question=user_question,
                    channel_id=channel_id,
                    thread_ts=thread_ts,  # Pass the correct thread identifier
                )
                logger.info(
                    f"SlackResponder workflow thread initiated for {channel_id}/{thread_ts}"
                )
                # Note: We don't await the result here, the workflow runs in the background.
                # The SlackResponder's respond_to_slack_thread tool handles sending the final answer.
            except Exception as e:
                logger.error(
                    f"Failed to start SlackResponder workflow thread: {e}",
                    exc_info=True,
                )
                # Optionally inform the user about the failure
                try:
                    await say(
                        text=f"Sorry {f'<@{user_id}>' if user_id else 'there'}, I couldn't start processing your request due to an internal error.",
                        thread_ts=thread_ts,
                    )
                except Exception as slack_e:
                    logger.error(
                        f"Failed to send processing error message to Slack: {slack_e}"
                    )

        # Attach handler (defined outside) to the globally defined bolt_app
        bolt_app.event("app_mention")(handle_app_mention)
        # Add other handlers here if needed
        # bolt_app.message("keyword")(handle_keyword_message)

        logger.info("RAGstar components and Bolt handlers initialized successfully.")

    except Exception as e:
        logger.error(
            f"Fatal error during API startup initialization: {e}", exc_info=True
        )
        # Depending on deployment, might want to exit or prevent app from starting
        raise RuntimeError(f"API Initialization Failed: {e}") from e


# --- Request Models ---
# Remove unused class definition
# class SlackQuestionInput(BaseModel):
#     question: str
#     channel_id: str
#     thread_ts: str
#     # Optional: Add user_id if needed for context or permissions
#     user_id: Optional[str] = None


# --- Bolt App Initialization ---
# Initialize Bolt app globally so handlers can be attached in startup
# Fetch config values needed for Bolt
slack_bot_token = get_config_value("SLACK_BOT_TOKEN")
slack_signing_secret = get_config_value("SLACK_SIGNING_SECRET")

if not slack_bot_token:
    raise ValueError("SLACK_BOT_TOKEN not found in environment variables or config.")
if not slack_signing_secret:
    raise ValueError(
        "SLACK_SIGNING_SECRET not found in environment variables or config."
    )

# Let Bolt manage its own client internally for simplicity
bolt_app = AsyncApp(
    token=slack_bot_token,
    signing_secret=slack_signing_secret,
)
bolt_handler = AsyncSlackRequestHandler(bolt_app)


# Mount Bolt handler to FastAPI app
@app.post("/slack/events")
async def endpoint(req: Request):
    return await bolt_handler.handle(req)


# Add other handlers if needed (e.g., for interactive components)
# Remove commented out interactive endpoint example
# @app.post("/slack/interactive")
# async def interactive_endpoint(req: Request):
#     return await bolt_handler.handle(req)


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    # Basic check: are components initialized?
    if not all(
        [
            llm_client,
            model_storage,
            vector_store,
            question_storage,
            slack_responder,
            slack_client,
        ]
    ):
        logger.warning("Health check failed: Components not fully initialized.")
        raise HTTPException(status_code=503, detail="Service components not ready")
    return {"status": "ok"}


# --- Running the App (for local development) ---
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting RAGstar Slack API with Bolt on port {port}")
    # Point uvicorn to the app object within this module
    uvicorn.run(
        "ragstar.api.slack_handler:app", host="0.0.0.0", port=port, reload=False
    )
