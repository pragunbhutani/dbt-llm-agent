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
from ragstar.core.agents import SlackResponder
from ragstar.utils.cli_utils import get_config_value
from ragstar.utils.logging import setup_logging
from ragstar.utils.slack import get_async_slack_client

# Import Slack Bolt components
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_sdk.errors import SlackApiError  # <<< ADD THIS IMPORT

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
        db_uri = get_config_value("database_url")
        if not db_uri:
            logger.error(
                "DATABASE_URL not found in environment variables or config. Postgres is required."
            )
            # Potentially raise an exception or handle appropriately if DB is essential
        model_storage = ModelStorage(connection_string=db_uri)
        vector_store = ModelEmbeddingStorage(connection_string=db_uri)
        question_storage = QuestionStorage(
            connection_string=db_uri, openai_api_key=openai_api_key
        )

        # Initialize the shared AsyncWebClient (needed by SlackResponder)
        slack_client = get_async_slack_client(slack_bot_token)

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
                return

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

        # --- NEW: Handle reaction_added event for feedback --- #
        @bolt_app.event("reaction_added")
        async def handle_reaction_added(event, client, ack):
            """Handles emoji reactions to capture feedback."""
            await ack()
            logger.info(f"Received reaction_added event: {event}")

            reaction = event.get("reaction")
            user_id = event.get("user")  # User who reacted
            item_user_id = event.get("item_user")  # User whose message was reacted to
            item_details = event.get("item", {})
            item_type = item_details.get("type")
            # channel_id = item_details.get("channel") # No longer needed for history lookup

            # --- DETERMINE BOT USER ID ---
            bot_user_id = None
            try:
                auth_test_res = await client.auth_test()
                bot_user_id = auth_test_res.get("user_id")
            except Exception as e:
                logger.error(f"Error getting bot user ID for reaction check: {e}")
                return  # Can't proceed without bot ID

            # --- CHECK IF REACTION IS ON BOT'S ITEM ---
            if item_user_id != bot_user_id:
                logger.debug(
                    f"Ignoring reaction to non-bot item (item_user: {item_user_id}, bot_user: {bot_user_id})"
                )
                return

            # --- DETERMINE ITEM IDENTIFIER & TYPE FOR DB (Simplified) ---
            item_identifier = None
            item_type_for_db = (
                None  # The type ('message' or 'file') to use for DB lookup
            )

            if item_type == "message":
                item_identifier = item_details.get("ts")
                if item_identifier:
                    item_type_for_db = "message"
                    logger.info(
                        f"Reaction on message item. Using ts: {item_identifier}"
                    )
                else:
                    logger.warning(
                        f"Missing 'ts' in item details for message reaction: {event}"
                    )
                    return
            elif item_type == "file":
                item_identifier = item_details.get("file_id")
                if item_identifier:
                    item_type_for_db = "file"
                    logger.info(
                        f"Reaction on file item. Using file_id: {item_identifier}"
                    )
                else:
                    logger.warning(
                        f"Missing 'file_id' in item details for file reaction: {event}"
                    )
                    return
            else:
                logger.debug(f"Ignoring reaction to unsupported item type: {item_type}")
                return

            # --- END DETERMINE ITEM IDENTIFIER & TYPE ---

            # Map reaction to feedback
            was_useful = None
            if reaction == "+1" or reaction == "thumbsup":  # Common positive reactions
                was_useful = True
            elif (
                reaction == "-1" or reaction == "thumbsdown"
            ):  # Common negative reactions
                was_useful = False
            else:
                # Ignore other reactions
                logger.debug(f"Ignoring non-feedback reaction: {reaction}")
                return

            if not question_storage:
                logger.error("QuestionStorage not initialized. Cannot record feedback.")
                return

            # Run the database update in a separate thread
            try:
                # Note: update_feedback runs synchronously
                success = await asyncio.to_thread(
                    question_storage.update_feedback,
                    item_identifier=item_identifier,  # Use determined identifier
                    item_type=item_type_for_db,  # Use determined type ('message' or 'file')
                    was_useful=was_useful,
                    feedback_provider_user_id=user_id,
                )
                if success:
                    logger.info(
                        f"Successfully recorded feedback ({reaction} -> was_useful={was_useful}) for {item_type_for_db} {item_identifier} from user {user_id}"
                    )
                else:
                    logger.warning(
                        f"Failed to record feedback for {item_type_for_db} {item_identifier} (record not found or DB error)"
                    )
            except Exception as e:
                logger.error(f"Error calling update_feedback: {e}", exc_info=True)

        # --- END NEW HANDLER --- #

        # --- NEW: Handle reaction_removed event to unset feedback --- #
        @bolt_app.event("reaction_removed")
        async def handle_reaction_removed(event, client, ack):
            """Handles removal of emoji reactions to unset feedback."""
            await ack()
            logger.info(f"Received reaction_removed event: {event}")

            reaction = event.get("reaction")
            user_id = event.get("user")  # User who removed the reaction
            item_user_id = event.get("item_user")  # User whose message had the reaction
            item_details = event.get("item", {})
            item_type = item_details.get("type")
            # channel_id = item_details.get("channel") # No longer needed for history lookup

            # --- DETERMINE BOT USER ID ---
            bot_user_id = None
            try:
                auth_test_res = await client.auth_test()
                bot_user_id = auth_test_res.get("user_id")
            except Exception as e:
                logger.error(
                    f"Error getting bot user ID for reaction removal check: {e}"
                )
                return

            # --- CHECK IF REACTION IS ON BOT'S ITEM ---
            if item_user_id != bot_user_id:
                logger.debug(
                    f"Ignoring reaction removal from non-bot item (item_user: {item_user_id}, bot_user: {bot_user_id})"
                )
                return

            # --- DETERMINE ITEM IDENTIFIER & TYPE FOR DB (Simplified) ---
            item_identifier = None
            item_type_for_db = (
                None  # The type ('message' or 'file') to use for DB lookup
            )

            if item_type == "message":
                item_identifier = item_details.get("ts")
                if item_identifier:
                    item_type_for_db = "message"
                    logger.info(
                        f"Reaction removal on message item. Using ts: {item_identifier}"
                    )
                else:
                    logger.warning(
                        f"Missing 'ts' in item details for message reaction removal: {event}"
                    )
                    return
            elif item_type == "file":
                item_identifier = item_details.get("file_id")
                if item_identifier:
                    item_type_for_db = "file"
                    logger.info(
                        f"Reaction removal on file item. Using file_id: {item_identifier}"
                    )
                else:
                    logger.warning(
                        f"Missing 'file_id' in item details for file reaction removal: {event}"
                    )
                    return
            else:
                logger.debug(
                    f"Ignoring reaction removal from unsupported item type: {item_type}"
                )
                return

            # --- END DETERMINE ITEM IDENTIFIER & TYPE ---

            # Check if the removed reaction was a feedback emoji
            if reaction not in ["+1", "thumbsup", "-1", "thumbsdown"]:
                logger.debug(f"Ignoring removal of non-feedback reaction: {reaction}")
                return

            if not question_storage:
                logger.error("QuestionStorage not initialized. Cannot remove feedback.")
                return

            # Run the database update in a separate thread to remove feedback (set was_useful=None)
            try:
                success = await asyncio.to_thread(
                    question_storage.update_feedback,
                    item_identifier=item_identifier,  # Use determined identifier
                    item_type=item_type_for_db,  # Use determined type ('message' or 'file')
                    was_useful=None,  # Set was_useful to None to remove feedback
                    feedback_provider_user_id=user_id,  # Still log who removed it
                )
                if success:
                    logger.info(
                        f"Successfully removed feedback (reaction: {reaction}) for {item_type_for_db} {item_identifier} based on removal by user {user_id}"
                    )
                else:
                    logger.warning(
                        f"Failed to remove feedback for {item_type_for_db} {item_identifier} (record not found or DB error)"
                    )
            except Exception as e:
                logger.error(
                    f"Error calling update_feedback for removal: {e}", exc_info=True
                )

        # --- END NEW HANDLER --- #

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
