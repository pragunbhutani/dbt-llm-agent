"""Slack integration for the ragstar."""

import os
import logging
import threading
from typing import Dict, Any, Optional, Callable

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from ragstar.core.agent import DBTAgent

logger = logging.getLogger(__name__)


class SlackBot:
    """Slack bot for interacting with the DBT agent."""

    def __init__(
        self,
        agent: DBTAgent,
        slack_bot_token: str,
        slack_app_token: str,
        signing_secret: Optional[str] = None,
    ):
        """Initialize the Slack bot.

        Args:
            agent: The DBT agent
            slack_bot_token: Slack bot token
            slack_app_token: Slack app token
            signing_secret: Slack signing secret
        """
        self.agent = agent

        # Initialize the Slack app
        self.app = App(token=slack_bot_token, signing_secret=signing_secret)

        self.slack_app_token = slack_app_token

        # Register event handlers
        self._register_event_handlers()

        logger.info("Initialized Slack bot")

    def _register_event_handlers(self) -> None:
        """Register event handlers for the Slack app."""

        # Handle app mentions
        @self.app.event("app_mention")
        def handle_app_mention(event, say):
            thread_ts = event.get("thread_ts", event.get("ts"))
            self._handle_question(event, say, thread_ts)

        # Handle direct messages
        @self.app.event("message")
        def handle_message(event, say):
            # Ignore messages from the bot itself or in channels
            if event.get("bot_id") or event.get("channel_type") != "im":
                return

            thread_ts = event.get("thread_ts", event.get("ts"))
            self._handle_question(event, say, thread_ts)

        # Handle slash commands
        @self.app.command("/dbt-doc")
        def handle_doc_command(ack, command, say):
            ack()

            model_name = command["text"].strip()
            if not model_name:
                say(text="Please specify a model name: `/dbt-doc model_name`")
                return

            thread_ts = command.get("thread_ts")

            # Acknowledge that we're generating documentation
            say(
                text=f"Generating documentation for model `{model_name}`...",
                thread_ts=thread_ts,
            )

            try:
                # Generate documentation
                result = self.agent.generate_documentation(model_name)

                if "error" in result:
                    say(
                        text=f"Error generating documentation: {result['error']}",
                        thread_ts=thread_ts,
                    )
                    return

                # Post the documentation
                say(text=result["full_documentation"], thread_ts=thread_ts)

                # Update the model documentation
                self.agent.update_model_documentation(model_name, result)

            except Exception as e:
                say(
                    text=f"Error generating documentation: {str(e)}",
                    thread_ts=thread_ts,
                )

    def _handle_question(self, event, say, thread_ts) -> None:
        """Handle a question from a user.

        Args:
            event: The Slack event
            say: The say function for responding
            thread_ts: Thread timestamp for threading replies
        """
        try:
            # Extract the question
            # For app mentions, remove the mention part
            user_id = event.get("user")
            text = event.get("text", "")

            if "<@" in text:
                # This is an app mention, extract the actual question
                question = text.split(">", 1)[1].strip() if ">" in text else text
            else:
                # This is a direct message
                question = text.strip()

            if not question:
                say(
                    text="Please ask a question about your dbt project.",
                    thread_ts=thread_ts,
                )
                return

            # Acknowledge that we're processing the question
            say(text="Thinking about your question...", thread_ts=thread_ts)

            # Answer the question
            result = self.agent.answer_question(question)

            # Format and post the answer
            say(text=result["answer"], thread_ts=thread_ts)

        except Exception as e:
            logger.error(f"Error handling question: {e}")
            say(
                text=f"I encountered an error while processing your question: {str(e)}",
                thread_ts=thread_ts,
            )

    def start(self, daemon: bool = False) -> None:
        """Start the Slack bot.

        Args:
            daemon: Whether to run in daemon mode (background thread)
        """
        try:
            logger.info("Starting Slack bot...")

            # Create the socket mode handler
            handler = SocketModeHandler(self.app, self.slack_app_token)

            if daemon:
                # Start in a background thread
                thread = threading.Thread(target=handler.start)
                thread.daemon = True
                thread.start()
                logger.info("Slack bot started in background")
            else:
                # Start in the main thread
                handler.start()

        except Exception as e:
            logger.error(f"Error starting Slack bot: {e}")
            raise
