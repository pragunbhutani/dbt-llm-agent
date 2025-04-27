"""LLM client for interacting with language models."""

import os
import logging
import tiktoken  # Added import
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage, LLMResult
from langchain.callbacks.base import BaseCallbackHandler

from ragstar.utils.config import get_config_value

logger = logging.getLogger(__name__)


# Added: Function to estimate tokens (using cl100k_base as a general default)
# TODO: Consider making encoding model-specific if needed
def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Returns the number of tokens in a text string using tiktoken."""
    try:
        # Using cl100k_base encoding as it's common for GPT-4, GPT-3.5, embeddings
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback or handle error if encoding not found
        logger.warning(
            "Tiktoken cl100k_base encoding not found. Falling back to basic split."
        )
        return len(text.split())  # Basic fallback

    num_tokens = len(encoding.encode(text))
    return num_tokens


# --- NEW: Callback Handler for Token Logging --- #
class TokenUsageLogger(BaseCallbackHandler):
    """Callback Handler that logs token usage from LLM calls."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log token usage when an LLM call ends."""
        # OpenAI specific token usage is usually in llm_output
        if response.llm_output is not None and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            # Get model name if available (might be nested differently depending on LLM)
            model_name = response.llm_output.get("model_name", "unknown")

            logger.info(
                f"LLM Call ({model_name}) Usage: "
                f"Prompt={prompt_tokens}, Completion={completion_tokens}, Total={total_tokens}"
            )
        else:
            # Fallback or log that usage info wasn't found
            logger.info("LLM Call Usage: Token usage data not found in response.")


# --- End Callback Handler --- #


class LLMClient:
    """Client for interacting with language models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        temperature: float = 1.0,
    ):
        """Initialize the LLM client.

        Args:
            api_key: The API key to use for the LLM service
            model: The model name to use for completions
            embedding_model: The model name to use for embeddings
            temperature: The temperature to use for completions (reads from config if available)
        """
        # Get API key from parameter or config
        self.api_key = api_key or get_config_value("openai_api_key")
        if not self.api_key:
            raise ValueError(
                "No API key provided and no openai_api_key found in config"
            )

        # Set model names
        self.model = model or get_config_value("openai_model", "gpt-4o")
        self.embedding_model = embedding_model or get_config_value(
            "openai_embedding_model", "text-embedding-3-small"
        )
        # Read temperature from config, falling back to the parameter (defaulting to 1.0)
        self.temperature = float(get_config_value("temperature", temperature))

        # Initialize OpenAI client (Raw client - might be needed elsewhere? Let's keep for now)
        # self.client = OpenAI(api_key=self.api_key) # COMMENTED OUT or REMOVE LATER if unused

        # ADDED: Initialize LangChain clients
        # MODIFIED: Always pass the determined temperature
        chat_client_kwargs = {
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,  # Always pass temperature
        }
        # if (
        #     self.model != "o3-mini"
        # ):
        #     chat_client_kwargs["temperature"] = self.temperature
        self.chat_client = ChatOpenAI(**chat_client_kwargs)
        # END MODIFIED

        self.embedding_client = OpenAIEmbeddings(
            api_key=self.api_key, model=self.embedding_model
        )
        # END ADDED

        logger.info(
            f"Initialized LLM client with models: {self.model} and {self.embedding_model}"
        )

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embedding for a text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding

        Raises:
            ValueError: If the embedding cannot be generated
        """
        try:
            # Use the initialized LangChain embedding client
            embedding = self.embedding_client.embed_query(text)
            return embedding
        except Exception as e:
            error_msg = f"Error getting embedding: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1000,
    ) -> str:
        """Get completion for a prompt.

        Args:
            prompt: The prompt to complete
            system_prompt: Optional system prompt to set context
            temperature: Optional override for the temperature
            max_tokens: Maximum number of tokens to generate

        Returns:
            The completed text
        """
        try:
            # Build messages in LangChain format
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            # Use the initialized LangChain chat client's invoke method
            chat_client_instance = self.chat_client
            if temperature is not None:
                chat_client_instance = ChatOpenAI(
                    api_key=self.api_key,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            response = chat_client_instance.invoke(messages)

            # Added: Log token usage from response metadata
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if (
                hasattr(response, "response_metadata")
                and "token_usage" in response.response_metadata
            ):
                token_usage = response.response_metadata["token_usage"]
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)
                total_tokens = token_usage.get("total_tokens", 0)
                logger.info(
                    f"LLM Call ({self.model}) Usage: "
                    f"Prompt={prompt_tokens}, Completion={completion_tokens}, Total={total_tokens}"
                )
            else:
                # Fallback: Estimate prompt tokens if metadata is unavailable
                full_prompt_text = "\n".join([m.content for m in messages])
                prompt_token_count = count_tokens(full_prompt_text, self.model)
                logger.info(
                    f"LLM Call ({self.model}) Usage: "
                    f"Prompt Estimate={prompt_token_count} (Response metadata unavailable)"
                )
            # End Added

            # Response is usually an AIMessage with content attribute
            return response.content
        except Exception as e:
            logger.error(f"Error getting completion: {e}")
            raise
