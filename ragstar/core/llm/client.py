"""LLM client for interacting with language models."""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage

from ragstar.utils.config import get_config_value

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with language models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Initialize the LLM client.

        Args:
            api_key: The API key to use for the LLM service
            model: The model name to use for completions
            embedding_model: The model name to use for embeddings
            temperature: The temperature to use for completions
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
        self.temperature = temperature

        # Initialize OpenAI client (Raw client - might be needed elsewhere? Let's keep for now)
        # self.client = OpenAI(api_key=self.api_key) # COMMENTED OUT or REMOVE LATER if unused

        # ADDED: Initialize LangChain clients
        # MODIFIED: Only pass temperature if model is not o3-mini (or potentially others)
        chat_client_kwargs = {
            "api_key": self.api_key,
            "model": self.model,
        }
        if (
            self.model != "o3-mini"
        ):  # Add other models here if they also don't support temp
            chat_client_kwargs["temperature"] = self.temperature

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

            # Response is usually an AIMessage with content attribute
            return response.content
        except Exception as e:
            logger.error(f"Error getting completion: {e}")
            raise
