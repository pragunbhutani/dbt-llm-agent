"""LLM client for interacting with language models."""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI

from dbt_llm_agent.utils.config import get_config_value

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

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        logger.info(
            f"Initialized LLM client with models: {self.model} and {self.embedding_model}"
        )

    def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Get embedding for a text.

        Args:
            text: The text to embed
            model: Optional override for the embedding model

        Returns:
            List of floats representing the embedding
        """
        try:
            # Use model parameter, otherwise use instance default
            use_model = model or self.embedding_model

            # Get embedding from OpenAI API
            response = self.client.embeddings.create(model=use_model, input=text)

            # Extract and return the embedding
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a zero embedding as fallback
            dim = 1536  # Default for text-embedding-3-small
            if "text-embedding-3" in (model or self.embedding_model):
                dim = 3072 if "large" in (model or self.embedding_model) else 1536
            return [0.0] * dim

    def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1000,
    ) -> str:
        """Get completion for a prompt.

        Args:
            prompt: The prompt to complete
            system_prompt: Optional system prompt to set context
            model: Optional override for the model
            temperature: Optional override for the temperature
            max_tokens: Maximum number of tokens to generate

        Returns:
            The completed text
        """
        try:
            # Use provided parameters or instance defaults
            use_model = model or self.model
            use_temperature = (
                temperature if temperature is not None else self.temperature
            )

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Get completion from OpenAI API
            response = self.client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=use_temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting completion: {e}")
            raise
