import os
import logging
from typing import List, Optional, Dict, Any

# Import necessary Langchain clients directly
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

# Unused imports removed for clarity:
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Import Django settings
from django.conf import settings

# Unused imports removed for clarity:
# import re
# import json

logger = logging.getLogger(__name__)


# Helper function to get API key
def get_openai_api_key() -> Optional[str]:
    """Retrieves the OpenAI API key from settings or environment variables."""
    api_key = getattr(settings, "OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not found in Django settings or environment variables."
        )
    return api_key


class EmbeddingService:
    """Service for generating text embeddings using configured provider."""

    def __init__(self):
        """
        Initializes the embedding service and its client.
        """
        self.client: Optional[Embeddings] = None
        api_key = get_openai_api_key()
        # TODO: Generalize for different providers
        model_name = getattr(
            settings, "LLM_EMBEDDING_MODEL", "text-embedding-3-small"  # Updated default
        )

        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Embedding generation disabled.")
        else:
            try:
                # TODO: Add logic here to support other providers based on settings.LLM_PROVIDER
                self.client = OpenAIEmbeddings(openai_api_key=api_key, model=model_name)
                logger.info(
                    f"EmbeddingService initialized OpenAIEmbeddings client with model: {model_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize OpenAIEmbeddings client (model: {model_name}): {e}",
                    exc_info=True,
                )

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates an embedding for the given text."""
        if not self.client:
            logger.warning("Embedding client not available.")
            return None
        if not text:
            return None

        try:
            return self.client.embed_query(text)
        except Exception as e:
            logger.error(
                f"Error generating embedding for text: '{text[:50]}...': {e}",
                exc_info=True,
            )
            return None

    def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generates embeddings for a list of texts."""
        if not self.client:
            logger.warning("Embedding client not available.")
            return [None] * len(texts)

        # Consider using embed_documents for efficiency if supported and useful
        results = []
        for text in texts:
            results.append(self.get_embedding(text))
        return results


class ChatService:
    """Service for providing access to the default chat LLM client."""

    def __init__(self):
        """
        Initializes the chat service and its client.
        """
        self.llm: Optional[BaseChatModel] = None
        api_key = get_openai_api_key()
        # TODO: Generalize for different providers
        model_name = getattr(settings, "LLM_CHAT_MODEL", "gpt-4o-mini")

        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Chat functionality disabled.")
        else:
            try:
                # TODO: Add logic here to support other providers based on settings.LLM_PROVIDER
                self.llm = ChatOpenAI(openai_api_key=api_key, model=model_name)
                logger.info(
                    f"ChatService initialized ChatOpenAI client with model: {model_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize ChatOpenAI client (model: {model_name}): {e}",
                    exc_info=True,
                )

    def get_client(self) -> Optional[BaseChatModel]:
        """Returns the initialized chat LLM client."""
        return self.llm


# --- Default Service Instances ---
# These instances provide easy access across the application
default_embedding_service = EmbeddingService()
default_chat_service = ChatService()
