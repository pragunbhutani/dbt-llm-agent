import os
import logging
from typing import List, Optional, Dict, Any

# Import necessary Langchain clients directly
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
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


class EmbeddingService:
    """Service for generating text embeddings using a configured provider."""

    def __init__(self, org_settings: "OrganisationSettings"):
        """
        Initializes the embedding service using settings from the provided
        OrganisationSettings object.
        """
        self.client: Optional[Embeddings] = None
        provider_name = org_settings.llm_embeddings_provider
        model_name = org_settings.llm_embeddings_model
        api_key: Optional[str] = None

        if provider_name == "openai":
            api_key = org_settings.get_llm_openai_api_key()
        elif provider_name == "google":
            api_key = org_settings.get_llm_google_api_key()
        # Anthropic not currently supported for embeddings in this basic setup

        if not api_key and provider_name in ["openai", "google"]:
            logger.warning(
                f"API key for embedding provider '{provider_name}' (settings.LLM_{provider_name.upper()}_API_KEY) not found. Embedding generation disabled."
            )
            return
        elif provider_name not in ["openai", "google"]:
            logger.warning(
                f"Unsupported or misconfigured LLM_EMBEDDINGS_PROVIDER_NAME: {provider_name}. Embedding generation disabled."
            )
            return

        try:
            if provider_name == "openai":
                self.client = OpenAIEmbeddings(openai_api_key=api_key, model=model_name)
                logger.info(
                    f"EmbeddingService initialized OpenAIEmbeddings client with model: {model_name}"
                )
            elif provider_name == "google":
                self.client = GoogleGenerativeAIEmbeddings(
                    google_api_key=api_key, model=model_name
                )
                logger.info(
                    f"EmbeddingService initialized GoogleGenerativeAIEmbeddings client with model: {model_name}"
                )
            # No else needed due to earlier return for unsupported providers
        except Exception as e:
            logger.error(
                f"Failed to initialize {provider_name} embedding client (model: {model_name}): {e}",
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
    """Service for providing access to a configured chat LLM client."""

    def __init__(self, org_settings: "OrganisationSettings"):
        """
        Initializes the chat service using settings from the provided
        OrganisationSettings object.
        """
        self.llm: Optional[BaseChatModel] = None
        provider_name = org_settings.llm_chat_provider
        model_name = org_settings.llm_chat_model
        api_key: Optional[str] = None

        if provider_name == "openai":
            api_key = org_settings.get_llm_openai_api_key()
        elif provider_name == "google":
            api_key = org_settings.get_llm_google_api_key()
        elif provider_name == "anthropic":
            api_key = org_settings.get_llm_anthropic_api_key()

        # TODO: Add temperature to OrganisationSettings model
        temperature = getattr(settings, "LLM_CHAT_CONFIG_TEMPERATURE", None)

        init_kwargs: Dict[str, Any] = {}
        if temperature is not None:
            init_kwargs["temperature"] = temperature

        if not api_key and provider_name in ["openai", "google", "anthropic"]:
            logger.warning(
                f"API key for chat provider '{provider_name}' (settings.LLM_{provider_name.upper()}_API_KEY) not found. Chat functionality disabled."
            )
            return
        elif provider_name not in ["openai", "google", "anthropic"]:
            logger.warning(
                f"Unsupported or misconfigured LLM_CHAT_PROVIDER_NAME: {provider_name}. Chat functionality disabled."
            )
            return

        try:
            if provider_name == "openai":
                self.llm = ChatOpenAI(
                    openai_api_key=api_key, model=model_name, **init_kwargs
                )
            elif provider_name == "google":
                self.llm = ChatGoogleGenerativeAI(
                    google_api_key=api_key, model=model_name, **init_kwargs
                )
            elif provider_name == "anthropic":
                self.llm = ChatAnthropic(
                    anthropic_api_key=api_key, model=model_name, **init_kwargs
                )
        except Exception as e:
            logger.error(
                f"Failed to initialize {provider_name} chat client (model: {model_name}, config: {init_kwargs}): {e}",
                exc_info=True,
            )

    def get_client(self) -> Optional[BaseChatModel]:
        """Returns the initialized chat LLM client."""
        return self.llm


# --- Default Service Instances ---
# These are now gone, as services must be initialized per-organisation.
# default_embedding_service = EmbeddingService()
# default_chat_service = ChatService()
