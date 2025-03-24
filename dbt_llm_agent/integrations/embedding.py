"""Embedding integration for various embedding models."""

import os
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def get_embedding(text: str, model: str = None) -> List[float]:
    """Get an embedding vector for a text string.

    Args:
        text: The text to embed
        model: The embedding model to use

    Returns:
        The embedding as a list of floats
    """
    # Get model from parameter, environment, or use default
    model = (
        model or os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
    )

    try:
        # Default to OpenAI's embedding model
        if "text-embedding" in model.lower():
            return get_openai_embedding(text, model)
        else:
            logger.warning(f"Unknown embedding model: {model}, falling back to OpenAI")
            return get_openai_embedding(text, "text-embedding-3-small")
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise


def get_openai_embedding(
    text: str, model: str = "text-embedding-3-small"
) -> List[float]:
    """Get an embedding from OpenAI's API.

    Args:
        text: The text to embed
        model: The OpenAI embedding model to use

    Returns:
        The embedding as a list of floats
    """
    try:
        from openai import OpenAI

        # Initialize client with API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment")

        client = OpenAI(api_key=api_key)

        # Get embedding from OpenAI API using new client format
        response = client.embeddings.create(model=model, input=text)

        # Extract and return the embedding
        embedding = response.data[0].embedding
        return embedding
    except ImportError:
        logger.error(
            "OpenAI package not installed. Install with 'pip install openai>=1.0.0'"
        )
        raise
    except Exception as e:
        logger.error(f"Error getting OpenAI embedding: {e}")
        raise
