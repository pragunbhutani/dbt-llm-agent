"""Storage module for dbt-llm-agent.

This module provides storage functionality for dbt-llm-agent.
"""

from dbt_llm_agent.storage.model_storage import ModelStorage
from dbt_llm_agent.storage.model_embedding_storage import ModelEmbeddingStorage
from dbt_llm_agent.storage.question_storage import QuestionStorage

# Removed backwards compatibility aliases as they reference non-existent modules

__all__ = [
    "ModelStorage",
    "ModelEmbeddingStorage",
    "QuestionStorage",
]
