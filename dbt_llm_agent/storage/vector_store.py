"""Vector store for storing and retrieving model embeddings using pgvector."""

import os
import logging
from typing import Dict, List, Any, Optional
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import numpy as np

from dbt_llm_agent.utils.config import load_config
from dbt_llm_agent.storage.models import Base, ModelEmbedding

logger = logging.getLogger(__name__)


class PostgresVectorStore:
    """Postgres vector store for storing and retrieving model embeddings using pgvector."""

    def __init__(
        self,
        connection_string: str = None,
        collection_name: str = "model_embeddings",
        embedding_model: str = None,
    ):
        """Initialize the Postgres vector store.

        Args:
            connection_string: SQLAlchemy connection string
            collection_name: Name of the collection/table to use
            embedding_model: Name of the embedding model to use
        """
        # Load config if connection_string is not provided
        if not connection_string:
            config = load_config()
            connection_string = config.get("postgres_connection_string")
            if not connection_string:
                raise ValueError(
                    "No connection_string provided and none found in config"
                )

        # Set embedding model
        self.embedding_model = embedding_model or os.environ.get(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )

        # Initialize SQLAlchemy engine and session
        self.engine = sa.create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)

        # Collection name
        self.collection_name = collection_name

        # Create tables
        self._create_tables()

        logger.info(
            f"Initialized Postgres vector store with model: {self.embedding_model}"
        )

    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        # Create pgvector extension if it doesn't exist
        with self.engine.connect() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

        # Create tables
        Base.metadata.create_all(self.engine)
        logger.info("Created vector store tables")

    def _get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a text.

        Args:
            text: The text to embed

        Returns:
            The embedding as a list of floats
        """
        # Import here to avoid circular imports
        from dbt_llm_agent.integrations.embedding import get_embedding

        return get_embedding(text, model=self.embedding_model)

    def store_model(
        self, model_name: str, model_text: str, metadata: Dict[str, Any] = None
    ) -> None:
        """Store a model in the vector store.

        Args:
            model_name: The name of the model
            model_text: The text content of the model
            metadata: Additional metadata for the model
        """
        session = self.Session()
        try:
            # Check if model already exists
            existing = (
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
                .first()
            )

            # Get embedding
            embedding = self._get_embedding(model_text)

            if existing:
                # Update existing model
                existing.document = model_text
                existing.embedding = embedding
                existing.model_metadata = metadata or {}
            else:
                # Create new model
                model_embedding = ModelEmbedding(
                    model_name=model_name,
                    document=model_text,
                    embedding=embedding,
                    model_metadata=metadata or {},
                )
                session.add(model_embedding)

            # Commit changes
            session.commit()
            logger.info(f"Stored model {model_name} in vector store")
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing model {model_name}: {e}")
            raise
        finally:
            session.close()

    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists in the vector store.

        Args:
            model_name: The name of the model

        Returns:
            Whether the model exists
        """
        session = self.Session()
        try:
            # Check if model exists
            existing = (
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
                .first()
            )
            return existing is not None
        except Exception as e:
            logger.error(f"Error checking if model {model_name} exists: {e}")
            return False
        finally:
            session.close()

    def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the metadata for a model.

        Args:
            model_name: The name of the model

        Returns:
            The model metadata, or None if not found
        """
        session = self.Session()
        try:
            # Get model
            model = (
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
                .first()
            )
            if not model:
                logger.debug(f"Model {model_name} not found")
                return None

            # Return metadata
            return model.model_metadata
        except Exception as e:
            logger.error(f"Error getting metadata for model {model_name}: {e}")
            return None
        finally:
            session.close()

    def search_models(
        self, query: str, n_results: int = 5, filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for models similar to a query.

        Args:
            query: The search query
            n_results: The number of results to return
            filter_metadata: Filter results by metadata

        Returns:
            A list of search results
        """
        session = self.Session()
        try:
            # Get embedding for query
            query_embedding = self._get_embedding(query)

            # Build query
            base_query = session.query(
                ModelEmbedding,
                ModelEmbedding.embedding.cosine_distance(query_embedding).label(
                    "distance"
                ),
            )

            # Apply filter if provided
            if filter_metadata:
                for key, value in filter_metadata.items():
                    base_query = base_query.filter(
                        ModelEmbedding.model_metadata[key].astext == str(value)
                    )

            # Order by distance and limit
            results = base_query.order_by(sa.text("distance")).limit(n_results).all()

            # Format results
            return [
                {
                    "model_name": result.ModelEmbedding.model_name,
                    "document": result.ModelEmbedding.document,
                    "metadata": result.ModelEmbedding.model_metadata,
                    "score": 1
                    - result.distance,  # Convert distance to similarity score
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []
        finally:
            session.close()

    def delete_model(self, model_name: str) -> None:
        """Delete a model from the vector store.

        Args:
            model_name: The name of the model to delete
        """
        session = self.Session()
        try:
            # Get model
            model = (
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
                .first()
            )
            if not model:
                logger.debug(f"Model {model_name} not found")
                return

            # Delete model
            session.delete(model)
            session.commit()
            logger.info(f"Deleted model {model_name} from vector store")
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting model {model_name}: {e}")
            raise
        finally:
            session.close()

    def store_models(
        self,
        models_dict: Dict[str, str],
        metadata_dict: Dict[str, Dict[str, Any]] = None,
    ) -> None:
        """Store multiple models in the vector store.

        Args:
            models_dict: Dictionary of model_name to model_text
            metadata_dict: Dictionary of model_name to metadata
        """
        metadata_dict = metadata_dict or {}
        for model_name, model_text in models_dict.items():
            self.store_model(model_name, model_text, metadata_dict.get(model_name, {}))
