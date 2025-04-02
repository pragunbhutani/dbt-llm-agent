"""Storage for model embeddings using pgvector in PostgreSQL."""

import os
import logging
from typing import Dict, List, Any, Optional
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy import cast
from pgvector.sqlalchemy import Vector
import numpy as np

from dbt_llm_agent.utils.config import get_config_value
from dbt_llm_agent.core.models import (
    Base,
    ModelEmbedding,
    ModelEmbeddingTable,
    DBTModel,
    ModelTable,
)
from dbt_llm_agent.integrations.llm.client import LLMClient

logger = logging.getLogger(__name__)


class ModelEmbeddingStorage:
    """Storage service for model embeddings using PostgreSQL and pgvector."""

    def __init__(
        self,
        connection_string: str = None,
        collection_name: str = "model_embeddings",
        embedding_model: str = None,
    ):
        """Initialize the model embedding storage service.

        Args:
            connection_string: SQLAlchemy connection string
            collection_name: Name of the collection/table to use
            embedding_model: Name of the embedding model to use
        """
        # Load config if connection_string is not provided
        if not connection_string:
            connection_string = get_config_value("postgres_uri")
            if not connection_string:
                raise ValueError("No postgres_uri found in config")

        # Set embedding model
        self.embedding_model = embedding_model or get_config_value(
            "openai_embedding_model", "text-embedding-3-small"
        )

        # Initialize SQLAlchemy engine and session
        self.engine = sa.create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)

        # Collection name
        self.collection_name = collection_name

        # Create tables
        self._create_tables()

        # Initialize LLM client for embeddings
        api_key = get_config_value("openai_api_key")
        self.llm_client = LLMClient(
            api_key=api_key, embedding_model=self.embedding_model
        )

        logger.info(
            f"Initialized model embedding storage with model: {self.embedding_model}"
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
        return self.llm_client.get_embedding(text, model=self.embedding_model)

    def store_model_embedding(
        self,
        model_name: str,
        model_text: Optional[str] = None,
        documentation_text: Optional[str] = None,
        interpretation_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        """Store a comprehensive model embedding in the vector store.

        This method creates a single embedding document that includes all available
        information about the model (YML description, LLM interpretation, documentation).

        Args:
            model_name: The name of the model
            model_text: Optional pre-formatted model text to embed
            documentation_text: Optional documentation text
            interpretation_text: Optional interpretation text
            metadata: Optional metadata to store with the model
            force: Whether to force update if model already exists
        """
        session = self.Session()
        try:
            # Check if model already exists in vector store
            existing = (
                session.query(ModelEmbeddingTable)
                .filter(ModelEmbeddingTable.model_name == model_name)
                .first()
            )

            if (
                not force
                and existing
                and not (model_text or documentation_text or interpretation_text)
            ):
                # If no new data provided and not forcing update, just return
                logger.info(f"No changes to model {model_name}, skipping update")
                return

            # Get existing metadata if available
            existing_metadata = {}
            if existing:
                domain_model = existing.to_domain()
                existing_metadata = domain_model.model_metadata or {}

            # Try to fetch the model from DBT model storage to get all descriptions
            model_record = (
                session.query(ModelTable).filter(ModelTable.name == model_name).first()
            )
            yml_description = model_record.yml_description if model_record else None
            stored_interpreted_description = (
                model_record.interpreted_description if model_record else None
            )

            # Use provided interpretation text or fall back to stored
            effective_interpretation = (
                interpretation_text or stored_interpreted_description
            )

            # Create a combined metadata object
            combined_metadata = existing_metadata.copy()
            if metadata:
                combined_metadata.update(metadata)

            # Ensure dbt_model section exists
            if "dbt_model" not in combined_metadata:
                combined_metadata["dbt_model"] = {}

            # Update metadata with descriptions
            if yml_description:
                combined_metadata["dbt_model"]["yml_description"] = yml_description
            if effective_interpretation:
                combined_metadata["dbt_model"][
                    "interpreted_description"
                ] = effective_interpretation
            if documentation_text:
                if "documentation" not in combined_metadata:
                    combined_metadata["documentation"] = {}
                combined_metadata["documentation"]["text"] = documentation_text
            if interpretation_text:
                if "interpretation" not in combined_metadata:
                    combined_metadata["interpretation"] = {}
                combined_metadata["interpretation"]["text"] = interpretation_text

            # Create embedding document
            if model_text and not (
                "YML Description:" in model_text
                and "Interpreted Description:" in model_text
            ):
                # If model_text is provided but doesn't have descriptions, we should regenerate
                logger.info(
                    f"Regenerating model text for {model_name} to include descriptions"
                )
                model_text = None

            if not model_text:
                # Generate model text from a dummy model
                dummy_model = DBTModel(
                    name=model_name,
                    description=yml_description,
                    interpreted_description=effective_interpretation,
                )
                model_text = dummy_model.get_text_representation(
                    include_documentation=True,
                    additional_documentation=documentation_text,
                )

            # Generate embedding
            try:
                embedding = self._get_embedding(model_text)
            except ValueError as e:
                logger.error(
                    f"Failed to generate embedding for model {model_name}: {e}"
                )
                raise ValueError(
                    f"Cannot store model {model_name} - embedding generation failed"
                ) from e

            # Create or update the model in the vector store
            if existing:
                # Update existing model
                existing.document = model_text
                existing.embedding = embedding
                existing.model_metadata = combined_metadata
            else:
                # Create new domain model
                domain_embedding = ModelEmbedding(
                    model_name=model_name,
                    document=model_text,
                    embedding=embedding,
                    model_metadata=combined_metadata,
                )

                # Convert to ORM and add to session
                orm_model = domain_embedding.to_orm()
                session.add(orm_model)

            # Commit changes
            session.commit()
            logger.info(f"Stored comprehensive embedding for model {model_name}")

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing embedding for model {model_name}: {e}")
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
            exists = (
                session.query(ModelEmbeddingTable)
                .filter(ModelEmbeddingTable.model_name == model_name)
                .first()
                is not None
            )
            return exists
        except Exception as e:
            logger.error(f"Error checking if model {model_name} exists: {e}")
            return False
        finally:
            session.close()

    def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a model.

        Args:
            model_name: The name of the model

        Returns:
            The model metadata, or None if the model doesn't exist
        """
        session = self.Session()
        try:
            # Get model
            model = (
                session.query(ModelEmbeddingTable)
                .filter(ModelEmbeddingTable.model_name == model_name)
                .first()
            )
            if not model:
                return None

            # Convert to domain model and return metadata
            domain_model = model.to_domain()
            return domain_model.model_metadata
        except Exception as e:
            logger.error(f"Error getting metadata for model {model_name}: {e}")
            return None
        finally:
            session.close()

    def search_models(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Dict[str, Any] = None,
        use_interpretation: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search for models similar to a query.

        Args:
            query: The query to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            use_interpretation: Whether to use interpretation in the response

        Returns:
            A list of model information dictionaries
        """
        session = self.Session()
        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(query)

            # Check if embedding generation failed
            if not query_embedding:
                logger.error(f"Could not generate embedding for query: {query}")
                return []  # Return empty list if embedding fails

            # Get embedding dimension
            embedding_dim = len(query_embedding)

            # Build query with explicit cast
            db_query = session.query(
                ModelEmbeddingTable,
                sa.func.cosine_distance(
                    ModelEmbeddingTable.embedding,
                    cast(
                        query_embedding, Vector(embedding_dim)
                    ),  # Cast to Vector with dimension
                ).label("distance"),
            )

            # Apply metadata filter if provided
            if filter_metadata:
                for key, value in filter_metadata.items():
                    db_query = db_query.filter(
                        ModelEmbeddingTable.model_metadata[key].astext == str(value)
                    )

            # Get results
            results = db_query.order_by(sa.text("distance")).limit(n_results).all()

            # Format results
            formatted_results = []
            for model, distance in results:
                # Convert to domain model
                domain_model = model.to_domain()

                result = {
                    "model_name": domain_model.model_name,
                    "document": domain_model.document if use_interpretation else None,
                    "metadata": domain_model.model_metadata,
                    "distance": float(distance),
                    "similarity_score": 1 - float(distance),
                }
                formatted_results.append(result)

            return formatted_results
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
                session.query(ModelEmbeddingTable)
                .filter(ModelEmbeddingTable.model_name == model_name)
                .first()
            )
            if not model:
                logger.warning(f"Model {model_name} not found for deletion")
                return

            # Delete model
            session.delete(model)
            session.commit()
            logger.info(f"Deleted model {model_name} from vector store")
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting model {model_name}: {e}")
        finally:
            session.close()

    def store_models(
        self,
        models_dict: Dict[str, str],
        metadata_dict: Dict[str, Dict[str, Any]] = None,
    ) -> None:
        """Store multiple models in the vector store.

        Args:
            models_dict: Dictionary mapping model names to model text
            metadata_dict: Optional dictionary mapping model names to metadata
        """
        for model_name, model_text in models_dict.items():
            metadata = metadata_dict.get(model_name) if metadata_dict else None
            self.store_model_embedding(
                model_name=model_name, model_text=model_text, metadata=metadata
            )
