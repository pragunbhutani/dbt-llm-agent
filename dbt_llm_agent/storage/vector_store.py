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


def create_homogeneous_document(
    model_name: str, model=None, documentation_text=None, interpretation_text=None
):
    """Create a homogeneous document structure for model embeddings.

    This ensures all models have the same document structure regardless of
    what information is available.

    The document structure follows this format:
    ```
    Model: <model_name>
    Description (YML): <model description>
    Interpretation (LLM): Available/Not Available
    Path: <file path>
    Schema: <schema>
    Database: <database>
    Materialization: <materialization>
    Depends on (via ref): <dependencies list>

    Columns from YML documentation (<count>):
    - <column_name>: <column description>
    - <column_name>: <column description>
    ...

    Interpreted columns from LLM (<count>):
    - <column_name>: <column description>
    - <column_name>: <column description>
    ...
    ```

    Args:
        model_name: The name of the model
        model: Optional DBTModel instance with model information
        documentation_text: Optional documentation text
        interpretation_text: Optional interpretation text

    Returns:
        A consistently structured document for embeddings
    """
    # Create a structured document with a consistent format
    embedding_text = f"Model: {model_name}\n"

    # Add model metadata
    if model:
        # Use documentation_text if provided, otherwise use model description
        description = (
            documentation_text
            if documentation_text and documentation_text.strip()
            else model.description or ""
        )
        embedding_text += f"Description (YML): {description}\n"

        # Use interpretation_text if provided, otherwise check if model has interpretation
        if interpretation_text and interpretation_text.strip():
            embedding_text += "Interpretation (LLM): Available\n"
        elif model.interpreted_description:
            embedding_text += "Interpretation (LLM): Available\n"
        else:
            embedding_text += "Interpretation (LLM): Not Available\n"

        embedding_text += f"Path: {model.path or ''}\n"
        embedding_text += f"Schema: {model.schema or ''}\n"
        embedding_text += f"Database: {model.database or ''}\n"
        embedding_text += f"Materialization: {model.materialization or ''}\n"

        # Add dependencies
        # Handle different possible structures for depends_on
        deps = []
        if hasattr(model, "depends_on"):
            if isinstance(model.depends_on, list):
                deps = model.depends_on
            elif isinstance(model.depends_on, dict) and "ref" in model.depends_on:
                deps = model.depends_on["ref"]

        embedding_text += f"Depends on (via ref): {', '.join(deps) if deps else ''}\n"
    else:
        embedding_text += "Description (YML): \n"
        embedding_text += "Interpretation (LLM): Not Available\n"
        embedding_text += "Path: \n"
        embedding_text += "Schema: \n"
        embedding_text += "Database: \n"
        embedding_text += "Materialization: \n"
        embedding_text += "Depends on (via ref): \n"

    # Add YML columns section
    if model and hasattr(model, "columns") and model.columns:
        column_count = len(model.columns)
        embedding_text += f"\nColumns from YML documentation ({column_count}):\n"

        # Handle both dictionary of Column objects and dictionary of dictionaries
        for col_name, col_info in model.columns.items():
            if hasattr(col_info, "description"):
                # This is a Column object
                col_desc = col_info.description or ""
            else:
                # This is a dictionary
                col_desc = col_info.get("description", "")
            embedding_text += f"- {col_name}: {col_desc}\n"
    else:
        embedding_text += "\nColumns from YML documentation (0):\n"

    # Add interpreted columns section
    if model and hasattr(model, "interpreted_columns") and model.interpreted_columns:
        interpreted_column_count = len(model.interpreted_columns)
        embedding_text += (
            f"\nInterpreted columns from LLM ({interpreted_column_count}):\n"
        )
        for col_name, col_desc in model.interpreted_columns.items():
            embedding_text += f"- {col_name}: {col_desc}\n"
    else:
        embedding_text += "\nInterpreted columns from LLM (0):\n"

    return embedding_text


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
            # First try to get postgres_uri (new standardized name)
            connection_string = config.get("postgres_uri")
            # Fallback to legacy name for backward compatibility
            if not connection_string:
                connection_string = config.get("postgres_connection_string")
            if not connection_string:
                raise ValueError(
                    "No connection_string provided and no postgres_uri found in config"
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

            # Import to avoid circular imports
            from dbt_llm_agent.storage.postgres_storage import PostgresStorage

            # Get model to access its documentation and interpretation
            postgres_storage = PostgresStorage(
                self.engine.url.render_as_string(hide_password=False)
            )
            model = postgres_storage.get_model(model_name)

            # Use the helper function to create a homogeneous document
            embedding_text = create_homogeneous_document(model_name, model)

            # Get embedding for the structured document
            embedding = self._get_embedding(embedding_text)

            if existing:
                # Update existing model
                existing.document = embedding_text
                existing.embedding = embedding
                existing.model_metadata = metadata or {}
            else:
                # Create new model
                model_embedding = ModelEmbedding(
                    model_name=model_name,
                    document=embedding_text,
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
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Dict[str, Any] = None,
        use_interpretation: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search for models similar to a query.

        Args:
            query: The search query
            n_results: The number of results to return
            filter_metadata: Filter results by metadata
            use_interpretation: Ignored, kept for backward compatibility

        Returns:
            A list of search results
        """
        session = self.Session()
        try:
            # Get embedding for query
            query_embedding = self._get_embedding(query)

            # Use regular embedding only (interpretation_embedding has been removed)
            logger.info("Searching using model embeddings")
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
                    "used_interpretation": False,  # Always false now
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

    def store_model_documentation(
        self, model_name: str, documentation_text: str, force: bool = False
    ) -> None:
        """Store the documentation embedding for a model.

        Args:
            model_name: The name of the model
            documentation_text: The documentation text to embed
            force: Whether to force re-embedding even if it already exists
        """
        try:
            # Check if the model already exists
            session = self.Session()
            existing = (
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
                .first()
            )

            # Import to avoid circular imports
            from dbt_llm_agent.storage.postgres_storage import PostgresStorage

            # Check if model has interpretation
            postgres_storage = PostgresStorage(
                self.engine.url.render_as_string(hide_password=False)
            )
            model = postgres_storage.get_model(model_name)

            # Use the helper function to create a homogeneous document
            embedding_text = create_homogeneous_document(
                model_name, model, documentation_text=documentation_text
            )

            # Get embedding for the structured document
            documentation_embedding = self._get_embedding(embedding_text)

            if existing:
                # Update the existing model
                if existing.embedding is not None and not force:
                    logger.info(
                        f"Model {model_name} already has documentation embedding and force=False"
                    )
                    session.close()
                    return

                existing.embedding = documentation_embedding
                existing.document = embedding_text
                session.commit()
            else:
                # Create a new model embedding
                new_embedding = ModelEmbedding(
                    model_name=model_name,
                    document=embedding_text,
                    embedding=documentation_embedding,
                )
                session.add(new_embedding)
                session.commit()

            logger.info(f"Stored documentation embedding for model {model_name}")
            session.close()
        except Exception as e:
            logger.error(
                f"Error storing documentation embedding for model {model_name}: {e}"
            )
            raise

    def store_model_interpretation(
        self, model_name: str, interpretation_text: str, force: bool = False
    ) -> None:
        """Store the interpretation embedding for a model.

        Args:
            model_name: The name of the model
            interpretation_text: The interpretation text to embed
            force: Whether to force re-embedding even if it already exists
        """
        try:
            # Check if the model already exists
            session = self.Session()
            existing = (
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
                .first()
            )

            # Import to avoid circular imports
            from dbt_llm_agent.storage.postgres_storage import PostgresStorage

            # Get model to access its documentation and columns
            postgres_storage = PostgresStorage(
                self.engine.url.render_as_string(hide_password=False)
            )
            model = postgres_storage.get_model(model_name)

            # Use the helper function to create a homogeneous document
            embedding_text = create_homogeneous_document(
                model_name, model, interpretation_text=interpretation_text
            )

            # Get embedding for the structured document
            interpretation_embedding = self._get_embedding(embedding_text)

            if existing:
                # Update the existing model
                if existing.embedding is not None and not force:
                    logger.info(
                        f"Model {model_name} already has embedding and force=False"
                    )
                    session.close()
                    return

                existing.embedding = interpretation_embedding
                existing.document = embedding_text
                session.commit()
            else:
                # Create a new model embedding
                new_embedding = ModelEmbedding(
                    model_name=model_name,
                    document=embedding_text,
                    embedding=interpretation_embedding,
                )
                session.add(new_embedding)
                session.commit()

            logger.info(f"Stored interpretation embedding for model {model_name}")
            session.close()
        except Exception as e:
            logger.error(
                f"Error storing interpretation embedding for model {model_name}: {e}"
            )
            raise
