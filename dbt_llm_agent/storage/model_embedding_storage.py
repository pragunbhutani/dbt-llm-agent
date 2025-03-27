"""Storage for model embeddings using pgvector in PostgreSQL."""

import os
import logging
from typing import Dict, List, Any, Optional
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector
import numpy as np

from dbt_llm_agent.utils.config import get_config_value
from dbt_llm_agent.core.models import Base, ModelEmbedding, ModelEmbeddingTable
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

    def _create_homogeneous_document(
        self,
        model_name: str,
        model=None,
        documentation_text=None,
        interpretation_text=None,
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

            embedding_text += (
                f"Depends on (via ref): {', '.join(deps) if deps else ''}\n"
            )
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
        if (
            model
            and hasattr(model, "interpreted_columns")
            and model.interpreted_columns
        ):
            interpreted_column_count = len(model.interpreted_columns)
            embedding_text += (
                f"\nInterpreted columns from LLM ({interpreted_column_count}):\n"
            )
            for col_name, col_desc in model.interpreted_columns.items():
                embedding_text += f"- {col_name}: {col_desc}\n"
        else:
            embedding_text += "\nInterpreted columns from LLM (0):\n"

        return embedding_text

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

    def store_model(
        self, model_name: str, model_text: str, metadata: Dict[str, Any] = None
    ) -> None:
        """Store a model in the vector store.

        Args:
            model_name: The name of the model
            model_text: The model text to embed
            metadata: Optional metadata to store with the model
        """
        session = self.Session()
        try:
            # Get embedding for the model
            embedding = self._get_embedding(model_text)

            # Check if model already exists
            existing_model = (
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
                .first()
            )

            # Create domain model
            domain_embedding = ModelEmbedding(
                model_name=model_name,
                document=model_text,
                embedding=embedding,
                model_metadata=metadata or {},
            )

            if existing_model:
                # Update existing model
                existing_model.document = model_text
                existing_model.embedding = embedding
                existing_model.model_metadata = metadata or {}
            else:
                # Convert to ORM model and add
                orm_model = ModelEmbedding.from_domain(domain_embedding)
                session.add(orm_model)

            # Commit
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
            exists = (
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
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
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
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

            # Build query
            db_query = session.query(
                ModelEmbedding,
                sa.func.cosine_distance(
                    ModelEmbedding.embedding, query_embedding
                ).label("distance"),
            )

            # Apply metadata filter if provided
            if filter_metadata:
                for key, value in filter_metadata.items():
                    db_query = db_query.filter(
                        ModelEmbedding.model_metadata[key].astext == str(value)
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
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
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
            self.store_model(model_name, model_text, metadata)

    def store_model_documentation(
        self, model_name: str, documentation_text: str, force: bool = False
    ) -> None:
        """Store a model's documentation in the vector store.

        This creates a structured document with all available information about the model.

        Args:
            model_name: The name of the model
            documentation_text: The documentation text
            force: Whether to force update if model already exists
        """
        session = self.Session()
        try:
            # Check if model already exists
            existing = (
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
                .first()
            )

            if existing and not force:
                # Get existing metadata
                domain_model = existing.to_domain()
                metadata = domain_model.model_metadata or {}

                # Create a new homogeneous document with existing data and new docs
                if "dbt_model" in metadata:
                    from dbt_llm_agent.core.models import DBTModel

                    # Create a new document with the updated documentation
                    if isinstance(metadata["dbt_model"], dict):
                        # Legacy format: metadata contains the model as a dict
                        document = self._create_homogeneous_document(
                            model_name, metadata["dbt_model"], documentation_text
                        )
                    else:
                        # Newer format: metadata contains the model object
                        document = self._create_homogeneous_document(
                            model_name, metadata["dbt_model"], documentation_text
                        )
                else:
                    # No model available, just create basic document
                    document = self._create_homogeneous_document(
                        model_name, None, documentation_text
                    )

                # Update embedding
                existing.document = document
                existing.embedding = self._get_embedding(document)

                # Update documentation in metadata
                if "documentation" not in metadata:
                    metadata["documentation"] = {}
                metadata["documentation"]["text"] = documentation_text
                existing.model_metadata = metadata

                session.commit()
                logger.info(f"Updated documentation for model {model_name}")
            else:
                # Create a basic document
                document = self._create_homogeneous_document(
                    model_name, None, documentation_text
                )

                # Create metadata with documentation
                metadata = {"documentation": {"text": documentation_text}}

                # Create new domain model
                domain_embedding = ModelEmbedding(
                    model_name=model_name,
                    document=document,
                    embedding=self._get_embedding(document),
                    model_metadata=metadata,
                )

                # Convert to ORM model and add to session
                new_model = ModelEmbedding.from_domain(domain_embedding)
                session.add(new_model)
                session.commit()
                logger.info(f"Stored documentation for model {model_name}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing documentation for model {model_name}: {e}")
        finally:
            session.close()

    def store_model_interpretation(
        self, model_name: str, interpretation_text: str, force: bool = False
    ) -> None:
        """Store a model's LLM interpretation in the vector store.

        This creates a structured document with all available information about the model.

        Args:
            model_name: The name of the model
            interpretation_text: The interpretation text
            force: Whether to force update if model already exists
        """
        session = self.Session()
        try:
            # Check if model already exists
            existing = (
                session.query(ModelEmbedding)
                .filter(ModelEmbedding.model_name == model_name)
                .first()
            )

            if existing:
                # Get domain model and metadata
                domain_model = existing.to_domain()
                metadata = domain_model.model_metadata or {}

                # Create a new homogeneous document with existing data and new interpretation
                if "dbt_model" in metadata:
                    # Create a new document with the updated interpretation
                    document = self._create_homogeneous_document(
                        model_name,
                        metadata["dbt_model"],
                        metadata.get("documentation", {}).get("text", ""),
                        interpretation_text,
                    )
                else:
                    # No model available, just create basic document
                    document = self._create_homogeneous_document(
                        model_name, None, None, interpretation_text
                    )

                # Update embedding
                existing.document = document
                existing.embedding = self._get_embedding(document)

                # Update interpretation in metadata
                if "interpretation" not in metadata:
                    metadata["interpretation"] = {}
                metadata["interpretation"]["text"] = interpretation_text
                existing.model_metadata = metadata

                session.commit()
                logger.info(f"Updated interpretation for model {model_name}")
            else:
                # Create a basic document
                document = self._create_homogeneous_document(
                    model_name, None, None, interpretation_text
                )

                # Create metadata with interpretation
                metadata = {"interpretation": {"text": interpretation_text}}

                # Create domain model
                domain_embedding = ModelEmbedding(
                    model_name=model_name,
                    document=document,
                    embedding=self._get_embedding(document),
                    model_metadata=metadata,
                )

                # Convert to ORM and store
                new_model = ModelEmbedding.from_domain(domain_embedding)
                session.add(new_model)
                session.commit()
                logger.info(f"Stored interpretation for model {model_name}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing interpretation for model {model_name}: {e}")
        finally:
            session.close()
