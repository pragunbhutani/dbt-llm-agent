"""Storage for dbt models in PostgreSQL."""

import logging
import json
from typing import Dict, List, Optional, Any
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, Session

from ragstar_archive.core.models import DBTModel, ModelTable, Base
from ragstar_archive.core.models import ModelMetadata
from ragstar_archive.core.models import QuestionTable, ModelEmbeddingTable

logger = logging.getLogger(__name__)


# Helper function to convert Pydantic column list to JSON dict expected by ModelTable
def _convert_metadata_columns_to_json(
    columns: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not columns:
        return None
    # The input list contains dicts like {"name": "col_a", "description": "...", ...}
    # The output should be a dict like {"col_a": {"name": "col_a", "description": "...", ...}}
    return {
        col_data["name"]: col_data
        for col_data in columns
        if "name" in col_data  # Ensure name key exists
    }


class ModelStorage:
    """Storage service for dbt models using PostgreSQL."""

    def __init__(self, connection_string: str, echo: bool = False):
        """Initialize the model storage service.

        Args:
            connection_string: SQLAlchemy connection string
            echo: Whether to echo SQL statements
        """
        self.engine = sa.create_engine(connection_string, echo=echo)
        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        self._create_tables()

        # Note: Migrations are no longer automatically applied in constructor
        # Call apply_migrations() explicitly when needed

        logger.info("Initialized model storage service")

    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Created model storage tables")

    def apply_migrations(self):
        """Apply database migrations using alembic.

        Returns:
            True if migrations were successful, False otherwise
        """
        try:
            from alembic import command
            from alembic.config import Config

            alembic_cfg = Config("alembic.ini")
            command.upgrade(alembic_cfg, "head")
            logger.info("Database migrations applied successfully")
            return True
        except Exception as e:
            logger.error(f"Error applying migrations: {e}")
            return False

    def store_model(self, model_metadata: ModelMetadata, force: bool = False):
        """Store a single model's metadata in the database.

        Args:
            model_metadata: A Pydantic ModelMetadata object containing model info.
            force: If True, overwrite existing model data.
        """
        session = self.Session()
        try:
            # Convert columns list to dict for storage
            columns_json_dict = _convert_metadata_columns_to_json(
                model_metadata.columns
            )

            # Check if model already exists by unique_id or name
            existing_model = None
            if model_metadata.unique_id:
                existing_model = (
                    session.query(ModelTable)
                    .filter(ModelTable.unique_id == model_metadata.unique_id)
                    .first()
                )
            elif model_metadata.name:
                existing_model = (
                    session.query(ModelTable)
                    .filter(ModelTable.name == model_metadata.name)
                    .first()
                )

            if existing_model:
                if force:
                    logger.info(
                        f"Model '{model_metadata.name}' (ID: {model_metadata.unique_id or 'N/A'}) already exists. Overwriting due to --force flag."
                    )
                    # --- Update existing model fields ---
                    existing_model.name = (
                        model_metadata.name
                    )  # Ensure name is updated if needed (though usually PK)
                    existing_model.yml_description = (
                        model_metadata.description
                    )  # Map to yml_description
                    existing_model.schema = (
                        model_metadata.schema_name
                    )  # Map schema_name to schema
                    existing_model.database = model_metadata.database
                    existing_model.materialization = model_metadata.materialization
                    existing_model.tags = model_metadata.tags
                    existing_model.yml_columns = columns_json_dict  # Use converted dict
                    existing_model.tests = model_metadata.tests  # Map tests to tests
                    existing_model.depends_on = model_metadata.depends_on
                    existing_model.path = model_metadata.path
                    existing_model.unique_id = (
                        model_metadata.unique_id
                    )  # Ensure unique_id is updated if needed
                    existing_model.raw_sql = model_metadata.raw_sql
                    existing_model.compiled_sql = model_metadata.compiled_sql
                    existing_model.all_upstream_models = (
                        model_metadata.all_upstream_models
                    )
                    # Optionally clear interpretation fields when overwriting?
                    # existing_model.interpreted_description = None
                    # existing_model.interpretation_details = None
                    session.add(existing_model)
                else:
                    logger.info(
                        f"Model '{model_metadata.name}' (ID: {model_metadata.unique_id or 'N/A'}) already exists. Skipping."
                    )
                    return
            else:
                # --- Create new model record ---
                logger.info(
                    f"Storing new model: '{model_metadata.name}' (ID: {model_metadata.unique_id or 'N/A'})"
                )
                new_model = ModelTable(
                    name=model_metadata.name,
                    yml_description=model_metadata.description,  # Map to yml_description
                    schema=model_metadata.schema_name,  # Map schema_name to schema
                    database=model_metadata.database,
                    materialization=model_metadata.materialization,
                    tags=model_metadata.tags,
                    yml_columns=columns_json_dict,  # Use converted dict
                    tests=model_metadata.tests,  # Map tests to tests
                    depends_on=model_metadata.depends_on,
                    path=model_metadata.path,
                    unique_id=model_metadata.unique_id,
                    raw_sql=model_metadata.raw_sql,
                    compiled_sql=model_metadata.compiled_sql,
                    all_upstream_models=model_metadata.all_upstream_models,
                    # Initialize interpretation fields as None
                    interpreted_description=None,
                    interpretation_details=None,
                    # yml_description and yml_columns are set above
                )
                session.add(new_model)

            session.commit()

        except Exception as e:
            logger.error(
                f"Error storing model {model_metadata.name}: {e}", exc_info=True
            )
            session.rollback()
        finally:
            session.close()

    def get_model(self, model_name: str) -> Optional[DBTModel]:
        """Get a DBT model from the database.

        Args:
            model_name: The name of the model to retrieve

        Returns:
            The DBT model, or None if not found
        """
        session = self.Session()
        try:
            # Get the model
            model_record = (
                session.query(ModelTable).filter(ModelTable.name == model_name).first()
            )
            if not model_record:
                logger.debug(f"Model {model_name} not found")
                return None

            # Convert to DBTModel using the to_domain method
            return model_record.to_domain()
        except Exception as e:
            logger.error(f"Error getting model {model_name}: {e}")
            return None
        finally:
            session.close()

    def get_all_models(self) -> List[DBTModel]:
        """Get all DBT models from the database.

        Returns:
            A list of all DBT models
        """
        session = self.Session()
        try:
            # Get all models
            model_records = session.query(ModelTable).all()
            # Convert to DBTModel using the to_domain method
            return [record.to_domain() for record in model_records]
        except Exception as e:
            logger.error(f"Error getting all models: {e}")
            return []
        finally:
            session.close()

    def update_model(self, model: DBTModel) -> bool:
        """Update an existing DBT model in the database.

        Args:
            model: The DBT model to update

        Returns:
            Whether the update was successful
        """
        # Just call store_model with force=True
        try:
            self.store_model(model, force=True)
            return True
        except Exception:
            return False

    def delete_model(self, model_name: str) -> bool:
        """Delete a DBT model from the database.

        Args:
            model_name: The name of the model to delete

        Returns:
            Whether the deletion was successful
        """
        session = self.Session()
        try:
            # Get the model
            model_record = (
                session.query(ModelTable).filter(ModelTable.name == model_name).first()
            )
            if not model_record:
                logger.debug(f"Model {model_name} not found")
                return False

            # Delete the model
            session.delete(model_record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
        finally:
            session.close()

    def execute_query(self, query: str) -> Optional[List[Dict]]:
        """Execute a raw SQL query on the database.

        Args:
            query: The SQL query to execute

        Returns:
            The query results as a list of dictionaries, or None on error
        """
        try:
            result = self.engine.execute(query)
            # Convert result to list of dicts
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None
