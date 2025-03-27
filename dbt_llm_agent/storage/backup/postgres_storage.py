"""Storage for dbt models in PostgreSQL."""

import logging
import json
from typing import Dict, List, Optional, Any
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, Session

from dbt_llm_agent.core.models import DBTModel, ModelTable, Base

logger = logging.getLogger(__name__)


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

    def store_model(self, model: DBTModel, force: bool = False) -> int:
        """Store a DBT model in the database.

        Args:
            model: The DBT model to store
            force: Whether to force update if model already exists

        Returns:
            The ID of the stored model
        """
        session = self.Session()
        try:
            # Check if the model already exists
            existing = (
                session.query(ModelTable).filter(ModelTable.name == model.name).first()
            )

            if existing and not force:
                logger.debug(f"Model {model.name} already exists, skipping")
                return existing.id

            if existing:
                # Update existing model with values from domain model
                orm_model = model.to_orm()
                orm_model.id = existing.id  # Make sure to preserve the ID

                # Copy all attributes from orm_model to existing
                for key, value in orm_model.__dict__.items():
                    if not key.startswith("_"):  # Skip SQLAlchemy internal attrs
                        setattr(existing, key, value)

                model_id = existing.id
            else:
                # Create new model from domain model
                orm_model = model.to_orm()
                session.add(orm_model)
                session.flush()  # Flush to get the model ID
                model_id = orm_model.id

            # Commit the changes
            session.commit()
            return model_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing model {model.name}: {e}")
            raise
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
