"""PostgreSQL storage for dbt model information."""

import logging
import json
from typing import Dict, List, Optional, Any
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, Session

from dbt_llm_agent.core.models import DBTModel
from dbt_llm_agent.storage.models import (
    Base,
    ModelTable,
    ColumnTable,
    TestTable,
    DependencyTable,
)

logger = logging.getLogger(__name__)


class PostgresStorage:
    """PostgreSQL storage for dbt model information."""

    def __init__(self, connection_string: str, echo: bool = False):
        """Initialize PostgreSQL storage.

        Args:
            connection_string: SQLAlchemy connection string
            echo: Whether to echo SQL statements
        """
        self.engine = sa.create_engine(connection_string, echo=echo)
        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        self._create_tables()

        logger.info("Initialized PostgreSQL storage")

    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Created model storage tables")

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

            # Prepare model data
            model_data = {
                "name": model.name,
                "path": model.path,
                "description": model.description,
                "schema": model.schema,
                "database": model.database,
                "materialization": model.materialization,
                "tags": model.tags,
                "depends_on": model.depends_on,
                "tests": [vars(test) for test in model.tests],
                "all_upstream_models": model.all_upstream_models,
                "meta": model.meta,
                "raw_sql": model.raw_sql,
                "documentation": model.documentation,
                "columns": (
                    {
                        col.name: {
                            "name": col.name,
                            "description": col.description,
                            "data_type": col.data_type,
                            "meta": col.meta,
                        }
                        for col in model.columns.values()
                    }
                    if model.columns
                    else None
                ),
            }

            if existing:
                # Update existing model
                for key, value in model_data.items():
                    setattr(existing, key, value)
                model_id = existing.id
            else:
                # Create new model
                new_model = ModelTable(**model_data)
                session.add(new_model)
                session.flush()  # Flush to get the model ID
                model_id = new_model.id

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

            # Convert to DBTModel
            return self._convert_to_dbt_model(model_record)
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
            # Convert to DBTModel
            return [self._convert_to_dbt_model(record) for record in model_records]
        except Exception as e:
            logger.error(f"Error getting all models: {e}")
            return []
        finally:
            session.close()

    def _convert_to_dbt_model(self, model_record: ModelTable) -> DBTModel:
        """Convert a ModelTable record to a DBTModel.

        Args:
            model_record: The ModelTable record

        Returns:
            The converted DBTModel
        """
        from dbt_llm_agent.core.models import DBTModel, Column

        # Convert columns to DBTColumn
        columns = {}
        if model_record.columns:
            for col_name, col_data in model_record.columns.items():
                columns[col_name] = Column(
                    name=col_data["name"],
                    description=col_data.get("description"),
                    data_type=col_data.get("data_type"),
                    meta=col_data.get("meta", {}),
                )

        # Create DBTModel
        model = DBTModel(
            name=model_record.name,
            path=model_record.path,
            description=model_record.description,
            schema=model_record.schema,
            database=model_record.database,
            materialization=model_record.materialization,
            tags=model_record.tags or [],
            depends_on=model_record.depends_on or [],
            tests=model_record.tests or [],
            all_upstream_models=model_record.all_upstream_models or [],
            meta=model_record.meta or {},
            raw_sql=model_record.raw_sql,
            documentation=model_record.documentation,
            columns=columns,
        )

        return model

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
