"""Data models for representing dbt projects and models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    JSON,
    Boolean,
    Table,
    TypeDecorator,
    ARRAY,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import registry
from sqlalchemy.sql import func
from datetime import datetime
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from pydantic import BaseModel, Field

# NOTE: This module serves as the single source of truth for all model definitions
# in the system. It provides bidirectional conversion between domain models and
# ORM models through the to_orm() and to_domain() methods.
#
# Future improvements could include:
# 1. Using SQLAlchemy's full ORM mapping capabilities through mapper_registry
# 2. Further refinement of the conversion methods

# Create a registry for mapping SQLAlchemy models to dataclasses
mapper_registry = registry()
Base = declarative_base()


# Define SQLAlchemy model classes (storage layer)
class ModelTable(Base):
    """SQLAlchemy model for the models table."""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    path = Column(String, nullable=False)
    schema = Column(String, nullable=True)
    database = Column(String, nullable=True)
    materialization = Column(String, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    depends_on = Column(ARRAY(String), nullable=True)
    tests = Column(JSONB, nullable=True)
    all_upstream_models = Column(ARRAY(String), nullable=True)
    meta = Column(JSON, nullable=True)
    raw_sql = Column(Text, nullable=True)
    compiled_sql = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    yml_description = Column(
        Text, nullable=True, comment="Description from YML documentation"
    )
    yml_columns = Column(JSON, nullable=True, comment="Columns from YML documentation")
    interpreted_columns = Column(
        JSON, nullable=True, comment="LLM-interpreted column descriptions"
    )
    interpreted_description = Column(
        Text, nullable=True, comment="LLM-generated description of the model"
    )
    interpretation_details = Column(JSONB, nullable=True)
    unique_id = Column(String, unique=True)
    # Removing compiled_sql, documentation, and unique_id as they're not in the DDL

    def to_domain(self):
        """Convert ORM model to domain model"""

        # Helper function to convert column JSON to domain objects
        def convert_columns(cols_json):
            if not cols_json:
                return {}
            return {
                name: Column(
                    name=data["name"],
                    description=data.get("description", ""),
                    data_type=data.get("data_type", ""),
                    meta=data.get("meta", {}),
                )
                for name, data in cols_json.items()
            }

        # Helper function to convert tests JSON to domain objects
        def convert_tests(tests_json):
            if not tests_json:
                return []
            return [
                Test(
                    name=test.get("name", ""),
                    column_name=test.get("column_name", ""),
                    test_type=test.get("test_type", ""),
                    unique_id=test.get("unique_id", ""),
                    meta=test.get("meta", {}),
                )
                for test in tests_json
            ]

        return DBTModel(
            id=self.id,
            name=self.name,
            path=self.path,
            description=self.yml_description or "",
            schema=self.schema or "",
            database=self.database or "",
            materialization=self.materialization or "",
            tags=self.tags or [],
            depends_on=self.depends_on or [],
            all_upstream_models=self.all_upstream_models or [],
            meta=self.meta or {},
            raw_sql=self.raw_sql or "",
            compiled_sql=self.compiled_sql or "",
            interpreted_description=self.interpreted_description or "",
            interpreted_columns=self.interpreted_columns or {},
            columns=convert_columns(self.yml_columns),
            tests=convert_tests(self.tests),
            unique_id=self.unique_id or "",
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_domain(cls, model):
        """Create an ORM model from a domain model"""

        # Helper function to convert column objects to JSON
        def convert_columns_to_json(columns):
            if not columns:
                return None
            return {
                name: {
                    "name": col.name,
                    "description": col.description,
                    "data_type": col.data_type,
                    "meta": col.meta,
                }
                for name, col in columns.items()
            }

        # Helper function to convert test objects to JSON
        def convert_tests_to_json(tests):
            if not tests:
                return None
            return [
                {
                    "name": test.name,
                    "column_name": test.column_name,
                    "test_type": test.test_type,
                    "unique_id": test.unique_id,
                    "meta": test.meta,
                }
                for test in tests
            ]

        return cls(
            id=model.id,
            name=model.name,
            path=model.path,
            yml_description=model.description,
            schema=model.schema,
            database=model.database,
            materialization=model.materialization,
            tags=model.tags,
            depends_on=model.depends_on,
            all_upstream_models=model.all_upstream_models,
            meta=model.meta,
            raw_sql=model.raw_sql,
            compiled_sql=model.compiled_sql,
            interpreted_description=model.interpreted_description,
            interpreted_columns=model.interpreted_columns,
            yml_columns=convert_columns_to_json(model.columns),
            tests=convert_tests_to_json(model.tests),
            created_at=model.created_at,
            updated_at=model.updated_at,
            interpretation_details=model.interpretation_details,
            unique_id=model.unique_id,
        )


# Question model relationships
# Removing unused association table definition
# question_models_assoc = Table(
#     "question_models_assoc",
#     Base.metadata,
#     Column("question_id", Integer, ForeignKey("questions.id"), primary_key=True),
#     Column("model_name", String, primary_key=True),
#     Column("relevance_score", Integer, nullable=True),
#     Column("created_at", DateTime(timezone=True), server_default=func.now()),
# )


class QuestionTable(Base):
    """SQLAlchemy model for storing questions and answers."""

    __tablename__ = "questions"

    id = Column(Integer, primary_key=True)
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=True)
    question_embedding = Column(
        Vector(1536),
        nullable=True,
        comment="Embedding vector for the question text using text-embedding-ada-002",
    )
    was_useful = Column(Boolean, nullable=True)
    feedback = Column(Text, nullable=True)
    question_metadata = Column(JSONB, nullable=True, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Define relationships - this would be customized based on your schema
    model_assocs = relationship("QuestionModelTable", back_populates="question")

    def to_domain(self):
        """Convert ORM model to domain model"""
        models_data = []
        if self.model_assocs:
            models_data = [
                {
                    "model_name": assoc.model_name,
                    "relevance_score": assoc.relevance_score,
                }
                for assoc in self.model_assocs
            ]

        return Question(
            id=self.id,
            question_text=self.question_text,
            answer_text=self.answer_text,
            question_embedding=self.question_embedding,
            was_useful=self.was_useful,
            feedback=self.feedback,
            question_metadata=self.question_metadata or {},
            created_at=self.created_at,
            updated_at=self.updated_at,
            models=models_data,
        )


class QuestionModelTable(Base):
    """SQLAlchemy model for tracking which models were used for each question."""

    __tablename__ = "question_models"

    question_id = Column(Integer, ForeignKey("questions.id"), primary_key=True)
    model_name = Column(String, primary_key=True)
    relevance_score = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Define relationships
    question = relationship("QuestionTable", back_populates="model_assocs")


class ModelEmbeddingTable(Base):
    """SQLAlchemy model for storing model embeddings."""

    __tablename__ = "model_embeddings"

    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    document = Column(Text, nullable=False)
    embedding = Column(
        Vector(1536), nullable=False, comment="Embedding based on model documentation"
    )
    model_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def to_domain(self):
        """Convert ORM model to domain model"""
        return ModelEmbedding(
            id=self.id,
            model_name=self.model_name,
            document=self.document,
            embedding=self.embedding,
            model_metadata=self.model_metadata or {},
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_domain(cls, embedding):
        """Create an ORM model from a domain model"""
        return cls(
            id=embedding.id,
            model_name=embedding.model_name,
            document=embedding.document,
            embedding=embedding.embedding,
            model_metadata=embedding.model_metadata,
        )


class ColumnTable(Base):
    """SQLAlchemy model for columns in dbt models."""

    __tablename__ = "columns"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    data_type = Column(String, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (sa.UniqueConstraint("model_id", "name", name="uix_model_column"),)


class DependencyTable(Base):
    """SQLAlchemy model for dependencies between dbt models."""

    __tablename__ = "dependencies"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    depends_on_name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        sa.UniqueConstraint("model_id", "depends_on_name", name="uix_model_dependency"),
    )


class TestTable(Base):
    """SQLAlchemy model for tests in dbt models."""

    __tablename__ = "tests"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    name = Column(String, nullable=False)
    column_name = Column(String, nullable=True)
    test_type = Column(String, nullable=True)
    unique_id = Column(String, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        sa.UniqueConstraint("model_id", "name", "column_name", name="uix_model_test"),
    )


# Define domain model classes
@dataclass
class Column:
    """Representation of a column in a dbt model."""

    name: str
    description: str = ""
    data_type: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Test:
    """Representation of a test in a dbt model."""

    name: str
    column_name: str = ""
    test_type: str = ""
    unique_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DBTModel:
    """Representation of a dbt model."""

    name: str
    description: str = ""  # From YML
    columns: Dict[str, Column] = field(default_factory=dict)  # From YML
    tests: List[Test] = field(default_factory=list)
    schema: str = ""
    database: str = ""
    materialization: str = ""
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    raw_sql: Optional[str] = None
    compiled_sql: Optional[str] = None
    # List of model names that this model depends on (names from ref() calls)
    depends_on: List[str] = field(default_factory=list)
    # List of all models upstream in the dependency chain
    all_upstream_models: List[str] = field(default_factory=list)
    path: str = ""
    unique_id: str = ""
    documentation: bool = False
    interpreted_description: str = ""  # LLM-generated description
    interpreted_columns: Dict[str, str] = field(
        default_factory=dict
    )  # LLM-interpreted column descriptions
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    interpretation_details: Dict[str, Any] = field(default_factory=dict)

    def get_text_representation(self, include_documentation: bool = False) -> str:
        """Get a text representation of the model for embedding.

        Args:
            include_documentation: Whether to include YML description and columns.
        """
        representation = f"Model: {self.name}\n\n"
        representation += f"Path: {self.path}\n\n"

        if self.depends_on:
            representation += f"Depends on: {', '.join(self.depends_on)}\n\n"
        else:
            representation += "Depends on: None\n\n"

        if include_documentation:
            representation += f"YML Description: {self.description or 'N/A'}\n\n"

        representation += f"Interpreted Description: {self.interpreted_description or 'N/A'}\n\n"

        if include_documentation:
            representation += "YML Columns:\n"
            if self.columns:
                for col_name, col in self.columns.items():
                    representation += f"  - {col_name}: {col.description or 'N/A'}\n"
                representation += "\n"  # Add newline after YML columns block
            else:
                representation += "  No YML column information available.\n\n"

        representation += "Interpreted Columns:\n"
        if self.interpreted_columns:
            for col_name, col_desc in self.interpreted_columns.items():
                representation += f"  - {col_name}: {col_desc or 'N/A'}\n"
            representation += "\n"  # Add newline after interpreted columns block
        else:
            representation += "  No interpreted column information available.\n\n"

        representation += f"Raw SQL:\n```sql\n{self.raw_sql or 'SQL not available'}\n```\n"
        # No extra newline needed after the final block

        return representation

    # ADDED: Simple summary representation for tool output
    def get_summary_representation(self) -> str:
        """Get a concise summary representation for tool output."""
        summary = ""
        # Use interpreted if available, fallback to YML
        description = self.interpreted_description or self.description
        if description:
            summary += f"  Description: {description}\n"
        else:
            summary += "  Description: N/A\n"

        # Show column names (interpreted or YML)
        column_names = []
        if self.interpreted_columns:
            column_names = list(self.interpreted_columns.keys())
        elif self.columns:
            column_names = list(self.columns.keys())

        if column_names:
            summary += f"  Columns: {", ".join(column_names)}\n"
        else:
            summary += "  Columns: N/A\n"

        return summary.strip()

    def to_dict(self, include_tests: bool = True) -> Dict[str, Any]:
        """Convert the model to a dictionary.

        Args:
            include_tests: Whether to include tests in the output

        Returns:
            Dict representation of the model
        """
        result = {
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
            "database": self.database,
            "materialization": self.materialization,
            "tags": self.tags,
            "meta": self.meta,
            "raw_sql": self.raw_sql,
            "compiled_sql": self.compiled_sql,
            "depends_on": self.depends_on,
            "all_upstream_models": self.all_upstream_models,
            "path": self.path,
            "unique_id": self.unique_id,
            "interpreted_description": self.interpreted_description,
            "interpreted_columns": self.interpreted_columns,
            "interpretation_details": self.interpretation_details,
            "columns": {name: vars(col) for name, col in self.columns.items()},
        }

        if include_tests:
            result["tests"] = [vars(test) for test in self.tests]

        return result

    def get_column_descriptions(self) -> Dict[str, str]:
        """Get a dictionary of column names and their descriptions.

        Returns:
            Dict mapping column names to descriptions
        """
        return {name: col.description for name, col in self.columns.items()}

    def debug_info(self) -> dict:
        """Get debug information about the model.

        Returns:
            A dictionary with debug information
        """
        info = {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "schema": self.schema,
            "database": self.database,
            "materialization": self.materialization,
            "tags": self.tags,
            "depends_on": self.depends_on,
            "depends_on_type": type(self.depends_on).__name__,
            "depends_on_length": len(self.depends_on) if self.depends_on else 0,
        }

        if hasattr(self, "all_upstream_models"):
            info["all_upstream_models"] = self.all_upstream_models
            info["all_upstream_models_type"] = type(self.all_upstream_models).__name__
            info["all_upstream_models_length"] = (
                len(self.all_upstream_models) if self.all_upstream_models else 0
            )

        return info

    def format_as_yaml(self) -> str:
        """Format the model as a dbt YAML document.

        Returns:
            A string containing the YAML document
        """
        # Start the YAML document
        yaml_lines = ["version: 2", "", "models:", f"  - name: {self.name}"]

        # Add description
        description = (
            self.interpreted_description
            if self.interpreted_description
            else self.description
        )
        if description:
            # Indent multiline descriptions correctly
            formatted_desc = description.replace("\n", "\n      ")
            yaml_lines.append(f"    description: >\n      {formatted_desc}")

        # Add model metadata
        if self.schema:
            yaml_lines.append(f"    schema: {self.schema}")

        if self.database:
            yaml_lines.append(f"    database: {self.database}")

        if self.materialization:
            yaml_lines.append(f"    config:")
            yaml_lines.append(f"      materialized: {self.materialization}")

        if self.tags:
            yaml_lines.append(f"    tags: [{', '.join(self.tags)}]")

        # Add column specifications
        if self.columns or self.interpreted_columns:
            yaml_lines.append("    columns:")

            # First try to use YML columns with their full specs
            if self.columns:
                for col_name, col in self.columns.items():
                    yaml_lines.append(f"      - name: {col_name}")
                    if col.description:
                        # Indent multiline descriptions correctly
                        col_desc = col.description.replace("\n", "\n          ")
                        yaml_lines.append(
                            f"        description: >\n          {col_desc}"
                        )
                    if col.data_type:
                        yaml_lines.append(f"        data_type: {col.data_type}")
            # Fall back to interpreted columns if available
            elif self.interpreted_columns:
                for col_name, description in self.interpreted_columns.items():
                    yaml_lines.append(f"      - name: {col_name}")
                    # Indent multiline descriptions correctly
                    col_desc = description.replace("\n", "\n          ")
                    yaml_lines.append(f"        description: >\n          {col_desc}")

        # Add tests if available
        if self.tests:
            test_lines = []
            for test in self.tests:
                if test.column_name:
                    # This is a column-level test, which should go under the column definition
                    continue
                else:
                    # This is a model-level test
                    test_name = test.test_type or test.name
                    if test_name:
                        test_lines.append(f"      - {test_name}")

            if test_lines:
                yaml_lines.append("    tests:")
                yaml_lines.extend(test_lines)

        return "\n".join(yaml_lines)

    def to_orm(self):
        """Convert domain model to ORM model"""
        return ModelTable.from_domain(self)


@dataclass
class DBTProject:
    """Representation of a dbt project."""

    name: str
    models: Dict[str, DBTModel]
    config: Dict[str, Any] = field(default_factory=dict)

    def get_model(self, model_name: str) -> Optional[DBTModel]:
        """Get a model by name.

        Args:
            model_name: Name of the model

        Returns:
            The model if found, None otherwise
        """
        return self.models.get(model_name)

    def get_model_names(self) -> List[str]:
        """Get a list of all model names.

        Returns:
            List of model names
        """
        return list(self.models.keys())


@dataclass
class Question:
    """Domain model for a question and its associated data."""

    question_text: str
    answer_text: Optional[str] = None
    question_embedding: Optional[List[float]] = None
    was_useful: Optional[bool] = None
    feedback: Optional[str] = None
    question_metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None
    models: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for API responses"""
        return {
            "id": self.id,
            "question_text": self.question_text,
            "answer_text": self.answer_text,
            "was_useful": self.was_useful,
            "feedback": self.feedback,
            "metadata": self.question_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "models": self.models,
        }

    def to_orm(self) -> QuestionTable:
        """Convert domain model to ORM model"""
        return QuestionTable(
            id=self.id,
            question_text=self.question_text,
            answer_text=self.answer_text,
            question_embedding=self.question_embedding,
            was_useful=self.was_useful,
            feedback=self.feedback,
            question_metadata=self.question_metadata,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


@dataclass
class ModelEmbedding:
    """Domain model for a model embedding."""

    model_name: str
    document: str
    embedding: Any  # Vector is a custom type
    id: Optional[int] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for API responses"""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "document": self.document,
            # Embedding is omitted for API responses as it's usually large
            "model_metadata": self.model_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def to_orm(self):
        """Convert domain model to ORM model"""
        return ModelEmbeddingTable.from_domain(self)

    @classmethod
    def from_domain(cls, embedding):
        """Create a domain model from another domain model instance
        This is added to handle the case where ModelEmbeddingTable.from_domain expects a ModelEmbedding
        """
        return cls(
            id=embedding.id,
            model_name=embedding.model_name,
            document=embedding.document,
            embedding=embedding.embedding,
            model_metadata=embedding.model_metadata,
            created_at=embedding.created_at,
            updated_at=embedding.updated_at,
        )


class ModelMetadata(BaseModel):
    """Pydantic model for storing extracted model metadata."""

    name: str = Field(..., description="Name of the dbt model")
    description: Optional[str] = Field(None, description="Description of the dbt model")
    schema_name: Optional[str] = Field(
        None, description="Schema the model is materialized into", alias="schema"
    )
    database: Optional[str] = Field(
        None, description="Database the model is materialized into"
    )
    materialization: Optional[str] = Field(
        None, description="Materialization type (table, view, etc.)"
    )
    tags: Optional[List[str]] = Field(
        default_factory=list, description="Tags associated with the model"
    )
    columns: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of columns with their name, description, and data type",
    )
    tests: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of tests associated with the model or its columns",
    )
    depends_on: Optional[List[str]] = Field(
        default_factory=list,
        description="List of models or sources this model directly depends on",
    )
    path: Optional[str] = Field(
        None, description="File path of the model relative to the project root"
    )
    unique_id: Optional[str] = Field(
        None, description="Unique ID of the model in the dbt project"
    )
    raw_sql: Optional[str] = Field(None, description="Raw SQL code of the model")
    compiled_sql: Optional[str] = Field(
        None, description="Compiled SQL code of the model"
    )
    all_upstream_models: Optional[List[str]] = Field(
        default_factory=list,
        description="List of all upstream model dependencies (recursive)",
    )
    interpreted_description: Optional[str] = Field(
        None, description="LLM-generated description of the model"
    )
    interpretation_details: Optional[Dict[str, Any]] = Field(
        None, description="Additional details about the interpretation process"
    )

    class Config:
        populate_by_name = True
