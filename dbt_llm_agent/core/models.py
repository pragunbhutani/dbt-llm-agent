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
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import registry
from sqlalchemy.sql import func
from datetime import datetime
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

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
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    schema = Column(String, nullable=True)
    database = Column(String, nullable=True)
    materialization = Column(String, nullable=True)
    tags = Column(JSON, nullable=True)
    depends_on = Column(JSON, nullable=True)
    tests = Column(JSON, nullable=True)
    all_upstream_models = Column(JSON, nullable=True)
    meta = Column(JSON, nullable=True)
    raw_sql = Column(Text, nullable=True)
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
            interpreted_description=self.interpreted_description or "",
            interpreted_columns=self.interpreted_columns or {},
            columns=convert_columns(self.yml_columns),
            tests=convert_tests(self.tests),
            # Removing compiled_sql, documentation, and unique_id
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
            # Removing compiled_sql, documentation, and unique_id
            interpreted_description=model.interpreted_description,
            interpreted_columns=model.interpreted_columns,
            yml_columns=convert_columns_to_json(model.columns),
            tests=convert_tests_to_json(model.tests),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


# Question model relationships
question_models_assoc = Table(
    "question_models_assoc",
    Base.metadata,
    Column("question_id", Integer, ForeignKey("questions.id"), primary_key=True),
    Column("model_name", String, primary_key=True),
    Column("relevance_score", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)


class QuestionTable(Base):
    """SQLAlchemy model for storing questions and answers."""

    __tablename__ = "questions"

    id = Column(Integer, primary_key=True)
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=True)
    was_useful = Column(Boolean, nullable=True)
    feedback = Column(Text, nullable=True)
    question_metadata = Column(JSON, nullable=True)  # For any additional metadata
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
            was_useful=self.was_useful,
            feedback=self.feedback,
            question_metadata=self.question_metadata or {},
            created_at=self.created_at,
            updated_at=self.updated_at,
            models=models_data,
        )

    @classmethod
    def from_domain(cls, question):
        """Create an ORM model from a domain model"""
        return cls(
            id=question.id,
            question_text=question.question_text,
            answer_text=question.answer_text,
            was_useful=question.was_useful,
            feedback=question.feedback,
            question_metadata=question.question_metadata,
            # Note: model_assocs would be handled separately
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
    raw_sql: str = ""
    # List of model names that this model depends on (names from ref() calls)
    depends_on: List[str] = field(default_factory=list)
    # List of all models upstream in the dependency chain
    all_upstream_models: List[str] = field(default_factory=list)
    path: str = ""
    interpreted_description: str = ""  # LLM-generated description
    interpreted_columns: Dict[str, str] = field(
        default_factory=dict
    )  # LLM-interpreted column descriptions
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_embedding_text(
        self,
        documentation_text: Optional[str] = None,
        interpretation_text: Optional[str] = None,
    ) -> str:
        """Create a standardized text representation for embeddings.

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
            documentation_text: Optional override for the model description
            interpretation_text: Optional override for the model interpretation

        Returns:
            A consistently structured document for embeddings
        """
        # Create a structured document with a consistent format
        embedding_text = f"Model: {self.name}\n"

        # Use documentation_text if provided, otherwise use model description
        description = (
            documentation_text
            if documentation_text and documentation_text.strip()
            else self.description or ""
        )
        embedding_text += f"Description (YML): {description if description.strip() else 'Not Available'}\n"

        # Use interpretation_text if provided, otherwise check if model has interpretation
        if interpretation_text and interpretation_text.strip():
            embedding_text += f"Interpretation (LLM): {interpretation_text}\n"
        elif self.interpreted_description:
            embedding_text += f"Interpretation (LLM): {self.interpreted_description}\n"
        else:
            embedding_text += "Interpretation (LLM): Not Available\n"

        embedding_text += f"Path: {self.path or ''}\n"
        embedding_text += f"Schema: {self.schema or ''}\n"
        embedding_text += f"Database: {self.database or ''}\n"
        embedding_text += f"Materialization: {self.materialization or ''}\n"

        # Add dependencies
        # Handle different possible structures for depends_on
        deps = []
        if isinstance(self.depends_on, list):
            deps = self.depends_on
        elif isinstance(self.depends_on, dict) and "ref" in self.depends_on:
            deps = self.depends_on["ref"]

        embedding_text += f"Depends on (via ref): {', '.join(deps) if deps else ''}\n"

        # Add YML columns section
        if self.columns:
            column_count = len(self.columns)
            embedding_text += f"\nColumns from YML documentation ({column_count}):\n"

            # Handle columns
            for col_name, col_info in self.columns.items():
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
        if self.interpreted_columns:
            interpreted_column_count = len(self.interpreted_columns)
            embedding_text += (
                f"\nInterpreted columns from LLM ({interpreted_column_count}):\n"
            )
            for col_name, col_desc in self.interpreted_columns.items():
                embedding_text += f"- {col_name}: {col_desc}\n"
        else:
            embedding_text += "\nInterpreted columns from LLM (0):\n"

        return embedding_text

    @staticmethod
    def embedding_text_to_json(
        embedding_text: str, tests: Optional[List[Test]] = None
    ) -> Dict[str, Any]:
        """Convert embedding text format to JSON.

        Args:
            embedding_text: The embedding text to convert
            tests: Optional list of tests to include in the output

        Returns:
            A structured dictionary representation of the embedding text
        """
        lines = embedding_text.strip().split("\n")
        result = {}

        current_section = None
        for line in lines:
            if line.startswith("Model:"):
                result["model_name"] = line.replace("Model:", "").strip()
            elif line.startswith("Description (YML):"):
                result["description"] = line.replace("Description (YML):", "").strip()
            elif line.startswith("Interpretation (LLM):"):
                result["has_interpretation"] = "Available" in line
            elif line.startswith("Path:"):
                result["path"] = line.replace("Path:", "").strip()
            elif line.startswith("Schema:"):
                result["schema"] = line.replace("Schema:", "").strip()
            elif line.startswith("Database:"):
                result["database"] = line.replace("Database:", "").strip()
            elif line.startswith("Materialization:"):
                result["materialization"] = line.replace("Materialization:", "").strip()
            elif line.startswith("Depends on (via ref):"):
                deps = line.replace("Depends on (via ref):", "").strip()
                result["depends_on"] = (
                    [d.strip() for d in deps.split(",")] if deps else []
                )
            elif "Columns from YML documentation" in line:
                current_section = "columns"
                result["columns"] = {}
            elif "Interpreted columns from LLM" in line:
                current_section = "interpreted_columns"
                result["interpreted_columns"] = {}
            elif line.startswith("- ") and current_section == "columns":
                # Parse column line: "- column_name: description"
                parts = line[2:].split(":", 1)
                if len(parts) == 2:
                    col_name, col_desc = parts
                    result["columns"][col_name.strip()] = col_desc.strip()
            elif line.startswith("- ") and current_section == "interpreted_columns":
                # Parse interpreted column line
                parts = line[2:].split(":", 1)
                if len(parts) == 2:
                    col_name, col_desc = parts
                    result["interpreted_columns"][col_name.strip()] = col_desc.strip()

        # Include tests in the output if provided
        if tests:
            result["tests"] = []
            for test in tests:
                result["tests"].append(
                    {
                        "test_type": test.test_type or "",
                        "column_name": test.column_name or "",
                    }
                )

        return result

    def to_embedding_json(self) -> Dict[str, Any]:
        """Convert the model to a JSON format for embeddings.

        Returns:
            A structured dictionary representation suitable for embeddings
        """
        embedding_text = self.to_embedding_text()
        return self.embedding_text_to_json(embedding_text, self.tests)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary.

        Returns:
            Dict representation of the model
        """
        return {
            "name": self.name,
            "description": self.description,
            "columns": {name: vars(col) for name, col in self.columns.items()},
            "tests": [vars(test) for test in self.tests],
            "schema": self.schema,
            "database": self.database,
            "materialization": self.materialization,
            "tags": self.tags,
            "meta": self.meta,
            "raw_sql": self.raw_sql,
            "depends_on": self.depends_on,
            "path": self.path,
            "interpreted_description": self.interpreted_description,
            "interpreted_columns": self.interpreted_columns,
        }

    def get_column_descriptions(self) -> Dict[str, str]:
        """Get a dictionary of column names and their descriptions.

        Returns:
            Dict mapping column names to descriptions
        """
        return {name: col.description for name, col in self.columns.items()}

    def get_readable_representation(self) -> str:
        """Get a readable representation of the model.

        Returns:
            A readable representation of the model
        """
        representation = f"Model: {self.name}\n"
        representation += (
            f"Description (YML): {self.description or 'No YML description'}\n"
        )

        if self.interpreted_description:
            representation += f"Interpretation (LLM): {self.interpreted_description}\n"
        else:
            representation += f"Interpretation (LLM): Not available\n"

        representation += f"Path: {self.path}\n"
        representation += f"Schema: {self.schema}\n"
        representation += f"Database: {self.database}\n"
        representation += f"Materialization: {self.materialization}\n"

        if self.tags:
            representation += f"Tags: {', '.join(self.tags)}\n"

        if self.depends_on:
            representation += f"Depends on (via ref): {', '.join(self.depends_on)}\n"

        # Add columns from YML
        if self.columns:
            representation += (
                f"\nColumns from YML documentation ({len(self.columns)}):\n"
            )
            for name, col in self.columns.items():
                representation += f"- {name}: {col.description or 'No description'}\n"

        # Add interpreted columns
        if self.interpreted_columns:
            representation += (
                f"\nInterpreted columns from LLM ({len(self.interpreted_columns)}):\n"
            )
            for name, description in self.interpreted_columns.items():
                representation += f"- {name}: {description}\n"

        return representation

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
    """Domain model for a question and its answer."""

    question_text: str
    id: Optional[int] = None
    answer_text: Optional[str] = None
    was_useful: Optional[bool] = None
    feedback: Optional[str] = None
    question_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
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

    def to_orm(self):
        """Convert domain model to ORM model"""
        return QuestionTable.from_domain(self)


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
