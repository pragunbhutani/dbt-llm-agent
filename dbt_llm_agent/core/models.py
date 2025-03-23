"""Data models for representing dbt projects and models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set


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
    compiled_sql: str = ""
    # List of model names that this model depends on (names from ref() calls)
    depends_on: List[str] = field(default_factory=list)
    # List of all models upstream in the dependency chain
    all_upstream_models: List[str] = field(default_factory=list)
    path: str = ""
    unique_id: str = ""
    documentation: str = ""  # Original documentation
    interpreted_description: str = ""  # LLM-generated description
    interpreted_columns: Dict[str, str] = field(
        default_factory=dict
    )  # LLM-interpreted column descriptions

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
            "compiled_sql": self.compiled_sql,
            "depends_on": self.depends_on,
            "path": self.path,
            "unique_id": self.unique_id,
            "documentation": self.documentation,
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
class ModelMetadata:
    """Metadata for a model, used for storage."""

    name: str
    description: str
    schema: str
    database: str
    materialization: str
    tags: List[str]
    columns: List[Dict[str, str]]
    tests: List[Dict[str, str]]
    depends_on: List[str]
    path: str
    unique_id: str
