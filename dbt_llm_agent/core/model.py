"""DBT model representation."""

from typing import Dict, List, Any, Optional


class DBTModel:
    """Represents a dbt model."""

    def __init__(
        self,
        name: str,
        path: str,
        description: str = "",
        columns: Dict[str, Dict[str, Any]] = None,
        schema: str = "",
        database: str = "",
        materialization: str = "",
        tags: List[str] = None,
        depends_on: List[str] = None,
        tests: List[Dict[str, Any]] = None,
        all_upstream_models: List[str] = None,
    ):
        """Initialize a dbt model.

        Args:
            name: Name of the model
            path: Path to the model file
            description: Description of the model
            columns: Columns in the model
            schema: Schema of the model
            database: Database of the model
            materialization: Materialization of the model
            tags: Tags for the model
            depends_on: Models this model depends on
            tests: Tests for the model
            all_upstream_models: All models that come upstream of this model (recursive)
        """
        self.id = None  # Will be set when stored in the database
        self.name = name
        self.path = path
        self.description = description
        self.columns = columns or {}
        self.schema = schema
        self.database = database
        self.materialization = materialization
        self.tags = tags or []
        self.depends_on = depends_on or []
        self.tests = tests or []
        self.all_upstream_models = all_upstream_models or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary.

        Returns:
            Dictionary representation of the model
        """
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "columns": self.columns,
            "schema": self.schema,
            "database": self.database,
            "materialization": self.materialization,
            "tags": self.tags,
            "depends_on": self.depends_on,
            "tests": self.tests,
            "all_upstream_models": self.all_upstream_models,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DBTModel":
        """Create a model from a dictionary.

        Args:
            data: Dictionary representation of the model

        Returns:
            DBTModel instance
        """
        model = cls(
            name=data["name"],
            path=data["path"],
            description=data.get("description", ""),
            columns=data.get("columns", {}),
            schema=data.get("schema", ""),
            database=data.get("database", ""),
            materialization=data.get("materialization", ""),
            tags=data.get("tags", []),
            depends_on=data.get("depends_on", []),
            tests=data.get("tests", []),
            all_upstream_models=data.get("all_upstream_models", []),
        )
        model.id = data.get("id")
        return model

    def get_readable_representation(self) -> str:
        """Get a readable representation of the model for embedding.

        Returns:
            Readable representation of the model
        """
        lines = [
            f"Model: {self.name}",
            f"Description: {self.description}",
            f"Schema: {self.schema}",
            f"Database: {self.database}",
            f"Materialization: {self.materialization}",
        ]

        if self.tags:
            lines.append(f"Tags: {', '.join(self.tags)}")

        if self.depends_on:
            lines.append(f"Depends on: {', '.join(self.depends_on)}")

        if self.all_upstream_models:
            lines.append(f"All upstream models: {', '.join(self.all_upstream_models)}")

        if self.columns:
            lines.append("\nColumns:")
            for col_name, col_info in self.columns.items():
                col_desc = col_info.get("description", "")
                lines.append(f"  - {col_name}: {col_desc}")

        if self.tests:
            lines.append("\nTests:")
            for test in self.tests:
                test_name = test.get("name", "unknown")
                test_column = test.get("column", "")
                if test_column:
                    lines.append(f"  - {test_name} on column {test_column}")
                else:
                    lines.append(f"  - {test_name} on model")

        return "\n".join(lines)
