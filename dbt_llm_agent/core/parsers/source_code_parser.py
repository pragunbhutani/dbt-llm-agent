"""Module for parsing dbt projects and extracting model information."""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
import re

from dbt_llm_agent.core.models import DBTModel, Column, Test, ModelMetadata, DBTProject

logger = logging.getLogger(__name__)


class SourceCodeParser:
    """Parser for dbt projects based on source code analysis."""

    def __init__(self, project_path: str):
        """Initialize the parser with a path to a dbt project.

        Args:
            project_path: Path to the dbt project directory
        """
        self.project_path = Path(project_path).resolve()
        self.models_path = self.project_path / "models"

        # Validate that this is a dbt project
        if not (self.project_path / "dbt_project.yml").exists():
            raise ValueError(f"No dbt_project.yml found in {project_path}")

        # Load project configuration
        self.project_config = self._load_project_config()

    def _load_project_config(self) -> Dict[str, Any]:
        """Load the dbt project configuration from dbt_project.yml.

        Returns:
            Dict containing the project configuration
        """
        try:
            with open(self.project_path / "dbt_project.yml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading project config: {e}")
            return {}

    def parse_project(self) -> DBTProject:
        """Parse the dbt project and extract all model information.

        Returns:
            DBTProject containing all project information
        """
        # Parse directly from files without using dbt compile
        return self._parse_from_files()

    def _parse_from_files(self) -> DBTProject:
        """Parse the dbt project from the source files.

        Returns:
            DBTProject containing all project information
        """
        project_name = self.project_config.get("name", "")
        project = DBTProject(name=project_name, models={}, config=self.project_config)

        # Find all YAML files with model definitions
        yaml_files = list(self.models_path.glob("**/*.yml"))
        yaml_files.extend(self.models_path.glob("**/*.yaml"))

        # Parse model definitions from YAML files
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r") as f:
                    yaml_content = yaml.safe_load(f)

                if yaml_content is None:
                    logger.warning(f"Empty or invalid YAML file: {yaml_file}")
                    continue

                # Check for different types of YAML files
                if "models" in yaml_content:
                    self._parse_models_yaml(yaml_content, yaml_file, project)
                elif "sources" in yaml_content:
                    # For now, we're just logging sources but not processing them
                    logger.info(f"Found source definitions in {yaml_file}")
                elif "macros" in yaml_content:
                    logger.info(f"Found macro definitions in {yaml_file}")
                else:
                    logger.info(f"Unknown YAML structure in {yaml_file}")

            except Exception as e:
                logger.error(f"Error parsing YAML file {yaml_file}: {e}")

        # Find and add SQL files
        sql_files = list(self.models_path.glob("**/*.sql"))
        for sql_file in sql_files:
            model_name = sql_file.stem

            # If model doesn't exist in YAML, create it
            if model_name not in project.models:
                model = DBTModel(
                    name=model_name,
                    description="",
                    columns={},
                    tests=[],
                    schema="",
                    database="",
                    materialization=self._extract_materialization_from_sql(sql_file),
                    tags=[],
                    meta={},
                    raw_sql="",
                    compiled_sql="",
                    depends_on=[],
                    path=str(sql_file.relative_to(self.project_path)),
                    unique_id=f"model.{project_name}.{model_name}",
                )
                project.models[model_name] = model

            # Add SQL content to the model
            try:
                with open(sql_file, "r") as f:
                    raw_sql = f.read()

                model = project.models[model_name]
                model.raw_sql = raw_sql
                model.path = str(sql_file.relative_to(self.project_path))

                # Extract description and column info from SQL comments if not already defined
                if not model.description:
                    model.description = self._extract_description_from_sql(raw_sql)

                # Extract dependencies from SQL
                model.depends_on = self._extract_dependencies_from_sql(
                    raw_sql, project_name
                )

                # Extract columns from SQL if not already defined
                if not model.columns:
                    model.columns = self._extract_columns_from_sql(raw_sql)

            except Exception as e:
                logger.error(f"Error reading SQL file {sql_file}: {e}")

        # Compute all_upstream_models for each model
        logger.info("Computing upstream dependency chains for all models")
        for model_name, model in project.models.items():
            try:
                # Get the upstream lineage for this model
                lineage = self.get_model_lineage(project, model_name, upstream=True)

                # Extract all upstream dependencies (excluding the model itself)
                all_upstream = set()
                for dep_model, deps in lineage.items():
                    if dep_model != model_name:  # Don't include the model itself
                        all_upstream.add(dep_model)
                    for dep in deps:
                        if dep != model_name:  # Don't include the model itself
                            all_upstream.add(dep)

                # Set all_upstream_models on the model
                model.all_upstream_models = list(all_upstream)
                logger.debug(
                    f"Model {model_name} has {len(model.all_upstream_models)} upstream models"
                )
            except Exception as e:
                logger.error(
                    f"Error computing upstream dependencies for model {model_name}: {e}"
                )
                model.all_upstream_models = []

        logger.info(f"Parsed {len(project.models)} models from files")
        return project

    def _parse_models_yaml(
        self, yaml_content: Dict[str, Any], yaml_file: Path, project: DBTProject
    ) -> None:
        """Parse models from a YAML file.

        Args:
            yaml_content: The parsed YAML content
            yaml_file: Path to the YAML file
            project: The DBTProject to add models to
        """
        project_name = project.name

        # Process models defined in the YAML
        models_list = yaml_content.get("models", [])
        if not isinstance(models_list, list):
            logger.warning(f"'models' in {yaml_file} is not a list, skipping")
            return

        for model_def in models_list:
            if not isinstance(model_def, dict):
                logger.warning(
                    f"Model definition in {yaml_file} is not a dictionary, skipping"
                )
                continue

            model_name = model_def.get("name", "")
            if not model_name:
                logger.warning(f"Model without name in {yaml_file}, skipping")
                continue

            # Parse columns
            columns = {}
            columns_list = model_def.get("columns", [])
            if isinstance(columns_list, list):
                for col_def in columns_list:
                    if not isinstance(col_def, dict):
                        continue

                    col_name = col_def.get("name", "")
                    if not col_name:
                        continue

                    columns[col_name] = Column(
                        name=col_name,
                        description=col_def.get("description", ""),
                        data_type="",  # Not available in YAML
                        meta=col_def.get("meta", {}) or {},
                    )

            # Parse tests
            tests = []
            if isinstance(columns_list, list):
                for col_def in columns_list:
                    if not isinstance(col_def, dict):
                        continue

                    col_name = col_def.get("name", "")

                    # Handle tests field which can be either a list or a dictionary
                    col_tests = col_def.get("tests", [])

                    if isinstance(col_tests, dict):
                        # Dictionary format: tests: { unique: true, not_null: true }
                        for test_name, test_config in col_tests.items():
                            test = Test(
                                name=test_name,
                                column_name=col_name,
                                test_type=test_name,
                                unique_id="",
                                meta={},
                            )
                            tests.append(test)
                    elif isinstance(col_tests, list):
                        # List format: tests: ['unique', 'not_null']
                        for test_item in col_tests:
                            if isinstance(test_item, str):
                                # Simple test name
                                test = Test(
                                    name=test_item,
                                    column_name=col_name,
                                    test_type=test_item,
                                    unique_id="",
                                    meta={},
                                )
                                tests.append(test)
                            elif isinstance(test_item, dict):
                                # Dictionary with a single key-value pair
                                for test_name, test_config in test_item.items():
                                    test = Test(
                                        name=test_name,
                                        column_name=col_name,
                                        test_type=test_name,
                                        unique_id="",
                                        meta={},
                                    )
                                    tests.append(test)

            # Also handle model-level tests
            model_tests = model_def.get("tests", [])
            if isinstance(model_tests, list):
                for test_item in model_tests:
                    if isinstance(test_item, str):
                        test = Test(
                            name=test_item,
                            column_name="",  # Model-level test
                            test_type=test_item,
                            unique_id="",
                            meta={},
                        )
                        tests.append(test)
                    elif isinstance(test_item, dict):
                        for test_name, test_config in test_item.items():
                            test = Test(
                                name=test_name,
                                column_name="",  # Model-level test
                                test_type=test_name,
                                unique_id="",
                                meta={},
                            )
                            tests.append(test)

            # Create a documentation string from the YAML content for this model
            documentation = ""
            if model_def.get("description"):
                documentation += (
                    f"## {model_name}\n\n{model_def.get('description')}\n\n"
                )

            if columns:
                documentation += "### Columns\n\n"
                for col_name, col in columns.items():
                    if col.description:
                        documentation += f"- {col_name}: {col.description}\n"

            # Create model (without SQL yet)
            model = DBTModel(
                name=model_name,
                description=model_def.get("description", ""),
                columns=columns,
                tests=tests,
                schema="",
                database="",
                materialization=self._extract_materialization_from_config(model_def),
                tags=model_def.get("tags", []),
                meta=model_def.get("meta", {}) or {},
                raw_sql="",
                compiled_sql="",
                depends_on=[],
                path=str(yaml_file.relative_to(self.project_path)),
                unique_id=f"model.{project_name}.{model_name}",
                documentation=documentation,  # Set the documentation field
            )

            project.models[model_name] = model

    def _extract_materialization_from_config(self, model_def: Dict[str, Any]) -> str:
        """Extract materialization from model config.

        Args:
            model_def: Model definition from YAML

        Returns:
            Materialization type (table, view, etc.)
        """
        config = model_def.get("config", {})
        if isinstance(config, dict):
            return config.get("materialized", "")
        return ""

    def _extract_materialization_from_sql(self, sql_file: Path) -> str:
        """Extract materialization from SQL file.

        Args:
            sql_file: Path to SQL file

        Returns:
            Materialization type (table, view, etc.)
        """
        try:
            with open(sql_file, "r") as f:
                content = f.read()

            # Look for materialization in config block
            materialization_match = re.search(
                r'{{\s*config\s*\(\s*materialized\s*=\s*[\'"](\w+)[\'"]', content
            )
            if materialization_match:
                return materialization_match.group(1)
        except Exception as e:
            logger.error(f"Error extracting materialization from {sql_file}: {e}")

        return ""

    def _extract_description_from_sql(self, sql_content: str) -> str:
        """Extract model description from SQL comments.

        Args:
            sql_content: SQL content

        Returns:
            Model description
        """
        # Look for description in comments
        description_match = re.search(r"--\s*description:\s*(.*?)(?:\n|$)", sql_content)
        if description_match:
            return description_match.group(1).strip()

        # Look for description in Jinja comment blocks
        jinja_desc_match = re.search(
            r'{{\s*config\s*\(\s*.*description\s*=\s*[\'"](.+?)[\'"]',
            sql_content,
            re.DOTALL,
        )
        if jinja_desc_match:
            return jinja_desc_match.group(1).strip()

        return ""

    def _extract_columns_from_sql(self, sql_content: str) -> Dict[str, Column]:
        """Extract column information from SQL content.

        Args:
            sql_content: SQL content

        Returns:
            Dict mapping column names to Column objects
        """
        columns = {}

        # Look for column definitions in comments
        column_matches = re.findall(
            r"--\s*column:\s*(\w+)\s*description:\s*(.*?)(?:\n|$)", sql_content
        )

        for match in column_matches:
            col_name, col_desc = match
            columns[col_name] = Column(
                name=col_name, description=col_desc.strip(), data_type="", meta={}
            )

        # If no columns found in comments, try to extract from SELECT statement
        if not columns:
            # Simple regex to find column names in SELECT statements
            # This is a basic implementation and might need refinement
            select_match = re.search(
                r"SELECT\s+(.+?)\s+FROM", sql_content, re.IGNORECASE | re.DOTALL
            )
            if select_match:
                select_clause = select_match.group(1)

                # Remove comments
                select_clause = re.sub(r"--.*?$", "", select_clause, flags=re.MULTILINE)

                # Split by commas, handling nested parentheses
                depth = 0
                current = ""
                parts = []

                for char in select_clause:
                    if char == "(" or char == "{":
                        depth += 1
                    elif char == ")" or char == "}":
                        depth -= 1

                    if char == "," and depth == 0:
                        parts.append(current.strip())
                        current = ""
                    else:
                        current += char

                if current:
                    parts.append(current.strip())

                # Extract column names
                for part in parts:
                    # Handle "AS" alias
                    as_match = re.search(r"(?:AS\s+)?(\w+)\s*$", part, re.IGNORECASE)
                    if as_match:
                        col_name = as_match.group(1)
                        columns[col_name] = Column(
                            name=col_name, description="", data_type="", meta={}
                        )

        return columns

    def _extract_dependencies_from_sql(
        self, sql_content: str, project_name: str
    ) -> List[str]:
        """Extract model dependencies from SQL content.

        Args:
            sql_content: SQL content
            project_name: Name of the dbt project

        Returns:
            List of model dependencies (only from ref() calls)
        """
        dependencies = []

        # Only look for {{ ref('model_name') }} or {{ ref("model_name") }} patterns
        # This regex matches both single and double quotes in the ref() function
        ref_matches = re.findall(
            r'{{\s*ref\s*\(\s*[\'"]([^\'"]*)[\'"]\s*\)\s*}}', sql_content
        )

        # In case there are multiple arguments to ref
        # e.g., {{ ref('model_name', 'version') }}
        multi_arg_ref_matches = re.findall(
            r'{{\s*ref\s*\(\s*[\'"]([^\'"]*)[\'"]\s*,\s*[\'"][^\'"]*[\'"]\s*\)\s*}}',
            sql_content,
        )

        # Combine all matches
        all_refs = ref_matches + multi_arg_ref_matches

        # Add all dependencies
        for model_name in set(all_refs):
            if model_name:  # Ensure we don't add empty strings
                dependencies.append(model_name)

        logger.debug(f"Found dependencies via ref(): {dependencies}")

        return dependencies

    def get_model_lineage(
        self, project: DBTProject, model_name: str, upstream: bool = True
    ) -> Dict[str, List[str]]:
        """Get the lineage of a model (its dependencies or dependents).

        Args:
            project: The dbt project
            model_name: Name of the model
            upstream: If True, get upstream dependencies; if False, get downstream dependents

        Returns:
            Dict mapping model names to their dependencies or dependents
        """
        result = {}
        visited = set()

        def traverse(current_model: str, depth: int = 0):
            if current_model in visited:
                return

            visited.add(current_model)
            model = project.models.get(current_model)

            if not model:
                return

            if upstream:
                # Get upstream dependencies - now they are just model names
                depends_on = model.depends_on
                result[current_model] = depends_on

                # Recursively traverse dependencies
                for dep in depends_on:
                    traverse(dep, depth + 1)
            else:
                # Get downstream dependents
                dependents = []
                for name, m in project.models.items():
                    if current_model in m.depends_on:
                        dependents.append(name)

                result[current_model] = dependents

                # Recursively traverse dependents
                for dep in dependents:
                    traverse(dep, depth + 1)

        traverse(model_name)
        return result

    def get_model_metadata(self, model: DBTModel) -> ModelMetadata:
        """Extract metadata for a model in a format suitable for storage.

        Args:
            model: The dbt model

        Returns:
            ModelMetadata containing extracted metadata
        """
        return ModelMetadata(
            name=model.name,
            description=model.description,
            schema=model.schema,
            database=model.database,
            materialization=model.materialization,
            tags=model.tags,
            columns=[
                {
                    "name": col.name,
                    "description": col.description,
                    "data_type": col.data_type,
                }
                for col in model.columns.values()
            ],
            tests=[
                {
                    "name": test.name,
                    "column_name": test.column_name,
                    "test_type": test.test_type,
                }
                for test in model.tests
            ],
            depends_on=model.depends_on,  # Dependencies are now already in the correct format
            path=model.path,
            unique_id=model.unique_id,
        )
