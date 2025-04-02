"""
Parser for dbt manifest.json files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from dbt_llm_agent.core.models import DBTModel, Column, Test, ModelMetadata, DBTProject

# Renamed internal representations to avoid confusion with SQLAlchemy models
from dbt_llm_agent.core.models import Column as ColumnInternal
from dbt_llm_agent.core.models import Test as TestInternal
from dbt_llm_agent.core.models import DBTModel as DBTModelInternal
from dbt_llm_agent.core.models import DBTProject as DBTProjectInternal

logger = logging.getLogger(__name__)


class ManifestParser:
    """Parses a dbt manifest.json file to extract project information."""

    def __init__(self, manifest_path: str):
        """Initialize the parser with the path to manifest.json.

        Args:
            manifest_path: Path to the manifest.json file.
        """
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.exists() or not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
        self.manifest_data: Dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the manifest.json file.

        Returns:
            Dictionary containing the manifest data.
        """
        try:
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding manifest JSON: {e}")
            raise ValueError(f"Invalid JSON in manifest file: {self.manifest_path}")
        except Exception as e:
            logger.error(f"Error loading manifest file: {e}")
            raise

    def parse_project(self) -> DBTProjectInternal:
        """Parse the manifest data and extract project information.

        Returns:
            DBTProjectInternal containing all project information derived from the manifest.
        """
        metadata = self.manifest_data.get("metadata", {})
        project_name = metadata.get("project_name", "")
        project = DBTProjectInternal(name=project_name, models={}, config={})

        nodes = self.manifest_data.get("nodes", {})
        sources = self.manifest_data.get("sources", {})
        macros = self.manifest_data.get("macros", {})
        exposures = self.manifest_data.get("exposures", {})
        metrics = self.manifest_data.get("metrics", {})
        tests = self.manifest_data.get("tests", {})

        model_count = 0
        for unique_id, node_data in nodes.items():
            if node_data.get("resource_type") == "model":
                try:
                    model = self._parse_model_node(unique_id, node_data, nodes, sources)
                    project.models[model.name] = model
                    model_count += 1
                except Exception as e:
                    # Log the full traceback for debugging
                    logger.warning(
                        f"Skipping node {unique_id} due to parsing error: {e}",
                        exc_info=True,
                    )

        # TODO: Potentially parse other resource types like sources, macros etc.
        # if needed for the application context.

        # Compute upstream dependencies (already handled within _parse_model_node)
        # Compute all_upstream_models for each model
        logger.info("Computing upstream dependency chains for all models from manifest")
        for model_name, model in project.models.items():
            try:
                lineage = self.get_model_lineage(project, model_name, upstream=True)
                all_upstream = set()
                for dep_model_name, deps in lineage.items():
                    if dep_model_name != model_name:
                        all_upstream.add(dep_model_name)
                    for dep_unique_id in deps:
                        # Resolve unique_id to model name for model dependencies
                        dep_node = nodes.get(dep_unique_id)
                        if dep_node and dep_node.get("resource_type") == "model":
                            dep_name = dep_node.get("name")
                            if dep_name and dep_name != model_name:
                                all_upstream.add(dep_name)
                        # We could add source names here too if needed

                model.all_upstream_models = list(all_upstream)
                logger.debug(
                    f"Model {model_name} has {len(model.all_upstream_models)} upstream models"
                )
            except Exception as e:
                logger.error(
                    f"Error computing upstream dependencies for model {model_name}: {e}",
                    exc_info=True,
                )
                model.all_upstream_models = []

        logger.info(f"Parsed {model_count} models from manifest: {self.manifest_path}")
        return project

    def _parse_model_node(
        self,
        unique_id: str,
        node_data: Dict[str, Any],
        all_nodes: Dict,
        all_sources: Dict,
    ) -> DBTModelInternal:
        """Parse a single model node from the manifest.

        Args:
            unique_id: The unique ID of the model node.
            node_data: The dictionary containing the model node data.
            all_nodes: All nodes from the manifest for dependency resolution.
            all_sources: All sources from the manifest for dependency resolution.

        Returns:
            DBTModelInternal object representing the parsed model.
        """
        model_name = node_data.get("name", "")

        # Parse columns
        columns = {}
        for col_name, col_data in node_data.get("columns", {}).items():
            columns[col_name] = ColumnInternal(
                name=col_name,
                description=col_data.get("description", ""),
                data_type=col_data.get("data_type", ""),
                meta=col_data.get("meta", {}) or {},
            )

        # Parse tests (Placeholder - Requires linking test nodes)
        tests = []

        # Extract dependencies
        depends_on_nodes = node_data.get("depends_on", {}).get("nodes", [])
        direct_dependencies = []
        for dep_id in depends_on_nodes:
            if dep_id.startswith("model.") or dep_id.startswith("source."):
                dep_node = all_nodes.get(dep_id)
                if dep_node and dep_node.get("resource_type") == "model":
                    direct_dependencies.append(dep_node.get("name"))
                elif dep_id.startswith("source."):
                    direct_dependencies.append(dep_id)  # Keep source unique_id

        # Map raw_code/compiled_code to raw_sql/compiled_sql
        raw_sql_content = node_data.get("raw_code") or node_data.get("raw_sql") or ""
        compiled_sql_content = (
            node_data.get("compiled_code") or node_data.get("compiled_sql") or ""
        )

        model = DBTModelInternal(
            name=model_name,
            description=node_data.get("description", ""),
            columns=columns,
            tests=tests,
            schema=node_data.get("schema", ""),
            database=node_data.get("database", ""),
            materialization=node_data.get("config", {}).get("materialized", ""),
            tags=node_data.get("tags", []),
            meta=node_data.get("meta", {}) or {},
            raw_sql=raw_sql_content,  # Use mapped value
            compiled_sql=compiled_sql_content,  # Use mapped value
            depends_on=direct_dependencies,
            path=node_data.get("path", ""),
            unique_id=unique_id,
            documentation=node_data.get("docs", {}).get("show", False),
            # all_upstream_models computed later
        )

        # Remove the check for raw_code/compiled_code here as mapping is done above
        # if model.raw_code and not model.raw_sql:
        #     model.raw_sql = model.raw_code
        # if model.compiled_code and not model.compiled_sql:
        #     model.compiled_sql = model.compiled_code

        return model

    def get_model_metadata(self, model: DBTModelInternal) -> ModelMetadata:
        """Extract metadata for a model in a format suitable for storage.

        Args:
            model: The dbt model parsed from the manifest.

        Returns:
            ModelMetadata containing extracted metadata.
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
            depends_on=model.depends_on,
            path=model.path,
            unique_id=model.unique_id,
            raw_sql=model.raw_sql,
            compiled_sql=model.compiled_sql,  # Ensure this is populated
            all_upstream_models=model.all_upstream_models,
        )

    # TODO: Adapt get_model_lineage from SourceCodeParser or reimplement for manifest structure
    # This version needs to work with unique_ids potentially before resolving all names
    def get_model_lineage(
        self, project: DBTProjectInternal, model_name: str, upstream: bool = True
    ) -> Dict[str, List[str]]:
        """Get the lineage of a model (its dependencies or dependents) based on manifest structure.

        Args:
            project: The dbt project (partially built from manifest).
            model_name: Name of the model.
            upstream: If True, get upstream dependencies; if False, get downstream dependents.

        Returns:
            Dict mapping model names to their direct dependencies (unique_ids) or dependents (names).
        """
        result = {}
        visited = set()
        nodes = self.manifest_data.get("nodes", {})

        # Find the unique_id for the starting model_name
        start_node_id = None
        for uid, node in nodes.items():
            if node.get("resource_type") == "model" and node.get("name") == model_name:
                start_node_id = uid
                break

        if not start_node_id:
            logger.warning(
                f"Could not find starting model '{model_name}' in manifest nodes."
            )
            return {}

        def traverse(current_node_id: str):
            if current_node_id in visited:
                return
            visited.add(current_node_id)

            current_node = nodes.get(current_node_id)
            if not current_node or current_node.get("resource_type") != "model":
                return

            current_model_name = current_node.get("name")
            if not current_model_name:
                return  # Should not happen for models

            if upstream:
                # Get upstream dependencies (unique_ids of models and sources)
                depends_on_nodes = current_node.get("depends_on", {}).get("nodes", [])
                direct_dependencies = []
                for dep_id in depends_on_nodes:
                    # Include only direct model/source dependencies
                    if dep_id.startswith("model.") or dep_id.startswith("source."):
                        direct_dependencies.append(dep_id)

                result[current_model_name] = direct_dependencies

                # Recursively traverse model dependencies
                for dep_id in direct_dependencies:
                    if dep_id.startswith("model."):
                        traverse(dep_id)
            else:  # downstream
                # Find nodes that depend on the current_node_id
                dependents = []
                for name, node_data in nodes.items():
                    # Check models, potentially others if needed
                    if node_data.get("resource_type") == "model":
                        node_deps = node_data.get("depends_on", {}).get("nodes", [])
                        if current_node_id in node_deps:
                            dependent_name = node_data.get("name")
                            if dependent_name:
                                dependents.append(dependent_name)

                result[current_model_name] = dependents

                # Recursively traverse dependents (using their names)
                # Need to find their unique_ids to continue traversal
                for dep_name in dependents:
                    dep_node_id = None
                    for uid, node in nodes.items():
                        if (
                            node.get("resource_type") == "model"
                            and node.get("name") == dep_name
                        ):
                            dep_node_id = uid
                            break
                    if dep_node_id:
                        traverse(dep_node_id)
                    else:
                        logger.warning(
                            f"Could not find unique_id for dependent model '{dep_name}' during downstream traversal."
                        )

        traverse(start_node_id)
        return result
