"""
Utility for processing dbt model selectors.

This module implements dbt's model selection syntax as described in
https://docs.getdbt.com/reference/node-selection/methods
"""

import re
import fnmatch
from typing import List, Dict, Set, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Implements dbt's model selector syntax for filtering models.

    Supports methods including:
    - tag: Models with the specified tag
    - +model_name: Include the model and all its parents (upstream models)
    - model_name+: Include the model and all its children (downstream models)
    - *: Wildcard matching for model names
    - path/to/folder: Models in the specified path
    - config.materialized: Models with the specified materialization

    Multiple selectors can be combined with commas, and the position of
    the + operator determines whether to include upstream or downstream models.
    """

    def __init__(self, models_dict: Dict[str, Any]):
        """
        Initialize the selector with a dictionary of models.

        Args:
            models_dict: Dictionary mapping model names to model objects
        """
        self.models_dict = models_dict

    def select(self, selection_str: str) -> List[str]:
        """
        Apply the selector string to filter models.

        Args:
            selection_str: Selection string in dbt selector syntax
                Examples:
                - "model_name"
                - "+model_name" (model and its upstream dependencies)
                - "model_name+" (model and its downstream dependencies)
                - "tag:marketing" (models with a specific tag)
                - "+tag:marketing" (models with tag and their upstream dependencies)
                - "tag:marketing+" (models with tag and their downstream dependencies)
                - "path/to/models" (models in a specific directory)

        Returns:
            List of selected model names
        """
        if not selection_str or selection_str.strip() == "*":
            # If no selection or just wildcard, return all models
            return list(self.models_dict.keys())

        # Split multiple selections (comma-separated)
        selectors = [s.strip() for s in selection_str.split(",")]

        # Process each selector and combine results
        selected_models = set()
        for selector in selectors:
            selected_models.update(self._process_selector(selector))

        return list(selected_models)

    def _process_selector(self, selector: str) -> Set[str]:
        """Process a single selector string."""
        selected = set()

        # Check for exclusion operator (!)
        exclude = selector.startswith("!")
        if exclude:
            selector = selector[1:]

        # Check for dependency modifiers based on their position
        # Pre-modifier (+model_name): Include upstream/parent models
        # Post-modifier (model_name+): Include downstream/child models
        include_upstream = selector.startswith("+")
        include_downstream = selector.endswith("+") and not selector.startswith("+")

        # Remove dependency modifiers for base selection
        base_selector = selector.strip("+")

        # Perform base selection
        base_selected = self._apply_base_selector(base_selector)

        # Start with base selection
        selected.update(base_selected)

        # Add dependencies if requested
        if include_upstream:
            upstream_models = set()
            for model in base_selected:
                upstream_models.update(self._get_parents(model))
            selected.update(upstream_models)

        if include_downstream:
            downstream_models = set()
            for model in base_selected:
                downstream_models.update(self._get_children(model))
            selected.update(downstream_models)

        # If exclusion operator was used, invert the selection
        if exclude:
            selected = set(self.models_dict.keys()) - selected

        return selected

    def _apply_base_selector(self, selector: str) -> Set[str]:
        """Apply the base selector without dependency modifiers."""
        # Tag selector: tag:my_tag
        if selector.startswith("tag:"):
            tag = selector[4:]
            return self._select_by_tag(tag)

        # Path selector: models/marketing/
        elif "/" in selector:
            return self._select_by_path(selector)

        # Materialization selector: config.materialized:table
        elif selector.startswith("config.materialized:"):
            materialization = selector[len("config.materialized:") :]
            return self._select_by_materialization(materialization)

        # Name selector (with possible wildcards)
        else:
            return self._select_by_name(selector)

    def _select_by_tag(self, tag: str) -> Set[str]:
        """Select models with the specified tag."""
        selected = set()
        for name, model in self.models_dict.items():
            if hasattr(model, "tags") and model.tags:
                # Handle both list and dict representations of tags
                if isinstance(model.tags, list) and tag in model.tags:
                    selected.add(name)
                elif isinstance(model.tags, dict) and tag in model.tags:
                    selected.add(name)
        return selected

    def _select_by_path(self, path: str) -> Set[str]:
        """
        Select models in the specified path.

        This handles both exact path matching and partial path inclusion,
        so you can select models by directory or by specific file path.
        """
        selected = set()
        for name, model in self.models_dict.items():
            if hasattr(model, "path") and model.path:
                # Handle path selectors that might represent directories
                if (
                    path in model.path
                    or fnmatch.fnmatch(model.path, path)
                    or fnmatch.fnmatch(model.path, f"{path}*")
                    or fnmatch.fnmatch(model.path, f"*/{path}/*")
                ):
                    selected.add(name)
        return selected

    def _select_by_materialization(self, materialization: str) -> Set[str]:
        """Select models with the specified materialization."""
        selected = set()
        for name, model in self.models_dict.items():
            if hasattr(model, "materialization") and model.materialization:
                if model.materialization == materialization:
                    selected.add(name)
        return selected

    def _select_by_name(self, name_pattern: str) -> Set[str]:
        """Select models by name, supporting wildcards."""
        if name_pattern == "*":
            return set(self.models_dict.keys())

        selected = set()
        for name in self.models_dict.keys():
            if fnmatch.fnmatch(name, name_pattern):
                selected.add(name)
        return selected

    def _get_children(self, model_name: str) -> Set[str]:
        """Get all downstream dependencies (children) of a model."""
        children = set()
        # Start with direct children
        direct_children = self._get_direct_children(model_name)
        children.update(direct_children)

        # Recursively add their children
        for child in direct_children:
            children.update(self._get_children(child))

        return children

    def _get_direct_children(self, model_name: str) -> Set[str]:
        """Get direct downstream dependencies of a model."""
        children = set()
        for name, model in self.models_dict.items():
            if hasattr(model, "depends_on") and model.depends_on:
                # Handle different dependency formats (list or dict)
                if (
                    isinstance(model.depends_on, list)
                    and model_name in model.depends_on
                ):
                    children.add(name)
                elif (
                    isinstance(model.depends_on, dict) and "models" in model.depends_on
                ):
                    if model_name in model.depends_on["models"]:
                        children.add(name)
        return children

    def _get_parents(self, model_name: str) -> Set[str]:
        """Get all upstream dependencies (parents) of a model."""
        parents = set()
        model = self.models_dict.get(model_name)

        if not model or not hasattr(model, "depends_on") or not model.depends_on:
            return parents

        # Handle different dependency formats
        if isinstance(model.depends_on, list):
            direct_parents = set(model.depends_on)
        elif isinstance(model.depends_on, dict) and "models" in model.depends_on:
            direct_parents = set(model.depends_on["models"])
        else:
            direct_parents = set()

        # Add direct parents
        parents.update(direct_parents)

        # Recursively add their parents
        for parent in direct_parents:
            if (
                parent in self.models_dict
            ):  # Only process if parent exists in our models
                parents.update(self._get_parents(parent))

        return parents
