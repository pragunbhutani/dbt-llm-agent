"""Simplified model interpreter workflow for dbt model interpretation."""

import logging
import json
from typing import Dict, Any, Optional, List

# Import models and services
from apps.knowledge_base.models import Model
from apps.llm_providers.services import ChatService
from apps.accounts.models import OrganisationSettings

# Import Langchain message classes
from langchain_core.messages import HumanMessage

# Import local components
from .prompts import create_interpretation_prompt
from . import ModelDocumentation

logger = logging.getLogger(__name__)


class ModelInterpreterWorkflow:
    """Simplified workflow for interpreting dbt models using a single comprehensive prompt."""

    def __init__(self, org_settings: OrganisationSettings, verbosity: int = 0):
        """Initialize the workflow with organization settings and verbosity level."""
        self.org_settings = org_settings
        self.verbosity = verbosity

        # Initialize Chat Service
        self.chat_service = ChatService(org_settings=org_settings)
        self.llm_client = self.chat_service.get_client()

        if not self.llm_client:
            raise ValueError("Failed to initialize LLM client for model interpretation")

    def _get_upstream_models_data(self, model: Model) -> List[Dict[str, Any]]:
        """Get detailed data for all upstream models."""
        if not model.all_upstream_models:
            return []

        upstream_models = Model.objects.filter(
            name__in=model.all_upstream_models, organisation=model.organisation
        ).values(
            "name",
            "raw_sql",
            "yml_description",
            "interpreted_description",
            "yml_columns",
            "interpreted_columns",
            "path",
            "materialization",
        )

        return list(upstream_models)

    def _parse_llm_response(
        self, response_content: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Parse the LLM response and extract JSON documentation."""
        try:
            # Try to extract JSON from response if it's wrapped in markdown
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
            elif response_content.strip().startswith("{"):
                json_content = response_content.strip()
            else:
                logger.error(f"Could not find valid JSON in response for {model_name}")
                return None

            return json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for {model_name}: {e}")
            logger.error(f"Response content: {response_content}")
            return None

    def interpret_model(self, model: Model) -> bool:
        """
        Run interpretation workflow for a single model.

        Args:
            model: The Model instance to interpret

        Returns:
            True if interpretation was successful and saved, False otherwise
        """
        logger.info(
            f"Starting simplified interpretation workflow for model: {model.name}"
        )

        if not model.raw_sql:
            logger.warning(
                f"Skipping interpretation for {model.name}: Raw SQL is missing."
            )
            return False

        try:
            # Get upstream models data
            upstream_models = self._get_upstream_models_data(model)
            if self.verbosity >= 1:
                logger.info(
                    f"Found {len(upstream_models)} upstream models for {model.name}"
                )

            # Create comprehensive prompt
            prompt = create_interpretation_prompt(model, upstream_models)

            if self.verbosity >= 2:
                logger.debug(
                    f"Generated prompt for {model.name} (length: {len(prompt)})"
                )

            # Get interpretation from LLM using proper Langchain message format
            messages = [HumanMessage(content=prompt)]
            response = self.llm_client.invoke(messages)
            response_content = (
                response.content if hasattr(response, "content") else str(response)
            )

            if self.verbosity >= 2:
                logger.debug(
                    f"LLM response for {model.name}: {response_content[:200]}..."
                )

            # Parse JSON response
            documentation_dict = self._parse_llm_response(response_content, model.name)
            if not documentation_dict:
                return False

            # Validate using Pydantic model
            try:
                validated_doc = ModelDocumentation(**documentation_dict)
            except Exception as pydantic_err:
                logger.error(
                    f"LLM returned invalid documentation structure for {model.name}: {pydantic_err}. Data: {documentation_dict}",
                    exc_info=True,
                )
                return False

            # Save results
            try:
                from django.db import transaction

                with transaction.atomic():
                    model.interpreted_description = validated_doc.description
                    model.interpreted_columns = {
                        col.name: col.description for col in validated_doc.columns
                    }
                    model.save(
                        update_fields=["interpreted_description", "interpreted_columns"]
                    )
                logger.info(f"Successfully saved interpretation for model {model.name}")
                return True
            except Exception as save_err:
                logger.error(
                    f"Error saving interpretation for model {model.name}: {save_err}",
                    exc_info=True,
                )
                return False

        except Exception as workflow_err:
            logger.error(
                f"Unexpected error during interpretation workflow for {model.name}: {workflow_err}",
                exc_info=True,
            )
            return False
