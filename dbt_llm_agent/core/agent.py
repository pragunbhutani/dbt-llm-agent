"""Agent for answering questions about dbt models."""

import logging
import re
import json
import yaml
from typing import Dict, List, Any, Optional, Union

from dbt_llm_agent.integrations.llm.client import LLMClient
from dbt_llm_agent.integrations.llm.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    MODEL_INTERPRETATION_PROMPT,
)
from dbt_llm_agent.storage.model_storage import ModelStorage
from dbt_llm_agent.storage.model_embedding_storage import ModelEmbeddingStorage
from dbt_llm_agent.core.models import DBTModel

logger = logging.getLogger(__name__)


class Agent:
    """Agent for interacting with dbt projects."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_storage: ModelStorage,
        vector_store: ModelEmbeddingStorage,
    ):
        """Initialize the agent.

        Args:
            llm_client: LLM client for generating text
            model_storage: Storage for dbt models
            vector_store: Vector store for semantic search
        """
        self.llm = llm_client
        self.model_storage = model_storage
        self.vector_store = vector_store

    def answer_question(
        self, question: str, use_interpretation: bool = False
    ) -> Dict[str, Any]:
        """Answer a question about the dbt project.

        Args:
            question: The question to answer
            use_interpretation: Whether to include model interpretations in the response

        Returns:
            Dict containing the answer and relevant model information
        """
        try:
            logger.info(f"Answering question: {question}")

            # Find relevant models using vector search
            search_results = self.vector_store.search_models(
                query=question, n_results=3, use_interpretation=use_interpretation
            )

            if not search_results:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant models to answer your question. Please try rephrasing or ask about specific models.",
                    "relevant_models": [],
                }

            logger.info(f"Found {len(search_results)} relevant models")

            # Get full model information from storage
            model_info = ""
            models_data = []

            for result in search_results:
                model_name = result["model_name"]
                model = self.model_storage.get_model(model_name)
                if model:
                    model_info += model.get_readable_representation() + "\n\n"
                    model_dict = model.to_dict()
                    # Add embedding search metadata
                    model_dict["search_score"] = result.get("similarity_score", 0)
                    models_data.append(model_dict)

            # Get the answer from LLM
            prompt = ANSWER_PROMPT_TEMPLATE.format(
                model_info=model_info, question=question
            )

            answer = self.llm.get_completion(
                prompt=question,
                system_prompt=prompt,
                max_tokens=1000,
            )

            # Return the answer and relevant models
            return {
                "question": question,
                "answer": answer,
                "relevant_models": models_data,
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "relevant_models": [],
            }

    def generate_documentation(self, model_name: str) -> Dict[str, Any]:
        """Generate documentation for a model.

        Args:
            model_name: Name of the model

        Returns:
            Dict containing the generated documentation
        """
        try:
            logger.info(f"Generating documentation for model: {model_name}")

            # Get model information
            model = self.model_storage.get_model(model_name)
            if not model:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found",
                }

            # Create custom prompt for model documentation
            prompt = f"""
            You are an AI assistant specialized in generating documentation for dbt models.
            You will be provided with information about a dbt model, and your task is to generate comprehensive documentation for it.
            
            Here is the information about the model:
            
            Model Name: {model_name}
            
            SQL Code:
            ```sql
            {model.raw_sql}
            ```
            
            Your documentation should include:
            1. A clear and concise description of what the model represents
            2. Descriptions for each column
            3. Information about any important business logic or transformations
            
            For each column, include:
            - What the column represents
            - The data type (if available)
            - Any important business rules or transformations
            
            Please format your response as follows:
            
            # Model Description
            [Your model description here]
            
            # Column Descriptions
            
            ## [Column Name 1]
            [Column 1 description]
            
            ## [Column Name 2]
            [Column 2 description]
            
            ...and so on for each column.
            """

            # Get the documentation from LLM
            documentation = self.llm.get_completion(
                prompt=f"Generate documentation for the model {model_name}",
                system_prompt=prompt,
                max_tokens=1500,
            )

            # Parse the documentation to extract model description and column descriptions
            model_description = ""
            column_descriptions = {}

            # Simple parsing - could be improved
            sections = documentation.split("# ")
            for section in sections:
                if section.startswith("Model Description"):
                    model_description = section.replace("Model Description", "").strip()
                elif section.startswith("Column Descriptions"):
                    column_section = section.replace("Column Descriptions", "").strip()
                    column_subsections = column_section.split("## ")
                    for col_subsection in column_subsections:
                        if col_subsection.strip():
                            col_lines = col_subsection.strip().split("\n", 1)
                            if len(col_lines) >= 2:
                                col_name = col_lines[0].strip()
                                col_desc = col_lines[1].strip()
                                column_descriptions[col_name] = col_desc

            return {
                "model_name": model_name,
                "description": model_description,
                "columns": column_descriptions,
                "raw_documentation": documentation,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error generating documentation for model {model_name}: {e}")
            return {
                "model_name": model_name,
                "error": f"Error generating documentation: {str(e)}",
                "success": False,
            }

    def update_model_documentation(
        self, model_name: str, documentation: Dict[str, Any]
    ) -> bool:
        """Update model documentation in the database.

        Args:
            model_name: Name of the model
            documentation: Documentation to update

        Returns:
            Whether the update was successful
        """
        try:
            # Get the existing model
            model = self.model_storage.get_model(model_name)
            if not model:
                logger.error(f"Model {model_name} not found")
                return False

            # Update the model with new documentation
            model.description = documentation.get("description", model.description)

            # Update columns
            if "columns" in documentation:
                for col_name, col_desc in documentation["columns"].items():
                    if col_name in model.columns:
                        model.columns[col_name].description = col_desc
                    else:
                        logger.warning(
                            f"Column {col_name} not found in model {model_name}"
                        )

            # Save the model
            success = self.model_storage.update_model(model)
            return success

        except Exception as e:
            logger.error(f"Error updating model documentation: {e}")
            return False

    def interpret_model(self, model_name: str) -> Dict[str, Any]:
        """Interpret a model and its columns by analyzing the model and its upstream models.

        This method creates a model documentation yaml format based on the model and its
        upstream dependencies, using an LLM to infer the meaning and structure.

        Args:
            model_name: Name of the model to interpret

        Returns:
            Dict containing the interpreted documentation in YAML format and other metadata
        """
        try:
            logger.info(f"Interpreting model: {model_name}")

            # Get model information
            model = self.model_storage.get_model(model_name)
            if not model:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found",
                }

            # Get upstream models
            upstream_models = []
            for upstream_name in model.all_upstream_models:
                upstream_model = self.model_storage.get_model(upstream_name)
                if upstream_model:
                    upstream_models.append(upstream_model)

            # Create information about upstream models
            upstream_info = ""
            for um in upstream_models:
                upstream_info += f"\nUpstream Model Name: {um.name}\n"
                upstream_info += f"Upstream Model Description: {um.description or 'No description'}\n"
                upstream_info += f"Upstream Model SQL:\n```sql\n{um.raw_sql}\n```\n"
                if um.columns:
                    upstream_info += "Upstream Model Columns:\n"
                    for col_name, col in um.columns.items():
                        upstream_info += (
                            f"  - {col_name}: {col.description or 'No description'}\n"
                        )

            # Create the prompt for model interpretation using our template
            prompt = MODEL_INTERPRETATION_PROMPT.format(
                model_name=model.name,
                model_sql=model.raw_sql,
                upstream_info=upstream_info,
            )

            # Log the prompt in verbose mode
            logger.debug(f"Interpretation prompt for model {model_name}:\n{prompt}")

            # Get the documentation from LLM
            yaml_documentation = self.llm.get_completion(
                prompt=f"Interpret and generate YAML documentation for the model {model_name} based on its SQL and upstream dependencies",
                system_prompt=prompt,
                max_tokens=2000,
            )

            # Log the raw response in verbose mode
            logger.debug(
                f"Raw LLM response for model {model_name}:\n{yaml_documentation}"
            )

            # Extract just the YAML content if wrapped in ```yaml ``` blocks
            if (
                "```yaml" in yaml_documentation
                and "```" in yaml_documentation.split("```yaml", 1)[1]
            ):
                yaml_documentation = (
                    yaml_documentation.split("```yaml", 1)[1].split("```", 1)[0].strip()
                )
            elif (
                "```" in yaml_documentation
                and "```" in yaml_documentation.split("```", 1)[1]
            ):
                yaml_documentation = (
                    yaml_documentation.split("```", 1)[1].split("```", 1)[0].strip()
                )

            return {
                "model_name": model_name,
                "yaml_documentation": yaml_documentation,
                "prompt": prompt,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error interpreting model {model_name}: {e}")
            return {
                "model_name": model_name,
                "error": f"Error interpreting model: {str(e)}",
                "success": False,
            }

    def save_interpreted_documentation(
        self, model_name: str, yaml_documentation: str, embed: bool = False
    ) -> Dict[str, Any]:
        """Save interpreted documentation for a model.

        Args:
            model_name: Name of the model
            yaml_documentation: YAML documentation as a string
            embed: Whether to embed the model in the vector store

        Returns:
            Dict containing the result of the operation
        """
        try:
            logger.info(f"Saving interpreted documentation for model {model_name}")

            # Parse the YAML documentation
            try:
                parsed_yaml = yaml.safe_load(yaml_documentation)
            except Exception as e:
                logger.error(f"Error parsing YAML: {e}")
                return {
                    "model_name": model_name,
                    "error": f"Invalid YAML format: {str(e)}",
                    "success": False,
                }

            # Extract model description and column descriptions
            model_description = ""
            column_descriptions = {}

            if (
                parsed_yaml
                and "models" in parsed_yaml
                and isinstance(parsed_yaml["models"], list)
                and len(parsed_yaml["models"]) > 0
            ):
                model_data = parsed_yaml["models"][0]
                if "description" in model_data:
                    model_description = model_data["description"]

                # Extract column descriptions
                if "columns" in model_data and isinstance(model_data["columns"], list):
                    for column in model_data["columns"]:
                        if "name" in column and "description" in column:
                            column_descriptions[column["name"]] = column["description"]
            else:
                logger.warning(
                    f"Missing or invalid YAML structure for model {model_name}"
                )

            # Get the existing model to validate
            existing_model = self.model_storage.get_model(model_name)
            if existing_model:
                logger.debug(
                    f"Existing model columns: {list(existing_model.columns.keys())}"
                )
                logger.debug(f"Tests in existing model: {existing_model.tests}")

                # Update the model with interpreted documentation
                existing_model.interpreted_description = model_description
                existing_model.interpreted_columns = column_descriptions

                # Save the model
                success = self.model_storage.update_model(existing_model)

                if success:
                    # Only update the vector store if embed is True
                    if embed:
                        logger.info(f"Embedding model {model_name} in vector store")
                        self.vector_store.store_model(
                            model_name=model_name,
                            model_text=existing_model.get_readable_representation(),
                        )
                    else:
                        logger.debug(f"Skipping embedding for model {model_name}")

                return {
                    "model_name": model_name,
                    "success": success,
                    "message": (
                        "Interpretation saved successfully"
                        if success
                        else "Failed to save interpretation"
                    ),
                }
            else:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found in the database",
                    "success": False,
                }

        except Exception as e:
            logger.error(f"Error saving interpreted documentation: {e}")
            return {
                "model_name": model_name,
                "error": f"Error saving interpreted documentation: {str(e)}",
                "success": False,
            }
