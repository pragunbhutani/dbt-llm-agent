"""Agent for answering questions and generating documentation about dbt projects."""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple

from openai import OpenAI
from langchain.prompts import PromptTemplate

from dbt_llm_agent.core.models import DBTModel, DBTProject, ModelMetadata
from dbt_llm_agent.storage.postgres_storage import PostgresStorage
from dbt_llm_agent.storage.vector_store import PostgresVectorStore

logger = logging.getLogger(__name__)


class DBTAgent:
    """Agent for interacting with dbt projects."""

    def __init__(
        self,
        postgres_storage: PostgresStorage,
        vector_store: PostgresVectorStore,
        openai_api_key: str,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.0,
    ):
        """Initialize the agent.

        Args:
            postgres_storage: PostgreSQL storage for models
            vector_store: Vector store for semantic search
            openai_api_key: OpenAI API key
            model_name: Name of the OpenAI model to use
            temperature: Temperature for the LLM
        """
        self.postgres = postgres_storage
        self.vector_store = vector_store
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.temperature = temperature

        # Define prompt templates
        self.question_prompt_template = """
        You are an AI assistant specialized in answering questions about dbt projects.
        You are given information about relevant dbt models that might help answer the user's question.
        
        Here is the information about the relevant dbt models:
        
        {model_info}
        
        Based on this information, please answer the following question:
        {question}
        
        Provide a clear, concise, and accurate answer. If the information provided is not sufficient to answer the question, explain what additional information would be needed.
        """

        self.documentation_prompt_template = """
        You are an AI assistant specialized in generating documentation for dbt models.
        You will be provided with information about a dbt model, and your task is to generate comprehensive documentation for it.
        
        Here is the information about the model:
        
        Model Name: {model_name}
        
        SQL Code:
        ```sql
        {raw_sql}
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

    def answer_question(
        self, question: str, use_interpretation: bool = False
    ) -> Dict[str, Any]:
        """Answer a question about the dbt project.

        Args:
            question: The question to answer
            use_interpretation: Ignored, kept for backward compatibility

        Returns:
            Dict containing the answer and relevant model information
        """
        try:
            logger.info(f"Answering question: {question}")

            # Find relevant models using vector search
            search_results = self.vector_store.search_models(
                query=question, n_results=3
            )

            if not search_results:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant models to answer your question. Please try rephrasing or ask about specific models.",
                    "relevant_models": [],
                }

            logger.info(f"Found {len(search_results)} relevant models")

            # Get full model information from PostgreSQL
            model_info = ""
            models_data = []

            for result in search_results:
                model_name = result["model_name"]
                model = self.postgres.get_model(model_name)
                if model:
                    model_info += model.get_readable_representation() + "\n\n"
                    model_dict = model.to_dict()
                    # Add embedding search metadata
                    model_dict["search_score"] = result.get("score", 0)
                    models_data.append(model_dict)

            # Create the prompt
            prompt = self.question_prompt_template.format(
                model_info=model_info, question=question
            )

            # Get the answer from OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
                temperature=self.temperature,
                max_tokens=1000,
            )

            answer = response.choices[0].message.content

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
            model = self.postgres.get_model(model_name)
            if not model:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found",
                }

            # Create the prompt
            prompt = self.documentation_prompt_template.format(
                model_name=model_name, raw_sql=model.raw_sql
            )

            # Get the documentation from OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"Generate documentation for the model {model_name}",
                    },
                ],
                temperature=self.temperature,
                max_tokens=1500,
            )

            documentation = response.choices[0].message.content

            # Parse the documentation to extract model description and column descriptions
            model_description = ""
            column_descriptions = {}

            # Very basic parsing - could be improved
            sections = documentation.split("# ")
            for section in sections:
                if section.startswith("Model Description"):
                    model_description = section.replace("Model Description", "").strip()
                elif section.startswith("Column Descriptions"):
                    columns_text = section.replace("Column Descriptions", "").strip()
                    column_sections = columns_text.split("## ")
                    for col_section in column_sections:
                        if col_section.strip():
                            lines = col_section.strip().split("\n")
                            if lines:
                                col_name = lines[0].strip()
                                col_desc = "\n".join(lines[1:]).strip()
                                column_descriptions[col_name] = col_desc

            return {
                "model_name": model_name,
                "model_description": model_description,
                "column_descriptions": column_descriptions,
                "full_documentation": documentation,
            }

        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return {
                "model_name": model_name,
                "error": f"Error generating documentation: {str(e)}",
            }

    def update_model_documentation(
        self, model_name: str, documentation: Dict[str, Any]
    ) -> bool:
        """Update the documentation for a model in the database.

        Args:
            model_name: Name of the model
            documentation: Documentation to update

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the model
            model = self.postgres.get_model(model_name)
            if not model:
                logger.error(f"Model {model_name} not found")
                return False

            # Update the model with the new documentation
            model.description = documentation.get(
                "model_description", model.description
            )

            # Update column descriptions
            column_descriptions = documentation.get("column_descriptions", {})
            for col_name, col_desc in column_descriptions.items():
                if col_name in model.columns:
                    model.columns[col_name].description = col_desc

            # Format documentation as markdown and store it in the documentation field
            formatted_doc = f"## {model_name}\n\n{model.description}\n\n"

            if model.columns:
                formatted_doc += "### Columns\n\n"
                for col_name, col in model.columns.items():
                    if col.description:
                        formatted_doc += f"- {col_name}: {col.description}\n"

            model.documentation = formatted_doc

            # Save the model
            success = self.postgres.update_model(model)

            if success:
                # Update the vector store with the updated model representation
                self.vector_store.store_model(
                    model_name=model_name,
                    model_text=model.get_readable_representation(),
                )

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
            model = self.postgres.get_model(model_name)
            if not model:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found",
                }

            # Get upstream models
            upstream_models = []
            for upstream_name in model.all_upstream_models:
                upstream_model = self.postgres.get_model(upstream_name)
                if upstream_model:
                    upstream_models.append(upstream_model)

            # Create a prompt that includes information about the model and its upstream models
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

            # Create the prompt for model interpretation
            prompt = f"""
            You are an AI assistant specialized in interpreting dbt models.
            You will be provided with information about a dbt model and its upstream dependencies.
            Your task is to analyze the SQL code and dependencies to generate comprehensive documentation for the model.
            
            Here is the information about the model to interpret:
            
            Model Name: {model.name}
            
            SQL Code:
            ```sql
            {model.raw_sql}
            ```
            
            Information about upstream models this model depends on:
            {upstream_info}
            
            Based on the SQL code and upstream models, please:
            1. Interpret what this model represents in the business context
            2. Identify and describe each column in the model
            3. Identify any important business logic or transformations
            
            Please format your response as a valid dbt YAML documentation in this format:
            
            ```yaml
            version: 2
            
            models:
              - name: {model.name}
                description: "[Your comprehensive model description]"
                columns:
                  - name: [column_name_1]
                    description: "[Column 1 description]"
                  
                  - name: [column_name_2]
                    description: "[Column 2 description]"
                  
                  ...and so on for each column.
            ```
            
            Make sure to:
            - Provide clear, concise, and accurate descriptions
            - Include all columns that appear in the SQL query's SELECT statement
            - Format the YAML correctly with proper indentation
            - Add business context where possible
            """

            # Log the prompt in verbose mode
            logger.debug(f"Interpretation prompt for model {model_name}:\n{prompt}")

            # Get the documentation from OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"Interpret and generate YAML documentation for the model {model_name} based on its SQL and upstream dependencies",
                    },
                ],
                temperature=self.temperature,
                max_tokens=2000,
            )

            yaml_documentation = response.choices[0].message.content

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
            logger.error(f"Error interpreting model: {e}")
            return {
                "model_name": model_name,
                "error": f"Error interpreting model: {str(e)}",
                "success": False,
            }

    def save_interpreted_documentation(
        self, model_name: str, yaml_documentation: str, embed: bool = False
    ) -> Dict[str, Any]:
        """Save the interpreted documentation to the database.

        This method parses the YAML documentation and updates the model in the database.

        Args:
            model_name: Name of the model
            yaml_documentation: YAML formatted documentation string
            embed: Whether to embed the model in the vector store

        Returns:
            Dict with status information
        """
        try:
            import yaml

            logger.info(f"Saving interpreted documentation for model: {model_name}")

            # Debug the input
            logger.debug(f"YAML documentation to save:\n{yaml_documentation}")

            # Parse the YAML documentation
            parsed_yaml = yaml.safe_load(yaml_documentation)

            # Debug the parsed YAML
            logger.debug(f"Parsed YAML: {parsed_yaml}")

            if not parsed_yaml or "models" not in parsed_yaml:
                return {
                    "model_name": model_name,
                    "error": "Invalid YAML format: 'models' key not found",
                    "success": False,
                }

            # Find the model in the YAML
            model_doc = None
            for model in parsed_yaml["models"]:
                if model.get("name") == model_name:
                    model_doc = model
                    break

            if not model_doc:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found in the YAML documentation",
                    "success": False,
                }

            # Debug the model document
            logger.debug(f"Found model in YAML: {model_doc}")

            # Extract model description and column descriptions
            model_description = model_doc.get("description", "")
            column_descriptions = {}

            if "columns" in model_doc:
                for column in model_doc["columns"]:
                    column_name = column.get("name")
                    column_description = column.get("description", "")
                    if column_name:
                        column_descriptions[column_name] = column_description

            # Debug the extracted information
            logger.debug(f"Extracted model description: {model_description}")
            logger.debug(f"Extracted column descriptions: {column_descriptions}")

            # Get the existing model to validate
            existing_model = self.postgres.get_model(model_name)
            if existing_model:
                logger.debug(
                    f"Existing model columns: {list(existing_model.columns.keys())}"
                )
                logger.debug(f"Tests in existing model: {existing_model.tests}")

                # Update the model with interpreted documentation
                existing_model.interpreted_description = model_description
                existing_model.interpreted_columns = column_descriptions

                # Save the model
                success = self.postgres.update_model(existing_model)

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
