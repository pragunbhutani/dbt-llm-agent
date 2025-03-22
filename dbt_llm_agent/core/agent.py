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

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about the dbt project.

        Args:
            question: The question to answer

        Returns:
            Dict containing the answer and relevant model information
        """
        try:
            logger.info(f"Answering question: {question}")

            # Find relevant models using vector search
            search_results = self.vector_store.search_models(
                query=question, n_results=3
            )
            relevant_models = [result["id"] for result in search_results]
            logger.info(f"Found {len(relevant_models)} relevant models")

            # Get full model information from PostgreSQL
            model_info = ""
            models_data = []

            for model_id in relevant_models:
                model = self.postgres.get_model(model_id)
                if model:
                    model_info += model.get_readable_representation() + "\n\n"
                    models_data.append(model.to_dict())

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

            # Update model description
            model.description = documentation.get(
                "model_description", model.description
            )

            # Update column descriptions
            for col_name, col_desc in documentation.get(
                "column_descriptions", {}
            ).items():
                if col_name in model.columns:
                    model.columns[col_name].description = col_desc

            # Format documentation as markdown and store it in the documentation field
            formatted_doc = f"## {model_name}\n\n{model.description}\n\n"

            if model.columns:
                formatted_doc += "### Columns\n\n"
                for col_name, col in model.columns.items():
                    if col.description:
                        formatted_doc += f"- **{col_name}**: {col.description}\n"

            model.documentation = formatted_doc

            # Save the updated model
            self.postgres.update_model(model)

            # Update the vector store with the updated model representation
            self.vector_store.store_model(
                model_name=model_name, model_text=model.get_readable_representation()
            )

            return True

        except Exception as e:
            logger.error(f"Error updating model documentation: {e}")
            return False
