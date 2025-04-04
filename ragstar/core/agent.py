"""Agent for answering questions about dbt models."""

import logging
import re
import json
import yaml
import sys
import os  # Import os
import openai  # Import openai
from typing import Dict, List, Any, Optional, Union, Set

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from ragstar.storage.question_storage import QuestionStorage

from ragstar.integrations.llm.client import LLMClient
from ragstar.integrations.llm.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    SEARCH_QUERY_REFINEMENT_PROMPT_TEMPLATE,
    MODEL_INTERPRETATION_PROMPT,
    PLANNING_PROMPT_TEMPLATE,
    ANALYSIS_PROMPT_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE,
    FEEDBACK_REFINEMENT_PROMPT_TEMPLATE,
)
from ragstar.storage.model_storage import ModelStorage
from ragstar.storage.model_embedding_storage import ModelEmbeddingStorage
from ragstar.core.models import DBTModel, ModelTable

logger = logging.getLogger(__name__)


class Agent:
    """Agent for interacting with dbt projects using an agentic workflow."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_storage: ModelStorage,
        vector_store: ModelEmbeddingStorage,
        question_storage: Optional[QuestionStorage] = None,
        console: Optional[Console] = None,
        temperature: float = 0.0,
        verbose: bool = False,
        openai_api_key: Optional[str] = None,  # Add API key parameter
    ):
        """Initialize the agent.

        Args:
            llm_client: LLM client for generating text
            model_storage: Storage for dbt models
            vector_store: Vector store for semantic search
            question_storage: Storage for question history (requires openai_api_key for embedding)
            console: Console for interactive prompts
            temperature: Temperature for LLM generation
            verbose: Whether to print verbose output
            openai_api_key: OpenAI API key for question embeddings.
        """
        self.llm = llm_client
        self.model_storage = model_storage
        self.vector_store = vector_store
        self.question_storage = question_storage
        self.console = console or Console()
        self.temperature = temperature
        self.verbose = verbose
        self.max_iterations = 3

    def _refine_search_query(self, original_query: str) -> str:
        """Refines the user's query for better vector search results."""
        self.console.print(
            f"[dim] Refining search query for: '{original_query}'...[/dim]"
        )
        refined_query = self.llm.get_completion(
            prompt=original_query,
            system_prompt=SEARCH_QUERY_REFINEMENT_PROMPT_TEMPLATE,
            temperature=0.0,
        )

        # Basic validation/cleanup
        refined_query = refined_query.strip().replace('"', "").replace("\n", " ")
        if (
            not refined_query or len(refined_query) < 5
        ):  # Guard against empty or too short results
            self.console.print(
                "[yellow]Query refinement failed, using original query.[/yellow]"
            )
            return original_query

        if self.verbose:
            self.console.print(f"[dim] Refined query: '{refined_query}'[/dim]")
        return refined_query

    def _search_and_fetch_models(
        self, query: str, current_context_models: List[str]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Searches for relevant models and fetches their details."""
        # Refine the query before searching
        refined_search_query = self._refine_search_query(query)

        self.console.print(
            f"[bold blue]🔍 Searching models relevant to: '{refined_search_query}'[/bold blue]"  # Use refined query
        )
        search_results = self.vector_store.search_models(
            query=refined_search_query, n_results=5
        )  # Use refined query

        model_info_str = ""
        models_data = []
        added_models = set(current_context_models)

        if not search_results:
            self.console.print(
                "[yellow]No additional relevant models found in this search.[/yellow]"
            )
            return "", []

        if self.verbose:
            self.console.print("[bold]Search Results:[/bold]")
            for res in search_results:
                score = res.get("similarity_score", "N/A")
                score_str = (
                    f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
                )
                self.console.print(f"- {res['model_name']} (Score: {score_str})")

        for result in search_results:
            model_name = result["model_name"]
            similarity = result.get("similarity_score", 0)

            if (
                model_name not in added_models
                and isinstance(similarity, (int, float))
                and similarity > 0.3  # Lowered threshold from 0.4 to 0.3
            ):
                model = self.model_storage.get_model(model_name)
                if model:
                    if self.verbose:
                        self.console.print(
                            f"[dim]  Fetching details for {model_name}...[/dim]"
                        )
                    model_info_str += model.get_text_representation() + "\n\n"
                    model_dict = model.to_dict()
                    model_dict["search_score"] = similarity
                    models_data.append(model_dict)
                    added_models.add(model_name)
                elif self.verbose:
                    self.console.print(
                        f"[yellow]  Could not fetch details for {model_name}[/yellow]"
                    )

        if self.verbose and models_data:
            self.console.print(
                f"[green]✅ Added {len(models_data)} new models to context.[/green]"
            )
        elif self.verbose and search_results:
            self.console.print(
                "[yellow]No new models met the criteria to be added to context.[/yellow]"
            )

        return model_info_str, models_data

    def run_agentic_workflow(self, question: str) -> Dict[str, Any]:
        """Runs the agentic workflow to answer a question with iterative search and analysis."""
        max_steps = 5
        accumulated_model_info_str = ""
        found_models_details = []
        found_model_names = set()
        current_search_query = ""
        original_question = question
        conversation_id = None
        relevant_feedback = []  # Store feedback found
        previous_search_queries = set()  # Track previous search queries

        self.console.print(f"[bold]🚀 Starting agentic workflow for:[/bold] {question}")

        try:
            # --- Feedback Check Step ---
            question_embedding = None
            if (
                self.question_storage
                and hasattr(self.question_storage, "_get_embedding")
                and self.question_storage.openai_client
            ):
                if self.verbose:
                    self.console.print(
                        "\n[blue]🔍 Checking for feedback on similar past questions...[/blue]"
                    )
                question_embedding = self.question_storage._get_embedding(
                    original_question
                )
                if question_embedding:
                    relevant_feedback = (
                        self.question_storage.find_similar_questions_with_feedback(
                            query_embedding=question_embedding,
                            limit=3,  # Limit feedback retrieved
                            similarity_threshold=0.75,  # Adjust as needed
                        )
                    )
                    if relevant_feedback:
                        if self.verbose:
                            self.console.print(
                                f"[dim] Found {len(relevant_feedback)} potentially relevant feedback item(s):[/dim]"
                            )
                            # Print details of each feedback item
                            for i, feedback_item in enumerate(relevant_feedback):
                                self.console.print(
                                    f"[dim] --- Feedback Item {i+1} ---[/dim]"
                                )
                                self.console.print(
                                    f"[dim]   Original Question: {feedback_item.question_text}[/dim]"
                                )
                                self.console.print(
                                    f"[dim]   Original Answer: {feedback_item.answer_text or 'N/A'}[/dim]"
                                )
                                self.console.print(
                                    f"[dim]   Was Useful: {feedback_item.was_useful}[/dim]"
                                )
                                self.console.print(
                                    f"[dim]   Feedback Text: {feedback_item.feedback or 'N/A'}[/dim]"
                                )
                        else:
                            self.console.print(
                                "[blue]... Found relevant past feedback.[/blue]"
                            )
                    elif self.verbose:
                        self.console.print("[dim] No relevant feedback found.[/dim]")
                elif self.verbose:
                    self.console.print(
                        "[yellow]Could not generate embedding for feedback check.[/yellow]"
                    )
            # --- End Feedback Check ---

            # 1. Planning Step: Determine the first search query
            if self.verbose:
                self.console.print(
                    "\n[bold blue]🧠 Planning initial search...[/bold blue]"
                )
            else:
                self.console.print("[blue]🧠 Planning initial search...[/blue]")
            planning_prompt = PLANNING_PROMPT_TEMPLATE.format(
                question=original_question
            )
            current_search_query = self.llm.get_completion(
                prompt=planning_prompt,
                system_prompt="You are an AI assistant specialized in dbt model analysis.",
                temperature=0.0,
                max_tokens=100,
            ).strip()

            if not current_search_query:
                self.console.print(
                    "[yellow]Could not determine an initial search query. Refining original query as fallback.[/yellow]"
                )
                # Fallback: Use the refinement logic on the original question
                current_search_query = self._refine_search_query(original_question)

            if self.verbose:
                self.console.print(
                    f"[dim] Initial search query: '{current_search_query}'[/dim]"
                )

            # 2. Iterative Search & Analysis Loop
            for step in range(max_steps):
                if self.verbose:
                    self.console.print(
                        f"\n[bold blue]🔄 Step {step + 1}/{max_steps}: Searching & Analyzing[/bold blue]"
                    )
                    self.console.print(
                        f"[dim] Current search query: '{current_search_query}'[/dim]"
                    )
                else:
                    self.console.print(
                        f"[blue]🔄 Step {step + 1}: Searching & Analyzing...[/blue]"
                    )

                # Search and fetch models based on current query
                newly_found_model_info_str, newly_found_models_details = (
                    self._search_and_fetch_models(
                        current_search_query, list(found_model_names)
                    )
                )

                # Build summary of models found *so far* for analysis prompt
                already_found_summary = ""
                if found_models_details:
                    already_found_summary += "Models found in previous steps:\n"
                    for model_detail in found_models_details:
                        columns = model_detail.get("columns", {})
                        column_names = list(columns.keys())
                        already_found_summary += f" - {model_detail['name']}: {model_detail.get('description', 'N/A')[:100]}... (Cols: {len(column_names)})\n"
                else:
                    already_found_summary = "No models found yet.\n"

                # Build details of *newly* found models for analysis prompt
                newly_found_model_details_for_prompt = ""
                current_step_model_names = []
                added_count = 0
                if newly_found_models_details:
                    newly_found_model_details_for_prompt = (
                        "Models found in the latest search:\n"
                    )
                    for model_data in newly_found_models_details:
                        model_name = model_data["name"]
                        if model_name not in found_model_names:
                            # Accumulate full details for synthesis
                            model_repr = (
                                self.model_storage.get_model(
                                    model_name
                                ).get_text_representation()
                                if self.model_storage.get_model(model_name)
                                else f"Details for {model_name} not fully available."
                            )
                            accumulated_model_info_str += model_repr + "\n\n"

                            # Create concise summary for analysis prompt
                            columns = model_data.get("columns", {})
                            column_names = list(columns.keys())
                            model_summary = f"Model: {model_name}\nDescription: {model_data.get('description', 'N/A')}\nColumns: {', '.join(column_names)}\n---"
                            newly_found_model_details_for_prompt += model_summary + "\n"

                            found_models_details.append(model_data)
                            found_model_names.add(model_name)
                            current_step_model_names.append(model_name)
                            added_count += 1
                else:
                    newly_found_model_details_for_prompt = (
                        "No new models found in the latest search.\n"
                    )

                if self.verbose and added_count > 0:
                    self.console.print(
                        f"[dim] -> Added {added_count} new models: {', '.join(current_step_model_names)}[/dim]"
                    )
                elif self.verbose and newly_found_models_details:
                    self.console.print(
                        "[dim] -> No *new* unique models added from this search.[/dim]"
                    )

                # Analysis Step
                if self.verbose:
                    self.console.print("[bold blue]🧠 Analyzing results...[/bold blue]")
                else:
                    self.console.print("[blue]🧠 Analyzing results...[/blue]")

                analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                    original_question=original_question,
                    already_found_models_summary=already_found_summary,
                    last_search_query=current_search_query,
                    newly_found_model_details_section=newly_found_model_details_for_prompt,
                )
                # Use the existing LLM client instance
                next_step_output = self.llm.get_completion(
                    prompt=analysis_prompt,
                    system_prompt="You are an AI assistant specialized in analyzing dbt models.",
                    temperature=0.0,  # Low temp for deterministic analysis
                    max_tokens=150,  # Allow slightly longer analysis/next query
                ).strip()

                if self.verbose:
                    self.console.print(
                        f"[dim] Analysis result: {next_step_output}[/dim]"
                    )

                # Check for termination condition
                if "ALL_INFO_FOUND" in next_step_output or not next_step_output:
                    if not found_models_details and step == 0:
                        self.console.print(
                            "[yellow]Initial search yielded no results and analysis could not determine next steps.[/yellow]"
                        )
                        final_answer = "I could not find relevant models based on the initial search. Please try rephrasing your question."
                        return {
                            "question": original_question,
                            "final_answer": final_answer,
                            "used_model_names": list(found_model_names),
                            "conversation_id": None,
                        }
                    else:
                        self.console.print(
                            "[green]✅ Analysis complete. Sufficient information gathered or loop finished.[/green]"
                        )
                        break
                else:
                    # Check if this search query was already used
                    if next_step_output in previous_search_queries:
                        if self.verbose:
                            self.console.print(
                                f"[yellow]⚠️ Skipping redundant search for query: '{next_step_output}'[/yellow]"
                            )
                        self.console.print(
                            "[green]✅ No new information to gather. Proceeding to synthesis.[/green]"
                        )
                        break
                    else:
                        # Add to previous queries and prepare for next iteration
                        previous_search_queries.add(next_step_output)
                        current_search_query = next_step_output

            # Check if loop finished without finding sufficient info
            if step == max_steps - 1 and "ALL_INFO_FOUND" not in next_step_output:
                self.console.print(
                    "[yellow]⚠️ Reached max search steps. Synthesizing answer with gathered info.[/yellow]"
                )
            # --- End Iterative Search & Analysis ---

            # 3. Synthesis Step
            if not found_models_details:
                if self.verbose:
                    self.console.print(
                        "[red]❌ No relevant models found after all steps.[/red]"
                    )
                final_answer = "I could not find any relevant dbt models to answer your question after completing the search process."
            else:
                if self.verbose:
                    self.console.print(
                        "\n[bold blue]✍️ Synthesizing the final answer...[/bold blue]"
                    )
                else:
                    self.console.print("[blue]✍️ Synthesizing final answer...[/blue]")

                synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                    question=original_question,
                    accumulated_model_info=accumulated_model_info_str,
                )
                final_answer = self.llm.get_completion(
                    prompt=synthesis_prompt,
                    system_prompt="You are an AI assistant specialized in dbt model analysis and explanation.",
                    temperature=self.temperature,
                    max_tokens=1500,
                )

            # --- Feedback Refinement Step (if feedback found) ---
            refined_answer = final_answer  # Start with the original answer
            if relevant_feedback:
                if self.verbose:
                    self.console.print(
                        "\n[bold blue]🔄 Refining answer based on past feedback...[/bold blue]"
                    )
                    # Print details (including answer) of each feedback item
                    for i, feedback_item in enumerate(relevant_feedback):
                        self.console.print(f"[dim] --- Feedback Item {i+1} ---[/dim]")
                        self.console.print(
                            f"[dim]   Original Question: {feedback_item.question_text}[/dim]"
                        )
                        self.console.print(
                            f"[dim]   Original Answer: {feedback_item.answer_text or 'N/A'}[/dim]"
                        )
                        self.console.print(
                            f"[dim]   Was Useful: {feedback_item.was_useful}[/dim]"
                        )
                        self.console.print(
                            f"[dim]   Feedback Text: {feedback_item.feedback or 'N/A'}[/dim]"
                        )
                else:
                    self.console.print(
                        "[blue]🔄 Refining answer based on past feedback...[/blue]"
                    )

                feedback_context = ""
                for item in relevant_feedback:
                    feedback_context += f"Past Question: {item.question_text}\n"
                    feedback_context += f"Past Answer: {item.answer_text}\n"
                    feedback_context += f"Feedback: Useful={item.was_useful}, Text={item.feedback or ''}\n---\n"

                refinement_prompt = f"""
                You are refining a dbt model interpretation based on verification feedback.
                
                Original model: {model_name}
                
                Original SQL:
                ```sql
                {model.raw_sql}
                ```
                
                Current interpretation:
                ```yaml
                {current_yaml_content}
                ```
                
                Verification feedback from iteration {iteration+1}:
                {verification_result}
                
                Upstream model information:
                {detailed_upstream_info}
                
                YOUR TASK IS TO CREATE A COMPLETE YAML INTERPRETATION THAT INCLUDES ALL COLUMNS FROM THE MODEL.
                
                This is absolutely critical:
                1. ADD ALL MISSING COLUMNS identified in the verification feedback - you must include EVERY column
                2. Even if there are 50+ columns to add, you must include all of them in your response
                3. The most common error is not including all columns from upstream models when SELECT * is used
                4. Be extremely thorough - lack of completeness is a critical issue
                5. Remove any incorrect columns identified in COLUMNS_TO_REMOVE
                6. Update descriptions for columns identified in COLUMNS_TO_MODIFY
                
                DO NOT OMIT ANY COLUMNS from your response - completeness is the highest priority.
                
                Your output should be complete, valid YAML for this model. Include all columns.
                """

                refined_answer = self.llm.get_completion(
                    prompt=refinement_prompt,
                    system_prompt="You are an AI assistant specialized in refining dbt model interpretations based on SQL analysis.",
                    temperature=self.temperature,  # Use original temperature
                    max_tokens=4000,  # Allow longer refined answer
                )

                if self.verbose:
                    self.console.print("[dim] -> Refined Answer:[/dim]")
                    self.console.print(f"[dim]{refined_answer}[/dim]")

            # Use the refined answer if it exists, otherwise use the original final answer
            final_answer_to_use = refined_answer
            # --- End Feedback Refinement Step ---

            # 4. Record the final question/answer pair
            if self.question_storage:
                try:
                    conversation_id = self.question_storage.record_question(
                        question_text=original_question,
                        answer_text=final_answer_to_use,
                        model_names=list(found_model_names),
                        # Initial recording has no feedback yet
                        was_useful=None,
                        feedback=None,
                        metadata={
                            "agent_steps": step + 1,
                            "initial_query": current_search_query,
                            # Add other relevant metadata
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to record question/answer: {e}")
                    self.console.print(
                        "[yellow]⚠️ Could not record question details to storage.[/yellow]"
                    )

            return {
                "question": original_question,
                "final_answer": final_answer_to_use,
                "used_model_names": list(found_model_names),
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.error(
                f"Error during agentic workflow: {str(e)}", exc_info=self.verbose
            )
            self.console.print(f"[bold red]Error during workflow:[/bold red] {str(e)}")
            return {
                "question": question,
                "final_answer": "An error occurred during the process. Please check logs or try again.",
                "used_model_names": list(found_model_names),
                "conversation_id": None,
            }

    def answer_question(
        self, question: str, use_interpretation: bool = False
    ) -> Dict[str, Any]:
        """Answer a question about the dbt project (Non-interactive version)."""
        try:
            logger.info(f"Answering question (simple mode): {question}")

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

            model_info = ""
            models_data = []

            for result in search_results:
                model_name = result["model_name"]
                model = self.model_storage.get_model(model_name)
                if model:
                    model_info += model.get_text_representation() + "\n\n"
                    model_dict = model.to_dict()
                    model_dict["search_score"] = result.get("similarity_score", 0)
                    models_data.append(model_dict)

            prompt = ANSWER_PROMPT_TEMPLATE.format(
                model_info=model_info, question=question
            )

            answer = self.llm.get_completion(
                prompt=question,
                system_prompt=prompt,
                temperature=self.temperature,
                max_tokens=1000,
            )

            return {
                "question": question,
                "answer": answer,
                "relevant_models": models_data,
            }

        except Exception as e:
            logger.error(f"Error answering question (simple mode): {e}")
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

            model = self.model_storage.get_model(model_name)
            if not model:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found",
                }

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

            documentation = self.llm.get_completion(
                prompt=f"Generate documentation for the model {model_name}",
                system_prompt=prompt,
                max_tokens=1500,
            )

            model_description = ""
            column_descriptions = {}

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
            model = self.model_storage.get_model(model_name)
            if not model:
                logger.error(f"Model {model_name} not found")
                return False

            model.description = documentation.get("description", model.description)

            if "columns" in documentation:
                for col_name, col_desc in documentation["columns"].items():
                    if col_name in model.columns:
                        model.columns[col_name].description = col_desc
                    else:
                        logger.warning(
                            f"Column {col_name} not found in model {model_name}"
                        )

            success = self.model_storage.update_model(model)
            return success

        except Exception as e:
            logger.error(f"Error updating model documentation: {e}")
            return False

    def interpret_model(
        self, model_name: str, max_verification_iterations: int = 1
    ) -> Dict[str, Any]:
        """Interpret a model and its columns using an agentic workflow.

        This method implements a step-by-step agentic approach to model interpretation:
        1. Read the source code of the model to interpret
        2. Identify upstream models that provide context
        3. Fetch details of the upstream models
        4. Create a draft interpretation
        5. Iteratively verify and refine the interpretation (configurable iterations):
           - Analyze upstream models' source code directly to ensure all columns are identified
           - Extract structured recommendations for columns to add, remove, or modify
           - Refine interpretation based on recommendations until verification passes or iterations complete
        6. Return final interpretation with column recommendations

        Args:
            model_name: Name of the model to interpret.
            max_verification_iterations: Maximum number of verification iterations to run (default: 1)

        Returns:
            Dict containing the interpreted documentation in YAML format and metadata,
            including structured column recommendations and verification iterations info.
        """
        try:
            logger.info(
                f"Starting agentic interpretation workflow for model: {model_name}"
            )

            # Step 1: Read the source code of the model to interpret
            model = self.model_storage.get_model(model_name)
            if not model:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found",
                    "success": False,
                }

            logger.info(f"Retrieved model {model_name} for interpretation")

            # Step 2: Identify upstream models from the model's source code
            logger.info(f"Analyzing upstream dependencies for {model_name}")

            # Get upstream models from the model's metadata
            upstream_model_names = model.all_upstream_models

            if not upstream_model_names:
                logger.warning(
                    f"No upstream models found for {model_name}. Using depends_on list."
                )
                upstream_model_names = model.depends_on

            logger.info(
                f"Found {len(upstream_model_names)} upstream models for {model_name}"
            )

            # Step 3: Fetch details of the upstream models for context
            upstream_models_data = {}
            upstream_info = ""
            for upstream_name in upstream_model_names:
                logger.info(f"Fetching details for upstream model: {upstream_name}")
                upstream_model = self.model_storage.get_model(upstream_name)
                if not upstream_model:
                    logger.warning(
                        f"Upstream model {upstream_name} not found. Skipping."
                    )
                    continue

                upstream_models_data[upstream_name] = upstream_model

                # Add upstream model information to context
                upstream_info += f"\n-- Upstream Model: {upstream_model.name} --\n"
                description = (
                    upstream_model.interpreted_description
                    or upstream_model.description
                    or "No description available."
                )
                upstream_info += f"Description: {description}\n"

                # Add column information from either interpreted columns or YML columns
                if upstream_model.interpreted_columns:
                    upstream_info += "Columns (from LLM interpretation):\n"
                    for (
                        col_name,
                        col_desc,
                    ) in upstream_model.interpreted_columns.items():
                        upstream_info += f"  - {col_name}: {col_desc}\n"
                elif upstream_model.columns:
                    upstream_info += "Columns (from YML):\n"
                    for col_name, col_obj in upstream_model.columns.items():
                        upstream_info += f"  - {col_name}: {col_obj.description or 'No description in YML'}\n"
                else:
                    upstream_info += "Columns: No column information available.\n"

                # Add SQL for context
                upstream_info += f"Raw SQL:\n```sql\n{upstream_model.raw_sql or 'SQL not available'}\n```\n"
                upstream_info += "-- End Upstream Model --\n"

            # Step 4: Create a draft interpretation using the model SQL and upstream info
            logger.info(f"Creating draft interpretation for {model_name}")

            prompt = MODEL_INTERPRETATION_PROMPT.format(
                model_name=model.name,
                model_sql=model.raw_sql,
                upstream_info=upstream_info,
            )

            draft_yaml_documentation = self.llm.get_completion(
                prompt=f"Interpret and generate YAML documentation for the model {model_name} based on its SQL and upstream dependencies",
                system_prompt=prompt,
                max_tokens=4000,
            )

            logger.debug(
                f"Draft interpretation for {model_name}:\n{draft_yaml_documentation}"
            )

            # Extract YAML content from the response
            match = re.search(
                r"```(?:yaml)?\n(.*?)```", draft_yaml_documentation, re.DOTALL
            )
            if match:
                draft_yaml_content = match.group(1).strip()
            else:
                # If no code block found, use the whole response but check for YAML tags
                draft_yaml_content = draft_yaml_documentation.strip()
                # Remove any potential YAML code fence markers at the beginning or end
                if draft_yaml_content.startswith("```yaml"):
                    draft_yaml_content = draft_yaml_content[7:]
                if draft_yaml_content.endswith("```"):
                    draft_yaml_content = draft_yaml_content[:-3]
                draft_yaml_content = draft_yaml_content.strip()

            logger.debug(
                f"Cleaned draft YAML content for {model_name}:\n{draft_yaml_content}"
            )

            # Parse the draft YAML to get column information
            try:
                # Additional safety check to ensure YAML content is clean
                if draft_yaml_content.startswith("```") or draft_yaml_content.endswith(
                    "```"
                ):
                    # Further cleaning if needed
                    lines = draft_yaml_content.strip().splitlines()
                    if lines and (
                        lines[0].startswith("```") or lines[0].startswith("---")
                    ):
                        lines = lines[1:]
                    if lines and (lines[-1] == "```" or lines[-1] == "---"):
                        lines = lines[:-1]
                    draft_yaml_content = "\n".join(lines).strip()

                draft_parsed = yaml.safe_load(draft_yaml_content)
                if draft_parsed is None:
                    logger.warning(
                        f"Draft YAML for {model_name} parsed as None, using empty dict"
                    )
                    draft_parsed = {}

                draft_model_data = draft_parsed.get("models", [{}])[0]
                draft_columns = draft_model_data.get("columns", [])
                if draft_columns is None:
                    draft_columns = []

                draft_column_names = [
                    col.get("name")
                    for col in draft_columns
                    if col and isinstance(col, dict) and "name" in col
                ]

                logger.info(
                    f"Draft interpretation contains {len(draft_column_names)} columns"
                )
            except Exception as e:
                logger.error(f"Error parsing draft YAML: {str(e)}")
                draft_columns = []
                draft_column_names = []

            # Step 5: Verify the draft interpretation against upstream models
            logger.info(
                f"Verifying draft interpretation for {model_name} against upstream models"
            )

            # Initialize results before the loop in case iterations = 0
            final_yaml_content = draft_yaml_content
            verification_result = "Verification skipped (iterations=0)"
            column_recommendations = {
                "columns_to_add": [],
                "columns_to_remove": [],
                "columns_to_modify": [],
            }
            iteration = -1  # To handle iteration + 1 in return value correctly

            # Enhanced verification with multiple iterations
            current_yaml_content = draft_yaml_content
            # final_yaml_content = draft_yaml_content # Already initialized above

            for iteration in range(max_verification_iterations):
                logger.info(
                    f"Starting verification iteration {iteration+1}/{max_verification_iterations}"
                )

                # Parse current YAML to get updated column information for verification prompt
                try:
                    current_parsed = yaml.safe_load(current_yaml_content)
                    if current_parsed is None:
                        logger.warning(
                            f"Current YAML for {model_name} parsed as None, using empty dict"
                        )
                        current_parsed = {}

                    current_model_data = current_parsed.get("models", [{}])[0]
                    current_columns = current_model_data.get("columns", [])
                    if current_columns is None:
                        current_columns = []

                    current_column_names = [
                        col.get("name")
                        for col in current_columns
                        if col and isinstance(col, dict) and "name" in col
                    ]
                    current_column_representation = (
                        ", ".join(current_column_names)
                        if current_column_names
                        else "No columns found"
                    )
                except Exception as e:
                    logger.error(
                        f"Error parsing current YAML in iteration {iteration+1}: {str(e)}"
                    )
                    current_column_representation = "Error parsing YAML"

                # Build a more detailed upstream model information with focus on SQL
                detailed_upstream_info = ""
                for upstream_name in upstream_model_names:
                    upstream_model = upstream_models_data.get(upstream_name)
                    if not upstream_model:
                        continue

                    detailed_upstream_info += (
                        f"\n-- Upstream Model: {upstream_model.name} --\n"
                    )

                    # Emphasize SQL analysis for column extraction
                    detailed_upstream_info += f"Raw SQL of {upstream_model.name}:\n```sql\n{upstream_model.raw_sql or 'SQL not available'}\n```\n"

                    # This model's columns - just as supplementary info
                    if upstream_model.interpreted_columns:
                        detailed_upstream_info += "Previously interpreted columns (for reference, SQL is definitive):\n"
                        for (
                            col_name,
                            col_desc,
                        ) in upstream_model.interpreted_columns.items():
                            detailed_upstream_info += f"  - {col_name}: {col_desc}\n"
                    elif upstream_model.columns:
                        detailed_upstream_info += (
                            "YML-defined columns (for reference, SQL is definitive):\n"
                        )
                        for col_name, col_obj in upstream_model.columns.items():
                            detailed_upstream_info += f"  - {col_name}: {col_obj.description or 'No description in YML'}\n"

                    detailed_upstream_info += "-- End Upstream Model --\n"

                verification_prompt = f"""
                You are validating a dbt model interpretation against its upstream model definitions to ensure it's complete and accurate.
                
                The model being interpreted is: {model_name}
                
                Original SQL of the model:
                ```sql
                {model.raw_sql}
                ```
                
                The current interpretation contains these columns:
                {current_column_representation}
                
                Current YAML content being verified (iteration {iteration+1}/{max_verification_iterations}):
                ```yaml
                {current_yaml_content}
                ```
                
                Here is information about the upstream models:
                {detailed_upstream_info}
                
                YOUR PRIMARY TASK IS TO COMPREHENSIVELY VERIFY THAT EVERY SINGLE COLUMN FROM THIS MODEL'S SQL OUTPUT IS CORRECTLY DOCUMENTED.
                
                Follow these specific steps in your verification:
                
                1. SQL ANALYSIS:
                   - Carefully trace the model's SQL to understand ALL columns in its output
                   - For any SELECT * statements, expand them by examining the source table/CTE's complete column list
                   - For JOINs, include columns from all joined tables that are in the SELECT
                   - For CTEs, carefully trace through each step to identify all columns
                
                2. UPSTREAM COLUMN VALIDATION:
                   - When a model uses SELECT * from an upstream model, carefully examine the SQL of that upstream model
                   - Count the total number of columns in each upstream model referenced with SELECT *
                   - Compare this count with the columns documented in the interpretation
                   - Missing columns in upstream models are the most common error - be extremely thorough
                
                3. COLUMN COUNT CHECK:
                   - Roughly estimate how many columns should appear in the output of this model
                   - Compare this estimate with the number of columns in the interpretation
                   - A significant discrepancy (e.g., interpretation has 5 columns but SQL output should have 60+) indicates missing columns
                
                4. COMPLETENESS CHECK:
                   - The interpretation must include EVERY column that will appear in the model's output
                   - Even if there are 50+ columns, all must be properly documented
                   - Any omission of columns is a critical error
                
                Based on your thorough analysis:
                
                If everything is correct, respond with "VERIFIED: The interpretation is complete and accurate."
                
                If there are issues, provide a structured response with the following format:
                
                VERIFICATION_RESULT:
                [Your general feedback and assessment here]
                [Include a count of total columns expected vs. documented]
                
                COLUMNS_TO_ADD:
                - name: [column_name_1]
                  description: [description]
                  reason: [reason this column should be added, specifically citing where in the SQL it comes from]
                - name: [column_name_2]
                  description: [description]
                  reason: [reason this column should be added, specifically citing where in the SQL it comes from]
                [List ALL missing columns, even if there are dozens]
                
                COLUMNS_TO_REMOVE:
                - name: [column_name_1]
                  reason: [reason this column should be removed, specifically citing evidence from the SQL]
                - name: [column_name_2]
                  reason: [reason this column should be removed, specifically citing evidence from the SQL]
                
                COLUMNS_TO_MODIFY:
                - name: [column_name_1]
                  current_description: [current description]
                  suggested_description: [suggested description]
                  reason: [reason for the change, with specific reference to the SQL]
                - name: [column_name_2]
                  current_description: [current description]
                  suggested_description: [suggested description]
                  reason: [reason for the change, with specific reference to the SQL]
                
                Only include sections that have actual entries (e.g., omit COLUMNS_TO_REMOVE if no columns need to be removed).
                """

                verification_result = self.llm.get_completion(
                    prompt=verification_prompt,
                    system_prompt="You are an AI assistant specialized in verifying dbt model interpretations. Your task is to carefully analyze SQL code to ensure all columns are correctly documented.",
                    max_tokens=4000,
                )

                logger.debug(
                    f"Verification result for {model_name} (iteration {iteration+1}):\n{verification_result}"
                )

                # Parse the verification result to extract recommended column changes
                column_recommendations = {
                    "columns_to_add": [],
                    "columns_to_remove": [],
                    "columns_to_modify": [],
                }

                # Only parse if verification found issues
                if "VERIFIED" not in verification_result:
                    logger.info(
                        f"Verification found issues for {model_name} in iteration {iteration+1}, extracting column recommendations"
                    )

                    # Extract columns to add
                    add_match = re.search(
                        r"COLUMNS_TO_ADD:\s*\n((?:.+\n)+?)(?:(?:COLUMNS_TO_REMOVE|COLUMNS_TO_MODIFY)|\Z)",
                        verification_result,
                        re.DOTALL,
                    )
                    if add_match:
                        add_section = add_match.group(1).strip()
                        # Parse the yaml-like format for columns to add
                        columns_to_add = []
                        current_column = {}
                        for line in add_section.split("\n"):
                            line = line.strip()
                            if line.startswith("- name:"):
                                if current_column and "name" in current_column:
                                    columns_to_add.append(current_column)
                                current_column = {
                                    "name": line.replace("- name:", "").strip()
                                }
                            elif line.startswith("description:") and current_column:
                                current_column["description"] = line.replace(
                                    "description:", ""
                                ).strip()
                            elif line.startswith("reason:") and current_column:
                                current_column["reason"] = line.replace(
                                    "reason:", ""
                                ).strip()
                        if current_column and "name" in current_column:
                            columns_to_add.append(current_column)
                        column_recommendations["columns_to_add"] = columns_to_add

                    # Extract columns to remove
                    remove_match = re.search(
                        r"COLUMNS_TO_REMOVE:\s*\n((?:.+\n)+?)(?:(?:COLUMNS_TO_ADD|COLUMNS_TO_MODIFY)|\Z)",
                        verification_result,
                        re.DOTALL,
                    )
                    if remove_match:
                        remove_section = remove_match.group(1).strip()
                        # Parse the yaml-like format for columns to remove
                        columns_to_remove = []
                        current_column = {}
                        for line in remove_section.split("\n"):
                            line = line.strip()
                            if line.startswith("- name:"):
                                if current_column and "name" in current_column:
                                    columns_to_remove.append(current_column)
                                current_column = {
                                    "name": line.replace("- name:", "").strip()
                                }
                            elif line.startswith("reason:") and current_column:
                                current_column["reason"] = line.replace(
                                    "reason:", ""
                                ).strip()
                        if current_column and "name" in current_column:
                            columns_to_remove.append(current_column)
                        column_recommendations["columns_to_remove"] = columns_to_remove

                    # Extract columns to modify
                    modify_match = re.search(
                        r"COLUMNS_TO_MODIFY:\s*\n((?:.+\n)+?)(?:(?:COLUMNS_TO_ADD|COLUMNS_TO_REMOVE)|\Z)",
                        verification_result,
                        re.DOTALL,
                    )
                    if modify_match:
                        modify_section = modify_match.group(1).strip()
                        # Parse the yaml-like format for columns to modify
                        columns_to_modify = []
                        current_column = {}
                        for line in modify_section.split("\n"):
                            line = line.strip()
                            if line.startswith("- name:"):
                                if current_column and "name" in current_column:
                                    columns_to_modify.append(current_column)
                                current_column = {
                                    "name": line.replace("- name:", "").strip()
                                }
                            elif (
                                line.startswith("current_description:")
                                and current_column
                            ):
                                current_column["current_description"] = line.replace(
                                    "current_description:", ""
                                ).strip()
                            elif (
                                line.startswith("suggested_description:")
                                and current_column
                            ):
                                current_column["suggested_description"] = line.replace(
                                    "suggested_description:", ""
                                ).strip()
                            elif line.startswith("reason:") and current_column:
                                current_column["reason"] = line.replace(
                                    "reason:", ""
                                ).strip()
                        if current_column and "name" in current_column:
                            columns_to_modify.append(current_column)
                        column_recommendations["columns_to_modify"] = columns_to_modify

                    logger.info(
                        f"Iteration {iteration+1}: Extracted column recommendations: {len(column_recommendations['columns_to_add'])} to add, "
                        f"{len(column_recommendations['columns_to_remove'])} to remove, "
                        f"{len(column_recommendations['columns_to_modify'])} to modify"
                    )

                    # If issues were found, refine the interpretation
                    logger.info(
                        f"Refining interpretation for {model_name} based on verification feedback (iteration {iteration+1})"
                    )

                    refinement_prompt = f"""
                    You are refining a dbt model interpretation based on verification feedback.
                    
                    Original model: {model_name}
                    
                    Original SQL:
                    ```sql
                    {model.raw_sql}
                    ```
                    
                    Current interpretation:
                    ```yaml
                    {current_yaml_content}
                    ```
                    
                    Verification feedback from iteration {iteration+1}:
                    {verification_result}
                    
                    Upstream model information:
                    {detailed_upstream_info}
                    
                    YOUR TASK IS TO CREATE A COMPLETE YAML INTERPRETATION THAT INCLUDES ALL COLUMNS FROM THE MODEL.
                    
                    This is absolutely critical:
                    1. ADD ALL MISSING COLUMNS identified in the verification feedback - you must include EVERY column
                    2. Even if there are 50+ columns to add, you must include all of them in your response
                    3. The most common error is not including all columns from upstream models when SELECT * is used
                    4. Be extremely thorough - lack of completeness is a critical issue
                    5. Remove any incorrect columns identified in COLUMNS_TO_REMOVE
                    6. Update descriptions for columns identified in COLUMNS_TO_MODIFY
                    
                    DO NOT OMIT ANY COLUMNS from your response - completeness is the highest priority.
                    
                    Your output should be complete, valid YAML for this model. Include all columns.
                    """

                    refined_yaml_documentation = self.llm.get_completion(
                        prompt=refinement_prompt,
                        system_prompt="You are an AI assistant specialized in refining dbt model interpretations based on SQL analysis.",
                        max_tokens=4000,
                    )

                    logger.debug(
                        f"Refined interpretation for {model_name} (iteration {iteration+1}):\n{refined_yaml_documentation}"
                    )

                    # Extract YAML content from the refined response
                    match = re.search(
                        r"```(?:yaml)?\n(.*?)```", refined_yaml_documentation, re.DOTALL
                    )
                    if match:
                        current_yaml_content = match.group(1).strip()
                    else:
                        # If no code block found, use the whole response but check for YAML tags
                        current_yaml_content = refined_yaml_documentation.strip()
                        # Remove any potential YAML code fence markers at the beginning or end
                        if current_yaml_content.startswith("```yaml"):
                            current_yaml_content = current_yaml_content[7:]
                        if current_yaml_content.endswith("```"):
                            current_yaml_content = current_yaml_content[:-3]
                        current_yaml_content = current_yaml_content.strip()

                    # Additional safety check for YAML content
                    if current_yaml_content.startswith(
                        "```"
                    ) or current_yaml_content.endswith("```"):
                        # Further cleaning if needed
                        lines = current_yaml_content.strip().splitlines()
                        if lines and (
                            lines[0].startswith("```") or lines[0].startswith("---")
                        ):
                            lines = lines[1:]
                        if lines and (lines[-1] == "```" or lines[-1] == "---"):
                            lines = lines[:-1]
                        current_yaml_content = "\n".join(lines).strip()

                    final_yaml_content = current_yaml_content

                    # If we found column issues but are not on last iteration, continue
                    has_column_changes = (
                        len(column_recommendations["columns_to_add"]) > 0
                        or len(column_recommendations["columns_to_remove"]) > 0
                        or len(column_recommendations["columns_to_modify"]) > 0
                    )

                    if (
                        has_column_changes
                        and iteration < max_verification_iterations - 1
                    ):
                        logger.info(
                            f"Moving to next verification iteration to check for additional issues"
                        )
                        continue
                else:
                    # Verification was successful - no issues found
                    logger.info(
                        f"Interpretation for {model_name} verified successfully in iteration {iteration+1}"
                    )
                    final_yaml_content = current_yaml_content
                    break

            # Store the verification history and column recommendations from the final iteration
            final_verification_result = verification_result
            final_column_recommendations = column_recommendations

            # Prepare the result
            result_data = {
                "model_name": model_name,
                "yaml_documentation": final_yaml_content,
                "draft_yaml": draft_yaml_content,
                "verification_result": final_verification_result,
                "column_recommendations": final_column_recommendations,
                "verification_iterations": iteration
                + 1,  # Will be 0 if loop didn't run
                "prompt": prompt,
                "success": True,
            }

            return result_data

        except Exception as e:
            logger.error(
                f"Error in agentic interpretation of model {model_name}: {e}",
                exc_info=True,
            )
            error_result = {
                "model_name": model_name,
                "error": f"Error in agentic interpretation: {str(e)}",
                "success": False,
            }
            return error_result

    def save_interpreted_documentation(
        self, model_name: str, yaml_documentation: str, embed: bool = False
    ) -> Dict[str, Any]:
        """Save interpreted documentation for a model.

        Args:
            model_name: Name of the model
            yaml_documentation: YAML documentation as a string. This should include
                any column additions, removals, or modifications that were recommended
                during the verification phase.
            embed: Whether to embed the model in the vector store

        Returns:
            Dict containing the result of the operation
        """
        logger.info(f"Saving interpreted documentation for model {model_name}")
        session = self.model_storage.Session()  # Get a session from storage
        try:
            # Parse the YAML
            try:
                parsed_yaml = yaml.safe_load(yaml_documentation)
                if (
                    not isinstance(parsed_yaml, dict)
                    or "models" not in parsed_yaml
                    or not isinstance(parsed_yaml["models"], list)
                    or not parsed_yaml["models"]
                ):
                    raise ValueError("Invalid YAML structure")
                model_data = parsed_yaml["models"][0]  # Assuming one model per YAML
                if model_data.get("name") != model_name:
                    logger.warning(
                        f"Model name mismatch in YAML ({model_data.get('name')}) and function call ({model_name}). Using {model_name}."
                    )

            except (yaml.YAMLError, ValueError) as e:
                logger.error(f"Error parsing YAML for {model_name}: {e}")
                return {
                    "success": False,
                    "error": f"Invalid YAML format: {e}",
                    "model_name": model_name,
                }

            # Extract interpretation data
            interpreted_description = model_data.get("description")
            interpreted_columns = {}
            if "columns" in model_data and isinstance(model_data["columns"], list):
                for col_data in model_data["columns"]:
                    if isinstance(col_data, dict) and "name" in col_data:
                        interpreted_columns[col_data["name"]] = col_data.get(
                            "description", ""
                        )

            # Fetch the existing ModelTable record
            model_record = (
                session.query(ModelTable).filter(ModelTable.name == model_name).first()
            )

            if not model_record:
                logger.error(f"Model {model_name} not found in database for saving.")
                return {
                    "success": False,
                    "error": f"Model {model_name} not found in database",
                    "model_name": model_name,
                }

            # Update the record directly
            model_record.interpreted_description = interpreted_description
            model_record.interpreted_columns = interpreted_columns
            # Add interpretation_details if needed (e.g., from agentic workflow)
            # model_record.interpretation_details = ...

            session.add(model_record)
            session.commit()
            logger.info(f"Successfully saved interpretation for model {model_name}")

            # Optional: Re-embed the model
            if embed:
                logger.info(
                    f"Re-embedding model {model_name} after saving interpretation"
                )
                # Fetch the updated DBTModel (using the existing storage method)
                updated_model = self.model_storage.get_model(model_name)
                if updated_model:
                    # Generate the text representation for embedding
                    model_text_for_embedding = updated_model.get_text_representation(
                        include_documentation=True
                    )
                    # Use the correct store_model_embedding method
                    self.vector_store.store_model_embedding(
                        model_name=updated_model.name,
                        model_text=model_text_for_embedding,
                        # Optionally add metadata if needed
                        # metadata=updated_model.to_dict() # Example if needed
                    )
                    logger.info(f"Successfully re-embedded model {model_name}")
                else:
                    logger.error(
                        f"Failed to fetch updated model {model_name} for re-embedding"
                    )
                    # Return success=True because saving worked, but warn about embedding
                    return {
                        "success": True,
                        "warning": f"Interpretation saved, but failed to re-embed model {model_name}",
                        "model_name": model_name,
                    }

            return {"success": True, "model_name": model_name}

        except Exception as e:
            logger.error(
                f"Error saving interpreted documentation for {model_name}: {e}",
                exc_info=True,
            )
            session.rollback()
            return {
                "success": False,
                "error": f"Internal error saving interpretation: {e}",
                "model_name": model_name,
            }
        finally:
            session.close()
