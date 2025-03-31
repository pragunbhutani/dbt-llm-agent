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
from dbt_llm_agent.storage.question_storage import QuestionStorage

from dbt_llm_agent.integrations.llm.client import LLMClient
from dbt_llm_agent.integrations.llm.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    SEARCH_QUERY_REFINEMENT_PROMPT_TEMPLATE,
    MODEL_INTERPRETATION_PROMPT,
    PLANNING_PROMPT_TEMPLATE,
    ANALYSIS_PROMPT_TEMPLATE,
    SYNTHESIS_PROMPT_TEMPLATE,
    FEEDBACK_REFINEMENT_PROMPT_TEMPLATE,
)
from dbt_llm_agent.storage.model_storage import ModelStorage
from dbt_llm_agent.storage.model_embedding_storage import ModelEmbeddingStorage
from dbt_llm_agent.core.models import DBTModel

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
                    model_info_str += model.get_readable_representation() + "\n\n"
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
                                ).get_readable_representation()
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

                refinement_prompt = FEEDBACK_REFINEMENT_PROMPT_TEMPLATE.format(
                    original_question=original_question,
                    original_answer=final_answer,
                    feedback_context=feedback_context,
                )

                refined_answer = self.llm.get_completion(
                    prompt=refinement_prompt,
                    system_prompt="You are an AI assistant refining answers based on feedback.",
                    temperature=self.temperature,  # Use original temperature
                    max_tokens=1500,  # Allow longer refined answer
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
                    model_info += model.get_readable_representation() + "\n\n"
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
        self,
        model_name: str,
        recursive: bool = False,
        force_recursive: bool = False,
        visited: Optional[Set[str]] = None,
        _recursive_results: Optional[
            Dict[str, Dict[str, Any]]
        ] = None,  # Internal tracking
    ) -> Dict[str, Any]:
        """Interpret a model and its columns, potentially recursively.

        Args:
            model_name: Name of the model to interpret.
            recursive: Whether to recursively interpret upstream models if they lack interpretation.
            force_recursive: Whether to force re-interpretation of all upstream models.
            visited: Set of model names visited in the current recursion path (for cycle detection).
            _recursive_results: Internal dictionary to store results from recursive calls.

        Returns:
            Dict containing the interpreted documentation in YAML format and other metadata for the *target* model_name.
        """
        if visited is None:
            visited = set()
        if _recursive_results is None:
            _recursive_results = (
                {}
            )  # Initialize storage for results from this branch downwards

        # --- Cycle Detection ---
        if model_name in visited:
            logger.warning(
                f"Circular dependency detected: Already processing {model_name}. Skipping further recursion."
            )
            # Return a specific indicator or just an error? For now, return error-like dict.
            return {
                "model_name": model_name,
                "error": f"Circular dependency detected involving {model_name}",
                "success": False,
                "skipped_circular": True,  # Add a flag
            }
        visited.add(model_name)

        try:
            logger.debug(
                f"Starting interpretation process for: {model_name} (Recursive: {recursive}, Force: {force_recursive})"
            )

            model = self.model_storage.get_model(model_name)
            if not model:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found",
                    "success": False,
                }

            # --- Recursive Interpretation of Upstream Models ---
            upstream_models_data: Dict[str, DBTModel] = (
                {}
            )  # Store fetched upstream models
            if recursive or force_recursive:
                logger.debug(f"Processing upstream models for {model_name}")
                for upstream_name in model.all_upstream_models:
                    upstream_model = self.model_storage.get_model(upstream_name)
                    if not upstream_model:
                        logger.warning(
                            f"Upstream model {upstream_name} not found for {model_name}. Skipping."
                        )
                        continue

                    upstream_models_data[upstream_name] = (
                        upstream_model  # Store for later use
                    )

                    # Check if interpretation is needed
                    needs_interpretation = False
                    if force_recursive:
                        needs_interpretation = True
                        logger.info(
                            f"Forcing recursive interpretation for upstream model: {upstream_name}"
                        )
                    elif (
                        recursive and not upstream_model.interpreted_description
                    ):  # Check if interpretation exists
                        needs_interpretation = True
                        logger.info(
                            f"Recursively interpreting upstream model {upstream_name} as it lacks interpretation."
                        )
                    else:
                        logger.debug(
                            f"Skipping interpretation for upstream model {upstream_name} (already interpreted or flags not set)."
                        )

                    if needs_interpretation:
                        # Make recursive call, passing a copy of visited set
                        logger.debug(
                            f"Calling interpret_model recursively for {upstream_name}"
                        )
                        recursive_result = self.interpret_model(
                            upstream_name,
                            recursive=recursive,  # Pass flags down
                            force_recursive=force_recursive,
                            visited=visited.copy(),  # Pass copy for independent path tracking
                            _recursive_results=_recursive_results,  # Pass the shared results dict
                        )

                        # Store the result (even if failed/skipped) to avoid re-processing and for context building
                        if upstream_name not in _recursive_results:
                            _recursive_results[upstream_name] = recursive_result

                        if not recursive_result.get("success"):
                            # Log error but continue, maybe partial info is better than none
                            error_msg = recursive_result.get(
                                "error", "Unknown error during recursive interpretation"
                            )
                            if not recursive_result.get(
                                "skipped_circular"
                            ):  # Don't log error again for cycles
                                logger.warning(
                                    f"Recursive interpretation failed for {upstream_name}: {error_msg}"
                                )

            # --- Generate Upstream Info String ---
            # Fetch any upstream models not already fetched during recursion check
            for upstream_name in model.all_upstream_models:
                if upstream_name not in upstream_models_data:
                    um = self.model_storage.get_model(upstream_name)
                    if um:
                        upstream_models_data[upstream_name] = um
                    else:
                        logger.warning(
                            f"Upstream model {upstream_name} (needed for context) not found for {model_name}. Skipping."
                        )

            upstream_info = ""
            for um_name, um in upstream_models_data.items():
                upstream_info += f"\n-- Upstream Model: {um.name} --\n"
                description = "No description available."
                columns_info = "Columns: No column information available."

                # Priority 1: Use result from the *current* recursive run if available and successful
                if um_name in _recursive_results and _recursive_results[um_name].get(
                    "success"
                ):
                    logger.debug(
                        f"Using fresh recursive result for upstream model {um_name}"
                    )
                    try:
                        yaml_doc = _recursive_results[um_name].get(
                            "yaml_documentation", ""
                        )
                        # Minimal parsing needed here, just description and columns
                        parsed = yaml.safe_load(yaml_doc)
                        model_data = parsed.get("models", [{}])[0]
                        description = model_data.get(
                            "description",
                            "Description missing in fresh interpretation.",
                        )
                        cols = model_data.get("columns", [])
                        if cols:
                            columns_info = "Columns (from fresh interpretation):\n"
                            for col in cols:
                                col_name = col.get("name", "unknown")
                                col_desc = col.get("description", "No description")
                                columns_info += f"  - {col_name}: {col_desc}\n"
                        else:
                            columns_info = (
                                "Columns: None found in fresh interpretation."
                            )

                    except Exception as e:
                        logger.warning(
                            f"Failed to parse fresh recursive result YAML for {um_name}: {e}. Falling back."
                        )
                        # Fallback logic starts here if parsing fails
                        description = (
                            um.interpreted_description
                            or um.description
                            or "No description provided."
                        )
                        if um.interpreted_columns:
                            columns_info = "Columns (from stored interpretation):\n"
                            for col_name, col_desc in um.interpreted_columns.items():
                                columns_info += f"  - {col_name}: {col_desc}\n"
                        elif um.columns:
                            columns_info = "Columns (from YML):\n"
                            for col_name, col_obj in um.columns.items():
                                columns_info += f"  - {col_name}: {col_obj.description or 'No description in YML'}\n"

                # Priority 2: Use stored interpretation if no fresh one exists
                elif um.interpreted_description or um.interpreted_columns:
                    logger.debug(
                        f"Using stored interpretation for upstream model {um_name}"
                    )
                    description = (
                        um.interpreted_description
                        or "No model description, using YML description if available."
                    )
                    # If model description was from interpretation, try using interpreted columns first
                    if um.interpreted_description and um.interpreted_columns:
                        columns_info = "Columns (from stored interpretation):\n"
                        for col_name, col_desc in um.interpreted_columns.items():
                            columns_info += f"  - {col_name}: {col_desc}\n"
                    # Fallback to YML columns if no interpreted columns
                    elif um.columns:
                        columns_info = "Columns (from YML):\n"
                        for col_name, col_obj in um.columns.items():
                            columns_info += f"  - {col_name}: {col_obj.description or 'No description in YML'}\n"
                    # Use model YML description if no interpreted description was found
                    if (
                        description
                        == "No model description, using YML description if available."
                    ):
                        description = um.description or "No description provided."

                # Priority 3: Use YML definition if no interpretation exists
                elif um.description or um.columns:
                    logger.debug(f"Using YML definition for upstream model {um_name}")
                    description = um.description or "No YML description provided."
                    if um.columns:
                        columns_info = "Columns (from YML):\n"
                        for col_name, col_obj in um.columns.items():
                            columns_info += f"  - {col_name}: {col_obj.description or 'No description in YML'}\n"

                upstream_info += f"Description: {description}\n"
                upstream_info += columns_info + "\n"  # Add newline for spacing
                upstream_info += (
                    f"Raw SQL:\n```sql\n{um.raw_sql or 'SQL not available'}\n```\n"
                )
                upstream_info += "-- End Upstream Model --\n"

            # --- Prepare and Call LLM for the Target Model ---
            prompt = MODEL_INTERPRETATION_PROMPT.format(
                model_name=model.name,
                model_sql=model.raw_sql,
                upstream_info=upstream_info,
            )

            logger.debug(
                f"Final interpretation prompt for model {model_name}:\n{prompt}"
            )

            yaml_documentation = self.llm.get_completion(
                prompt=f"Interpret and generate YAML documentation for the model {model_name} based on its SQL and upstream dependencies",
                system_prompt=prompt,
                max_tokens=2000,  # Consider if this needs adjustment
            )

            logger.debug(
                f"Raw LLM response for model {model_name}:\n{yaml_documentation}"
            )

            # Extract YAML content
            match = re.search(r"```(?:yaml)?\n(.*?)```", yaml_documentation, re.DOTALL)
            yaml_content = (
                match.group(1).strip() if match else yaml_documentation.strip()
            )

            # Store own result in the recursive results dict as well
            result_data = {
                "model_name": model_name,
                "yaml_documentation": yaml_content,
                "prompt": prompt,
                "success": True,
            }
            _recursive_results[model_name] = result_data
            return result_data

        except Exception as e:
            logger.error(
                f"Error interpreting model {model_name}: {e}", exc_info=True
            )  # Add traceback
            error_result = {
                "model_name": model_name,
                "error": f"Error interpreting model: {str(e)}",
                "success": False,
            }
            _recursive_results[model_name] = error_result  # Store error result
            return error_result
        finally:
            # Clean up visited set for this specific path after returning
            # This is important so siblings in the DAG don't see this node as visited
            # Note: Because we pass visited.copy() down, this removal only affects the current stack frame upwards
            if model_name in visited:
                visited.remove(model_name)

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

            # Clean the YAML string from markdown code fences
            lines = yaml_documentation.strip().splitlines()
            if lines and lines[0].strip().startswith("```yaml"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_yaml_documentation = "\n".join(lines)

            try:
                # Use the cleaned string for parsing
                parsed_yaml = yaml.safe_load(cleaned_yaml_documentation)
            except Exception as e:
                logger.error(f"Error parsing YAML: {e}")
                return {
                    "model_name": model_name,
                    "error": f"Invalid YAML format: {str(e)}",
                    "success": False,
                }

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

                if "columns" in model_data and isinstance(model_data["columns"], list):
                    for column in model_data["columns"]:
                        # Add a check to ensure 'column' is a dictionary and not None
                        if (
                            isinstance(column, dict)
                            and "name" in column
                            and "description" in column
                        ):
                            column_descriptions[column["name"]] = column["description"]
                        elif (
                            column is not None
                        ):  # Log a warning if it's not a dict but also not None
                            logger.warning(
                                f"Skipping invalid column entry in YAML for model {model_name}: {column}"
                            )
            else:
                logger.warning(
                    f"Missing or invalid YAML structure for model {model_name}"
                )

            existing_model = self.model_storage.get_model(model_name)
            if existing_model:
                logger.debug(
                    f"Existing model columns: {list(existing_model.columns.keys())}"
                )
                logger.debug(f"Tests in existing model: {existing_model.tests}")

                existing_model.interpreted_description = model_description
                existing_model.interpreted_columns = column_descriptions

                success = self.model_storage.update_model(existing_model)

                if success:
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

    def interpret_model_agentic(self, model_name: str) -> Dict[str, Any]:
        """Interpret a model and its columns using an agentic workflow.

        This method implements a step-by-step agentic approach to model interpretation:
        1. Read the source code of the model to interpret
        2. Identify upstream models that provide context
        3. Fetch details of the upstream models
        4. Create a draft interpretation
        5. Verify the draft against upstream models to ensure completeness and correctness

        Args:
            model_name: Name of the model to interpret.

        Returns:
            Dict containing the interpreted documentation in YAML format and metadata.
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
                max_tokens=2000,
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

            # Prepare a representation of columns for verification
            column_representation = "No columns found in draft"
            if draft_column_names:
                column_representation = ", ".join(draft_column_names)
            elif draft_yaml_content:
                # If we couldn't parse columns but have YAML content, include raw content
                column_representation = f"YAML parsing failed, raw content available but columns couldn't be extracted."

            verification_prompt = f"""
            You are validating a dbt model interpretation against its upstream model definitions to ensure it's complete and accurate.
            
            The model being interpreted is: {model_name}
            
            The draft interpretation contains these columns:
            {column_representation}
            
            Draft YAML content:
            {draft_yaml_content}
            
            Here is information about the upstream models:
            {upstream_info}
            
            Based on the SQL and upstream model information, verify:
            1. Does the draft interpretation correctly identify all columns from the model's SQL?
            2. Are there any missing columns that should be included based on the SQL and upstream references?
            3. Is the description of each column accurate based on upstream models?
            
            If everything is correct, respond with "VERIFIED: The interpretation is complete and accurate."
            If there are issues, provide specific feedback on what needs to be fixed, including any missing columns or inaccuracies.
            """

            verification_result = self.llm.get_completion(
                prompt=verification_prompt,
                system_prompt="You are an AI assistant specialized in verifying dbt model interpretations.",
                max_tokens=1000,
            )

            logger.debug(
                f"Verification result for {model_name}:\n{verification_result}"
            )

            # If verification identified issues, refine the interpretation
            if "VERIFIED" not in verification_result:
                logger.info(
                    f"Refining interpretation for {model_name} based on verification feedback"
                )

                refinement_prompt = f"""
                You are refining a dbt model interpretation based on verification feedback.
                
                Original model: {model_name}
                
                Original SQL:
                ```sql
                {model.raw_sql}
                ```
                
                Draft interpretation:
                ```yaml
                {draft_yaml_content}
                ```
                
                Verification feedback:
                {verification_result}
                
                Upstream model information:
                {upstream_info}
                
                Please create an improved YAML interpretation that addresses all the issues identified in the verification feedback.
                Use the exact same YAML format as the draft interpretation but with corrected content.
                """

                refined_yaml_documentation = self.llm.get_completion(
                    prompt=refinement_prompt,
                    system_prompt="You are an AI assistant specialized in refining dbt model interpretations.",
                    max_tokens=2000,
                )

                logger.debug(
                    f"Refined interpretation for {model_name}:\n{refined_yaml_documentation}"
                )

                # Extract YAML content from the refined response
                match = re.search(
                    r"```(?:yaml)?\n(.*?)```", refined_yaml_documentation, re.DOTALL
                )
                if match:
                    final_yaml_content = match.group(1).strip()
                else:
                    # If no code block found, use the whole response but check for YAML tags
                    final_yaml_content = refined_yaml_documentation.strip()
                    # Remove any potential YAML code fence markers at the beginning or end
                    if final_yaml_content.startswith("```yaml"):
                        final_yaml_content = final_yaml_content[7:]
                    if final_yaml_content.endswith("```"):
                        final_yaml_content = final_yaml_content[:-3]
                    final_yaml_content = final_yaml_content.strip()

                # Additional safety check for YAML content
                if final_yaml_content.startswith("```") or final_yaml_content.endswith(
                    "```"
                ):
                    # Further cleaning if needed
                    lines = final_yaml_content.strip().splitlines()
                    if lines and (
                        lines[0].startswith("```") or lines[0].startswith("---")
                    ):
                        lines = lines[1:]
                    if lines and (lines[-1] == "```" or lines[-1] == "---"):
                        lines = lines[:-1]
                    final_yaml_content = "\n".join(lines).strip()

                logger.debug(
                    f"Cleaned final YAML content for {model_name}:\n{final_yaml_content}"
                )
            else:
                logger.info(f"Interpretation for {model_name} verified successfully")
                final_yaml_content = draft_yaml_content

            # Prepare the result
            result_data = {
                "model_name": model_name,
                "yaml_documentation": final_yaml_content,
                "draft_yaml": draft_yaml_content,
                "verification_result": verification_result,
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
