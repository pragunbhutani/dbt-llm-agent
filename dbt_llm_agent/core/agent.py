"""Agent for answering questions about dbt models."""

import logging
import re
import json
import yaml
import sys
from typing import Dict, List, Any, Optional, Union, Set

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from dbt_llm_agent.storage.question_storage import QuestionStorage

from dbt_llm_agent.integrations.llm.client import LLMClient
from dbt_llm_agent.integrations.llm.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    SEARCH_QUERY_REFINEMENT_PROMPT_TEMPLATE,
)
from dbt_llm_agent.storage.model_storage import ModelStorage
from dbt_llm_agent.storage.model_embedding_storage import ModelEmbeddingStorage
from dbt_llm_agent.core.models import DBTModel

logger = logging.getLogger(__name__)

# Define a new prompt template for refining the answer based on feedback
REFINE_PROMPT_TEMPLATE = """
You are an AI assistant helping users understand their dbt project.
The user asked the following question: {question}

Based on the available dbt models, you previously generated this answer:
{previous_answer}

The user provided the following feedback indicating the answer was not sufficient:
{user_feedback}

Here is the context about relevant dbt models:
{model_info}

Based on the user's feedback and the model context, please generate an improved and more complete answer to the original question: {question}
"""


class Agent:
    """Agent for interacting with dbt projects using an agentic workflow."""

    # Placeholder for the new planning prompt
    PLANNING_PROMPT_TEMPLATE = """
You are a data analyst assistant planning how to answer a user's question about dbt models.
User question: "{question}"

1. Identify the primary metric or entity the user is asking about.
2. Identify any grouping, filtering, or specific dimensions requested.
3. Based on this, determine the *first* key piece of information to search for in the dbt models.
4. Formulate a concise search query focused *only* on finding that first piece of information.

Respond with just the search query.
Search Query: """

    # Placeholder for the new analysis prompt
    ANALYSIS_PROMPT_TEMPLATE = """You are a data analyst assistant analyzing dbt models to answer a user's question. Be concise.
Original user question: "{original_question}"

We have already gathered information on these models:
{already_found_models_summary}

We just searched for "{last_search_query}".
{newly_found_model_details_section}

Analyze the newly found model(s) if any:
1. What key information do they provide relevant to the original question?
2. What key entities (e.g., customer IDs, timestamps, dimensions) do they contain or relate to?
3. Based on the original question and *all* the models found so far, what is the *single most important* specific piece of information we still need?
4. If all necessary information seems present across the found models to answer the original question, respond ONLY with `ALL_INFO_FOUND`.
5. Otherwise, formulate a *concise* search query for the next piece of missing information.

Respond *concisely* with the next search query OR `ALL_INFO_FOUND`.
Next Step: """

    # Placeholder for the new synthesis prompt
    SYNTHESIS_PROMPT_TEMPLATE = """You are a data analyst assistant synthesizing an answer to a user's question using information from dbt models. Be concise.
User question: "{question}"

We have gathered information about the following relevant dbt models:
{accumulated_model_info}

Based *strictly* and *only* on the provided model information (`accumulated_model_info`), construct a step-by-step answer for the user.
- Explain how the models can be used together to answer the question.
- Mention necessary joins (including join keys) between models *if applicable and derivable from the provided info*.
- Describe any required calculations or aggregations.
- If possible, provide a sample SQL query demonstrating the process, using *only* models and columns present in the provided info.
- **Do not invent table names, column names, or relationships** that are not explicitly present in the provided model information. Do not assume the existence of dimension tables unless they are listed in `accumulated_model_info`.
- If the gathered models are insufficient to fully answer the question, *clearly state this*. Explain exactly what information (e.g., specific columns, relationships) is missing and which of the provided models contain related partial information. **Do not provide a sample query using hypothetical/invented tables or columns.**

Answer: """

    def __init__(
        self,
        llm_client: LLMClient,
        model_storage: ModelStorage,
        vector_store: ModelEmbeddingStorage,
        question_storage: Optional[QuestionStorage] = None,
        console: Optional[Console] = None,
        temperature: float = 0.0,
        verbose: bool = False,
    ):
        """Initialize the agent.

        Args:
            llm_client: LLM client for generating text
            model_storage: Storage for dbt models
            vector_store: Vector store for semantic search
            question_storage: Storage for question history
            console: Console for interactive prompts
            temperature: Temperature for LLM generation
            verbose: Whether to print verbose output
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
        system_prompt = SEARCH_QUERY_REFINEMENT_PROMPT_TEMPLATE
        refined_query = self.llm.get_completion(
            prompt=original_query, system_prompt=system_prompt, temperature=0.0
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
                and similarity > 0.4  # Lowered threshold from 0.5 to 0.4
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
        max_steps = 5  # Max search/analysis steps
        accumulated_model_info_str = ""
        found_models_details = []  # List to store dicts of found models
        found_model_names = set()  # Set to track names of found models
        current_search_query = ""
        original_question = question
        conversation_id = None  # Keep track for potential history saving

        self.console.print(f"[bold]🚀 Starting agentic workflow for:[/bold] {question}")

        try:
            # 1. Planning Step: Determine the first search query
            if self.verbose:
                self.console.print(
                    "\n[bold blue]🧠 Planning initial search...[/bold blue]"
                )
            else:
                self.console.print("[blue]🧠 Planning initial search...[/blue]")
            planning_prompt = self.PLANNING_PROMPT_TEMPLATE.format(
                question=original_question
            )
            # Use the existing LLM client instance
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
                        f"\n[bold magenta]🔍 Step {step + 1}/{max_steps}: Searching for '{current_search_query}'[/bold magenta]"
                    )

                # Search for models based on the current query
                new_model_info_str, new_models_data = self._search_and_fetch_models(
                    query=current_search_query,
                    current_context_models=list(
                        found_model_names
                    ),  # Pass names of already found models
                )

                newly_found_model_details_for_prompt = ""
                if not new_models_data:
                    if self.verbose:
                        self.console.print(
                            "[yellow]No new relevant models found in this step.[/yellow]"
                        )
                    newly_found_model_details_for_prompt = "No new models found."
                else:
                    # Process newly found models
                    added_count = 0
                    current_step_model_names = []
                    for model_data in new_models_data:
                        # Ensure model_data is a dict and has a name
                        if isinstance(model_data, dict):
                            model_name = model_data.get(
                                "name"
                            )  # Prioritize 'name' if available
                            if (
                                not model_name and "model_name" in model_data
                            ):  # Fallback to 'model_name'
                                model_name = model_data["model_name"]

                            if model_name and model_name not in found_model_names:
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
                                columns = model_data.get("columns", [])
                                column_names = [
                                    c.get("name", "N/A")
                                    for c in columns
                                    if isinstance(c, dict)
                                ]
                                model_summary = f"Model: {model_name}\\nDescription: {model_data.get('description', 'N/A')}\\nColumns: {', '.join(column_names)}\\n---"
                                newly_found_model_details_for_prompt += (
                                    model_summary + "\\n"
                                )

                                found_models_details.append(
                                    model_data
                                )  # Store the dict
                                found_model_names.add(model_name)
                                current_step_model_names.append(model_name)
                                added_count += 1

                    if self.verbose and added_count > 0:
                        self.console.print(
                            f"[green]✅ Added {added_count} models this step: {', '.join(current_step_model_names)}[/green]"
                        )
                    elif (
                        self.verbose and new_models_data and not added_count
                    ):  # If search returned something but we didn't add new ones (and verbose)
                        self.console.print(
                            "[dim]Models found but already in context or invalid format.[/dim]"
                        )
                    elif (
                        not new_models_data and self.verbose
                    ):  # Explicitly check verbose for no new models log
                        self.console.print(
                            "[yellow]No new relevant models found in this step.[/yellow]"
                        )

                # Analysis Step: Analyze results and determine next step
                if self.verbose:
                    self.console.print(
                        "\n[bold blue]🤔 Analyzing results and planning next step...[/bold blue]"
                    )

                # Prepare summary of already found models for the prompt
                # Exclude models that were *just* added in this step from the "already found" summary
                already_found_summary = ""
                models_found_before_this_step = [
                    m
                    for m in found_models_details
                    if m.get("name", m.get("model_name"))
                    not in current_step_model_names
                ]

                if models_found_before_this_step:
                    already_found_summary = "Previously found models summary:\\n" + "\\n".join(
                        f"- {m.get('name', m.get('model_name', 'N/A'))}: {m.get('description', 'N/A')[:100]}..."
                        for m in models_found_before_this_step
                    )
                elif (
                    not found_models_details
                ):  # If it's the first step and nothing was found
                    already_found_summary = "No models found yet."
                else:  # Only found models this step
                    already_found_summary = "No *other* models found previously."

                analysis_prompt = self.ANALYSIS_PROMPT_TEMPLATE.format(
                    original_question=original_question,
                    already_found_models_summary=already_found_summary,
                    last_search_query=current_search_query,
                    newly_found_model_details_section=newly_found_model_details_for_prompt
                    or "No new models found this step.",  # Ensure it's never empty
                )

                # Use the existing LLM client instance
                next_step_output = (
                    self.llm.get_completion(
                        prompt=analysis_prompt,
                        system_prompt="You are an AI assistant specialized in dbt model analysis.",
                        temperature=0.0,
                        max_tokens=500,  # Increased max_tokens from 150 to 500
                    )
                    .strip()
                    .replace('"', "")
                )  # Clean quotes often added by LLM

                if self.verbose:
                    self.console.print(
                        f"[dim] Analysis result: '{next_step_output}'[/dim]"
                    )

                # Check for termination condition
                if "ALL_INFO_FOUND" in next_step_output or not next_step_output:
                    if (
                        not found_models_details and step == 0
                    ):  # Nothing found on first step, analysis couldn't help
                        self.console.print(
                            "[yellow]Initial search yielded no results and analysis could not determine next steps.[/yellow]"
                        )
                        final_answer = "I could not find relevant models based on the initial search. Please try rephrasing your question."
                        return {  # Return early
                            "question": original_question,
                            "final_answer": final_answer,
                            "used_model_names": list(found_model_names),
                            "conversation_id": None,
                        }
                    else:
                        self.console.print(
                            "[green]✅ Analysis complete. Sufficient information gathered or loop finished.[/green]"
                        )
                        break  # Exit loop to synthesize answer
                else:
                    # Update search query for the next iteration
                    # Extract the query part if the LLM added explanation
                    query_match = re.search(
                        r"(?:Next Step|Search Query):\s*(.*)",
                        next_step_output,
                        re.IGNORECASE,
                    )
                    if query_match:
                        current_search_query = query_match.group(1).strip()
                    else:
                        current_search_query = (
                            next_step_output  # Assume the whole output is the query
                        )

                    if (
                        not current_search_query
                    ):  # Safety check if regex fails or output is weird
                        self.console.print(
                            "[yellow]⚠️ Analysis did not provide a valid next search query. Stopping iteration.[/yellow]"
                        )
                        break

                    if self.verbose:
                        self.console.print(
                            f"[dim] Next search query: '{current_search_query}'[/dim]"
                        )

                    if step == max_steps - 1:
                        self.console.print(
                            "[yellow]⚠️ Max search steps reached. Proceeding with gathered information.[/yellow]"
                        )

            # 3. Synthesis Step
            if not found_models_details:
                # This case should ideally be caught earlier, but acts as a final fallback
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
                synthesis_prompt = self.SYNTHESIS_PROMPT_TEMPLATE.format(
                    question=original_question,
                    accumulated_model_info=accumulated_model_info_str,  # Pass the full, readable details
                )
                # Use the existing LLM client instance
                final_answer = self.llm.get_completion(
                    prompt=synthesis_prompt,
                    system_prompt="You are an AI assistant specialized in dbt model analysis and explanation.",
                    temperature=self.temperature,  # Use configured temperature
                    max_tokens=1500,  # Allow longer answer
                )

            # Save history (optional, keep existing logic if question_storage exists)
            if self.question_storage:
                try:
                    conversation_id = self.question_storage.record_question(
                        question_text=original_question,
                        answer_text=final_answer,
                        model_names=list(found_model_names),
                        # Could add more context like search steps later if needed
                    )
                    if self.verbose:
                        self.console.print(
                            f"[dim] Saved conversation with ID {conversation_id}[/dim]"
                        )
                except Exception as e:
                    logger.warning(f"Could not save question history: {e}")

            # 4. Return results
            return {
                "question": original_question,
                "final_answer": final_answer,
                "used_model_names": list(found_model_names),
                "conversation_id": conversation_id,
                # "search_steps": [], # Could potentially return the sequence of queries/results later
            }

        except Exception as e:
            logger.error(
                f"Error during agentic workflow: {str(e)}", exc_info=self.verbose
            )
            self.console.print(f"[bold red]Error during workflow:[/bold red] {str(e)}")
            return {
                "question": question,
                "final_answer": "An error occurred during the process. Please check logs or try again.",
                "used_model_names": list(
                    found_model_names
                ),  # Return what was found, if any
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

            model = self.model_storage.get_model(model_name)
            if not model:
                return {
                    "model_name": model_name,
                    "error": f"Model {model_name} not found",
                }

            upstream_models = []
            for upstream_name in model.all_upstream_models:
                upstream_model = self.model_storage.get_model(upstream_name)
                if upstream_model:
                    upstream_models.append(upstream_model)

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

            prompt = MODEL_INTERPRETATION_PROMPT.format(
                model_name=model.name,
                model_sql=model.raw_sql,
                upstream_info=upstream_info,
            )

            logger.debug(f"Interpretation prompt for model {model_name}:\n{prompt}")

            yaml_documentation = self.llm.get_completion(
                prompt=f"Interpret and generate YAML documentation for the model {model_name} based on its SQL and upstream dependencies",
                system_prompt=prompt,
                max_tokens=2000,
            )

            logger.debug(
                f"Raw LLM response for model {model_name}:\n{yaml_documentation}"
            )

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

            try:
                parsed_yaml = yaml.safe_load(yaml_documentation)
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
                        if "name" in column and "description" in column:
                            column_descriptions[column["name"]] = column["description"]
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
