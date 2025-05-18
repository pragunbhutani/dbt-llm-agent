import logging
import json
from typing import Dict, List, Any, Optional, Set, Union

# Import Django model types for type hinting (optional but helpful)
try:
    from apps.knowledge_base.models import Model
    from apps.workflows.models import Question
except ImportError:
    # Handle case where models might not be available during standalone testing
    Model = Any
    Question = Any  # type: ignore

from apps.workflows.rules_loader import get_agent_rules  # Updated import

logger = logging.getLogger(__name__)

# TODO: Refine these functions based on actual state data format


def _format_model_for_prompt(model_data: Dict[str, Any]) -> str:
    """Helper to format a single model's details for the prompt."""
    name = model_data.get("name", "Unknown Model")
    desc = (
        model_data.get("interpreted_description")
        or model_data.get("yml_description")
        or "No description provided."
    )
    cols_data = (
        model_data.get("yml_columns") or model_data.get("interpreted_columns") or {}
    )

    cols_str = ""
    if isinstance(cols_data, dict) and cols_data:
        col_items = []
        for col_name, col_info in cols_data.items():
            col_desc = (
                col_info.get("description", "") if isinstance(col_info, dict) else ""
            )
            col_items.append(
                f"    - `{col_name}`: {col_desc}" if col_desc else f"    - `{col_name}`"
            )
        cols_str = "\n".join(col_items)
    elif isinstance(
        cols_data, list
    ):  # Handle simple list of names case, though dict is expected
        cols_str = ", ".join([f"`{c}`" for c in cols_data])
    else:
        cols_str = "  Column details not available or in unexpected format."

    fetch_method = model_data.get("fetch_method", "unknown")
    search_score = model_data.get("search_score")
    score_info = (
        f" (Similarity: {search_score:.2f})"
        if fetch_method == "vector_search" and search_score is not None
        else ""
    )

    formatted = f"\n*   **Model:** `{name}`{score_info}\n"
    formatted += f"    *   Description: {desc}\n"
    formatted += f"    *   Columns:\n{cols_str}\n"

    # Optionally include raw_sql snippet
    # raw_sql = model_data.get('raw_sql', '')
    # if raw_sql:
    #     formatted += f"    *   Raw SQL Snippet: ```sql\n{raw_sql[:150]}...\n```\n"

    return formatted


def _format_question_for_prompt(q: Union[Question, Dict[str, Any]], prefix: str) -> str:
    """Helper to format Question instance or dict for prompt context."""
    if isinstance(q, dict):
        q_id = q.get("id", "N/A")
        q_text = q.get("question_text", q.get("original_message_text", "N/A"))
        a_text = q.get("answer_text", "N/A")
        f_text = q.get("feedback")
        was_useful = q.get("was_useful")
    else:  # Assume Django model instance
        q_id = q.pk
        q_text = q.question_text or q.original_message_text or "N/A"
        a_text = q.answer_text or "N/A"
        f_text = q.feedback
        was_useful = q.was_useful

    formatted = f"{prefix} ID: {q_id}, Question: {q_text}\n"
    if a_text and a_text != "N/A":
        formatted += f"     Answer Provided: {a_text[:200]}...\n"
    if f_text:
        formatted += f"     Feedback Text: {f_text[:150]}...\n"
    if was_useful is not None:
        formatted += (
            f"     Feedback Rating: {'👍 Useful' if was_useful else '👎 Not Useful'}\n"
        )
    return formatted


def create_system_prompt(
    all_models_summary: List[Dict[str, str]],
    relevant_feedback_by_question: List[Any],  # Can be Question instances or dicts
    relevant_feedback_by_content: List[Any],  # Can be Question instances or dicts
    similar_original_messages: List[Dict[str, Any]],  # Should be dicts from tool
    accumulated_models: List[Dict[str, Any]],  # Should be dicts from tools
    search_model_calls: int,
    max_vector_searches: int,
    data_warehouse_type: Optional[str] = None,
) -> str:
    """Generates the system prompt for the QuestionAnswerer agent."""

    # --- Basic Prompt Structure ---
    sql_dialect_instruction = (
        f"The SQL you generate MUST be compatible with {data_warehouse_type}."
        if data_warehouse_type
        else "Assume standard SQL (compatible with common warehouses like Snowflake, BigQuery, Redshift)."
    )

    prompt = f"""You are an expert AI assistant specialized in answering questions about dbt models using SQL. Your goal is to generate accurate SQL queries based *only* on the provided dbt models and context.

**Constraints & Style Guide:**
*   **Grounding:** Base your SQL query and explanations *strictly* on the provided dbt models' structure (columns, descriptions). DO NOT HALLUCINATE tables, columns, or relationships not present in the provided model details.
*   **Table Referencing:** Always use fully qualified table names (e.g., `database_name.schema_name.table_name`) in your SQL queries. Do NOT use dbt `{{ ref() }}` macros or similar Jinja templating.
*   **SQL Dialect:** {sql_dialect_instruction}
*   **Clarity:** Prefer CTEs for complex logic. Alias tables clearly. Use `SAFE_CAST` or similar functions for potentially problematic type conversions.
*   **SQL Styling:** Strictly adhere to the formatting shown in the 'SQL Style Guide Example' below. This includes placing comments describing CTEs *above* the CTE definition, formatted as shown (e.g., `--\n-- comment about CTE\n--`), and using inline comments (`-- explanation`) for specific lines or logic within the SQL where appropriate.
*   **Explanations:** Add concise comments within the SQL (`-- explanation`) to clarify joins or complex logic.
*   **Limitations:** If the available models are insufficient to fully answer the question, generate the best possible query using *only* the available models and clearly state these limitations as a list of bullet points (e.g., `* Limitation 1\n* Limitation 2`) *after* the SQL block. Do NOT add a "Footnotes:", "Notes:", or any other heading to this list of bullet points; only provide the raw bullet points. The system will handle the appropriate header.
*   **Output Format:** The SQL query itself should be plain text, adhering to the style guide below. Do NOT wrap the SQL query in triple backticks or any other Slack markdown within the `final_answer` parameter of the `finish_workflow` call; the display will be handled by the Slack integration.
*   **Tool Use:**
    *   Use `fetch_model_details` if you need the schema or detailed info for models identified from the initial list.
    *   Use `model_similarity_search` *only* as a fallback if the initial list is insufficient or you need models related to specific concepts not obvious from names/descriptions. You have a limit of {max_vector_searches} vector searches (used: {search_model_calls}).
    *   Use `search_past_feedback`, `search_feedback_content`, and `search_organizational_context` sparingly to clarify ambiguous terms or find established definitions, if the initial context is insufficient.
    *   Use `finish_workflow` ONLY when you have the complete final answer (SQL + explanations/footnotes).
"""

    # --- SQL Style Guide Example ---
    sql_style_example_content = """
with

--
cte_without_comment as (
    select
        *

    from
        database_name.schema_name.table_name

    where
        this_condition is true
        and (
            that_condition is true
            or other_condition is false
        )
),

-- 
-- this cte only has a small comment
--
cte_with_small_comment as (
    select
        *

    from
        cte_without_comment      
)

--
-- this cte has a few more styling examples and comes with a comment that_condition
-- spans multiple lines
--
cte_with_comment as (
    select
        this_columns,

        case
            when that_condition is true
            then this_value
            else that_value
        end as new_column,

        count(distinct this_column) over (
            partition by that_column
            order by this_column
            rows between 1 preceding and current row
        ) as new_column_from_window_function

    from
        cte_without_comment
)

--
select 
    wc.this_column,
    wc.that_column,
    wsc.new_column,
    wsc.new_column_from_window_function
    
from 
    cte_with_comment as wc
left join
    cte_with_small_comment as wsc
        on wc.this_column = wsc.this_column
    """

    prompt += f"""

**SQL Style Guide Example:**
```sql
{{sql_style_example_content}}
```
"""

    prompt += f"""

**Available dbt Models (Summary):**
You have access to the following models. Use `fetch_model_details` to get column details if needed.
"""

    if all_models_summary:
        for model in all_models_summary:
            prompt += (
                f"- **{model.get('name', 'N/A')}**: {model.get('description', 'N/A')}\n"
            )
    else:
        prompt += "- No models available in summary. Use search tools if necessary.\n"

    # --- Context Sections ---
    if similar_original_messages:
        prompt += "\n**Potentially Relevant Organizational Context (from past user questions):**\n"
        for i, interaction in enumerate(similar_original_messages[:3]):  # Limit display
            prompt += _format_question_for_prompt(interaction, prefix=f"{i+1}.")

    if relevant_feedback_by_question or relevant_feedback_by_content:
        prompt += "\n**Potentially Relevant Past Feedback:**\n"
        if relevant_feedback_by_question:
            prompt += "*From similar past questions:*\n"
            for i, feedback_item in enumerate(relevant_feedback_by_question[:2]):
                prompt += _format_question_for_prompt(
                    feedback_item, prefix=f"  {i+1}. "
                )
        if relevant_feedback_by_content:
            prompt += "*Mentioning related terms:*\n"
            for i, feedback_item in enumerate(relevant_feedback_by_content[:2]):
                prompt += _format_question_for_prompt(
                    feedback_item, prefix=f"  {i+1}. "
                )

    # --- Accumulated Models ---
    if accumulated_models:
        prompt += (
            "\n**Retrieved Model Details (Your SOLE source for SQL generation):**\n"
        )
        prompt += "You MUST limit your SQL query to use ONLY the tables and columns detailed in this section. Do not use any other tables, even if mentioned in the initial summary or elsewhere, unless their full details are retrieved and listed here.\n"
        for model_data in accumulated_models:
            prompt += _format_model_for_prompt(model_data)

    # Append custom rules
    custom_rules = get_agent_rules("question_answerer")
    if custom_rules:
        prompt += (
            f"\n\n**Additional Instructions (from .ragstarrules.yml):**\n{custom_rules}"
        )

    prompt += "\nBased on the user's question and the available information/tools, decide the next step. If ready, call `finish_workflow` with the final SQL query and explanations."
    return prompt


def create_guidance_message(
    search_model_calls: int,
    max_vector_searches: int,
    accumulated_models: List[Dict[str, Any]],
) -> Optional[str]:
    """Generates a guidance message for the LLM if needed."""
    guidance_parts = []

    if search_model_calls >= max_vector_searches:
        guidance_parts.append(
            f"You have reached the maximum ({max_vector_searches}) vector searches."
        )
    else:
        guidance_parts.append(
            f"You have {max_vector_searches - search_model_calls} vector searches remaining."
        )

    if not accumulated_models:
        guidance_parts.append(
            "You haven't retrieved any specific model details yet. Use `fetch_model_details` or `model_similarity_search`."
        )
    else:
        guidance_parts.append(
            f"You have retrieved details for: {', '.join([m.get('name', '?') for m in accumulated_models])}."
        )

    # Combine guidance parts if any exist
    if guidance_parts:
        return (
            "**Guidance:** "
            + " ".join(guidance_parts)
            + " Decide your next step or call `finish_workflow` if ready."
        )
    else:
        return None  # No specific guidance needed
