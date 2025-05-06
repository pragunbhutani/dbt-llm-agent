"""Prompts for the QuestionAnswerer Agent."""

from typing import List, Dict, Any, Optional, Set

# --- NEW IMPORT ---
from ragstar.core.agents.rules import load_ragstar_rules

# --- END NEW IMPORT ---


def create_system_prompt(
    all_models_summary: List[Dict[str, str]],
    relevant_feedback_by_question: List[Any],
    relevant_feedback_by_content: List[Any],
    similar_original_messages: List[Dict[str, Any]],
    accumulated_models: List[Dict[str, Any]],
    search_model_calls: int,
    max_vector_searches: int,
) -> str:
    """Generates the system prompt for the QuestionAnswerer agent node."""

    # Convert model summary list to a more readable string format for the prompt
    model_list_str = "\n".join(
        [f"- `{m['name']}`: {m['description']}" for m in all_models_summary]
    )
    if not model_list_str:
        model_list_str = "(No usable models found in storage)"

    # Format pre-fetched context/feedback for the prompt
    # Safely format feedback by question
    feedback_q_list = []
    for item in relevant_feedback_by_question:
        q_text = getattr(item, "question_text", "N/A")
        a_text = getattr(item, "answer_text", "N/A")
        f_text = getattr(item, "feedback", "N/A")
        useful = getattr(item, "was_useful", "N/A")
        feedback_q_list.append(
            f"  - Question: {q_text}\n    Answer: {a_text[:100]}...\n    Feedback: {f_text}\n    Useful: {useful}"
        )
    feedback_q_str = "\n".join(feedback_q_list) if feedback_q_list else "None found."

    # Safely format feedback by content
    feedback_c_list = []
    for item in relevant_feedback_by_content:
        q_text = getattr(item, "question_text", "N/A")
        a_text = getattr(item, "answer_text", "N/A")
        f_text = getattr(item, "feedback", "N/A")
        useful = getattr(item, "was_useful", "N/A")
        feedback_c_list.append(
            f"  - Found in Feedback for Question: {q_text}\n    Answer: {a_text[:100]}...\n    Feedback Snippet: {f_text}...\n    Useful: {useful}"
        )
    feedback_c_str = "\n".join(feedback_c_list) if feedback_c_list else "None found."

    # Safely format similar original messages
    similar_msg_list = []
    for (
        item_dict
    ) in similar_original_messages:  # Already converted to dicts in pre-fetch
        orig_q = item_dict.get("original_message_text", "N/A")
        ans_text = item_dict.get("answer_text", "N/A")
        similar_msg_list.append(
            f"  - Past Question: {orig_q}\n    Past Answer: {ans_text[:100]}..."
        )
    similar_msg_str = "\n".join(similar_msg_list) if similar_msg_list else "None found."

    system_prompt = f"""You are an AI assistant specialized in analyzing dbt projects and generating SQL queries based ONLY on provided dbt model context.

    **Overall Goal:** Answer the user's question (`original_question`) by:
    1. Understanding the question, the provided Slack thread context (`thread_context`), and the pre-fetched historical context/feedback (provided below).
    2. **Examining the list of available models** (provided below) to identify potentially relevant ones based on names and descriptions.
    3. **Prioritizing `fetch_model_details`:** Use this tool first to get details for models selected from the list.
    4. **Using `model_similarity_search` as a fallback:** If the initial list doesn't reveal clear candidates, or if `fetch_model_details` doesn't yield enough information, use vector similarity search to find models related to specific concepts or calculations.
    5. Synthesizing information from models, pre-fetched context/feedback, and the thread context.
    6. Generating a final, grounded SQL query and explanation using the `finish_workflow` tool.
    7. **Using Search Tools Sparingly:** Tools like `search_organizational_context`, `search_past_feedback`, and `search_feedback_content` are available but should only be used if the initial pre-fetched information is insufficient and you need to search for *different* or *more specific* information based on your intermediate findings (e.g., a model detail mentions a term not in the pre-fetched context).

    **CRITICAL RULE:** Your final action in this workflow MUST be a call to the `finish_workflow` tool. Do NOT output plain text or ask clarification questions as your final response. If you cannot fully answer the question with the available information, you MUST still call `finish_workflow` and explain the limitations in the 'Footnotes' section of the `final_answer`.

    **Input Context:**
    - `original_question`: The primary question, potentially compiled from the thread.
    - `thread_context`: The history of the Slack conversation leading to the question. Use this to understand nuances, definitions, or constraints provided by the user.
    - **Pre-fetched Similar Original Messages:**
{similar_msg_str}
    - **Pre-fetched Feedback (Similar Questions):**
{feedback_q_str}
    - **Pre-fetched Feedback (Similar Content):**
{feedback_c_str}
    - **Available Models:**
{model_list_str}

    **Tool Usage Strategy:**
    1.  **Analyze 'Available Models' list AND Pre-fetched Context/Feedback first.** Based on `original_question`, `thread_context`, and the pre-fetched data, identify models whose names/descriptions seem relevant. Determine if pre-fetched info helps clarify the question or provides relevant definitions.
    2.  **Use `fetch_model_details`** to get schemas for models identified from the list.
    3.  **Use `model_similarity_search` ONLY if:**
        *   The 'Available Models' list and fetched details don't contain obvious candidates.
        *   You need to search for models based on a specific concept or calculation not covered by pre-fetched context (e.g., 'how is revenue calculated?'). Refine the query based on context found.
        *   Limit: {max_vector_searches} calls. Use specific table/column names from `thread_context` if available.
    4.  **Use Context/Feedback Search Tools (`search_organizational_context`, `search_past_feedback`, `search_feedback_content`) ONLY if** the initial pre-fetched data is insufficient and you need to probe for *different* or *more specific* information based on your intermediate findings (e.g., a model detail mentions a term not in the pre-fetched context). Use targeted queries.
    5.  **Use `finish_workflow`** ONLY when ready for the complete, final answer. Base SQL *strictly* on retrieved models, considering context from `thread_context` and *all* available feedback/context (pre-fetched and potentially from tool calls).

    **SQL Generation Rules (CRITICAL):**
    - **Grounding:** ONLY use tables, columns, relationships from `accumulated_models`. DO NOT HALLUCINATE.
    - **Completeness:** Generate best possible SQL using available info. State limitations (from missing models or unclear `thread_context` requests) in comments or 'Footnotes'.
    - **Style:** Follow the provided SQL style guide example.
    - **No Slack Mrkdwn:** Do not use Slack formatting in `final_answer`.

    **Current State:**
    - Models Found: {len(accumulated_models)}
    - Vector Similarity Search Calls Used: {search_model_calls} / {max_vector_searches}
    - Pre-fetched Feedback (Qs): {len(relevant_feedback_by_question)} items
    - Pre-fetched Feedback (Content): {len(relevant_feedback_by_content)} items
    - Pre-fetched Org Context (Msgs): {len(similar_original_messages)} items
    """

    # --- MODIFIED: Embed SQL style example directly --- #
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

    system_prompt += f"""

**SQL Style Guide Example:**
```sql
{sql_style_example_content}
```"""
    # --- END MODIFIED --- #

    # --- NEW: Load and append custom rules ---
    all_rules = load_ragstar_rules()
    custom_rules = all_rules.get("question_answerer", "")

    if custom_rules:
        final_prompt = (
            system_prompt + "\n\n**Additional Instructions:**\n" + custom_rules
        )
        return final_prompt
    else:
        return system_prompt
    # --- END NEW ---


def create_guidance_message(
    search_model_calls: int,
    max_vector_searches: int,
    accumulated_models: List[Dict[str, Any]],
) -> Optional[str]:
    """Generates the guidance message for the QuestionAnswerer agent node."""
    guidance_items = []

    if search_model_calls >= max_vector_searches:
        guidance_items.append(
            f"You have reached the maximum ({max_vector_searches}) vector similarity searches."
        )
        if not accumulated_models:
            guidance_items.append(
                "No relevant models were found. You MUST now call `finish_workflow` and explain in the 'Footnotes' that you cannot answer the question due to missing model context."
            )
        else:
            guidance_items.append(
                "You MUST now use the `finish_workflow` tool to generate the final answer based *only* on the models found so far, noting any limitations."
            )
    else:
        guidance_items.append(
            f"You have used the vector similarity search tool {search_model_calls} times (max {max_vector_searches})."
        )
        if not accumulated_models:
            guidance_items.append(
                "No models found yet. Examine the 'Available Models' list. If relevant models seen, use `fetch_model_details`. Consider `search_organizational_context` if the question has specific terms needing definition. Otherwise, consider `model_similarity_search` with a specific query, or `search_past_feedback` if appropriate."
            )
        else:
            model_names = [m["name"] for m in accumulated_models]
            guidance_items.append(
                f"Analyze models found so far: {model_names}. Do these + context/feedback answer the question? If yes, call `finish_workflow`. If not, can `fetch_model_details` get more? Is `search_organizational_context` needed for terms? Only use `model_similarity_search` if needed and under limit ({max_vector_searches}). Finish or fetch/search."
            )

    if guidance_items:
        return "Guidance: " + " ".join(guidance_items)
    else:
        return None
