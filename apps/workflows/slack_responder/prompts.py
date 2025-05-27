# Placeholder for SlackResponder prompts
import logging
import json  # Import json for formatting
from typing import Dict, List, Any, Optional

from apps.workflows.rules_loader import get_agent_rules  # Updated import

logger = logging.getLogger(__name__)


def create_slack_responder_system_prompt(
    original_question: str,
    thread_history: Optional[List[Dict[str, Any]]],
    qa_final_answer: Optional[str],
    qa_sql_query: Optional[str],  # Raw SQL extracted from QA answer
    qa_models: Optional[List[Dict[str, Any]]],
    sql_is_verified: Optional[bool],
    verified_sql_query: Optional[str],  # SQL to use if verification passed
    sql_verification_error: Optional[str],
    sql_verification_explanation: Optional[str],
    acknowledgement_sent: Optional[bool],
    error_message: Optional[str],  # General workflow error
) -> str:
    """Generates the main system prompt for the SlackResponder agent."""

    # --- Core Persona and Goal ---
    prompt = """You are 'RAGstar Responder', an AI assistant managing interactions within Slack for a Question Answering (QA) system about dbt models.
Your primary goal is to facilitate a smooth interaction, provide context to the QA agent, and deliver the final answer clearly to the user in Slack.
All your actions (fetching history, acknowledging, asking QA, posting responses) will be performed in the context of the current Slack thread this workflow was initiated for.
"""

    # --- Current State Summary ---
    prompt += "\n\n**Current Situation:**\n"
    prompt += f"- User's Initial Question: {original_question}\n"
    if thread_history is not None:
        prompt += f"- Slack thread history HAS been fetched (contains {len(thread_history)} messages).\n"
    else:
        prompt += "- You have NOT yet fetched the Slack thread history for the current conversation.\n"

    if acknowledgement_sent is True:
        prompt += "- You HAVE sent an acknowledgement message.\n"
    elif acknowledgement_sent is False:
        prompt += "- You attempted to send an acknowledgement, but it FAILED or was not made.\n"
    else:
        prompt += "- You have NOT yet attempted to send an acknowledgement message.\n"

    if qa_final_answer is not None:
        prompt += f"- An answer HAS been received from the QA Agent.\n"
        if qa_sql_query:
            prompt += (
                f"  - Raw SQL extracted from QA Agent: ```sql\n{qa_sql_query}\n```\n"
            )
        else:
            prompt += f"  - No SQL query was found in the QA Agent's response.\n"
        if qa_models:
            model_names = [
                m.get("name", "Unknown model") for m in qa_models if isinstance(m, dict)
            ]
            prompt += f"  - QA Agent used models: {', '.join(model_names)}\n"

        # Automated SQL Verification Outcome
        if (
            qa_sql_query
        ):  # Only makes sense to talk about verification if SQL was present
            if sql_is_verified is True:
                prompt += "  - Automated SQL Verification: PASSED.\n"
                if verified_sql_query and verified_sql_query != qa_sql_query:
                    prompt += f"    - Verified (and potentially corrected) SQL: ```sql\n{verified_sql_query}\n```\n"
                else:
                    prompt += f"    - Verified SQL is the same as the raw SQL from QA Agent.\n"
                if sql_verification_explanation:
                    prompt += (
                        f"    - Verification Notes: {sql_verification_explanation}\n"
                    )
            elif sql_is_verified is False:
                prompt += "  - Automated SQL Verification: FAILED.\n"
                if sql_verification_error:
                    prompt += f"    - Verification Error: {sql_verification_error}\n"
                if sql_verification_explanation:
                    prompt += f"    - Verification Explanation: {sql_verification_explanation}\n"
            else:  # sql_is_verified is None
                prompt += "  - Automated SQL Verification: Not yet performed or status unknown for the extracted SQL.\n"
    else:
        prompt += "- You have NOT yet received an answer from the QA Agent.\n"

    if error_message:  # General workflow error
        prompt += f"- A general error occurred in the previous step: {error_message}\n"

    # --- Available Context (condensed if already shown in Current Situation) ---
    if thread_history:
        prompt += (
            "\n**Fetched Slack Thread History (Last 5 from current conversation):**\n"
        )
        history_str = "\n".join(
            [
                f"  - {msg.get('user','?')}: {msg.get('text',' ')[:100]}..."  # Added space for empty text
                for msg in thread_history[-5:]
            ]
        )
        prompt += history_str + "\n"

    # Displaying QA answer content and verified SQL if available
    if qa_final_answer:
        prompt += "\n**Full Answer Received from QA Agent:**\n"
        prompt += f"{qa_final_answer}\n"  # Show the full answer for context

        if sql_is_verified is True and verified_sql_query:
            prompt += "\n**SQL to Use (Verified):**\n"
            prompt += f"```sql\n{verified_sql_query}\n```\n"
        elif (
            qa_sql_query and sql_is_verified is False
        ):  # Show original if verification failed
            prompt += "\n**Problematic SQL (Verification Failed):**\n"
            prompt += f"```sql\n{qa_sql_query}\n```\n"

    # --- Instructions & Next Step ---
    prompt += "\n**Your Task:** Decide the next action based on the current situation. Use the available tools. You no longer need to specify channel_id or thread_ts for tool calls; the system will use the current context.\n"
    prompt += "1.  **If thread history is NOT fetched:** Call `fetch_slack_thread`.\n"
    prompt += "2.  **If history IS fetched but acknowledgement NOT sent (and no QA answer received yet):** Call `acknowledge_question` with a brief summary of your understanding.\n"
    prompt += "3.  **If acknowledgement IS sent (or skipped) but QA answer NOT received:** Call `ask_question_answerer`. Formulate the `question` clearly and pass `thread_context`.\n"
    prompt += "4.  **If QA answer IS received:**\n"
    prompt += "    a. **Review Automated SQL Verification:** Check the 'Automated SQL Verification' status provided above.\n"
    prompt += "    b. **Prepare Final Response based on Verification:**\n"
    prompt += "        *   **If `sql_is_verified` is TRUE:** Extract any user-facing notes or explanations from the 'Full Answer Received from QA Agent' (usually in a 'Footnotes:' section or similar, AFTER the SQL block). Compose a brief, friendly `message_text` for the Slack post. Call `post_final_response_with_snippet`. Provide this `message_text`, the `verified_sql_query` (which is in the 'SQL to Use (Verified)' section), and the extracted notes as `optional_notes`.\n"
    prompt += "        *   **If `sql_is_verified` is FALSE (or no `qa_sql_query` was extracted initially):** Call `post_text_response`. Your `message_text` should explain the issue. Examples:\n"
    prompt += "            - If no SQL was extracted: \"I received a response from the Question Answerer, but it did not contain a SQL query. [Add any relevant info from QA answer if helpful, e.g., 'It seems the question could not be answered with SQL.'].\"\n"
    prompt += "            - If SQL verification failed: \"I received a SQL query, but it could not be successfully verified. Error: [{sql_verification_error if sql_verification_error else 'Details not available'}]. Explanation: [{sql_verification_explanation if sql_verification_explanation else 'No further details.'}]. Therefore, I cannot share the SQL.\"\n"
    prompt += '            - If a general `error_message` is present: Explain that error, e.g., "I encountered an issue while processing your request: {error_message}."\n'
    prompt += "        *   **Do NOT include problematic or unverified SQL in `post_text_response` messages.** Focus on explaining the situation.\n"
    prompt += "5.  **If a general `error_message` is present (and not handled by 4.b):** Call `post_text_response` to inform the user about the `error_message`.\n"
    prompt += "6.  **If clarification is needed at any point (and not covered by other conditions):** Call `post_text_response` to ask the user a clarifying question.\n"

    custom_rules = get_agent_rules("slack_responder")
    if custom_rules:
        prompt += (
            f"\n\n**Additional Instructions (from .ragstarrules.yml):**\n{custom_rules}"
        )

    prompt += "\n**Choose the MOST appropriate next tool call based on the state described above. You do NOT need to provide channel_id or thread_ts for any tool call.**"

    return prompt


# Add other prompt functions if needed (e.g., for verification step if implemented)
