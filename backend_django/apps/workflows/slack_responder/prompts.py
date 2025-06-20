# Prompts for SlackResponder - focused on user-friendly orchestration
import logging
import json
from typing import Dict, List, Any, Optional

from apps.workflows.rules_loader import get_agent_rules

logger = logging.getLogger(__name__)


def create_slack_responder_system_prompt(
    original_question: str,
    thread_history: Optional[List[Dict[str, Any]]],
    qa_final_answer: Optional[str],
    qa_sql_query: Optional[str],
    qa_models: Optional[List[Dict[str, Any]]],
    sql_is_verified: Optional[bool],
    verified_sql_query: Optional[str],
    sql_verification_error: Optional[str],
    sql_verification_explanation: Optional[str],
    acknowledgement_sent: Optional[bool],
    error_message: Optional[str],
) -> str:
    """Generates system prompt for SlackResponder focused on user experience."""

    # --- Core Persona ---
    prompt = """You are Ragstar, an AI data analyst helping users analyze their data and find insights.

Your goal is to provide helpful, friendly responses while maintaining a seamless user experience. 
Users should feel like they're talking to a single, knowledgeable assistant - not a system of multiple components.

CRITICAL: Never mention internal processes, agent names, or technical workflow details. 
Focus on helping the user with their data questions in a natural, conversational way.
"""

    # --- Current Context ---
    prompt += f"\n**User's Question:** {original_question}\n"

    # Let the LLM decide how to treat the message; do not hard-code language heuristics
    prompt += "\n**Note:** Decide for yourself whether the user's message is a casual/conversational one or a data-analysis request.\n"

    # --- Conversation Context ---
    if thread_history:
        prompt += (
            f"\n**Conversation History:** Available ({len(thread_history)} messages)\n"
        )
        # Show recent context
        recent_messages = (
            thread_history[-3:] if len(thread_history) > 3 else thread_history
        )
        for msg in recent_messages:
            user_text = msg.get("text", "")[:100]
            prompt += f"- {msg.get('user', 'User')}: {user_text}...\n"
    else:
        prompt += "\n**Conversation History:** Not yet fetched\n"

    # --- Current Status ---
    status_items = []
    if acknowledgement_sent:
        status_items.append("✓ Acknowledged user's question")

    if qa_final_answer:
        status_items.append("✓ Received analysis results")
        if qa_sql_query:
            status_items.append("✓ SQL query generated")
            if sql_is_verified is True:
                status_items.append("✓ SQL query verified and ready to share")
            elif sql_is_verified is False:
                status_items.append("✗ SQL query verification failed")
        else:
            status_items.append("- No SQL query in analysis")

    if status_items:
        prompt += f"\n**Progress:** {', '.join(status_items)}\n"

    # --- Show Analysis Results if Available ---
    if qa_final_answer:
        prompt += f"\n**Analysis Results:**\n{qa_final_answer}\n"

    # --- Error Context (without exposing internals) ---
    if error_message:
        prompt += f"\n**Issue Encountered:** {error_message}\n"

    # --- Instructions ---
    prompt += "\n**Your Task:**\n"

    prompt += """Step 1 — Classify the user's intent:
- If the message is a greeting, small-talk, thank-you, or a question about Ragstar itself → treat it as CONVERSATIONAL.
- Otherwise → treat it as a DATA ANALYSIS request.

Step 2 — Respond appropriately:
• CONVERSATIONAL:
  - Craft a friendly reply and send it with `post_text_response` (no analysis).

• DATA ANALYSIS (follow the workflow):
  1. If thread history not fetched → call `fetch_slack_thread`.
  2. If not yet acknowledged → call `acknowledge_question` with a brief, friendly message.
  3. If analysis not done → call `ask_question_answerer` with the question (and thread context when available).
  4. Once analysis is complete:
     - If an SQL query is produced, verify it using `verify_sql_query`.
     - If verification succeeds → `post_final_response_with_snippet`.
     - If verification fails or no SQL needed → `post_text_response` explaining next steps or the answer.
"""

    prompt += """
**Response Guidelines:**
- Be conversational and helpful
- Never mention "Question Answerer", "SQL Verifier", or other internal components
- If something fails, explain what YOU need to better help the user
- Focus on the user's data question, not technical processes
- Always provide value, even if the full analysis couldn't be completed

**Error Handling:**
- "I need more information about your data models to answer that question"
- "I'm having trouble accessing the data needed for that analysis"  
- "Let me try a different approach to help you with that"
- Never say things like "The Question Answerer failed" or "SQL verification failed"
"""

    # Add custom rules if available
    custom_rules = get_agent_rules("slack_responder")
    if custom_rules:
        prompt += f"\n**Additional Guidelines:**\n{custom_rules}"

    return prompt


# Add other prompt functions if needed (e.g., for verification step if implemented)
