import logging
import json
import asyncio  # Added for potential LLM client delays
import uuid
from datetime import datetime, timezone
from typing import Optional  # Added for type hinting bot_user_id

# Assuming BaseChatModel is the type hint for the llm_client
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
)  # For type hinting if llm_client.ainvoke returns AIMessage

logger = logging.getLogger(__name__)


class KnowledgeExtractorWorkflow:
    def __init__(
        self, llm_client: BaseChatModel, bot_user_id_to_ignore: Optional[str] = None
    ):
        """
        Initializes the workflow with an LLM client.
        llm_client: An initialized chat LLM client (e.g., from llm_providers.ChatService).
        bot_user_id_to_ignore: Optional bot user ID to ignore messages from.
        """
        self.llm_client = llm_client
        self.bot_user_id_to_ignore = bot_user_id_to_ignore
        if self.bot_user_id_to_ignore:
            logger.info(
                f"KnowledgeExtractorWorkflow initialized to ignore messages from bot_user_id: {self.bot_user_id_to_ignore}"
            )

    def _format_thread_for_llm(self, thread_messages: list) -> str:
        """
        Formats the Slack thread messages into a single string for the LLM.
        Each message includes the user and the text.
        """
        intro_message = (
            f"The following is a conversation thread from Slack. Please analyze it to extract key learnings based on human statements and the query debugging process.\n"
            f"Each message is prefixed with 'User (user_id):'.\n"
        )
        if self.bot_user_id_to_ignore:
            intro_message += (
                f"Messages from the bot itself (User ({self.bot_user_id_to_ignore})) represent the system's actions or reports. \n"
                f"Do NOT treat the bot's rephrasing of a user's question or general conversational chatter as learned knowledge. \n"
                f"However, DO pay attention to bot messages that report specific outcomes of data operations, such as query errors, corrections made during query execution, or final data results if they are then discussed or validated by humans. These can be sources of learned knowledge.\n\n"
            )
        else:
            intro_message += "\n\n"

        processed_messages = []
        for message in thread_messages:
            user = message.get("user")
            text = message.get("text", "").strip()

            if self.bot_user_id_to_ignore and user == self.bot_user_id_to_ignore:
                # Heuristic to include bot messages that seem to convey factual data from operations
                if any(
                    keyword in text.lower()
                    for keyword in [
                        "error:",
                        "failed:",
                        "database error:",
                        "query result:",
                        "correction made:",
                        "identified issue:",
                        "query succeeded after",
                    ]
                ):
                    logger.info(
                        f"Including potentially important message from bot {user}: '{text[:100]}...'"
                    )
                    processed_messages.append(
                        f"Bot ({user}) reported data/outcome: {text}\n---\n"
                    )
                else:
                    logger.info(
                        f"Ignoring general conversational message from bot_user_id: {user}"
                    )
                continue

            processed_messages.append(f"User ({user or 'UnknownUser'}): {text}\n---\n")

        if not processed_messages:
            logger.warning(
                "No messages (human or key bot reports) found in the thread after filtering. LLM might not have context."
            )
            return (
                intro_message + "No relevant messages found in the thread for analysis."
            )

        return intro_message + "".join(processed_messages)

    def _construct_llm_prompt(self, formatted_thread_content: str) -> str:
        """
        Constructs the detailed prompt for the LLM, asking for a JSON array of learning objects.
        """
        prompt = f"""{formatted_thread_content}

        Based on the Slack conversation thread above, your task is to identify and structure knowledge that is **explicitly stated, corrected, or confirmed by humans**, OR knowledge that arises from **factual reports from the Bot about the query debugging process** (e.g., errors encountered, corrections it applied, reasons for query failure/success if stated as fact by the bot and subsequently acted upon or acknowledged by humans).

        **Crucially, IGNORE the Bot's own interpretations, rephrasing of user questions, or general conversational messages as sources of learned knowledge.** Focus on human statements and the Bot's factual reports on data operations and their outcomes.

        Format your response as a single, valid JSON array. Each object in the array should represent a distinct piece of learning and must conform to the following structure:
        {{
          "knowledge_type": "string (must be one of: 'organisational_knowledge', 'column_particularity', 'model_structure_change', 'model_comparison')",
          "content": {{ /* Object whose structure depends on knowledge_type */ }},
          "keywords": ["list", "of", "relevant", "search", "terms"],
          "potential_source_indicators": ["User ID (e.g., U12345) who stated this", "Direct quote or close paraphrase from a human supporting this learning", "Reference to a specific error message discussed", "Bot (BOT_USER_ID) reported factual outcome of an operation"],
          "confidence_estimate": 0.0 // A float between 0.0 (low) and 1.0 (high), reflecting certainty based on explicit human statements or clear evidence from the debugging process.
        }}

        Here are the specific `knowledge_type` categories and the expected structure for their `content` object:

        1.  `knowledge_type: "organisational_knowledge"` (Extract ONLY if a human explicitly defines or explains a term/concept in the thread. Bot explanations of terms are NOT organisational knowledge unless confirmed by a human as definitive.)
            `content` structure:
            {{
              "term": "The specific term, acronym, or concept defined by a human user",
              "definition_explanation": "The detailed definition or explanation as provided by a human user.",
              "aliases_or_related_terms": ["Optional", "list", "of", "aliases mentioned by a human user"]
            }}

        2.  `knowledge_type: "model_comparison"` (Extract ONLY if a human explicitly compares models or states a preference for a specific context. Bot suggestions are NOT knowledge unless confirmed by a human.)
            `content` structure:
            {{
              "subject_description": "Brief description of what is being compared, as stated by a human user (e.g., 'Choosing between sales_v1 and sales_v2 for daily reports').",
              "models": [
                {{"name": "Model A Name", "description": "Role or characteristic of Model A as stated by a human user"}},
                {{"name": "Model B Name", "description": "Role or characteristic of Model B as stated by a human user"}}
              ],
              "comparison_insight": "The core insight or conclusion from the comparison, as stated by a human user.",
              "preferred_model": "Optional: Name of the preferred model if explicitly stated by a human user for the context."
            }}

        3.  `knowledge_type: "column_particularity"` (Focus on human corrections/clarifications OR factual outcomes reported by the Bot from query execution/debugging if these outcomes are then discussed or implicitly accepted by humans.)
            `content` structure:
            {{
              "entity_name": "Name of the table, model, or data entity as identified by a user, in an error message, or in a bot's factual report from query execution",
              "column_name": "Name of the column discussed",
              "property_type": "Type of particularity (e.g., 'valid_values_mapping', 'data_type_correction', 'usage_correction', 'non_existent_column', 'business_rule_clarification')",
              "property_details": {{ /* Flexible object for specifics. Example: {{"incorrect_usage_or_value": "value X was used", "correct_usage_or_value": "value Y should be used", "reason_for_correction": "User Z explained that X is outdated OR Bot reported correcting X to Y after query failure."}} */ }}
            }}

        4.  `knowledge_type: "model_structure_change"` (Identify based on human statements OR Bot reports of schema changes/discoveries from query errors that are then acted upon or acknowledged by humans.)
            `content` structure:
            {{
              "entity_name": "Name of the table or model affected",
              "element_type": "Type of element changed (e.g., 'column', 'table')",
              "element_name": "Name of the element (e.g., column name)",
              "change_type": "Type of change discussed (e.g., 'deprecation_confirmed_by_human', 'type_mismatch_identified_by_bot_error_report', 'column_missing_per_db_error', 'column_to_be_added_per_human_request')",
              "description_of_change": "Detailed description of what was identified by a human or factually reported by the Bot from a data operation.",
              "old_value_or_schema": "Optional: Previous value or schema detail if mentioned by a user or in a bot-reported error/correction",
              "new_value_or_schema": "Optional: New/correct value or schema detail if mentioned by a user or in a bot-reported correction",
              "reason_for_change": "Optional: Reason if stated by a user or implied by a bot's error correction report"
            }}

        Example JSON Array Output (focus on human-derived facts and factual bot reports from query debugging):
        ```json
        [
          {{
            "knowledge_type": "column_particularity",
            "content": {{
              "entity_name": "dim_user_location",
              "column_name": "current_state",
              "property_type": "valid_values_mapping",
              "property_details": {{
                "context": "Filtering users by state for Rajasthan.",
                "incorrect_value_used": "Rajasthan",
                "correct_value_stated": "RJ",
                "correction_source": "Human user U06TEJP6D99 pointed out the error, which was then confirmed by human user U08MY1XK2EM."
              }}
            }},
            "keywords": ["current_state", "Rajasthan", "RJ", "state code", "dim_user_location", "filter correction"],
            "potential_source_indicators": ["U06TEJP6D99: ...you should use current_state = \'RJ\'", "U08MY1XK2EM: I understand that the previous query used `current_state = \'Rajasthan\'` incorrectly..."],
            "confidence_estimate": 1.0
          }},
          {{
            "knowledge_type": "model_structure_change",
            "content": {{
              "entity_name": "Referred to in a Snowflake query context (specific table name not explicitly stated by human, but implied by error)",
              "element_type": "column",
              "element_name": "IS_SUBSCRIBED",
              "change_type": "non_existent_column_identified_by_bot_error_report",
              "description_of_change": "The Bot (U08MY1XK2EM) reported a Snowflake SQL query failed due to an 'invalid identifier IS_SUBSCRIBED' error. This indicates the column, as named, does not exist in the context it was queried.",
              "old_value_or_schema": "IS_SUBSCRIBED (attempted use)"
            }},
            "keywords": ["IS_SUBSCRIBED", "invalid identifier", "SQL error", "Snowflake", "schema issue", "column missing"],
            "potential_source_indicators": ["Bot (U08MY1XK2EM) reported data/outcome: Snowflake Error: 000904 (42000): ... SQL compilation error: error line 8 at position 12 invalid identifier \'IS_SUBSCRIBED\'"],
            "confidence_estimate": 0.98
          }},
          {{
            "knowledge_type": "organisational_knowledge",
            "content": {{
              "term": "Resurrection Rate (Daily, for dormant cohort)",
              "definition_explanation": "Human user U08MY1XK2EM explained that for a specific dormant user cohort, this is the daily percentage of those users who resumed watching content.",
              "aliases_or_related_terms": ["daily resurrection rate"]
            }},
            "keywords": ["resurrection rate", "daily resurrection rate", "dormant users", "cohort analysis"],
            "potential_source_indicators": ["U08MY1XK2EM's explanation in response to U06TEJP6D99s query."],
            "confidence_estimate": 0.95
          }}
        ]
        ```

        Ensure your output is ONLY the JSON array, starting with `[` and ending with `]`.
        """
        return prompt

    async def _call_llm(self, prompt: str) -> list:
        """
        Calls the provided LLM client with the prompt and expects a JSON array string response.
        Returns the parsed list of learning objects or an empty list if errors occur.
        """
        if not self.llm_client:
            logger.error(
                "KnowledgeExtractorWorkflow: LLM client not provided or not initialized."
            )
            return []

        logger.info(
            "KnowledgeExtractorWorkflow: Calling LLM for knowledge extraction..."
        )
        llm_response_content = ""
        try:
            response_message = await self.llm_client.ainvoke(prompt)

            if not isinstance(response_message, AIMessage) or not hasattr(
                response_message, "content"
            ):
                logger.error(
                    f"KnowledgeExtractorWorkflow: Unexpected LLM response type: {type(response_message)}"
                )
                return []

            llm_response_content = response_message.content
            logger.info(
                f"KnowledgeExtractorWorkflow: Raw LLM Response content (first 500 chars): {llm_response_content[:500]}..."
            )

            # Strip markdown code block markers if present.
            if llm_response_content.strip().startswith("```json"):
                llm_response_content = llm_response_content.strip()[7:-3].strip()
            elif llm_response_content.strip().startswith("```"):
                llm_response_content = llm_response_content.strip()[3:-3].strip()

            # Ensure the response is treated as an array
            parsed_json = json.loads(llm_response_content)
            if not isinstance(parsed_json, list):
                logger.error(
                    f"KnowledgeExtractorWorkflow: LLM response was not a JSON array as expected. Type: {type(parsed_json)}"
                )
                # Optionally, try to wrap it in a list if it's a single dict that matches a learning object structure, but this is risky.
                return []

            logger.info(
                f"KnowledgeExtractorWorkflow: Parsed list of {len(parsed_json)} learning objects from LLM."
            )
            return parsed_json
        except json.JSONDecodeError as e:
            logger.error(
                f"KnowledgeExtractorWorkflow: Failed to parse JSON array from LLM response: {e}. Response was: {llm_response_content[:500]}...",
                exc_info=True,
            )
            return []
        except Exception as e:
            logger.error(
                f"KnowledgeExtractorWorkflow: Error during LLM call: {e}", exc_info=True
            )
            return []

    async def extract_from_thread(
        self,
        thread_messages: list,
        source_slack_channel_id: str,
        source_slack_thread_ts: str,
    ) -> dict:
        """
        Main method to orchestrate the knowledge extraction from a Slack thread.
        Generates metadata and wraps the LLM-extracted learnings.
        """
        if not self.llm_client:
            logger.error(
                "KnowledgeExtractorWorkflow: LLM client not configured for extraction."
            )
            return {"error": "LLM client not configured properly."}

        if not thread_messages:
            logger.warning(
                "KnowledgeExtractorWorkflow: No messages provided to extract knowledge from."
            )
            return {}

        logger.info(
            f"KnowledgeExtractorWorkflow: Starting extraction from {len(thread_messages)} messages for thread {source_slack_channel_id}/{source_slack_thread_ts}."
        )

        formatted_content = self._format_thread_for_llm(thread_messages)
        llm_prompt = self._construct_llm_prompt(formatted_content)

        extracted_learnings_list = await self._call_llm(llm_prompt)

        # Determine LLM model used, safely
        llm_model_name = "unknown"
        if hasattr(self.llm_client, "model_name") and self.llm_client.model_name:
            llm_model_name = str(self.llm_client.model_name)
        elif (
            hasattr(self.llm_client, "model") and self.llm_client.model
        ):  # Some clients might use 'model'
            llm_model_name = str(self.llm_client.model)

        # Prepare final structured output with metadata
        final_output = {
            "extraction_metadata": {
                "source_slack_channel_id": source_slack_channel_id,
                "source_slack_thread_ts": source_slack_thread_ts,
                "extraction_timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),  # Changed to timezone.utc
                "llm_model_used": llm_model_name,
            },
            "learnings": [],
        }

        if (
            not extracted_learnings_list
        ):  # Handles errors from _call_llm or empty extractions
            logger.warning(
                "KnowledgeExtractorWorkflow: No learning objects were extracted or an error occurred in LLM call."
            )
        else:
            # Add system-generated IDs to each learning object
            for learning_obj in extracted_learnings_list:
                # Basic validation: check if it's a dict and has a knowledge_type
                if isinstance(learning_obj, dict) and learning_obj.get(
                    "knowledge_type"
                ):
                    learning_obj["id"] = str(uuid.uuid4())  # Generate unique ID
                    final_output["learnings"].append(learning_obj)
                else:
                    logger.warning(
                        f"KnowledgeExtractorWorkflow: LLM returned an invalid learning object: {str(learning_obj)[:200]}..."
                    )

        logger.info(
            f"KnowledgeExtractorWorkflow: Extraction complete. Returning {len(final_output['learnings'])} processed learning objects."
        )
        return final_output
