import os
import yaml
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_ragstar_rules(rules_file_path: str = ".ragstarrules.yml") -> Dict[str, str]:
    """
    Loads custom instructions for agents from the YAML .ragstarrules file.

    The file format expected is YAML:
    agent_name:
      - Instruction line 1
      - Instruction line 2
    another_agent:
      - Another instruction

    Args:
        rules_file_path: The path to the rules file (relative to project root).

    Returns:
        A dictionary mapping agent names (e.g., 'slack_responder') to their
        custom instruction strings (joined from the list). Returns an empty dict
        if file not found, cannot be parsed, or PyYAML is not installed.
    """
    rules = {}
    if not yaml:
        logger.warning(
            "PyYAML is not installed. Cannot load custom rules from .ragstarrules. "
            "Please install it (`pip install pyyaml`)."
        )
        return rules

    if not os.path.exists(rules_file_path):
        logger.debug(
            f"Rules file not found at '{rules_file_path}'. No custom rules loaded."
        )
        return rules

    try:
        with open(rules_file_path, "r") as f:
            loaded_yaml = yaml.safe_load(f)

        if not isinstance(loaded_yaml, dict):
            logger.warning(
                f"Rules file '{rules_file_path}' does not contain a valid YAML dictionary at the top level. "
                "No custom rules loaded."
            )
            return {}

        # Validate and extract rules
        for agent_name, instructions_list in loaded_yaml.items():
            if isinstance(agent_name, str) and isinstance(instructions_list, list):
                # Ensure all items in the list are strings and join them
                processed_instructions = []
                valid_list = True
                for item in instructions_list:
                    if isinstance(item, str):
                        processed_instructions.append(item.strip())  # Strip each line
                    else:
                        logger.warning(
                            f"Skipping invalid non-string item in list for agent '{agent_name}' in '{rules_file_path}': {type(item)}"
                        )
                        valid_list = False
                        break  # Stop processing this agent if list is invalid

                if valid_list:
                    # Join the list of strings with newlines
                    rules[agent_name] = "\n".join(processed_instructions).strip()
            elif isinstance(agent_name, str) and instructions_list is None:
                # Handle empty sections like 'agent:' without any list items
                logger.debug(
                    f"Agent '{agent_name}' has no rules defined in '{rules_file_path}'. Skipping."
                )
            else:
                logger.warning(
                    f"Skipping invalid entry in '{rules_file_path}': Key must be string, Value must be a list of strings or None. Found: {type(agent_name)}: {type(instructions_list)}"
                )

        if rules:
            logger.info(f"Loaded custom rules for agents: {list(rules.keys())}")
        else:
            logger.debug(
                f"No valid agent rule sections found or file was empty/invalid in '{rules_file_path}'."
            )

    except yaml.YAMLError as e:
        logger.error(
            f"Error parsing YAML rules file '{rules_file_path}': {e}", exc_info=True
        )
        return {}  # Return empty dict on YAML parsing error
    except Exception as e:
        logger.error(
            f"Error reading rules file '{rules_file_path}': {e}", exc_info=True
        )
        return {}  # Return empty dict on other errors

    return rules
