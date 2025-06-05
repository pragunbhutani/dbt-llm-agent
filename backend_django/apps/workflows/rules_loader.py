import yaml
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_cached_rules = None


def load_ragstar_rules():
    """Loads custom instructions for agents from .ragstarrules.yml.

    Returns:
        dict: A dictionary where keys are agent names and values are their custom instructions.
              Returns an empty dictionary if the file is not found or is invalid.
    """
    global _cached_rules
    if _cached_rules is not None:
        return _cached_rules

    rules = {}
    try:
        current_path = Path(__file__).resolve()  # apps/workflows/rules_loader.py
        project_root = (
            current_path.parent.parent
        )  # Go up two levels to the workspace root (ragstar/)

        rules_file_path = project_root / ".ragstarrules.yml"

        if rules_file_path.exists():
            with open(rules_file_path, "r") as f:
                loaded_yaml = yaml.safe_load(f)
                if isinstance(loaded_yaml, dict):
                    for agent_name, instructions in loaded_yaml.items():
                        if isinstance(instructions, list):
                            rules[agent_name] = "\n".join(instructions)
                        elif isinstance(instructions, str):
                            rules[agent_name] = instructions
                        else:
                            rules[agent_name] = (
                                str(instructions) if instructions is not None else ""
                            )
                    _cached_rules = rules
                    logger.info(
                        f"Successfully loaded RAGstar rules from {rules_file_path}"
                    )
                else:
                    logger.warning(
                        f".ragstarrules.yml was found but is not a valid dictionary. Loaded: {type(loaded_yaml)}"
                    )
                    _cached_rules = {}
        else:
            logger.info(
                f".ragstarrules.yml not found at {rules_file_path}. No custom rules will be applied."
            )
            _cached_rules = {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing .ragstarrules.yml: {e}")
        _cached_rules = {}
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading .ragstarrules.yml: {e}"
        )
        _cached_rules = {}
    return _cached_rules


def get_agent_rules(agent_name: str) -> str:
    """Returns custom rules for a specific agent."""
    all_rules = load_ragstar_rules()
    return all_rules.get(agent_name, "")


# Example usage (can be removed or kept for testing):
if __name__ == "__main__":
    print("Loading rules from apps/workflows/rules_loader.py...")
    rules = load_ragstar_rules()
    if rules:
        print("Available rules:")
        for agent, rule_text in rules.items():
            print(f"  {agent}: {rule_text[:50]}...")
    else:
        print("No rules loaded or file not found/invalid.")

    print("\nTesting get_agent_rules:")
    print(f"Rules for 'model_interpreter': {get_agent_rules('model_interpreter')}")
    print(f"Rules for 'non_existent_agent': {get_agent_rules('non_existent_agent')}")
