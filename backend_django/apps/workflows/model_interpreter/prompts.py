"""Prompts for the simplified model interpreter workflow."""

from typing import List, Dict, Any
from apps.knowledge_base.models import Model
from apps.workflows.rules_loader import get_agent_rules


def create_interpretation_prompt(
    model: Model, upstream_models: List[Dict[str, Any]]
) -> str:
    """Create a comprehensive prompt for model interpretation.

    Args:
        model: The target Model instance to interpret
        upstream_models: List of upstream model data dictionaries

    Returns:
        A comprehensive prompt string for the LLM
    """
    # Get custom rules
    custom_rules = get_agent_rules("model_interpreter")
    custom_rules_section = (
        f"\n\n**Additional Instructions:**\n{custom_rules}" if custom_rules else ""
    )

    # Build upstream models section
    upstream_section = ""
    if upstream_models:
        upstream_section = "\n\n**Upstream Models (Dependencies):**\n"
        for i, upstream in enumerate(upstream_models, 1):
            upstream_section += (
                f"\n{i}. **{upstream['name']}** (Path: {upstream['path'] or 'N/A'})\n"
            )
            if upstream["yml_description"] or upstream["interpreted_description"]:
                desc = (
                    upstream["interpreted_description"] or upstream["yml_description"]
                )
                upstream_section += f"   Description: {desc}\n"
            if upstream["raw_sql"]:
                upstream_section += (
                    f"   SQL:\n   ```sql\n{upstream['raw_sql']}\n   ```\n"
                )
            else:
                upstream_section += "   SQL: Not available\n"

    prompt = f"""You are an expert dbt model interpreter. Analyze the provided dbt model and generate CONCISE documentation suitable for dbt YAML files.

**Target Model: {model.name}**
Path: {model.path}
Schema: {model.schema_name or 'N/A'}
Database: {model.database or 'N/A'}
Materialization: {model.materialization or 'N/A'}

**Target Model SQL:**
```sql
{model.raw_sql}
```
{upstream_section}

**Task:**
Generate a structured documentation object for the target model '{model.name}' that includes:

1. **name**: The exact model name
2. **description**: A brief, 1-2 sentence description summarizing the model's purpose, suitable for a dbt description field
3. **columns**: A complete list of all output columns from the target model's final SELECT statement, with concise descriptions for each column

**Requirements:**
- Generate **concise** descriptions suitable for dbt documentation
- Focus on the business purpose and meaning of each column
- Consider the upstream model context to understand data lineage
- Only document columns that are actually output by the final SELECT statement
- Use clear, professional language appropriate for technical documentation

**Output Format:**
Respond with a JSON object in this exact structure:
```json
{{
  "name": "model_name",
  "description": "Brief description of the model's purpose",
  "columns": [
    {{
      "name": "column_name",
      "description": "Brief description of the column"
    }}
  ]
}}
```{custom_rules_section}"""

    return prompt
