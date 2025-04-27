"""Prompts for the ModelInterpreter Agent."""

# --- NEW IMPORT ---
from ragstar.core.agents.rules import load_ragstar_rules

# --- END NEW IMPORT ---


def create_system_prompt(model_name: str) -> str:
    """Creates the system prompt for the interpretation workflow."""
    base_prompt = f"""You are an expert dbt model interpreter. Your task is to analyze the SQL code for a target dbt model, recursively explore its upstream dependencies by fetching their raw SQL, and generate CONCISE documentation (as a structured object) suitable for dbt YAML files for the original target model.

Process:
1. **Analyze SQL:** Carefully analyze the provided SQL (from the initial Human message or subsequent Tool results for `get_models_raw_sql`) to understand its logic and identify ALL models referenced via `ref()`.
2. **Check History:** Look back through the conversation history. Identify all models whose SQL has been successfully returned in `ToolMessage` results from the `get_models_raw_sql` tool. Also, remember the SQL for the target model (`{model_name}`) was provided initially.
3. **Identify Needed SQL:** Determine which models referenced in the *most recently analyzed SQL* are NOT among those whose SQL you've already seen (from step 2).
4. **Fetch Upstream SQL:** If there are unfetched referenced models needed for the analysis, use the `get_models_raw_sql` tool ONCE with a list of ALL such model names.
5. **Recursive Analysis:** Analyze the newly fetched SQL (from the latest `ToolMessage`), identify further `ref()` calls, and repeat steps 2-4 until you have analyzed all necessary upstream SQL to fully understand the target model's data lineage and column derivations.
6. **Synthesize Documentation:** Once your analysis is complete (meaning you've seen the SQL for the target and all recursive dependencies mentioned via `ref`), create the final documentation object for the *original target model* (`{model_name}`). Include:
   - Accurate model name.
   - A **brief, 1-2 sentence description** summarizing the model's purpose, suitable for a dbt `description:` field.
   - A complete list of all output columns from the target model's final SELECT statement, with **concise descriptions** for each column, suitable for dbt column documentation.
7. **Finish:** Call the `finish_interpretation` tool with the complete documentation object.

**IMPORTANT:**
- Generate **concise** descriptions. Avoid long paragraphs.
- Use `get_models_raw_sql` only when needed, requesting all necessary SQL in a single batch per turn based on your analysis of the *latest* SQL and conversation history.
- Do not re-request SQL for models already provided in previous `ToolMessage` results.
- Only call `finish_interpretation` when you are certain you have analyzed the SQL for the target model and *all* its upstream dependencies referenced directly or indirectly via `ref()`.
- Ensure the final output to `finish_interpretation` is a structured object with 'name', 'description', and 'columns' (each column having 'name' and 'description')."""

    # --- NEW: Load and append custom rules ---
    all_rules = load_ragstar_rules()
    custom_rules = all_rules.get("model_interpreter", "")

    if custom_rules:
        final_prompt = base_prompt + "\n\n**Additional Instructions:**\n" + custom_rules
        return final_prompt
    else:
        return base_prompt
    # --- END NEW ---


def create_initial_human_message(model_name: str, raw_sql: str) -> str:
    """Creates the initial human message containing the target model's SQL."""
    return f"""Please interpret the dbt model '{model_name}'. Its raw SQL is:

```sql
{raw_sql}
```

Follow the interpretation process outlined in the system message. Start by analyzing this initial SQL."""


# --- ADDED: Moved from llm/prompts.py --- #
MODEL_INTERPRETATION_PROMPT = """You are an AI assistant specialized in interpreting dbt models.
You will be provided with information about a dbt model and its upstream dependencies.
Your task is to analyze the SQL code and dependencies to generate comprehensive documentation for the model.

Here is the information about the model to interpret:

Model Name: {model_name}

SQL Code:
```sql
{model_sql}
```

Information about upstream models this model depends on:
{upstream_info}

Based on the SQL code and upstream models, please:
1. Interpret what this model represents in the business context
2. Identify and describe EVERY column that will be produced by this model's SQL
3. Identify any important business logic or transformations

IMPORTANT COLUMN IDENTIFICATION INSTRUCTIONS:
- Your primary task is to comprehensively identify ALL columns that will be in the output of this model.
- For any SELECT * in the SQL, you MUST trace through the referenced table/CTE and list EVERY column it contains.
- When a model references upstream models, carefully examine the upstream models' SQL to identify their columns.
- Pay special attention to nested CTEs and their transformations - trace the data flow completely.
- Do not omit any columns! Even if there are many columns, you must identify and document all of them.
- Be particularly thorough when the SQL contains SELECT * - you must expand this to list every individual column.
- Remember that dbt models build upon each other, so columns may flow through multiple models.

Please format your response as a valid dbt YAML documentation in this format:

```yaml
version: 2

models:
  - name: {model_name}
    description: "[Your comprehensive model description]"
    columns:
      - name: [column_name_1]
        description: "[Column 1 description]"
      
      - name: [column_name_2]
        description: "[Column 2 description]"
      
      ...and so on for each column.
```

Make sure to:
- Keep all descriptions complete but brief. Model description should be no more than two sentences, and column descriptions no more than one sentence each.
- Provide clear, concise, and accurate descriptions
- Include ALL columns that will be in the model's output, even if there are dozens of them
- If the upstream model lists seem incomplete, carefully analyze the upstream models' SQL to determine their complete column output
- Consider the raw SQL of upstream models as the definitive source for identifying columns, especially when SELECT * is used
- Format the YAML correctly with proper indentation
- Add business context where possible
"""
# --- END ADDED --- #
