# Prompts for the Query Executor Workflow

# SQL_DEBUG_PROMPT_TEMPLATE has been moved to apps/workflows/sql_verifier/prompts.py
# BUT is still imported by QueryExecutorWorkflow. Uncommenting for now.
SQL_DEBUG_PROMPT_TEMPLATE = """ 
An SQL query failed with the following error:
{error_message}

Original Query:
```sql
{sql_query}
```

Available table schemas (if any):
{table_schemas}

List of relevant column values (if any):
{column_values}

Please analyze the error and the query.
If you can correct the query, provide the corrected SQL.
Provide a concise explanation of the problem and your solution.
If you cannot correct it, explain why.
When providing a corrected query or an explanation, please be concise as your response will be used in a debugging log.

Your response MUST be a single, valid JSON object that conforms to the following Pydantic model structure:

```json
{{
  "corrected_sql": "string | null",
  "explanation": "string",
  "confidence_of_fix": "float | null",
  "summary_of_changes": "list[string] | null"
}}
```

Detailed field descriptions:
- `corrected_sql`: The corrected SQL query as a single string, if a correction was possible. If not, this field should be `null`. Provide ONLY the raw SQL code, without any markdown, backticks, or "sql" language specifiers.
- `explanation`: A concise string explaining the error, the fix applied (if any), or why a fix was not possible. This explanation will be shown to the user and should be easy to understand.
- `confidence_of_fix`: An optional float between 0.0 and 1.0 indicating your confidence in the correction. If not applicable or unsure, this field can be `null`.
- `summary_of_changes`: An optional list of 1 to 3 short strings. Each string should be a bullet point summarizing a specific change made to the SQL query. If no changes were made or it's not applicable, this field can be `null` or an empty list. These will be used to build a concise debugging log for the user.

Example of a good `summary_of_changes` item: "Replaced incorrect column `IS_SUBSCRIBED` with `SUBSCRIPTION_STATUS`."

Do NOT include any text outside of the JSON object in your response.
"""

# Placeholder for other prompts that might be needed.
