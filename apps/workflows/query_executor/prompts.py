# Prompts for the Query Executor Workflow

# Placeholder for a potential debug prompt if we use an LLM to help fix SQL queries
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

Please provide a corrected SQL query. If you cannot correct it, explain why.
"""

# Placeholder for other prompts that might be needed.
