"""Prompt templates for dbt-llm-agent."""

# Prompt for answering questions about dbt models
ANSWER_PROMPT_TEMPLATE = """
You are an AI assistant specialized in answering questions about dbt projects.
You are given information about relevant dbt models that might help answer the user's question.

Here is the information about the relevant dbt models:

{model_info}

Based on this information, please answer the following question:
{question}

Provide a clear, concise, and accurate answer. If the information provided is not sufficient to answer the question, explain what additional information would be needed.
"""

# Prompt for interpreting dbt models
MODEL_INTERPRETATION_PROMPT = """
You are an AI assistant specialized in interpreting dbt models.
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
2. Identify and describe each column in the model
3. Identify any important business logic or transformations

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
- Provide clear, concise, and accurate descriptions
- Identify **all** columns selected by the final `SELECT` statement. This includes columns explicitly named, columns created through expressions/functions, and **all columns implicitly selected using `*` from upstream models or CTEs**.
- If the provided column lists for an upstream model seem incomplete or are missing, analyze its Raw SQL to determine the columns implicitly selected by `*`.
- **Highest priority:** The raw SQL of upstream models is the definitive source for understanding which columns are available from them, especially when `SELECT *` is used. Rely on the SQL over potentially incomplete YML column lists or previous interpretations.
- Format the YAML correctly with proper indentation
- Add business context where possible
"""

# Prompt for refining user queries for semantic search
SEARCH_QUERY_REFINEMENT_PROMPT_TEMPLATE = """
You are an AI assistant helping to refine user questions into effective search queries for a vector database containing dbt model documentation.

The user's original question is:
{prompt}

Your task is to extract the key entities, metrics, dimensions, and concepts from the user's question and formulate a concise and focused search query. The query should ideally be a few keywords or a short phrase that captures the core semantic meaning relevant to finding dbt models.

Example 1:
User Question: "how can I find the mrr per month grouped by industry type?"
Refined Query: "monthly mrr by industry"

Example 2:
User Question: "show me total sales and number of orders for the retail segment in Q4"
Refined Query: "Q4 retail sales orders"

Example 3:
User Question: "which models contain information about customer lifetime value?"
Refined Query: "customer lifetime value ltv"

Return ONLY the refined search query, without any explanation or preamble.
"""

# Prompt for refining based on past feedback
FEEDBACK_REFINEMENT_PROMPT_TEMPLATE = """
You are an AI assistant refining an answer based on past user feedback.
Original Question: {original_question}

Initial Answer Generated:
{original_answer}

Previously, similar questions received the following feedback (most relevant/recent first):
{feedback_context}

Please refine the initial answer.
- Incorporate the specific points from the feedback text, if any.
- **If a past answer was marked 'not useful' even without specific text, it strongly suggests the previous approach, the models used, or the interpretation was incorrect or insufficient. Re-evaluate the initial answer and the models used. Consider alternative models from the context or different ways to combine them.**
- **Conversely, if past feedback was positive or marked useful (even without specific text), it indicates the previous approach was likely on the right track. Reinforce that approach or build upon the successful aspects.**
- Aim to make the answer more accurate and helpful based on past user interactions.

Refined Answer:
"""

# Define a new prompt template for refining the answer based on feedback (Old, keeping for reference if needed)
REFINE_PROMPT_TEMPLATE = """
You are an AI assistant helping users understand their dbt project.
The user asked the following question: {question}

Based on the available dbt models, you previously generated this answer:
{previous_answer}

The user provided the following feedback indicating the answer was not sufficient:
{user_feedback}

Here is the context about relevant dbt models:
{model_info}

Based on the user's feedback and the model context, please generate an improved and more complete answer to the original question: {question}
"""

# Agentic Workflow Prompts
PLANNING_PROMPT_TEMPLATE = """
You are a data analyst assistant planning how to answer a user's question about dbt models.
User question: "{question}"

1. Identify the primary metric or entity the user is asking about.
2. Identify any grouping, filtering, or specific dimensions requested.
3. Based on this, determine the *first* key piece of information to search for in the dbt models.
4. Formulate a concise set of *keywords or a short phrase* focused *only* on finding that first piece of information. This should be suitable for a *semantic search* to find relevant dbt models, **not a SQL query**.

Respond with just the semantic search query (keywords/phrase).
Search Query: """

ANALYSIS_PROMPT_TEMPLATE = """
You are a data analyst assistant analyzing dbt models to answer a user's question. Be concise.
Original user question: "{original_question}"

We have already gathered information on these models:
{already_found_models_summary}

We just searched for "{last_search_query}".
{newly_found_model_details_section}

Analyze the newly found model(s) if any:
1. What key information do they provide relevant to the original question?
2. What key entities (e.g., customer IDs, timestamps, dimensions) do they contain or relate to?
3. Based on the original question and *all* the models found so far, what is the *single most important* specific piece of information we still need to find in other dbt models?
4. If all necessary information seems present across the found models to answer the original question, respond ONLY with `ALL_INFO_FOUND`.
5. Otherwise, formulate a *concise set of keywords or a short phrase* representing the next piece of missing information. This should be suitable as a *semantic search query* to find relevant dbt models, **not a SQL query**.

Respond *concisely* with the next semantic search query (keywords/phrase) OR `ALL_INFO_FOUND`.
Next Step Search Query: """

SYNTHESIS_PROMPT_TEMPLATE = """
You are a data analyst assistant synthesizing an answer to a user's question using information from dbt models. Be concise.
User question: "{question}"

We have gathered information about the following relevant dbt models:
{accumulated_model_info}

Based *strictly* and *only* on the provided model information (`accumulated_model_info`), construct a step-by-step answer for the user.
- Explain how the models can be used together to answer the question.
- Mention necessary joins (including join keys) between models *if applicable and derivable from the provided info*.
- Describe any required calculations or aggregations.
- If possible, provide a sample SQL query demonstrating the process, using *only* models and columns present in the provided info.
- **Do not invent table names, column names, or relationships** that are not explicitly present in the provided model information. Do not assume the existence of dimension tables unless they are listed in `accumulated_model_info`.
- If the gathered models are insufficient to fully answer the question, *clearly state this*. Explain exactly what information (e.g., specific columns, relationships) is missing and which of the provided models contain related partial information. **Do not provide a sample query using hypothetical/invented tables or columns.**

Answer: """
