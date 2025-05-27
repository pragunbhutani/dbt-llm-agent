from typing import Optional, List, Any
from pydantic import BaseModel, Field


# NEW Pydantic model for structured LLM response in SQL debugging
# class SQLDebugResult(BaseModel):
#     corrected_sql: Optional[str] = Field(
#         default=None,
#         description="The corrected SQL query, if a correction was possible. Null if no correction or not applicable. Provide ONLY the SQL code without any markdown or backticks.",
#     )
#     explanation: str = Field(
#         description="A concise explanation of the error, the fix applied, or why a fix was not possible. This will be shown to the user."
#     )
#     confidence_of_fix: Optional[float] = Field(
#         default=None,
#         description="A score between 0.0 and 1.0 indicating the LLM's confidence in the correction. Optional.",
#     )
#     summary_of_changes: Optional[List[str]] = Field(
#         default=None,
#         description="A short bullet-point list (1-3 items) summarizing the changes made to the SQL query. Each item should be a concise string. Optional.",
#     )


# --- Tool Input Schemas (for tools the LLM might call, or for clarity) ---
class PostMessageInput(BaseModel):
    message_text: str = Field(
        description="The text message to post to the Slack thread."
    )


class PostCsvInput(BaseModel):
    csv_data: str = Field(description="The CSV data as a string.")
    initial_comment: str = Field(description="A comment to accompany the CSV file.")
    filename: str = Field(
        default="query_results.csv", description="The filename for the CSV."
    )


class DescribeTableInput(BaseModel):
    table_name: str = Field(description="The name of the table to describe.")


class ListColumnValuesInput(BaseModel):
    table_name: str = Field(description="The name of the table.")
    column_name: str = Field(
        description="The name of the column to list distinct values for."
    )
    limit: Optional[int] = Field(
        default=20, description="Max number of distinct values to return."
    )
