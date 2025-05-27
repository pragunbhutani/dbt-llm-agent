from typing import Optional, List
from pydantic import BaseModel, Field


class SQLDebugResult(BaseModel):
    corrected_sql: Optional[str] = Field(
        default=None,
        description="The corrected SQL query, if a correction was possible. Null if no correction or not applicable. Provide ONLY the SQL code without any markdown or backticks.",
    )
    explanation: str = Field(
        description="A concise explanation of the error, the fix applied, or why a fix was not possible. This will be shown to the user."
    )
    confidence_of_fix: Optional[float] = Field(
        default=None,
        description="A score between 0.0 and 1.0 indicating the LLM's confidence in the correction. Optional.",
    )
    summary_of_changes: Optional[List[str]] = Field(
        default=None,
        description="A short bullet-point list (1-3 items) summarizing the changes made to the SQL query. Each item should be a concise string. Optional.",
    )
    debugging_log: Optional[List[str]] = Field(
        default=None,
        description="A log of actions and observations during the SQL verification and debugging process. Optional.",
    )


class SQLSchemaValidationResult(BaseModel):
    is_valid: bool = Field(
        description="Whether the SQL query is valid based on the provided dbt models information."
    )
    discrepancies: Optional[List[str]] = Field(
        default=None,
        description="A list of discrepancies found, e.g., missing tables or columns.",
    )


class SQLStyleCheckResult(BaseModel):
    is_style_compliant: bool = Field(
        description="Whether the SQL query adheres to the provided style guide."
    )
    styled_sql: Optional[str] = Field(
        default=None,
        description="The re-styled SQL query if modifications were made, or the original SQL if already compliant. Should be null if the LLM cannot restyle it.",
    )
    style_violations: Optional[List[str]] = Field(
        default=None, description="A list of specific style violations found, if any."
    )
