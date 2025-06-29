from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SQLVerificationResponse(BaseModel):
    """Canonical contract between SQLVerifierWorkflow and SlackResponder.

    Every exit path from the verifier **must** populate these fields so the
    responder can rely on a stable shape.
    """

    success: bool = Field(
        description="Whether the verifier completed its run without internal crashes."
    )
    # True / False when success, None when verifier crashed before deciding.
    is_valid: Optional[bool] = Field(
        default=None,
        description="Whether the query was deemed valid/executable after checks.",
    )
    verified_sql: Optional[str] = Field(
        default=None, description="Final form of the SQL query (corrected or original)."
    )
    error: Optional[str] = Field(
        default=None,
        description="Human-readable error if the query is invalid or verifier failed.",
    )
    explanation: Optional[str] = Field(
        default=None, description="Debug/LLM explanation for corrections or failures."
    )
    debugging_log: List[str] = Field(
        default_factory=list,
        description="Step-by-step log lines generated by the verifier for transparency.",
    )
    # Extra optional context flags
    was_executed: Optional[bool] = Field(
        default=None,
        description="Indicates whether the query was actually run against the warehouse.",
    )
    is_style_compliant: Optional[bool] = Field(default=None)
    style_violations: Optional[List[str]] = Field(default=None)


class QAResponse(BaseModel):
    """Canonical contract object exchanged from QuestionAnswerer → SlackResponder.

    Using an explicit model means we have static validation at the boundary and can
    evolve the payload in a backwards-compatible way via optional fields.
    """

    answer: str = Field(..., description="End-user facing answer (plain text)")
    sql_query: Optional[str] = Field(
        default=None,
        description="SQL query that supports the answer, when applicable",
    )
    models_used: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Metadata about dbt models referenced in the answer or query",
    )

    # Non-critical, informational fields
    warning: Optional[str] = Field(
        default=None,
        description="Optional warning (e.g., limitations, partial answer)",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the QA workflow completed with issues",
    )
