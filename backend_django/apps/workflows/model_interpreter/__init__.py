"""Simplified model interpreter components for dbt model interpretation."""

import logging
from typing import List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ColumnDocumentation(BaseModel):
    name: str = Field(description="The name of the column.")
    description: str = Field(description="The description of the column.")


class ModelDocumentation(BaseModel):
    name: str = Field(description="The name of the dbt model.")
    description: str = Field(
        description="A brief, 1-2 sentence description summarizing the model's purpose, suitable for a dbt description field."
    )
    columns: List[ColumnDocumentation] = Field(
        description="A detailed list of columns returned by the model."
    )


# Export workflow class for easier imports
from .workflow import ModelInterpreterWorkflow

__all__ = ["ModelDocumentation", "ColumnDocumentation", "ModelInterpreterWorkflow"]
