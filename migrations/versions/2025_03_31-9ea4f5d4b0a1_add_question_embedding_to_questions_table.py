"""Add question_embedding to questions table

Revision ID: 9ea4f5d4b0a1
Revises: e84e91c77c08
Create Date: 2025-03-31 16:27:14.043515

"""

from typing import Sequence, Union
import logging

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from dbt_llm_agent.utils.logging import get_logger

# Get logger
logger = get_logger("alembic.migration")

# revision identifiers, used by Alembic.
revision: str = "9ea4f5d4b0a1"
down_revision: Union[str, None] = "e84e91c77c08"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    logger.info(f"Starting {revision} upgrade")

    # ### commands manually added ###
    op.add_column(
        "questions",
        sa.Column(
            "question_embedding",
            Vector(1536),
            nullable=True,
            comment="Embedding vector for the question text",
        ),
    )
    logger.info(f"Added question_embedding column to questions table.")
    # ### end commands ###

    logger.info(f"Completed {revision} upgrade")


def downgrade() -> None:
    """Downgrade schema."""
    logger.info(f"Starting {revision} downgrade")

    # ### commands manually added ###
    op.drop_column("questions", "question_embedding")
    logger.info(f"Dropped question_embedding column from questions table.")
    # ### end commands ###

    logger.info(f"Completed {revision} downgrade")
