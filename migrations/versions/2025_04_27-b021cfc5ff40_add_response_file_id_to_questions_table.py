"""Add response_file_id to questions table

Revision ID: b021cfc5ff40
Revises: 47c0826a871f
Create Date: 2025-04-27 20:24:19.350293

"""

from typing import Sequence, Union
import logging

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from ragstar.utils.logging import get_logger

# Get logger
logger = get_logger(__name__)

# revision identifiers, used by Alembic.
revision: str = "b021cfc5ff40"
down_revision: Union[str, None] = "47c0826a871f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    logger.info(f"Starting {revision}: Add response_file_id to questions table")

    op.add_column("questions", sa.Column("response_file_id", sa.Uuid(), nullable=True))
    op.create_index(
        op.f("ix_questions_response_file_id"),
        "questions",
        ["response_file_id"],
        unique=False,
    )

    logger.info(f"Completed {revision}: Add response_file_id to questions table")


def downgrade() -> None:
    """Downgrade schema."""
    logger.info(f"Starting downgrade for {revision}")

    op.drop_constraint(
        "questions_response_file_id_fkey", "questions", type_="foreignkey"
    )
    op.drop_column("questions", "response_file_id")

    logger.info(f"Completed downgrade for {revision}")
