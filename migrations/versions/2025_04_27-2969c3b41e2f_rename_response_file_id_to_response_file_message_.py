"""Rename response_file_id to response_file_message_ts in questions table

Revision ID: 2969c3b41e2f
Revises: b021cfc5ff40
Create Date: 2025-04-27 23:26:59.904244

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
revision: str = "2969c3b41e2f"
down_revision: Union[str, None] = "b021cfc5ff40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    logger.info(
        f"Starting {revision}: Rename response_file_id to response_file_message_id"
    )

    op.add_column(
        "questions", sa.Column("response_file_message_ts", sa.String(), nullable=True)
    )
    op.drop_index("ix_questions_response_file_id", table_name="questions")
    op.create_index(
        op.f("ix_questions_response_file_message_ts"),
        "questions",
        ["response_file_message_ts"],
        unique=False,
    )
    op.drop_column("questions", "response_file_id")
    # ### end Alembic commands ###

    logger.info(
        f"Completed {revision}: Rename response_file_id to response_file_message_id"
    )


def downgrade() -> None:
    """Downgrade schema."""
    logger.info(f"Starting downgrade for {revision}")

    op.alter_column(
        "questions",
        "response_file_message_ts",
        new_column_name="response_file_id",
    )

    logger.info(f"Completed downgrade for {revision}")
