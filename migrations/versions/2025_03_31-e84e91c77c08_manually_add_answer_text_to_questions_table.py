"""Manually add answer_text to questions table

Revision ID: e84e91c77c08
Revises: 2025_03_24
Create Date: 2025-03-31 16:24:04.304629

"""

from typing import Sequence, Union
import logging

from alembic import op
import sqlalchemy as sa

from dbt_llm_agent.utils.logging import get_logger

# Get logger
logger = get_logger("alembic.migration")

# revision identifiers, used by Alembic.
revision: str = "e84e91c77c08"
down_revision: Union[str, None] = "2025_03_24"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    logger.info(f"Starting {revision} upgrade")

    # ### commands manually added ###
    op.add_column("questions", sa.Column("answer_text", sa.Text(), nullable=True))
    logger.info(f"Added answer_text column to questions table.")
    # ### end commands ###

    logger.info(f"Completed {revision} upgrade")


def downgrade() -> None:
    """Downgrade schema."""
    logger.info(f"Starting {revision} downgrade")

    # ### commands manually added ###
    op.drop_column("questions", "answer_text")
    logger.info(f"Dropped answer_text column from questions table.")
    # ### end commands ###

    logger.info(f"Completed {revision} downgrade")
