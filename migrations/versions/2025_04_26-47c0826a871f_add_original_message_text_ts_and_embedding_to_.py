"""Add original message text, ts, and embedding to QuestionTable

Revision ID: 47c0826a871f
Revises: f95fc2f21b41
Create Date: 2025-04-26 19:44:04.310841

"""

from typing import Sequence, Union
import logging

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import pgvector.sqlalchemy

from ragstar.utils.logging import get_logger

# Get logger
logger = get_logger(__name__)

# revision identifiers, used by Alembic.
revision: str = "47c0826a871f"
down_revision: Union[str, None] = "f95fc2f21b41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    logger.info(
        f"Starting Add original message text, ts, and embedding to QuestionTable upgrade"
    )

    op.add_column(
        "questions", sa.Column("original_message_text", sa.Text(), nullable=True)
    )
    op.add_column(
        "questions", sa.Column("original_message_ts", sa.String(), nullable=True)
    )
    op.add_column(
        "questions",
        sa.Column(
            "original_message_embedding",
            pgvector.sqlalchemy.Vector(dim=1536),
            nullable=True,
            comment="Embedding vector for the original message text using text-embedding-ada-002",
        ),
    )

    logger.info(
        f"Completed Add original message text, ts, and embedding to QuestionTable upgrade"
    )


def downgrade() -> None:
    """Downgrade schema."""
    logger.info(
        f"Starting Add original message text, ts, and embedding to QuestionTable downgrade"
    )

    op.drop_column("questions", "original_message_embedding")
    op.drop_column("questions", "response_message_ts")
    op.drop_column("questions", "original_message_text")

    logger.info(
        f"Completed Add original message text, ts, and embedding to QuestionTable downgrade"
    )
