"""Add can_be_used_for_answers to model_embeddings

Revision ID: f95fc2f21b41
Revises: 865a53a7440e
Create Date: 2025-04-26 12:11:30.482502

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
revision: str = "f95fc2f21b41"
down_revision: Union[str, None] = "865a53a7440e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    logger.info(f"Starting {revision}: Add can_be_used_for_answers to model_embeddings")

    op.add_column(
        "model_embeddings",
        sa.Column(
            "can_be_used_for_answers",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
            comment="Whether this embedding can be used for answering questions",
        ),
    )
    # ### end Alembic commands ###

    logger.info(
        f"Completed {revision}: Add can_be_used_for_answers to model_embeddings"
    )


def downgrade() -> None:
    """Downgrade schema."""
    logger.info(f"Starting downgrade for {revision}")

    op.drop_column("model_embeddings", "can_be_used_for_answers")

    logger.info(f"Completed downgrade for {revision}")
