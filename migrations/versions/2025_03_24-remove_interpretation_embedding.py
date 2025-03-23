"""Remove interpretation_embedding column

Revision ID: 2025_03_24
Revises: b15533d80afc
Create Date: 2025-03-24 12:00:00.000000

"""

from typing import Sequence, Union
import logging

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

from dbt_llm_agent.utils.logging import get_logger

# Get logger
logger = get_logger("alembic.migration")

# revision identifiers, used by Alembic.
revision: str = "2025_03_24"
down_revision: Union[str, None] = "b15533d80afc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema by removing the redundant interpretation_embedding column."""
    logger.info("Starting schema update to remove interpretation_embedding column")

    # Drop the index for interpretation_embedding
    logger.info("Dropping index for interpretation_embedding")
    op.drop_index("idx_model_embeddings_interpretation", table_name="model_embeddings")

    # Drop interpretation_embedding column
    logger.info("Dropping interpretation_embedding column")
    op.drop_column("model_embeddings", "interpretation_embedding")

    logger.info("Schema update completed")


def downgrade() -> None:
    """Downgrade schema by adding back the interpretation_embedding column."""
    logger.info("Starting schema downgrade")

    # Add interpretation_embedding column to model_embeddings table
    logger.info("Adding interpretation_embedding column to model_embeddings table")
    op.add_column(
        "model_embeddings",
        sa.Column(
            "interpretation_embedding",
            Vector(1536),
            nullable=True,
            comment="Embedding based on model interpretation",
        ),
    )

    # Create index for the new column
    logger.info("Creating index for interpretation_embedding")
    op.create_index(
        "idx_model_embeddings_interpretation",
        "model_embeddings",
        ["interpretation_embedding"],
        postgresql_using="ivfflat",
    )

    logger.info("Schema downgrade completed")
