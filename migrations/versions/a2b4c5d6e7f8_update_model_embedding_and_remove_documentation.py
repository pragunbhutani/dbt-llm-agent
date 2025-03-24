"""Update model embedding and remove documentation

Revision ID: a2b4c5d6e7f8
Revises: 542cd72c270f
Create Date: 2023-03-23 10:48:15.123456

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
revision: str = "a2b4c5d6e7f8"
down_revision: Union[str, None] = "542cd72c270f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    logger.info("Starting schema update for model embeddings and documentation")

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

    # Add comment to the existing embedding column
    logger.info("Adding comment to existing embedding column")
    op.alter_column(
        "model_embeddings",
        "embedding",
        existing_type=Vector(1536),
        comment="Embedding based on model documentation",
        existing_nullable=False,
    )

    # Check if documentation column exists in models table
    conn = op.get_bind()
    insp = sa.inspect(conn)
    columns = insp.get_columns("models")
    column_names = [col["name"] for col in columns]

    if "documentation" in column_names:
        logger.info("Removing documentation column from models table")
        op.drop_column("models", "documentation")

    logger.info("Schema update completed")


def downgrade() -> None:
    """Downgrade schema."""
    logger.info("Starting schema downgrade")

    # Add back documentation column to models table
    logger.info("Adding back documentation column to models table")
    op.add_column(
        "models",
        sa.Column(
            "documentation",
            sa.TEXT(),
            nullable=True,
            comment="Original model documentation",
        ),
    )

    # Remove comment from the embedding column
    logger.info("Removing comment from embedding column")
    op.alter_column(
        "model_embeddings",
        "embedding",
        existing_type=Vector(1536),
        comment=None,
        existing_comment="Embedding based on model documentation",
        existing_nullable=False,
    )

    # Drop the index for interpretation_embedding
    logger.info("Dropping index for interpretation_embedding")
    op.drop_index("idx_model_embeddings_interpretation", table_name="model_embeddings")

    # Drop interpretation_embedding column
    logger.info("Dropping interpretation_embedding column")
    op.drop_column("model_embeddings", "interpretation_embedding")

    logger.info("Schema downgrade completed")
