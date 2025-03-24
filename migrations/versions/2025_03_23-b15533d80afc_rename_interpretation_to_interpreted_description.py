"""rename_interpretation_to_interpreted_description

Revision ID: b15533d80afc
Revises: a2b4c5d6e7f8
Create Date: 2025-03-23 16:22:15.415794

"""

from typing import Sequence, Union
import logging

from alembic import op
import sqlalchemy as sa


from dbt_llm_agent.utils.logging import get_logger

# Get logger
logger = get_logger("alembic.migration")

# revision identifiers, used by Alembic.
revision: str = "b15533d80afc"
down_revision: Union[str, None] = "a2b4c5d6e7f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    logger.info(f"Starting rename_interpretation_to_interpreted_description upgrade")

    # Add the new column
    op.add_column(
        "models",
        sa.Column(
            "interpreted_description",
            sa.Text(),
            nullable=True,
            comment="LLM-generated description of the model",
        ),
    )

    # Copy data from interpretation to interpreted_description
    # Extract just the model description from the markdown format
    conn = op.get_bind()
    models_table = sa.Table(
        "models",
        sa.MetaData(),
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("interpretation", sa.Text()),
        sa.Column("interpreted_description", sa.Text()),
    )

    # Get all rows that have an interpretation
    rows = conn.execute(
        sa.select(models_table.c.id, models_table.c.interpretation).where(
            models_table.c.interpretation.isnot(None)
        )
    ).fetchall()

    # For each row, extract the description from the markdown and update the model
    for row in rows:
        model_id = row[0]
        interpretation = row[1]

        # The description is in the markdown between the header and the Columns section
        # Format is typically:
        # ## model_name (LLM Interpretation)
        #
        # Description text here
        #
        # ### Columns
        description = ""
        if interpretation:
            lines = interpretation.strip().split("\n")
            description_lines = []
            capture = False
            for line in lines:
                if line.startswith("## ") and "(LLM Interpretation)" in line:
                    capture = True
                    continue
                elif line.startswith("### Columns"):
                    break
                elif capture and line.strip():
                    description_lines.append(line)

            description = "\n".join(description_lines).strip()

        # Update the row with the extracted description
        conn.execute(
            models_table.update()
            .where(models_table.c.id == model_id)
            .values(interpreted_description=description)
        )

    # Drop the old column
    op.drop_column("models", "interpretation")

    logger.info(f"Completed rename_interpretation_to_interpreted_description upgrade")


def downgrade() -> None:
    """Downgrade schema."""
    logger.info(f"Starting rename_interpretation_to_interpreted_description downgrade")

    # Add back the interpretation column
    op.add_column(
        "models",
        sa.Column(
            "interpretation",
            sa.Text(),
            nullable=True,
            comment="LLM-generated interpretation of the model",
        ),
    )

    # Copy data from interpreted_description back to interpretation
    # Format it as markdown
    conn = op.get_bind()
    models_table = sa.Table(
        "models",
        sa.MetaData(),
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String),
        sa.Column("interpretation", sa.Text()),
        sa.Column("interpreted_description", sa.Text()),
        sa.Column("interpreted_columns", sa.JSON),
    )

    # Get all rows that have an interpreted_description
    rows = conn.execute(
        sa.select(
            models_table.c.id,
            models_table.c.name,
            models_table.c.interpreted_description,
            models_table.c.interpreted_columns,
        ).where(models_table.c.interpreted_description.isnot(None))
    ).fetchall()

    # For each row, format the description as markdown
    for row in rows:
        model_id = row[0]
        model_name = row[1]
        description = row[2]
        columns = row[3]

        # Format as markdown
        formatted_doc = f"## {model_name} (LLM Interpretation)\n\n{description}\n\n"

        if columns:
            formatted_doc += "### Columns\n\n"
            for col_name, col_desc in columns.items():
                formatted_doc += f"- **{col_name}**: {col_desc}\n"

        # Update the row with the formatted doc
        conn.execute(
            models_table.update()
            .where(models_table.c.id == model_id)
            .values(interpretation=formatted_doc)
        )

    # Drop the new column
    op.drop_column("models", "interpreted_description")

    logger.info(f"Completed rename_interpretation_to_interpreted_description downgrade")
