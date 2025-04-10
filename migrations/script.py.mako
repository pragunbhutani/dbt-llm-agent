"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union
import logging

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

from ragstar.utils.logging import get_logger

# Get logger
logger = get_logger(__name__)

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    """Upgrade schema."""
    logger.info(f"Starting ${message} upgrade")
    
    ${upgrades if upgrades else "pass"}
    
    logger.info(f"Completed ${message} upgrade")


def downgrade() -> None:
    """Downgrade schema."""
    logger.info(f"Starting ${message} downgrade")
    
    ${downgrades if downgrades else "pass"}
    
    logger.info(f"Completed ${message} downgrade")
