"""SQLAlchemy models for vector storage."""

from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from dbt_llm_agent.storage.models import Base


class ModelEmbedding(Base):
    """SQLAlchemy model for storing model embeddings."""

    __tablename__ = "model_embeddings"

    id = Column(Integer, primary_key=True)
    model_name = Column(String, unique=True, nullable=False, index=True)
    document = Column(Text, nullable=False)
    embedding = Column(
        Vector(1536), nullable=False
    )  # Default dimension for text-embedding-ada-002
    model_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Create vector index for fast similarity search
    __table_args__ = (
        Index("idx_model_embeddings_embedding", embedding, postgresql_using="ivfflat"),
    )

    def __repr__(self):
        return f"<ModelEmbedding(model_name='{self.model_name}')>"
