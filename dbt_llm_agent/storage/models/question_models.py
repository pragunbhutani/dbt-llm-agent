"""Models for tracking questions and answers."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
    Table,
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from dbt_llm_agent.storage.models import Base


# Association table for questions and models
question_models = Table(
    "question_models_assoc",
    Base.metadata,
    Column("question_id", Integer, ForeignKey("questions.id"), primary_key=True),
    Column("model_name", String, primary_key=True),
    Column("relevance_score", Integer, nullable=True),
)


class Question(Base):
    """SQLAlchemy model for storing questions and answers."""

    __tablename__ = "questions"

    id = Column(Integer, primary_key=True)
    question_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=True)
    was_useful = Column(Boolean, nullable=True)
    feedback = Column(Text, nullable=True)
    question_metadata = Column(JSONB, nullable=True)  # For any additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Used models - accessible via relationship
    models = relationship(
        "QuestionModel", backref="questions", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Question(id={self.id})>"


class QuestionModel(Base):
    """SQLAlchemy model for tracking which models were used for each question."""

    __tablename__ = "question_models"

    question_id = Column(Integer, ForeignKey("questions.id"), primary_key=True)
    model_name = Column(String, primary_key=True)
    relevance_score = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<QuestionModel(question_id={self.question_id}, model_name='{self.model_name}')>"
