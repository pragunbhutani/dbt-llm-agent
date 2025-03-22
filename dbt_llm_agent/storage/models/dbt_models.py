"""SQLAlchemy models for DBT model storage."""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.schema import UniqueConstraint

from dbt_llm_agent.storage.models import Base


class ModelTable(Base):
    """SQLAlchemy model for the models table."""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)
    path = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    schema = Column(String, nullable=True)
    database = Column(String, nullable=True)
    materialization = Column(String, nullable=True)
    tags = Column(JSONB, nullable=True)
    depends_on = Column(JSONB, nullable=True)
    tests = Column(JSONB, nullable=True)
    all_upstream_models = Column(JSONB, nullable=True)
    columns = Column(JSONB, nullable=True)  # Add columns directly to the model table
    meta = Column(JSONB, nullable=True)
    raw_sql = Column(Text, nullable=True)
    documentation = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self):
        return f"<Model(name='{self.name}')>"


class ColumnTable(Base):
    """SQLAlchemy model for the columns table."""

    __tablename__ = "columns"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    data_type = Column(String, nullable=True)
    meta = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (UniqueConstraint("model_id", "name", name="uix_model_column"),)

    def __repr__(self):
        return f"<Column(name='{self.name}')>"


class TestTable(Base):
    """SQLAlchemy model for the tests table."""

    __tablename__ = "tests"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    name = Column(String, nullable=False)
    column_name = Column(String, nullable=True)
    test_type = Column(String, nullable=True)
    unique_id = Column(String, nullable=True)
    meta = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("model_id", "name", "column_name", name="uix_model_test"),
    )

    def __repr__(self):
        return f"<Test(name='{self.name}')>"


class DependencyTable(Base):
    """SQLAlchemy model for the dependencies table."""

    __tablename__ = "dependencies"

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    depends_on_name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("model_id", "depends_on_name", name="uix_model_dependency"),
    )

    def __repr__(self):
        return f"<Dependency(depends_on='{self.depends_on_name}')>"
