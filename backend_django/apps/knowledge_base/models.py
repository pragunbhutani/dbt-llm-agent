from django.db import models
from django.contrib.postgres.fields import ArrayField
from pgvector.django import (
    VectorField,
)  # Assuming this might still be used, if not, can be removed later
from apps.accounts.models import OrganisationScopedModelMixin  # Import the mixin
from apps.data_sources.models import DbtProject

# Assuming PostgreSQL for ArrayField/JSONField, JSONField is standard in recent Django


# Model representing a dbt artifact and its interpretation
class Model(OrganisationScopedModelMixin, models.Model):
    # Organisation field will be inherited from OrganisationScopedModelMixin
    dbt_project = models.ForeignKey(
        DbtProject, on_delete=models.CASCADE, related_name="models", null=True
    )
    name = models.CharField(max_length=255, null=False)
    path = models.CharField(max_length=1024, null=False)
    schema_name = models.CharField(
        max_length=255, null=True, blank=True
    )  # Renamed from schema
    database = models.CharField(max_length=255, null=True, blank=True)
    materialization = models.CharField(max_length=50, null=True, blank=True)
    tags = ArrayField(models.CharField(max_length=255), null=True, blank=True)
    depends_on = ArrayField(models.CharField(max_length=255), null=True, blank=True)
    tests = models.JSONField(null=True, blank=True)
    all_upstream_models = ArrayField(
        models.CharField(max_length=255), null=True, blank=True
    )
    meta = models.JSONField(null=True, blank=True)
    raw_sql = models.TextField(null=True, blank=True)
    compiled_sql = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    yml_description = models.TextField(
        null=True, blank=True, help_text="Description from YML documentation"
    )
    yml_columns = models.JSONField(
        null=True, blank=True, help_text="Columns from YML documentation"
    )
    interpreted_columns = models.JSONField(
        null=True, blank=True, help_text="LLM-interpreted column descriptions"
    )
    interpreted_description = models.TextField(
        null=True, blank=True, help_text="LLM-generated description of the model"
    )
    interpretation_details = models.JSONField(null=True, blank=True)
    unique_id = models.CharField(max_length=255, null=True, blank=True)

    def get_text_representation(self, include_documentation: bool = False) -> str:
        """Get a text representation of the model for embedding.

        Args:
            include_documentation: Whether to include YML description and columns.
        """
        representation = f"Model: {self.name}\n\n"
        representation += f"Path: {self.path}\n\n"

        if self.depends_on:
            representation += f"Depends on: {', '.join(self.depends_on)}\n\n"
        else:
            representation += "Depends on: None\n\n"

        if include_documentation:
            representation += f"YML Description: {self.yml_description or 'N/A'}\n\n"

        representation += (
            f"Interpreted Description: {self.interpreted_description or 'N/A'}\n\n"
        )

        if include_documentation and self.yml_columns:
            representation += "YML Columns:\n"
            if isinstance(self.yml_columns, dict):
                for col_name, col_data in self.yml_columns.items():
                    col_desc = (
                        col_data.get("description", "N/A")
                        if isinstance(col_data, dict)
                        else "N/A"
                    )
                    representation += f"  - {col_name}: {col_desc}\n"
            elif isinstance(self.yml_columns, list):  # Handle list case just in case
                for col_data in self.yml_columns:
                    if isinstance(col_data, dict) and "name" in col_data:
                        col_name = col_data["name"]
                        col_desc = col_data.get("description", "N/A")
                        representation += f"  - {col_name}: {col_desc}\n"
            else:
                representation += "  No YML column information available.\n"
            representation += "\n"

        representation += "Interpreted Columns:\n"
        if self.interpreted_columns and isinstance(self.interpreted_columns, dict):
            for col_name, col_desc in self.interpreted_columns.items():
                representation += f"  - {col_name}: {col_desc or 'N/A'}\n"
            representation += "\n"
        else:
            representation += "  No interpreted column information available.\n\n"

        representation += (
            f"Raw SQL:\n```sql\n{self.raw_sql or 'SQL not available'}\n```\n"
        )

        return representation

    def __str__(self):
        return self.name

    class Meta:
        db_table = "models"  # Match existing table name
        verbose_name = "DBT Model"
        verbose_name_plural = "DBT Models"
        constraints = [
            models.UniqueConstraint(
                fields=["organisation", "dbt_project", "name"],
                name="unique_model_per_org_project",
            ),
            models.UniqueConstraint(
                fields=["organisation", "unique_id"],
                name="unique_model_id_per_org",
                condition=models.Q(unique_id__isnull=False),
            ),
        ]


# Note: Removed Question, QuestionModel, ModelEmbedding classes from here.
# Removed VectorField import if not needed by Model itself. Check dependencies.
# Added comment about unique_id nullability.
