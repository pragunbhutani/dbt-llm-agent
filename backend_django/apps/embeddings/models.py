from django.db import models
from pgvector.django import VectorField

from apps.knowledge_base.models import Model
from apps.accounts.models import OrganisationScopedModelMixin  # Import the mixin
from apps.data_sources.models import DbtProject


class ModelEmbedding(OrganisationScopedModelMixin, models.Model):
    """Stores model embeddings based on documentation or other text."""

    # Organisation field will be inherited from OrganisationScopedModelMixin
    dbt_project = models.ForeignKey(
        DbtProject, on_delete=models.CASCADE, related_name="embeddings", null=True
    )
    model = models.ForeignKey(
        Model, on_delete=models.CASCADE, related_name="embeddings", null=True
    )
    document = models.TextField(null=False)
    embedding = VectorField(
        dimensions=3072, null=False, help_text="Embedding based on model documentation"
    )
    model_metadata = models.JSONField(null=True, blank=True)
    can_be_used_for_answers = models.BooleanField(
        null=False,
        default=True,
        help_text="Whether this embedding can be used for answering questions",
    )
    is_processing = models.BooleanField(
        null=False,
        default=False,
        help_text="Whether this model is currently being processed/trained",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Embedding for {self.model.name} (ID: {self.id}) - Org: {self.organisation_id}"

    class Meta:
        db_table = "model_embeddings"  # Match existing table name
        verbose_name = "Model Embedding"
        verbose_name_plural = "Model Embeddings"


# Note: Removed Model, Question, QuestionModel classes.
