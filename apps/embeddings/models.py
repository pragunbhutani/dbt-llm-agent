from django.db import models
from pgvector.django import VectorField

# Potentially import Model from knowledge_base if model_name becomes a ForeignKey
# from apps.knowledge_base.models import Model


class ModelEmbedding(models.Model):
    """Stores model embeddings based on documentation or other text."""

    # Consider adding ForeignKey(Model, ...) if model_name should reference Model.name
    model_name = models.CharField(max_length=255, null=False)
    document = models.TextField(null=False)
    embedding = VectorField(
        dimensions=1536, null=False, help_text="Embedding based on model documentation"
    )
    model_metadata = models.JSONField(null=True, blank=True)
    can_be_used_for_answers = models.BooleanField(
        null=False,
        default=True,
        help_text="Whether this embedding can be used for answering questions",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Embedding for {self.model_name} (ID: {self.id})"

    class Meta:
        db_table = "model_embeddings"  # Match existing table name
        verbose_name = "Model Embedding"
        verbose_name_plural = "Model Embeddings"


# Note: Removed Model, Question, QuestionModel classes.
