from django.db import models
from pgvector.django import VectorField
from apps.knowledge_base.models import Model
from apps.accounts.models import OrganisationScopedModelMixin


class Question(OrganisationScopedModelMixin, models.Model):
    question_text = models.TextField(null=False)
    answer_text = models.TextField(null=True, blank=True)
    question_embedding = VectorField(
        dimensions=3072,
        null=True,
        blank=True,
        help_text="Embedding vector for the question text",
    )
    was_useful = models.BooleanField(null=True, blank=True)
    feedback = models.TextField(null=True, blank=True)
    feedback_embedding = VectorField(
        dimensions=3072,
        null=True,
        blank=True,
        help_text="Embedding vector for the feedback text",
    )
    question_metadata = models.JSONField(null=True, blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    original_message_text = models.TextField(null=True, blank=True)
    original_message_ts = models.CharField(max_length=50, null=True, blank=True)
    response_message_ts = models.CharField(max_length=50, null=True, blank=True)
    original_message_embedding = VectorField(
        dimensions=3072,
        null=True,
        blank=True,
        help_text="Embedding vector for the original message text",
    )
    response_file_message_ts = models.CharField(
        max_length=50, null=True, blank=True, db_index=True
    )

    models_used = models.ManyToManyField(
        Model,
        through="QuestionModel",
        related_name="questions_used_in",
    )

    def __str__(self):
        return f"Q{self.id}: {self.question_text[:50]}..."

    class Meta:
        db_table = "questions"  # Match existing table name
        verbose_name = "Question"
        verbose_name_plural = "Questions"


class QuestionModel(models.Model):
    """Association table tracking which Models were used for each Question."""

    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    model = models.ForeignKey(Model, on_delete=models.CASCADE)
    relevance_score = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "question_models"  # Match existing table name
        unique_together = (("question", "model"),)
        verbose_name = "Question Model Usage"
        verbose_name_plural = "Question Model Usages"


# Note: Removed Model and ModelEmbedding classes.
