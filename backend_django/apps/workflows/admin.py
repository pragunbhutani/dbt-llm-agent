from django.contrib import admin

# Local model imports
from .models import Question, QuestionModel


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "question_text",
        "was_useful",
        "created_at",
        "response_message_ts",
    )
    search_fields = (
        "question_text",
        "answer_text",
        "feedback",
        "original_message_text",
    )
    list_filter = ("was_useful", "created_at")
    readonly_fields = (
        "created_at",
        "updated_at",
        "question_embedding",
        "feedback_embedding",
        "original_message_embedding",
    )  # Embeddings are not editable here


@admin.register(QuestionModel)
class QuestionModelAdmin(admin.ModelAdmin):
    list_display = ("question_id", "model", "relevance_score", "created_at")
    list_select_related = ("question", "model")  # Optimize query
    readonly_fields = ("created_at",)
