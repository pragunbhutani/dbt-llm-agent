from django.contrib import admin

# Local model import
from .models import ModelEmbedding


@admin.register(ModelEmbedding)
class ModelEmbeddingAdmin(admin.ModelAdmin):
    list_display = ("model_name", "can_be_used_for_answers", "created_at", "updated_at")
    search_fields = ("model_name", "document")
    list_filter = ("can_be_used_for_answers", "created_at")
    readonly_fields = ("created_at", "updated_at", "embedding")
    # Explicitly define fields to show in the change form, excluding 'embedding'
    fields = (
        "model_name",
        "document",
        "model_metadata",
        "can_be_used_for_answers",
        "created_at",
        "updated_at",
    )


# Note: The 'embed_models' admin action lives on ModelAdmin in knowledge_base,
# but its logic involves creating ModelEmbedding instances.
# Consider refactoring services for cleaner separation.
