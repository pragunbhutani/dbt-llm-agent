import logging
from django.conf import settings
from django.contrib import admin, messages
from django.db import transaction
from django.utils import timezone

# Local model import
from .models import Model

# Imports for Admin Actions
# Remove direct service/agent imports used only by actions
# Import the new service functions
from apps.embeddings.services import embed_knowledge_model
from apps.workflows.services import trigger_model_interpretation
from apps.accounts.models import OrganisationSettings

logger = logging.getLogger(__name__)


@admin.register(Model)
class ModelAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "unique_id",
        "schema_name",
        "materialization",
        "created_at",
        "updated_at",
    )
    search_fields = ("name", "unique_id", "yml_description", "path")
    list_filter = ("materialization", "schema_name", "database")
    readonly_fields = (
        "created_at",
        "updated_at",
    )
    actions = ["embed_models", "interpret_models", "interpret_and_embed_models"]

    # --- embed_models action ---
    def embed_models(self, request, queryset):
        """Triggers embedding for selected models using the embedding service."""
        success_count = 0
        error_count = 0
        for model in queryset:
            try:
                # Call the service function
                if embed_knowledge_model(model=model, include_docs=True):
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                # Catch unexpected errors during service call
                logger.error(
                    f"Unexpected error embedding model {model.name}: {e}", exc_info=True
                )
                messages.error(request, f"Unexpected error embedding {model.name}: {e}")
                error_count += 1

        msg = f"Successfully triggered embedding for {success_count} model(s)."
        if error_count > 0:
            msg += f" Failed to trigger/complete embedding for {error_count} model(s). Check logs."
            messages.warning(request, msg)
        else:
            messages.success(request, msg)

    embed_models.short_description = "Embed selected models (using Service)"

    # --- interpret_models action ---
    def interpret_models(self, request, queryset):
        """Triggers interpretation for selected models using the workflow service."""
        success_count = 0
        error_count = 0
        # Get verbosity level once
        admin_verbosity = getattr(settings, "AGENT_DEFAULT_VERBOSITY", 0)

        for model in queryset:
            try:
                # Get the organisation settings
                try:
                    org_settings = OrganisationSettings.objects.get(
                        organisation=model.organisation
                    )
                except OrganisationSettings.DoesNotExist:
                    logger.error(
                        f"OrganisationSettings not found for organisation: {model.organisation}"
                    )
                    messages.error(
                        request,
                        f"OrganisationSettings not found for model {model.name}",
                    )
                    error_count += 1
                    continue

                # Call the service function
                if trigger_model_interpretation(
                    model=model, org_settings=org_settings, verbosity=admin_verbosity
                ):
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                # Catch unexpected errors during service call
                logger.error(
                    f"Unexpected error interpreting model {model.name}: {e}",
                    exc_info=True,
                )
                messages.error(
                    request, f"Unexpected error interpreting {model.name}: {e}"
                )
                error_count += 1

        msg = f"Successfully triggered interpretation for {success_count} model(s)."
        if error_count > 0:
            msg += f" Failed to trigger/complete interpretation for {error_count} model(s). Check logs."
            messages.warning(request, msg)
        else:
            messages.success(request, msg)

    interpret_models.short_description = "Interpret selected models (using Service)"

    # --- interpret_and_embed_models action ---
    def interpret_and_embed_models(self, request, queryset):
        """Triggers interpretation and then embedding for selected models."""
        interpret_success_count = 0
        interpret_error_count = 0
        embed_success_count = 0
        embed_error_count = 0
        admin_verbosity = getattr(settings, "AGENT_DEFAULT_VERBOSITY", 0)

        for model in queryset:
            # Interpret
            try:
                # Get the organisation settings
                try:
                    org_settings = OrganisationSettings.objects.get(
                        organisation=model.organisation
                    )
                except OrganisationSettings.DoesNotExist:
                    logger.error(
                        f"OrganisationSettings not found for organisation: {model.organisation}"
                    )
                    messages.error(
                        request,
                        f"OrganisationSettings not found for model {model.name}",
                    )
                    interpret_error_count += 1
                    continue

                if trigger_model_interpretation(
                    model=model, org_settings=org_settings, verbosity=admin_verbosity
                ):
                    interpret_success_count += 1
                    # Embed only if interpretation was successful
                    try:
                        if embed_knowledge_model(model=model, include_docs=True):
                            embed_success_count += 1
                        else:
                            embed_error_count += 1
                    except Exception as e:
                        logger.error(
                            f"Unexpected error embedding model {model.name} after interpretation: {e}",
                            exc_info=True,
                        )
                        messages.error(
                            request,
                            f"Unexpected error embedding {model.name} after interpretation: {e}",
                        )
                        embed_error_count += 1
                else:
                    interpret_error_count += 1
            except Exception as e:
                logger.error(
                    f"Unexpected error interpreting model {model.name}: {e}",
                    exc_info=True,
                )
                messages.error(
                    request, f"Unexpected error interpreting {model.name}: {e}"
                )
                interpret_error_count += 1

        # Prepare messages
        interpret_msg = f"Interpretation: {interpret_success_count} succeeded, {interpret_error_count} failed."
        embed_msg = f"Embedding: {embed_success_count} succeeded, {embed_error_count} failed (tried for {interpret_success_count} models)."

        final_message = f"{interpret_msg} {embed_msg}"

        if interpret_error_count > 0 or embed_error_count > 0:
            messages.warning(request, final_message)
        else:
            messages.success(request, final_message)

    interpret_and_embed_models.short_description = (
        "Interpret and Embed selected models (using Service)"
    )
