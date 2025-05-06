import logging
import re
import json  # Import json for processing agent output
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from typing import List, Dict, Any, Optional  # Add Any, Optional

# Django Imports
from apps.knowledge_base.models import Model
from apps.llm_providers.services import default_chat_service

# Agent Import
from apps.workflows.model_interpreter import (
    ModelInterpreterAgent,
    ModelDocumentation,  # Import Pydantic model for validation
    ColumnDocumentation,
)

# Rich console for agent verbosity (optional)
try:
    from rich.console import Console as RichConsole

    console = RichConsole()
except ImportError:
    console = None
    RichConsole = None

logger = logging.getLogger(__name__)

# Removed find_refs helper function - Agent handles refs internally


class Command(BaseCommand):
    help = (
        "Generates agentic LLM interpretations (description, columns) for dbt models."
    )

    def add_arguments(self, parser):
        # Keep existing arguments, maybe add agent-specific ones later if needed
        parser.add_argument(
            "--models",
            nargs="*",
            type=str,
            help="Specific model names to interpret. If not provided, processes all models.",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Interpret all models found in the database that have raw SQL.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force regeneration of interpretations even if they already exist.",
        )
        # Remove skip flags? The agent workflow generates both.
        # For now, keep them, but maybe disable interpretation steps based on them.
        parser.add_argument(
            "--skip-desc",
            action="store_true",
            help="(Currently Ignored by Agent) Skip interpreting the model description.",
        )
        parser.add_argument(
            "--skip-cols",
            action="store_true",
            help="(Currently Ignored by Agent) Skip interpreting the model columns.",
        )

    def handle(self, *args, **options):
        model_names = options["models"]
        interpret_all = options["all"]
        force_update = options["force"]
        skip_desc = options["skip_desc"]  # Currently unused by agent workflow
        skip_cols = options["skip_cols"]  # Currently unused by agent workflow
        verbosity = options["verbosity"]

        # Basic validation (keep as is)
        if not model_names and not interpret_all:
            raise CommandError("Specify --models or use --all.")
        if model_names and interpret_all:
            raise CommandError("Cannot use both --models and --all.")
        # if skip_desc and skip_cols:
        #     self.stdout.write(self.style.WARNING("Agent generates both desc/cols. Ignoring skip flags."))
        # return

        # Select models (keep as is)
        base_queryset = Model.objects.exclude(raw_sql__isnull=True).exclude(raw_sql="")
        if interpret_all:
            models_to_interpret = base_queryset.all()
            self.stdout.write(
                "Attempting agentic interpretation for all models with raw SQL..."
            )
        else:
            models_to_interpret = base_queryset.filter(name__in=model_names)
            found_names = set(models_to_interpret.values_list("name", flat=True))
            missing_names = set(model_names) - found_names
            if missing_names:
                self.stdout.write(
                    self.style.WARNING(f"Models not found/no SQL: {missing_names}")
                )
            self.stdout.write(f"Processing specified models: {list(found_names)}")

        if not models_to_interpret.exists():
            self.stdout.write(self.style.WARNING("No models found to interpret."))
            return

        # --- Initialize Agent --- #
        # Pass the integer verbosity level directly
        # agent_verbose = verbosity > 1 # Old logic
        try:
            # Pass chat service and console (if available and verbosity > 0) to agent
            interpreter_agent = ModelInterpreterAgent(
                chat_service=default_chat_service,
                verbosity=verbosity,  # Pass integer verbosity directly
                console=(
                    console if verbosity > 0 else None
                ),  # Pass console if verbosity > 0
            )
        except ValueError as e:  # Catch init errors (e.g., no LLM client)
            raise CommandError(f"Failed to initialize ModelInterpreterAgent: {e}")
        # --- End Initialize Agent --- #

        processed_count = 0
        updated_count = 0  # Count successful updates
        failed_count = 0

        total_to_process = models_to_interpret.count()
        self.stdout.write(f"Found {total_to_process} models to process.")

        for model in models_to_interpret:
            processed_count += 1
            self.stdout.write(
                f"\nProcessing ({processed_count}/{total_to_process}): {model.name}"
            )

            # Check force flag or if interpretation is missing
            should_process = (
                force_update
                or not model.interpreted_description
                or not model.interpreted_columns
            )
            if not should_process:
                self.stdout.write(
                    f"  Skipping {model.name}, interpretation exists and --force not used."
                )
                continue

            if not model.raw_sql:
                self.stdout.write(
                    self.style.WARNING(f"  Skipping {model.name}, raw_sql is missing.")
                )
                failed_count += 1  # Count as failure if SQL missing
                continue

            # --- Run Agent Workflow --- #
            if verbosity > 0:
                self.stdout.write(
                    f"  Running agent interpretation workflow for {model.name}..."
                )

            agent_result = interpreter_agent.run_interpretation_workflow(
                model_name=model.name, raw_sql=model.raw_sql
            )

            # --- Process Agent Result --- #
            if agent_result["success"] and agent_result["documentation"]:
                documentation_dict = agent_result["documentation"]

                # Basic validation using Pydantic model (optional but good practice)
                try:
                    validated_doc = ModelDocumentation(**documentation_dict)
                except Exception as pydantic_err:
                    self.stdout.write(
                        self.style.ERROR(
                            f"  Agent returned invalid documentation structure for {model.name}: {pydantic_err}"
                        )
                    )
                    logger.error(
                        f"Pydantic validation failed for {model.name}. Data: {documentation_dict}",
                        exc_info=True,
                    )
                    failed_count += 1
                    continue  # Skip saving invalid data

                # Extract data for saving
                interpreted_description = validated_doc.description
                interpreted_columns = {
                    col.name: col.description for col in validated_doc.columns
                }

                # Save to Django Model
                try:
                    with transaction.atomic():
                        model.interpreted_description = interpreted_description
                        model.interpreted_columns = interpreted_columns
                        model.save(
                            update_fields=[
                                "interpreted_description",
                                "interpreted_columns",
                            ]
                        )
                        updated_count += 1
                        if verbosity > 0:
                            self.stdout.write(
                                self.style.SUCCESS(
                                    f"  Successfully saved interpretation for {model.name}"
                                )
                            )
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f"  Error saving interpretation for {model.name}: {e}"
                        )
                    )
                    logger.error(
                        f"Failed to save interpretation for {model.name}", exc_info=True
                    )
                    failed_count += 1

            else:  # Agent workflow failed
                failed_count += 1
                error_msg = agent_result.get("error", "Unknown agent error")
                self.stdout.write(
                    self.style.ERROR(
                        f"  Agent failed to interpret {model.name}: {error_msg}"
                    )
                )
                # Optionally log agent messages on failure
                if verbosity > 2 and agent_result.get("messages"):
                    self.stdout.write("  Agent messages on failure:")
                    try:
                        # Pretty print messages if rich console available
                        if console:
                            console.print(agent_result["messages"])
                        else:
                            self.stdout.write(str(agent_result["messages"]))
                    except Exception:
                        self.stdout.write("  (Could not display agent messages)")

            # --- End Process Agent Result ---

        # Final summary
        self.stdout.write(
            self.style.SUCCESS("\nAgentic interpretation process complete.")
        )
        self.stdout.write(f"  Models processed: {processed_count}")
        self.stdout.write(
            f"  Interpretations successfully generated/updated: {updated_count}"
        )
        self.stdout.write(
            f"  Failures (missing SQL, agent error, save error): {failed_count}"
        )
