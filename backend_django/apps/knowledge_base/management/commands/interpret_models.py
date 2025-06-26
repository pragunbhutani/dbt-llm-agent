import logging
from django.core.management.base import BaseCommand, CommandError, CommandParser
from typing import List

# Django Imports
from apps.knowledge_base.models import Model
from apps.accounts.models import Organisation, OrganisationSettings

# Import simplified service
from apps.workflows.services import trigger_model_interpretation

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Generates LLM interpretations (description, columns) for dbt models using a simplified single-prompt approach."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--model-name",
            type=str,
            help="The name of a specific model to interpret. If not provided, all models will be interpreted.",
        )
        parser.add_argument(
            "--organisation",
            type=str,
            required=True,
            help="The name or ID of the organisation whose settings to use.",
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

    def handle(self, *args, **options):
        model_names = options["model_name"]
        interpret_all = options["all"]
        force_update = options["force"]
        verbosity = options["verbosity"]
        organisation_identifier = options["organisation"]

        if not model_names and not interpret_all:
            raise CommandError("Specify --model-name or use --all.")
        if model_names and interpret_all:
            raise CommandError("Cannot use both --model-name and --all.")

        base_queryset = Model.objects.exclude(raw_sql__isnull=True).exclude(raw_sql="")
        if interpret_all:
            models_to_process = base_queryset.all()
        else:
            models_to_process = base_queryset.filter(name__in=model_names)

        try:
            if len(organisation_identifier) > 30:
                organisation = Organisation.objects.get(pk=organisation_identifier)
            else:
                organisation = Organisation.objects.get(name=organisation_identifier)
            org_settings = OrganisationSettings.objects.get(organisation=organisation)
        except (Organisation.DoesNotExist, OrganisationSettings.DoesNotExist):
            raise CommandError(
                f"Organisation '{organisation_identifier}' or its settings not found."
            )

        processed_count = 0
        updated_count = 0
        failed_count = 0
        total_to_process = models_to_process.count()

        self.stdout.write(
            f"Found {total_to_process} models to process for organisation '{organisation.name}'."
        )

        for model in models_to_process:
            processed_count += 1
            self.stdout.write(
                f"Processing ({processed_count}/{total_to_process}): {model.name}"
            )

            should_process = (
                force_update
                or not model.interpreted_description
                or not model.interpreted_columns
            )
            if not should_process:
                self.stdout.write(
                    "  Skipping, interpretation exists and --force not used."
                )
                continue

            success = trigger_model_interpretation(
                model=model, org_settings=org_settings, verbosity=verbosity
            )

            if success:
                updated_count += 1
                self.stdout.write(
                    self.style.SUCCESS("  -> Successfully interpreted and saved.")
                )
            else:
                failed_count += 1
                self.stdout.write(self.style.ERROR("  -> Failed to interpret."))

        self.stdout.write(
            self.style.SUCCESS(
                f"Finished. Updated: {updated_count}, Failed: {failed_count}, Skipped: {total_to_process - processed_count}"
            )
        )
