import logging
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.knowledge_base.models import Model
from apps.embeddings.models import ModelEmbedding
from apps.llm_providers.services import default_embedding_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Generates and stores embeddings for specified dbt models."

    def add_arguments(self, parser):
        parser.add_argument(
            "--models",
            nargs="*",  # 0 or more model names
            type=str,
            help="Specific model names to embed. If not provided, processes all models.",
        )
        parser.add_argument(
            "--all", action="store_true", help="Embed all models found in the database."
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force regeneration of embeddings even if they already exist.",
        )
        parser.add_argument(
            "--include-docs",
            action="store_true",
            help="Include YML description and columns in the text used for embedding.",
        )

    def handle(self, *args, **options):
        model_names = options["models"]
        embed_all = options["all"]
        force_update = options["force"]
        include_docs = options["include_docs"]
        verbosity = options["verbosity"]

        if not model_names and not embed_all:
            raise CommandError(
                "You must specify model names using --models or use --all."
            )
        if model_names and embed_all:
            raise CommandError("You cannot use both --models and --all.")

        # Select models
        if embed_all:
            models_to_embed = Model.objects.all()
            self.stdout.write("Processing all models...")
        else:
            models_to_embed = Model.objects.filter(name__in=model_names)
            found_names = set(models_to_embed.values_list("name", flat=True))
            missing_names = set(model_names) - found_names
            if missing_names:
                self.stdout.write(
                    self.style.WARNING(
                        f"Could not find models: {', '.join(missing_names)}"
                    )
                )
            self.stdout.write(f"Processing specified models: {', '.join(found_names)}")

        if not models_to_embed.exists():
            self.stdout.write(self.style.WARNING("No models found to embed."))
            return

        processed_count = 0
        created_count = 0
        updated_count = 0
        skipped_count = 0
        error_count = 0

        total_to_process = models_to_embed.count()
        self.stdout.write(f"Found {total_to_process} models to process.")

        for model in models_to_embed:
            processed_count += 1
            if verbosity > 1:
                self.stdout.write(
                    f"Processing ({processed_count}/{total_to_process}): {model.name}"
                )

            # Check if embedding exists and if force is not set
            existing_embedding = ModelEmbedding.objects.filter(
                model_name=model.name
            ).first()
            if existing_embedding and not force_update:
                if verbosity > 1:
                    self.stdout.write(
                        f"  Skipping {model.name}, embedding exists and --force not used."
                    )
                skipped_count += 1
                continue

            # Generate text representation
            try:
                document_text = model.get_text_representation(
                    include_documentation=include_docs
                )
                if not document_text:
                    self.stdout.write(
                        self.style.WARNING(
                            f"  Skipping {model.name}, generated empty text representation."
                        )
                    )
                    skipped_count += 1
                    continue
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"  Error generating text for {model.name}: {e}")
                )
                error_count += 1
                continue

            # Generate embedding
            embedding_vector = default_embedding_service.get_embedding(document_text)
            if not embedding_vector:
                self.stdout.write(
                    self.style.WARNING(
                        f"  Skipping {model.name}, failed to generate embedding vector."
                    )
                )
                # Consider if this should be an error or just skipped
                error_count += 1  # Count as error for now
                continue

            # Prepare metadata for ModelEmbedding
            embedding_metadata = {
                "schema": model.schema_name,
                "database": model.database,
                "materialization": model.materialization,
                "tags": model.tags,
                # Add other relevant fields from Model if needed
            }

            # Save embedding
            try:
                with transaction.atomic():  # Ensure atomic update/create
                    embedding_instance, created = (
                        ModelEmbedding.objects.update_or_create(
                            model_name=model.name,
                            defaults={
                                "document": document_text,
                                "embedding": embedding_vector,
                                "model_metadata": embedding_metadata,
                                "can_be_used_for_answers": True,  # Default to true
                            },
                        )
                    )
                    if created:
                        created_count += 1
                    else:
                        updated_count += 1
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"  Error saving embedding for {model.name}: {e}")
                )
                error_count += 1

        # Final summary
        self.stdout.write(self.style.SUCCESS("Embedding process complete."))
        self.stdout.write(f"  Total models considered: {total_to_process}")
        self.stdout.write(f"  Embeddings created: {created_count}")
        self.stdout.write(f"  Embeddings updated: {updated_count}")
        self.stdout.write(f"  Skipped (already exists): {skipped_count}")
        self.stdout.write(f"  Errors: {error_count}")
