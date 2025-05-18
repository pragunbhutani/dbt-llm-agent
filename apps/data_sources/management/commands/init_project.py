import logging
from pathlib import Path
import subprocess
import requests
import tempfile
import json
import os
import sys
from django.core.management.base import BaseCommand, CommandError
from django.core.management import call_command
from django.db import transaction
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Assuming Model is importable like this, adjust if necessary
# from ragstar.core.models import Model
# Or get it via apps registry if models aren't directly importable
from django.apps import apps

# Import the new services
from apps.data_sources.services import import_from_manifest, import_from_source

logger = logging.getLogger(__name__)
console = Console()


class Command(BaseCommand):
    help = "Initializes the database by parsing dbt artifacts using cloud, core (compile), or source methods."

    def add_arguments(self, parser):
        parser.add_argument(
            "--method",
            type=str,
            choices=["cloud", "core", "source"],
            default="core",
            help="Method for parsing: 'cloud' (fetches from dbt Cloud API via env vars), 'core' (runs dbt compile locally), 'source' (parses local source files).",
        )
        parser.add_argument(
            "--mode",
            type=str,
            choices=["insert", "update", "upsert", "overwrite", "reset"],
            default="upsert",
            help="Import mode: 'insert' (new only), 'update' (existing only), 'upsert' (add/update), 'overwrite'/'reset' (clear then import). Note: 'insert'/'update' behavior depends on underlying command support.",
        )
        parser.add_argument(
            "--project-path",
            type=str,
            # envvar="DBT_PROJECT_PATH", # Prefer explicit fetching for clarity
            help="Path to the root dbt project directory (required for --method core and --method source). Can also be set via DBT_PROJECT_PATH env var.",
        )
        # Removed dbt Cloud specific CLI args, will rely on env vars
        # parser.add_argument(...)

    def _clear_data(self):
        """Clears existing Model data from the database."""
        try:
            Model = apps.get_model(app_label="knowledge_base", model_name="Model")
            with transaction.atomic():
                deleted_count, _ = Model.objects.all().delete()
                self.stdout.write(
                    self.style.SUCCESS(f"Cleared {deleted_count} existing models.")
                )
        except LookupError:
            raise CommandError(
                "Could not find the 'Model' model. Ensure 'core' app is correctly configured."
            )
        except Exception as e:
            raise CommandError(f"Failed to clear existing model data: {e}")

    def _get_dbt_cloud_manifest(
        self, dbt_cloud_url, dbt_cloud_account_id, dbt_cloud_api_key
    ):
        """Fetches the latest manifest.json from dbt Cloud using provided credentials."""
        # Validation happens in the handle method before calling this
        headers = {
            "Authorization": f"Token {dbt_cloud_api_key}",
            "Content-Type": "application/json",
        }
        base_api_url = (
            f"{dbt_cloud_url.rstrip('/')}/api/v2/accounts/{dbt_cloud_account_id}"
        )
        latest_run_id = None
        manifest_content = None
        temp_manifest_path = None

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=console,
            ) as progress:
                # 1. Find latest successful run with artifacts
                runs_task = progress.add_task(
                    "Finding latest dbt Cloud run with manifest...", total=None
                )
                runs_url = f"{base_api_url}/runs/"
                params = {
                    "order_by": "-finished_at",
                    "status": 10,
                    "limit": 10,
                }  # status 10 = success
                response = requests.get(
                    runs_url, headers=headers, params=params, timeout=30
                )
                response.raise_for_status()
                runs_data = response.json().get("data", [])

                if not runs_data:
                    raise CommandError(
                        f"No recent successful runs found for account ID {dbt_cloud_account_id}."
                    )

                for run in runs_data:
                    run_id = run.get("id")
                    if not run_id:
                        continue
                    artifact_url = (
                        f"{base_api_url}/runs/{run_id}/artifacts/manifest.json"
                    )
                    try:
                        artifact_response = requests.get(
                            artifact_url, headers=headers, timeout=10
                        )
                        if artifact_response.status_code == 200:
                            latest_run_id = run_id
                            logger.info(
                                f"Found manifest artifact in run ID: {latest_run_id}"
                            )
                            break
                        # else: logger.debug(f"Run {run_id} manifest status: {artifact_response.status_code}")
                    except requests.exceptions.RequestException as e:
                        logger.warning(
                            f"Error checking artifacts for run {run_id}: {e}. Trying next."
                        )

                if not latest_run_id:
                    limit_val = params.get("limit", 10)  # Get limit value safely
                    raise CommandError(
                        f"Could not find manifest.json in the latest {limit_val} successful runs."
                    )

                # Create description string separately using concatenation
                description_str = "Found manifest in run " + str(latest_run_id)
                progress.update(runs_task, completed=True, description=description_str)

                # 2. Fetch the manifest
                artifact_task = progress.add_task(
                    "Fetching manifest.json...", total=None
                )
                artifact_url = (
                    f"{base_api_url}/runs/{latest_run_id}/artifacts/manifest.json"
                )
                response = requests.get(artifact_url, headers=headers, timeout=60)
                response.raise_for_status()
                manifest_content = response.json()
                progress.update(
                    artifact_task, completed=True, description="Fetched manifest.json"
                )

                # 3. Save to temp file
                save_task = progress.add_task(
                    "Saving manifest temporarily...", total=None
                )
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".json", encoding="utf-8"
                ) as temp_file:
                    json.dump(manifest_content, temp_file)
                    temp_manifest_path = temp_file.name
                progress.update(
                    save_task,
                    completed=True,
                    description=f"Saved manifest to {os.path.basename(temp_manifest_path)}",
                )

            return temp_manifest_path

        except requests.exceptions.RequestException as e:
            raise CommandError(f"dbt Cloud API request failed: {e}")
        except json.JSONDecodeError:
            raise CommandError("Failed to decode manifest.json from dbt Cloud.")
        except Exception as e:
            # Clean up temp file if created before error
            if temp_manifest_path and os.path.exists(temp_manifest_path):
                try:
                    os.remove(temp_manifest_path)
                except OSError:
                    pass
            raise CommandError(f"Error during dbt Cloud interaction: {e}")

    def _run_dbt_compile(self, project_path_obj):
        """Runs 'dbt compile' in the specified project path."""
        manifest_path = project_path_obj / "target" / "manifest.json"
        self.stdout.write(f"Running 'dbt compile' in {project_path_obj}...")
        try:
            # Try common ways to run dbt
            commands_to_try = [
                ["dbt", "compile"],
                ["poetry", "run", "dbt", "compile"],
                # Add other potential wrappers if necessary
            ]
            process = None
            cmd_used = None
            for cmd in commands_to_try:
                try:
                    process = subprocess.run(
                        cmd,
                        cwd=project_path_obj,
                        capture_output=True,
                        text=True,
                        check=True,
                        encoding="utf-8",
                        errors="ignore",  # Avoid decode errors on unusual output
                    )
                    cmd_used = cmd  # Store the command list itself
                    logger.info(f"Successfully ran: {' '.join(cmd)}")
                    logger.debug(f"dbt compile stdout:\n{process.stdout}")
                    logger.debug(f"dbt compile stderr:\n{process.stderr}")
                    break  # Stop trying commands if one succeeds
                except FileNotFoundError:
                    logger.debug(f"Command not found: {' '.join(cmd)}. Trying next.")
                    continue  # Try the next command
                except subprocess.CalledProcessError as e:
                    # This command exists but failed
                    cmd_str = " ".join(cmd)
                    logger.error(f"'{cmd_str}' failed with exit code {e.returncode}")
                    logger.error(f"Stdout:\n{e.stdout}")
                    logger.error(f"Stderr:\n{e.stderr}")
                    raise CommandError(
                        f"'{cmd_str}' failed. Check dbt logs or project setup. Error: {e.stderr[:500]}"
                    )  # Show truncated error

            if process is None or process.returncode != 0:
                raise CommandError(
                    "Could not successfully execute 'dbt compile'. Ensure dbt is installed and runnable in the project directory (e.g., via 'dbt' or 'poetry run dbt')."
                )

            cmd_str = " ".join(cmd_used)  # Create cmd string outside f-string
            self.stdout.write(
                self.style.SUCCESS(f"'{cmd_str} compile' completed successfully.")
            )

        except subprocess.CalledProcessError as e:
            # Handled above but catch here just in case
            raise CommandError(
                f"'dbt compile' failed. Check dbt logs. Error: {e.stderr[:500]}"
            )
        except Exception as e:
            raise CommandError(
                f"An unexpected error occurred while running dbt compile: {e}"
            )

        if not manifest_path.exists():
            raise CommandError(
                f"manifest.json not found at {manifest_path} after running 'dbt compile'."
            )

        self.stdout.write(f"Found manifest: {manifest_path}")
        return str(manifest_path)

    def handle(self, *args, **options):
        method = options["method"]
        mode = options["mode"]
        project_path_cli = options["project_path"]
        verbosity = options["verbosity"]

        # Fetch project_path from CLI arg or ENV var
        project_path = project_path_cli or os.getenv("DBT_PROJECT_PATH")

        self.stdout.write(
            f"Initializing project using method: '{method}', mode: '{mode}'"
        )

        # Determine flags based on mode
        clear_db = mode in ["overwrite", "reset"]
        force_update = mode in ["update", "upsert", "overwrite", "reset"]

        # --- Pre-run Checks & Variable Fetching ---
        dbt_cloud_url = None
        dbt_cloud_account_id = None
        dbt_cloud_api_key = None

        if method == "cloud":
            dbt_cloud_url = os.getenv("DBT_CLOUD_URL")
            dbt_cloud_account_id_str = os.getenv("DBT_CLOUD_ACCOUNT_ID")
            dbt_cloud_api_key = os.getenv("DBT_CLOUD_API_KEY")
            if not all([dbt_cloud_url, dbt_cloud_account_id_str, dbt_cloud_api_key]):
                raise CommandError(
                    "Environment variables DBT_CLOUD_URL, DBT_CLOUD_ACCOUNT_ID, and DBT_CLOUD_API_KEY are required when method is 'cloud'."
                )
            try:
                dbt_cloud_account_id = int(dbt_cloud_account_id_str)
            except (ValueError, TypeError):
                raise CommandError(
                    "DBT_CLOUD_ACCOUNT_ID environment variable must be an integer."
                )

        elif method in ["core", "source"]:
            if not project_path:
                raise CommandError(
                    f"Argument --project-path (or DBT_PROJECT_PATH env var) is required when method is '{method}'."
                )
            project_p = Path(project_path).resolve()
            if not (project_p / "dbt_project.yml").exists():
                raise CommandError(
                    f"dbt_project.yml not found in specified project path: {project_path}"
                )
            project_path = str(project_p)  # Use resolved path

        temp_manifest_to_clean = None
        results = None

        try:
            # --- Clear Data if Requested ---
            if clear_db:
                self.stdout.write(
                    self.style.WARNING(
                        f"Mode '{mode}' selected. Clearing existing model data first..."
                    )
                )
                self._clear_data()

            # --- Execute Based on Method ---
            if method == "cloud":
                manifest_p = None
                try:
                    manifest_p = self._get_dbt_cloud_manifest(
                        dbt_cloud_url, dbt_cloud_account_id, dbt_cloud_api_key
                    )
                    temp_manifest_to_clean = manifest_p
                    self.stdout.write("Importing from downloaded manifest...")
                    # Call service directly
                    results = import_from_manifest(
                        manifest_path_str=manifest_p,
                        clear_data=False,  # Already cleared above if requested
                    )
                finally:
                    # Clean up temp file
                    if temp_manifest_to_clean and os.path.exists(
                        temp_manifest_to_clean
                    ):
                        try:
                            os.remove(temp_manifest_to_clean)
                        except OSError:
                            pass

            elif method == "core":
                if not project_path:
                    raise CommandError("Project path is required.")
                manifest_p = self._run_dbt_compile(Path(project_path))
                self.stdout.write("Importing from compiled manifest...")
                # Call service directly
                results = import_from_manifest(
                    manifest_path_str=manifest_p,
                    clear_data=False,  # Already cleared above if requested
                )

            elif method == "source":
                if not project_path:
                    raise CommandError("Project path is required.")
                self.stdout.write("Importing from source files...")
                # Call service directly
                results = import_from_source(project_path_str=project_path)

            # --- Report Results ---
            if results:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Project initialization complete (method: {method}).\n"
                        f"  Models processed: {results.get('total', 0)}\n"
                        f"  Models created: {results.get('created', 0)}\n"
                        f"  Models updated: {results.get('updated', 0)}\n"
                        f"  Models skipped: {results.get('skipped', 0)}"
                    )
                )
            else:
                # Should not happen unless an error occurred before service call
                self.stdout.write(
                    self.style.WARNING(
                        "Initialization completed, but no import results generated."
                    )
                )

        except CommandError as e:
            raise e
        except Exception as e:
            logger.exception(f"An unexpected error occurred during initialization: {e}")
            raise CommandError(
                f"Initialization failed. See logs for details. Error: {e}"
            )
        finally:
            if temp_manifest_to_clean and os.path.exists(temp_manifest_to_clean):
                try:
                    os.remove(temp_manifest_to_clean)
                    logger.info(
                        f"Cleaned up temporary manifest file on exit: {temp_manifest_to_clean}"
                    )
                except OSError as e:
                    logger.warning(
                        f"Could not remove temporary manifest file {temp_manifest_to_clean} on exit: {e}"
                    )
