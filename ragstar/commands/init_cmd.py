"""
Command to initialize the ragstar project by parsing dbt artifacts.
"""

import click
import sys
import os
import subprocess
import requests
import tempfile
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Dict, List, Optional

from ragstar.utils.logging import get_logger
from ragstar.utils.cli_utils import get_config_value, set_logging_level
from ragstar.storage.model_storage import ModelStorage
from ragstar.core.parsers.source_code_parser import SourceCodeParser
from ragstar.core.parsers.manifest_parser import ManifestParser

# Initialize logger
logger = get_logger(__name__)

# Initialize console for rich output
console = Console()


@click.group()
def init():
    """Initialize the project by parsing dbt artifacts."""
    pass


@init.command()
@click.option(
    "--dbt-cloud-url",
    envvar="DBT_CLOUD_URL",
    help="URL of your dbt Cloud instance (e.g., https://cloud.getdbt.com)",
    required=True,
)
@click.option(
    "--dbt-cloud-account-id",
    envvar="DBT_CLOUD_ACCOUNT_ID",
    help="Your dbt Cloud account ID (required for API v2)",
    required=True,
    type=int,
)
@click.option(
    "--dbt-cloud-api-key",
    envvar="DBT_CLOUD_API_KEY",
    help="Your dbt Cloud API key (User Token or Service Token)",
    required=True,
)
@click.option("--force", is_flag=True, help="Force reimport of models")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cloud(
    dbt_cloud_url,
    dbt_cloud_account_id,
    dbt_cloud_api_key,
    force,
    verbose,
):
    """Initialize using dbt Cloud API. Fetches manifest from the latest successful run in the account."""
    set_logging_level(verbose)
    logger.info("Initializing using dbt Cloud API...")

    # Load configuration
    postgres_uri = get_config_value("postgres_uri")
    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    headers = {
        "Authorization": f"Token {dbt_cloud_api_key}",
        "Content-Type": "application/json",
    }
    base_api_url = f"{dbt_cloud_url.rstrip('/')}/api/v2/accounts/{dbt_cloud_account_id}"

    manifest_content = None
    temp_manifest_path = None
    latest_run_id = None

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # 1. Find the latest successful run for the account with artifacts
            runs_task = progress.add_task(
                f"Fetching latest successful run for account {dbt_cloud_account_id}...",
                total=None,
            )
            runs_url = f"{base_api_url}/runs/"
            params = {
                "order_by": "-finished_at",
                "status": 10,
                "limit": 10,
            }
            try:
                response = requests.get(
                    runs_url, headers=headers, params=params, timeout=30
                )
                response.raise_for_status()
                runs_data = response.json()

                if not runs_data or not runs_data.get("data"):
                    logger.error(
                        f"No recent successful runs found for account ID {dbt_cloud_account_id}"
                    )
                    console.print(
                        f"[red]Error: No recent successful runs found for account ID {dbt_cloud_account_id}. Ensure at least one job/run has completed successfully in the account.[/red]"
                    )
                    sys.exit(1)

                # Iterate through runs to find one with a manifest
                for run in runs_data["data"]:
                    run_id = run.get("id")
                    if not run_id:
                        continue

                    logger.info(f"Checking run ID {run_id} for manifest artifact...")
                    # Use the correct artifact URL format (no project_id needed here)
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
                        elif artifact_response.status_code == 404:
                            logger.info(
                                f"Run ID {run_id} does not contain manifest.json, trying next..."
                            )
                        else:
                            logger.warning(
                                f"Unexpected status {artifact_response.status_code} when checking artifacts for run {run_id}"
                            )
                    except requests.exceptions.RequestException as e:
                        logger.warning(
                            f"Error checking artifacts for run {run_id}: {e}. Trying next run."
                        )

                if not latest_run_id:
                    logger.error(
                        f"Could not find manifest.json in the latest {params['limit']} successful runs for account {dbt_cloud_account_id}"
                    )
                    console.print(
                        f"[red]Error: Could not find a recent successful run containing manifest.json for account {dbt_cloud_account_id}.[/red]"
                    )
                    sys.exit(1)

                progress.update(runs_task, completed=True)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching dbt Cloud runs: {e}")
                console.print(f"[red]Error connecting to dbt Cloud API: {e}[/red]")
                sys.exit(1)
            except KeyError as e:
                logger.error(
                    f"Unexpected response structure from dbt Cloud API (runs): {e}"
                )
                console.print(
                    f"[red]Unexpected response from dbt Cloud API when fetching runs.[/red]"
                )
                sys.exit(1)

            # 2. Get the manifest.json artifact from the identified run
            artifact_task = progress.add_task(
                f"Fetching manifest.json from run {latest_run_id}...", total=None
            )
            # Use the corrected artifact URL again
            artifact_url = (
                f"{base_api_url}/runs/{latest_run_id}/artifacts/manifest.json"
            )
            try:
                response = requests.get(artifact_url, headers=headers, timeout=60)
                response.raise_for_status()
                manifest_content = response.json()
                logger.info("Successfully fetched manifest.json content.")

                # --- Add Debug Logging ---
                if verbose and manifest_content and "nodes" in manifest_content:
                    logger.debug("Sample model nodes from fetched manifest:")
                    count = 0
                    for node_id, node_data in manifest_content["nodes"].items():
                        if node_data.get("resource_type") == "model":
                            # Log relevant keys for debugging
                            log_data = {
                                "unique_id": node_data.get("unique_id"),
                                "name": node_data.get("name"),
                                "raw_sql_present": "raw_sql" in node_data,
                                "raw_code_present": "raw_code" in node_data,
                                "compiled_sql_present": "compiled_sql" in node_data,
                                "compiled_code_present": "compiled_code" in node_data,
                                "keys": list(
                                    node_data.keys()
                                ),  # Log all keys for inspection
                            }
                            logger.debug(
                                f"Node {node_id}: {json.dumps(log_data, indent=2)}"
                            )
                            count += 1
                            if count >= 3:  # Log first 3 models
                                break
                # --- End Debug Logging ---

                progress.update(artifact_task, completed=True)
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Error fetching manifest.json artifact from run {latest_run_id}: {e}"
                )
                console.print(
                    f"[red]Error fetching manifest.json from dbt Cloud (run {latest_run_id}): {e}[/red]"
                )
                sys.exit(1)
            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding manifest.json content from dbt Cloud: {e}"
                )
                console.print(
                    f"[red]Invalid JSON received for manifest.json from dbt Cloud.[/red]"
                )
                sys.exit(1)

            # 3. Save manifest to temporary file
            save_task = progress.add_task(
                "Preparing manifest for parsing...", total=None
            )
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".json"
                ) as temp_file:
                    json.dump(manifest_content, temp_file)
                    temp_manifest_path = temp_file.name
                logger.info(f"Saved manifest to temporary file: {temp_manifest_path}")
                progress.update(save_task, completed=True)
            except Exception as e:
                logger.error(f"Error saving manifest to temporary file: {e}")
                console.print(f"[red]Error saving temporary manifest file: {e}[/red]")
                sys.exit(1)

            # 4. Parse the manifest
            parse_task = progress.add_task("Parsing manifest...", total=None)
            parser = ManifestParser(manifest_path=temp_manifest_path)
            dbt_project = parser.parse_project()
            progress.update(parse_task, completed=True)

            # 5. Store models
            store_task = progress.add_task(
                f"Storing {len(dbt_project.models)} models in database...", total=None
            )
            model_storage = ModelStorage(postgres_uri)
            for model in dbt_project.models.values():
                metadata = parser.get_model_metadata(model)
                model_storage.store_model(metadata, force=force)
            progress.update(store_task, completed=True)

        console.print(
            f"[green]Successfully initialized with {len(dbt_project.models)} models using dbt Cloud (from run {latest_run_id}).[/green]"
        )
        console.print("Next steps:")
        console.print('1. Embed models for semantic search: dbt-llm embed --select "*"')
        console.print('2. Ask questions: dbt-llm ask "What models do we have?"')

    except Exception as e:
        logger.error(f"Error during dbt Cloud initialization: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        if temp_manifest_path and os.path.exists(temp_manifest_path):
            try:
                os.remove(temp_manifest_path)
                logger.info(f"Cleaned up temporary manifest file: {temp_manifest_path}")
            except OSError as e:
                logger.warning(
                    f"Could not remove temporary manifest file {temp_manifest_path}: {e}"
                )


@init.command()
@click.option(
    "--project-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    envvar="DBT_PROJECT_PATH",
    help="Path to your local dbt project directory.",
    required=True,
)
@click.option("--force", is_flag=True, help="Force reimport of models")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def local(project_path, force, verbose):
    """Initialize using local dbt project and 'dbt compile'."""
    set_logging_level(verbose)
    project_path_obj = Path(project_path).resolve()
    manifest_path = project_path_obj / "target" / "manifest.json"

    logger.info(f"Initializing using local dbt project at: {project_path_obj}")

    # Load configuration
    postgres_uri = get_config_value("postgres_uri")
    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        # Run dbt compile
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            compile_task = progress.add_task("Running 'dbt compile'...", total=None)
            logger.info(f"Running 'poetry run dbt compile' in {project_path_obj}")
            try:
                process = subprocess.run(
                    ["poetry", "run", "dbt", "compile"],
                    cwd=project_path_obj,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.debug(f"dbt compile stdout:\n{process.stdout}")
                logger.debug(f"dbt compile stderr:\n{process.stderr}")
                progress.update(compile_task, completed=True)
                console.print("[green]dbt compile completed successfully.[/green]")
            except subprocess.CalledProcessError as e:
                logger.error(f"'dbt compile' failed with exit code {e.returncode}")
                logger.error(f"Stdout:\n{e.stdout}")
                logger.error(f"Stderr:\n{e.stderr}")
                console.print(
                    "[red]Error running 'dbt compile'. Check logs for details.[/red]"
                )
                sys.exit(1)
            except FileNotFoundError:
                logger.error(
                    "Could not find 'poetry'. Ensure Poetry is installed and in your PATH."
                )
                console.print("[red]Error: 'poetry' command not found.[/red]")
                sys.exit(1)

        # Check if manifest.json was generated
        if not manifest_path.exists():
            logger.error(
                f"manifest.json not found at {manifest_path} after running 'dbt compile'"
            )
            console.print(
                f"[red]Error: Could not find manifest.json at expected location:[/red] {manifest_path}"
            )
            sys.exit(1)

        logger.info(f"Found manifest.json at: {manifest_path}")

        # Parse manifest and store models
        model_storage = ModelStorage(postgres_uri)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            parse_task = progress.add_task("Parsing manifest.json...", total=None)
            parser = ManifestParser(manifest_path=str(manifest_path))
            dbt_project = parser.parse_project()
            progress.update(parse_task, completed=True)

            store_task = progress.add_task(
                f"Storing {len(dbt_project.models)} models in database...", total=None
            )
            for model in dbt_project.models.values():
                metadata = parser.get_model_metadata(model)
                model_storage.store_model(metadata, force=force)
            progress.update(store_task, completed=True)

        console.print(
            f"[green]Successfully initialized with {len(dbt_project.models)} models using local dbt project.[/green]"
        )
        console.print("Next steps:")
        console.print('1. Embed models for semantic search: dbt-llm embed --select "*"')
        console.print('2. Ask questions: dbt-llm ask "What models do we have?"')

    except FileNotFoundError as e:
        logger.error(f"Initialization error: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing from local dbt project: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)


@init.command()
@click.argument(
    "project_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    required=False,
)
@click.option("--force", is_flag=True, help="Force reimport of models")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def source(project_path, force, verbose):
    """Initialize using local dbt project source code only (Fallback)."""
    set_logging_level(verbose)

    # Try to use DBT_PROJECT_PATH env var if path not provided
    if not project_path:
        project_path = get_config_value("dbt_project_path")

    # Try to use current directory if still not found
    if not project_path:
        if os.path.exists("dbt_project.yml"):
            project_path = "."
        else:
            logger.error("No dbt project path provided or found in current directory.")
            console.print("Please specify the path to your dbt project:")
            console.print("Example: dbt-llm init source /path/to/dbt/project")
            sys.exit(1)

    logger.info(f"Initializing using source code from: {project_path}")

    # Load configuration from environment
    postgres_uri = get_config_value("postgres_uri")
    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        # Initialize storage
        model_storage = ModelStorage(postgres_uri)

        # Create spinner
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            parse_task = progress.add_task(
                f"Parsing dbt project source code at {project_path}...", total=None
            )

            parser = SourceCodeParser(project_path=project_path)
            dbt_project = parser.parse_project()

            progress.update(parse_task, completed=True)

            store_task = progress.add_task(
                f"Storing {len(dbt_project.models)} models in database...", total=None
            )

            for model in dbt_project.models.values():
                metadata = parser.get_model_metadata(model)
                model_storage.store_model(metadata, force=force)

            progress.update(store_task, completed=True)

        console.print(
            f"[green]Successfully initialized with {len(dbt_project.models)} models using source code.[/green]"
        )
        # Suggest next steps
        console.print("Next steps:")
        console.print('1. Embed models for semantic search: dbt-llm embed --select "*"')
        console.print('2. Ask questions: dbt-llm ask "What models do we have?"')

    except (
        ValueError
    ) as e:  # Catch specific error from parser if dbt_project.yml is missing
        logger.error(f"Initialization error: {e}")
        console.print(f"[red]Error: {e}[/red]")
        console.print("Ensure the provided path contains a valid dbt project.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error initializing from source code: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
