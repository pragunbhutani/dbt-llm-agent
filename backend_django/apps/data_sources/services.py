import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from django.db import transaction
from django.core.management.base import CommandError  # Keep for raising errors
import re
import yaml
import requests
import tempfile
import os

from apps.knowledge_base.models import Model  # Import Model from correct app

logger = logging.getLogger(__name__)


def initialize_dbt_cloud_project(
    dbt_project: Any,
    organisation: Any,
):
    """
    Initializes a project from dbt Cloud.

    Args:
        dbt_project: The DbtProject instance.
        organisation: The organisation to associate the models with.

    Returns:
        A dictionary containing counts of created, updated, and skipped models.
    """
    manifest_path = None
    try:
        manifest_path = _fetch_dbt_cloud_manifest(
            dbt_project.dbt_cloud_url,
            dbt_project.dbt_cloud_account_id,
            dbt_project.dbt_cloud_api_key,
        )
        results = import_from_manifest(
            manifest_path_str=manifest_path,
            organisation=organisation,
            dbt_project=dbt_project,
        )
        return results
    finally:
        if manifest_path and os.path.exists(manifest_path):
            try:
                os.remove(manifest_path)
            except OSError as e:
                logger.warning(
                    f"Could not remove temporary manifest file {manifest_path}: {e}"
                )


def import_from_github_repository(dbt_project: Any, organisation: Any):
    """
    Initializes a project from a GitHub repository.

    Args:
        dbt_project: The DbtProject instance.
        organisation: The organisation to associate the models with.
    """
    import subprocess
    from apps.integrations.models import OrganisationIntegration

    try:
        integration = OrganisationIntegration.objects.get(
            organisation=organisation, integration_key="github", is_enabled=True
        )
        access_token = integration.get_credential("access_token")
        if not access_token:
            raise CommandError("GitHub access token not found.")
    except OrganisationIntegration.DoesNotExist:
        raise CommandError("GitHub integration not found or not enabled.")

    repo_url = dbt_project.github_repository_url
    branch = dbt_project.github_branch

    # Inject token into URL for https authentication
    repo_url_with_auth = repo_url.replace("https://", f"https://oauth2:{access_token}@")

    from git import Repo  # GitPython

    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "repo"
        try:
            Repo.clone_from(
                repo_url_with_auth,
                repo_path,
                branch=branch,
                depth=1,
            )

            project_folder = dbt_project.github_project_folder or ""
            dbt_project_path = repo_path / project_folder

            try:
                # Try normal dbt workflow first
                subprocess.run(["dbt", "deps"], cwd=dbt_project_path, check=True)
                subprocess.run(["dbt", "compile"], cwd=dbt_project_path, check=True)

                manifest_path = dbt_project_path / "target" / "manifest.json"
                if manifest_path.exists():
                    return import_from_manifest(
                        manifest_path_str=str(manifest_path),
                        organisation=organisation,
                        dbt_project=dbt_project,
                    )
                logger.warning(
                    "manifest.json missing after compile – falling back to raw source parse"
                )
            except Exception as compile_err:
                logger.warning(
                    f"dbt compile failed ({compile_err}); falling back to raw source parse"
                )

            # Fallback: parse raw source files
            return import_from_source(
                project_path_str=str(dbt_project_path),
                organisation=organisation,
                dbt_project=dbt_project,
            )

        except Exception as e:
            # Git errors, dbt errors, etc.
            logger.error(f"Error importing from GitHub: {e}")
            raise CommandError(f"Error importing from GitHub: {e}")


def _fetch_dbt_cloud_manifest(
    dbt_cloud_url: str, dbt_cloud_account_id: int, dbt_cloud_api_key: str
) -> str:
    """Fetches the latest manifest.json from dbt Cloud."""
    headers = {
        "Authorization": f"Token {dbt_cloud_api_key}",
        "Content-Type": "application/json",
    }
    base_api_url = f"{dbt_cloud_url.rstrip('/')}/api/v2/accounts/{dbt_cloud_account_id}"
    latest_run_id = None

    runs_url = f"{base_api_url}/runs/"
    params = {
        "order_by": "-finished_at",
        "status": 10,
        "limit": 10,
    }
    response = requests.get(runs_url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    runs_data = response.json().get("data", [])

    if not runs_data:
        raise Exception(
            f"No recent successful runs found for account ID {dbt_cloud_account_id}."
        )

    for run in runs_data:
        run_id = run.get("id")
        if not run_id:
            continue
        artifact_url = f"{base_api_url}/runs/{run_id}/artifacts/manifest.json"
        try:
            artifact_response = requests.get(artifact_url, headers=headers, timeout=10)
            if artifact_response.status_code == 200:
                latest_run_id = run_id
                logger.info(f"Found manifest artifact in run ID: {latest_run_id}")
                break
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Error checking artifacts for run {run_id}: {e}. Trying next."
            )

    if not latest_run_id:
        raise Exception(
            "Could not find manifest.json in the latest 10 successful runs."
        )

    artifact_url = f"{base_api_url}/runs/{latest_run_id}/artifacts/manifest.json"
    response = requests.get(artifact_url, headers=headers, timeout=60)
    response.raise_for_status()
    manifest_content = response.json()

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json", encoding="utf-8"
    ) as temp_file:
        json.dump(manifest_content, temp_file)
        temp_manifest_path = temp_file.name

    return temp_manifest_path


# --- Service for Importing from Manifest ---
def import_from_manifest(
    manifest_path_str: str,
    organisation: Any,
    dbt_project: Any,
    clear_data: bool = False,
) -> Dict[str, int]:
    """Imports dbt models from a manifest file into the Django database.

    Args:
        manifest_path_str: Path to the manifest.json file.
        organisation: The organisation to associate the models with.
        dbt_project: The DbtProject to associate the models with.
        clear_data: If True, clears existing Model data first.

    Returns:
        A dictionary containing counts of created, updated, and skipped models.

    Raises:
        CommandError: If the file is not found, invalid, or other errors occur.
    """
    manifest_path = Path(manifest_path_str).resolve()
    if not manifest_path.exists() or not manifest_path.is_file():
        raise CommandError(f"Manifest file not found at {manifest_path}")

    logger.info(f"Loading manifest from: {manifest_path}")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except json.JSONDecodeError:
        raise CommandError(f"Invalid JSON in manifest file: {manifest_path}")
    except Exception as e:
        raise CommandError(f"Error loading manifest file: {e}")

    nodes = manifest_data.get("nodes", {})
    unique_id_to_name = {}
    parsed_models = {}

    # --- Step 1: Initial Parse and Map Building ---
    logger.info("Parsing manifest nodes...")
    for unique_id, node_data in nodes.items():
        if node_data.get("resource_type") == "model":
            model_name = node_data.get("name")
            if not model_name:
                logger.warning(f"Skipping node {unique_id} - missing name.")
                continue

            unique_id_to_name[unique_id] = model_name
            parsed_models[unique_id] = {
                "raw_node": node_data,
                "data_for_db": {
                    "name": model_name,
                    "path": node_data.get("path"),
                    "schema_name": node_data.get("schema"),
                    "database": node_data.get("database"),
                    "materialization": node_data.get("config", {}).get("materialized"),
                    "tags": node_data.get("tags"),
                    "depends_on": [],
                    "tests": node_data.get("tests"),
                    "all_upstream_models": [],
                    "meta": node_data.get("meta"),
                    "raw_sql": node_data.get("raw_code") or node_data.get("raw_sql"),
                    "compiled_sql": node_data.get("compiled_code")
                    or node_data.get("compiled_sql"),
                    "yml_description": node_data.get("description"),
                    "yml_columns": node_data.get("columns"),
                    "interpreted_columns": None,
                    "interpreted_description": None,
                    "interpretation_details": None,
                    "unique_id": unique_id,
                    "dbt_project": dbt_project,
                },
            }
    logger.info(f"Found {len(parsed_models)} models in manifest.")

    # --- Step 2: Refine depends_on ---
    logger.info("Resolving direct dependencies...")
    for unique_id, model_info in parsed_models.items():
        raw_deps = model_info["raw_node"].get("depends_on", {}).get("nodes", [])
        resolved_deps = []
        for dep_id in raw_deps:
            if dep_id.startswith("model."):
                dep_name = unique_id_to_name.get(dep_id)
                if dep_name:
                    resolved_deps.append(dep_name)
                else:
                    logger.warning(
                        f"Could not resolve model dependency ID: {dep_id} for model {model_info['data_for_db']['name']}"
                    )
        model_info["data_for_db"]["depends_on"] = list(set(resolved_deps))

    # --- Step 3: Calculate all_upstream_models ---
    logger.info("Calculating upstream dependency chains...")
    memo = {}

    def find_all_upstream(start_unique_id):
        if start_unique_id in memo:
            return memo[start_unique_id]
        upstream_names = set()
        visited_ids = set()

        def traverse(current_unique_id):
            if (
                current_unique_id in visited_ids
                or current_unique_id not in parsed_models
            ):
                return
            visited_ids.add(current_unique_id)
            model_info = parsed_models[current_unique_id]
            direct_dep_names = model_info["data_for_db"]["depends_on"]
            for dep_name in direct_dep_names:
                dep_unique_id = next(
                    (
                        uid
                        for uid, name in unique_id_to_name.items()
                        if name == dep_name
                    ),
                    None,
                )
                if dep_unique_id:
                    upstream_names.add(dep_name)
                    traverse(dep_unique_id)

        traverse(start_unique_id)
        memo[start_unique_id] = list(upstream_names)
        return memo[start_unique_id]

    for unique_id, model_info in parsed_models.items():
        model_info["data_for_db"]["all_upstream_models"] = find_all_upstream(unique_id)

    # --- Step 4: Save to Database ---
    models_created = 0
    models_updated = 0
    models_skipped = 0

    with transaction.atomic():  # Ensure all or nothing
        if clear_data:
            logger.warning(
                "Clearing existing Model data for organisation %s...", organisation.id
            )
            Model.objects.for_organisation(organisation).delete()

        existing_models = {
            m.unique_id: m
            for m in Model.objects.for_organisation(organisation).filter(
                unique_id__in=parsed_models.keys()
            )
        }

        models_to_create = []
        models_to_update = []

        for unique_id, model_info in parsed_models.items():
            db_data = model_info["data_for_db"]
            if unique_id in existing_models:
                # Update existing model
                model_instance = existing_models[unique_id]
                for key, value in db_data.items():
                    setattr(model_instance, key, value)
                models_to_update.append(model_instance)
            else:
                # Create new model
                models_to_create.append(Model(**db_data, organisation=organisation))

        if models_to_create:
            Model.objects.bulk_create(models_to_create)
            models_created = len(models_to_create)
            logger.info(f"Created {models_created} new models.")

        if models_to_update:
            # Note: bulk_update doesn't call save() or send signals.
            # This is fine for this use case as we are just syncing data.
            update_fields = list(
                parsed_models[next(iter(parsed_models))]["data_for_db"].keys()
            )
            Model.objects.bulk_update(models_to_update, update_fields)
            models_updated = len(models_to_update)
            logger.info(f"Updated {models_updated} existing models.")

        # This logic can be simplified as we are creating/updating all parsed models
        models_skipped = 0

    return {
        "created": models_created,
        "updated": models_updated,
        "skipped": models_skipped,
    }


# --- Service for Importing from Source Code ---
def import_from_source(
    project_path_str: str,
    organisation: Any,
    dbt_project: Any | None = None,
) -> Dict[str, int]:
    """
    Parses a dbt project from a local file path and stores the models
    in the database, associating them with the given organisation.

    Args:
        project_path_str: The absolute path to the dbt project directory.
        organisation: The Organisation instance to associate models with.

    Returns:
        A dictionary with counts of created and updated models.
    """
    project_path = Path(project_path_str).resolve()
    models_path = project_path / "models"
    dbt_project_file = project_path / "dbt_project.yml"

    if not dbt_project_file.exists():
        raise CommandError(f"dbt_project.yml not found in {project_path}.")
    if not models_path.is_dir():
        raise CommandError(f"Models directory not found at {models_path}.")

    logger.info(f"Parsing dbt project source at: {project_path}")
    try:
        with open(dbt_project_file, "r", encoding="utf-8") as f:
            project_config = yaml.safe_load(f)
        project_name = project_config.get("name", "unknown_project")
    except Exception as e:
        raise CommandError(f"Error loading dbt_project.yml: {e}")

    # --- Step 1: Parse YAML files ---
    parsed_models = _parse_yaml_files(models_path, project_name)
    logger.info(f"Parsed {len(parsed_models)} models/definitions from YAML files.")

    # --- Step 2: Parse SQL files ---
    parsed_models = _parse_sql_files(
        models_path, project_path, project_name, parsed_models
    )
    logger.info(f"Processed SQL files. Total models found: {len(parsed_models)}.")

    # --- Step 3: Calculate all_upstream_models ---
    logger.info("Calculating upstream dependencies...")
    _calculate_all_upstream(parsed_models)

    # --- Step 4: Save to Database ---
    models_created = 0
    models_updated = 0
    models_skipped = 0

    with transaction.atomic():
        logger.info("Storing models from source parsing in database...")
        for model_name, model_data in parsed_models.items():
            model_data["organisation"] = organisation  # Set organisation
            if dbt_project is not None:
                model_data["dbt_project"] = dbt_project  # Associate with project
            try:
                lookup_kwargs = {
                    "name": model_name,
                    "organisation": organisation,
                }
                if dbt_project is not None:
                    lookup_kwargs["dbt_project"] = dbt_project

                _, created = Model.objects.update_or_create(
                    **lookup_kwargs, defaults=model_data
                )
                if created:
                    models_created += 1
                else:
                    models_updated += 1
            except Exception as e:
                logger.error(
                    f"Error saving model {model_name} from source: {e}", exc_info=True
                )
                models_skipped += 1

    logger.info(
        f"Source import complete. Created: {models_created}, Updated: {models_updated}, Skipped: {models_skipped}"
    )
    return {
        "created": models_created,
        "updated": models_updated,
        "skipped": models_skipped,
        "total": models_created + models_updated + models_skipped,
    }


# --- Helper Parsing Methods (Internal to this service module) ---

# Helper Regex patterns (adapted from SourceCodeParser)
RE_REF = re.compile(r'{{\s*ref\s*\(\s*[\'"]([^\'"]*)[\'"]\s*\)\s*}}', re.IGNORECASE)
RE_REF_MULTI = re.compile(
    r'{{\s*ref\s*\(\s*[\'"]([^\'"]*)[\'"]\s*,\s*[\'"][^\'"]*[\'"]\s*\)\s*}}',
    re.IGNORECASE,
)
RE_CONFIG_MATERIALIZED = re.compile(
    r'{{\s*config\s*\(\s*materialized\s*=\s*[\'"](\w+)[\'"]', re.IGNORECASE
)
RE_COMMENT_DESC = re.compile(r"--\s*description:\s*(.*?)(?:\n|$)")
RE_COMMENT_COL = re.compile(r"--\s*column:\s*(\w+)\s*description:\s*(.*?)(?:\n|$)")


def _parse_yaml_files(models_path: Path, project_name: str) -> Dict[str, Dict]:
    # (Implementation from parse_source.py command - pasted below)
    parsed_models = {}
    yaml_files = list(models_path.glob("**/*.yml"))
    yaml_files.extend(models_path.glob("**/*.yaml"))
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
            if (
                not yaml_content
                or "models" not in yaml_content
                or not isinstance(yaml_content["models"], list)
            ):
                continue
            for model_def in yaml_content["models"]:
                if not isinstance(model_def, dict):
                    continue
                model_name = model_def.get("name")
                if not model_name:
                    continue
                columns_dict = {}
                if isinstance(model_def.get("columns"), list):
                    for col_def in model_def["columns"]:
                        if isinstance(col_def, dict) and col_def.get("name"):
                            col_name = col_def["name"]
                            columns_dict[col_name] = {
                                "name": col_name,
                                "description": col_def.get("description", ""),
                                "data_type": col_def.get("data_type"),
                                "meta": col_def.get("meta"),
                            }
                tests_list = []
                if isinstance(model_def.get("tests"), list):
                    tests_list.extend(model_def["tests"])
                if isinstance(model_def.get("columns"), list):
                    for col_def in model_def["columns"]:
                        if (
                            isinstance(col_def, dict)
                            and col_def.get("name")
                            and col_def.get("tests")
                        ):
                            tests_list.append(
                                {"column": col_def["name"], "tests": col_def["tests"]}
                            )
                parsed_models[model_name] = {
                    "name": model_name,
                    "yml_description": model_def.get("description", ""),
                    "yml_columns": columns_dict,
                    "tests": tests_list,
                    "tags": model_def.get("tags"),
                    "meta": model_def.get("meta"),
                    "path": None,
                    "schema_name": None,
                    "database": None,
                    "materialization": None,
                    "depends_on": [],
                    "all_upstream_models": [],
                    "raw_sql": None,
                    "compiled_sql": None,
                    "interpreted_columns": None,
                    "interpreted_description": None,
                    "interpretation_details": None,
                    "unique_id": None,
                }
        except Exception as e:
            logger.error(f"Error parsing YAML file {yaml_file}: {e}", exc_info=True)
    return parsed_models


def _parse_sql_files(
    models_path: Path, project_path: Path, project_name: str, parsed_models: Dict
) -> Dict:
    # (Implementation from parse_source.py command - pasted below)
    sql_files = list(models_path.glob("**/*.sql"))
    for sql_file in sql_files:
        model_name = sql_file.stem
        relative_path = str(sql_file.relative_to(project_path))
        try:
            with open(sql_file, "r", encoding="utf-8") as f:
                sql_content = f.read()
            model_data = parsed_models.get(
                model_name,
                {"name": model_name, "depends_on": [], "all_upstream_models": []},
            )
            model_data["path"] = relative_path
            model_data["raw_sql"] = sql_content
            if not model_data.get("yml_description"):
                model_data["yml_description"] = _extract_description_from_sql(
                    sql_content
                )
            if not model_data.get("materialization"):
                model_data["materialization"] = _extract_materialization_from_sql(
                    sql_content
                )
            sql_deps = _extract_dependencies_from_sql(sql_content)
            model_data["depends_on"] = list(
                set(model_data.get("depends_on", []) + sql_deps)
            )
            if not model_data.get("yml_columns"):
                model_data["yml_columns"] = _extract_columns_from_sql(sql_content)
            parsed_models[model_name] = model_data
        except Exception as e:
            logger.error(f"Error reading SQL file {sql_file}: {e}", exc_info=True)
    return parsed_models


def _extract_materialization_from_sql(sql_content: str) -> Optional[str]:
    match = RE_CONFIG_MATERIALIZED.search(sql_content)
    return match.group(1) if match else None


def _extract_description_from_sql(sql_content: str) -> str:
    match = RE_COMMENT_DESC.search(sql_content)
    return match.group(1).strip() if match else ""


def _extract_dependencies_from_sql(sql_content: str) -> List[str]:
    deps = set(RE_REF.findall(sql_content))
    deps.update(RE_REF_MULTI.findall(sql_content))
    return [d for d in deps if d]


def _extract_columns_from_sql(sql_content: str) -> Dict:
    columns = {}
    matches = RE_COMMENT_COL.findall(sql_content)
    for name, desc in matches:
        columns[name] = {"name": name, "description": desc.strip()}
    return columns


def _calculate_all_upstream(parsed_models: Dict):
    # (Implementation from parse_source.py command - pasted below)
    memo = {}
    model_names = set(parsed_models.keys())

    def find_upstream_recursive(model_name: str) -> Set[str]:
        if model_name in memo:
            return memo[model_name]
        if model_name not in parsed_models:
            return set()
        visited_during_recursion = {model_name}
        all_deps = set()
        direct_deps = parsed_models[model_name].get("depends_on", [])
        for dep_name in direct_deps:
            if dep_name not in model_names:
                continue
            if dep_name in visited_during_recursion:
                logger.warning(
                    f"Circular dependency detected involving {model_name} and {dep_name}. Breaking cycle."
                )
                continue
            all_deps.add(dep_name)
            upstream_of_dep = find_upstream_recursive(dep_name)
            all_deps.update(upstream_of_dep)
            # Remove the dependency name from the current recursion path after exploring its upstreams
            visited_during_recursion.discard(dep_name)
        memo[model_name] = all_deps
        return all_deps

    for model_name in parsed_models.keys():
        parsed_models[model_name]["all_upstream_models"] = list(
            find_upstream_recursive(model_name)
        )
