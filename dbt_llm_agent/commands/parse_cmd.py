"""
Command to parse dbt project files and import models.
"""

import click
import sys
import os
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Dict, List, Optional

from dbt_llm_agent.utils.logging import get_logger
from dbt_llm_agent.utils.cli_utils import get_config_value, set_logging_level

# Initialize logger
logger = get_logger(__name__)

# Initialize console for rich output
console = Console()


@click.command()
@click.argument("project_path", type=click.Path(exists=True), required=False)
@click.option(
    "--manifest", type=click.Path(exists=True), help="Path to dbt manifest.json file"
)
@click.option("--force", is_flag=True, help="Force reimport of models")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def parse(project_path, manifest, force, verbose):
    """Parse a dbt project and import models.

    This command parses dbt project files to extract model metadata and SQL,
    then imports the models into the database.

    You can specify either a dbt project directory or a manifest file.

    Examples:
        dbt-llm parse /path/to/dbt/project
        dbt-llm parse --manifest /path/to/manifest.json
        dbt-llm parse /path/to/dbt/project --force
    """
    set_logging_level(verbose)

    # Check if we have a project path or manifest path
    if not project_path and not manifest:
        # Try to use current directory as project path if not specified
        if os.path.exists("dbt_project.yml"):
            project_path = "."
        else:
            logger.error("No project path or manifest file provided")
            console.print("Please specify either a project path or manifest file.")
            console.print("Example: dbt-llm parse /path/to/dbt/project")
            console.print("Example: dbt-llm parse --manifest /path/to/manifest.json")
            sys.exit(1)

    # Load configuration from environment
    postgres_uri = get_config_value("postgres_uri")

    if not postgres_uri:
        logger.error("PostgreSQL URI not provided in environment variables (.env file)")
        sys.exit(1)

    try:
        # Import necessary modules
        from dbt_llm_agent.storage.model_storage import ModelStorage
        from dbt_llm_agent.parsers.dbt_manifest_parser import DBTManifestParser

        # Initialize storage
        model_storage = ModelStorage(postgres_uri)

        # Create spinner for long-running operations
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Parse the project
            parse_task = progress.add_task("Parsing dbt project...", total=None)

            if manifest:
                logger.info(f"Loading manifest from {manifest}")
                parser = DBTManifestParser(manifest_path=manifest)
            else:
                logger.info(f"Parsing project at {project_path}")
                parser = DBTManifestParser(project_path=project_path)

            # Run the parser
            parser.parse()
            progress.update(parse_task, completed=True)

            # Store models in database
            store_task = progress.add_task(
                f"Storing {len(parser.models)} models in database...", total=None
            )

            for model_name, model in parser.models.items():
                model_storage.store_model(model, force=force)

            progress.update(store_task, completed=True)

        # Display success message
        console.print(
            f"[green]Successfully imported {len(parser.models)} models[/green]"
        )
        console.print("You can now:")
        console.print("1. List models: dbt-llm list")
        console.print("2. Get model details: dbt-llm model-details [model_name]")
        console.print('3. Embed models for semantic search: dbt-llm embed --select "*"')
        console.print('4. Ask questions: dbt-llm ask "What models do we have?"')

    except Exception as e:
        logger.error(f"Error parsing project: {str(e)}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        sys.exit(1)
