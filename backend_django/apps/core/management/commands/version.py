import logging
import toml
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings  # To find BASE_DIR

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Displays the project version from pyproject.toml."

    # No arguments needed for this command

    def handle(self, *args, **options):
        try:
            pyproject_path = settings.BASE_DIR / "pyproject.toml"
            if not pyproject_path.exists():
                raise CommandError(f"pyproject.toml not found at: {pyproject_path}")

            data = toml.load(pyproject_path)

            # Try finding version under [tool.poetry] or [project]
            version = None
            if (
                "tool" in data
                and "poetry" in data["tool"]
                and "version" in data["tool"]["poetry"]
            ):
                version = data["tool"]["poetry"]["version"]
            elif "project" in data and "version" in data["project"]:
                version = data["project"]["version"]

            if version:
                self.stdout.write(version)
            else:
                raise CommandError(
                    "Version could not be found in pyproject.toml under [tool.poetry.version] or [project.version]."
                )

        except CommandError as e:
            raise e
        except Exception as e:
            logger.exception(f"Error reading version from pyproject.toml: {e}")
            raise CommandError(f"Could not read project version: {e}")
