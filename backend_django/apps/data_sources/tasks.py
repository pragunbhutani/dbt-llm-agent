from celery import shared_task
from .services import import_from_github_repository
from .models import DbtProject
import logging

logger = logging.getLogger(__name__)


@shared_task
def parse_github_project_task(project_id: int):
    """
    Celery task to parse a dbt project from a GitHub repository.
    """
    try:
        project = DbtProject.objects.get(id=project_id)
        logger.info(f"Starting GitHub project parsing for project: {project.name}")
        import_from_github_repository(project, project.organisation)
        logger.info(f"Successfully parsed GitHub project: {project.name}")
    except DbtProject.DoesNotExist:
        logger.error(f"DbtProject with id {project_id} not found.")
    except Exception as e:
        logger.error(f"Failed to parse GitHub project {project_id}: {e}", exc_info=True)
