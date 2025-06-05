import logging
from django.conf import settings
from django.contrib import admin, messages

logger = logging.getLogger(__name__)

# Register your core models here, if any are created later.

# All admin registrations previously in this file have been moved:
# - ModelAdmin -> apps.knowledge_base.admin
# - QuestionAdmin, QuestionModelAdmin -> apps.agents.admin
# - ModelEmbeddingAdmin -> apps.embeddings.admin
