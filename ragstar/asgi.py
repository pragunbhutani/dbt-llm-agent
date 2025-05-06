"""
ASGI config for ragstar_django project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os
from django.core.asgi import get_asgi_application
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ragstar.settings")

application = get_asgi_application()
