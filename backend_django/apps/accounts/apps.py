from django.apps import AppConfig


class AccountsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.accounts"

    def ready(self):
        # Import signal handlers to ensure they are registered when the app is
        # loaded. The import must be inside ready() to avoid side-effects
        # during Django's app loading and to prevent circular-import issues in
        # tests or management commands.
        from . import signals  # noqa: F401
