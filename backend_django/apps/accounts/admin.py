import logging
from django.conf import settings
from django.contrib import admin, messages
from django.contrib.auth.admin import UserAdmin
from .models import User, Organisation

logger = logging.getLogger(__name__)

# Register your core models here, if any are created later.

# All admin registrations previously in this file have been moved:
# - ModelAdmin -> apps.knowledge_base.admin
# - QuestionAdmin, QuestionModelAdmin -> apps.agents.admin
# - ModelEmbeddingAdmin -> apps.embeddings.admin

# Register your models here.


class CustomUserAdmin(UserAdmin):
    # Add or override fieldsets if you want to customize the admin form for users
    # For example, to add the 'organisation' field:
    fieldsets = UserAdmin.fieldsets + (("Organisation", {"fields": ("organisation",)}),)
    add_fieldsets = UserAdmin.add_fieldsets + (
        ("Organisation", {"fields": ("organisation",)}),
    )
    list_display = (
        "username",
        "email",
        "first_name",
        "last_name",
        "is_staff",
        "organisation",
    )
    list_filter = UserAdmin.list_filter + ("organisation",)
    search_fields = UserAdmin.search_fields + ("organisation__name",)


admin.site.register(User, CustomUserAdmin)
admin.site.register(Organisation)
