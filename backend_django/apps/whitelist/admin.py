from django.contrib import admin
from .models import SignupWhitelist


@admin.register(SignupWhitelist)
class SignupWhitelistAdmin(admin.ModelAdmin):
    list_display = ("email",)
    search_fields = ("email",)
