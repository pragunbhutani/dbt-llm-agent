from django.contrib import admin
from .models import (
    OrganisationIntegration,
    MCPOAuthClient,
    MCPOAuthAuthorizationCode,
    MCPOAuthAuthorizationRequest,
)


@admin.register(OrganisationIntegration)
class OrganisationIntegrationAdmin(admin.ModelAdmin):
    list_display = (
        "organisation",
        "integration_key",
        "is_enabled",
        "connection_status",
        "created_at",
    )
    list_filter = ("integration_key", "is_enabled", "created_at")
    search_fields = ("organisation__name", "integration_key")
    readonly_fields = ("created_at", "updated_at", "credentials_path")
    ordering = ("-created_at",)

    fieldsets = (
        (None, {"fields": ("organisation", "integration_key", "is_enabled")}),
        (
            "Configuration",
            {
                "fields": ("configuration",),
                "classes": ("collapse",),
            },
        ),
        (
            "Credentials",
            {
                "fields": ("credentials_path",),
                "classes": ("collapse",),
            },
        ),
        (
            "Status",
            {
                "fields": ("last_test_result", "last_tested_at"),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )


@admin.register(MCPOAuthClient)
class MCPOAuthClientAdmin(admin.ModelAdmin):
    list_display = (
        "client_id",
        "client_name",
        "auto_registered",
        "organisation",
        "created_at",
    )
    list_filter = ("auto_registered", "created_at")
    search_fields = ("client_id", "client_name", "organisation__name")
    readonly_fields = ("created_at", "updated_at")
    ordering = ("-created_at",)

    fieldsets = (
        (None, {"fields": ("client_id", "client_name", "organisation")}),
        (
            "OAuth Configuration",
            {
                "fields": ("redirect_uris", "grant_types", "response_types", "scope"),
            },
        ),
        (
            "Security",
            {
                "fields": ("client_secret", "auto_registered"),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("collapse",),
            },
        ),
    )

    def get_readonly_fields(self, request, obj=None):
        readonly = list(self.readonly_fields)
        if obj:  # Editing existing object
            readonly.extend(["client_id"])  # Don't allow changing client_id
        return readonly


@admin.register(MCPOAuthAuthorizationCode)
class MCPOAuthAuthorizationCodeAdmin(admin.ModelAdmin):
    list_display = ("code_short", "client", "expires_at", "is_expired", "created_at")
    list_filter = ("created_at", "expires_at")
    search_fields = ("code", "client__client_name", "client__client_id")
    readonly_fields = ("code", "created_at", "is_expired")
    ordering = ("-created_at",)

    def code_short(self, obj):
        return f"{obj.code[:10]}..." if obj.code else ""

    code_short.short_description = "Code"

    fieldsets = (
        (None, {"fields": ("code", "client", "expires_at")}),
        (
            "PKCE",
            {
                "fields": ("code_challenge", "redirect_uri"),
                "classes": ("collapse",),
            },
        ),
        (
            "User Data",
            {
                "fields": ("user_data", "scopes"),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "is_expired"),
                "classes": ("collapse",),
            },
        ),
    )

    actions = ["delete_expired_codes"]

    def delete_expired_codes(self, request, queryset):
        expired_count = 0
        for code in queryset:
            if code.is_expired:
                code.delete()
                expired_count += 1

        self.message_user(
            request,
            f"Successfully deleted {expired_count} expired authorization codes.",
        )

    delete_expired_codes.short_description = "Delete expired authorization codes"


@admin.register(MCPOAuthAuthorizationRequest)
class MCPOAuthAuthorizationRequestAdmin(admin.ModelAdmin):
    list_display = (
        "request_id_short",
        "client_id",
        "expires_at",
        "is_expired",
        "created_at",
    )
    list_filter = ("created_at", "expires_at")
    search_fields = ("request_id", "client_id")
    readonly_fields = ("request_id", "created_at", "is_expired")
    ordering = ("-created_at",)

    def request_id_short(self, obj):
        return f"{obj.request_id[:10]}..." if obj.request_id else ""

    request_id_short.short_description = "Request ID"

    fieldsets = (
        (None, {"fields": ("request_id", "client_id", "expires_at")}),
        (
            "OAuth Parameters",
            {
                "fields": ("redirect_uri", "response_type", "scope", "state"),
            },
        ),
        (
            "PKCE",
            {
                "fields": ("code_challenge", "code_challenge_method"),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "is_expired"),
                "classes": ("collapse",),
            },
        ),
    )

    actions = ["delete_expired_requests"]

    def delete_expired_requests(self, request, queryset):
        expired_count = 0
        for auth_request in queryset:
            if auth_request.is_expired:
                auth_request.delete()
                expired_count += 1

        self.message_user(
            request,
            f"Successfully deleted {expired_count} expired authorization requests.",
        )

    delete_expired_requests.short_description = "Delete expired authorization requests"
