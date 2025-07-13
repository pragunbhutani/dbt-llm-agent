"""
Management command to clean up expired OAuth authorization codes and requests.
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.integrations.models import (
    MCPOAuthAuthorizationCode,
    MCPOAuthAuthorizationRequest,
)


class Command(BaseCommand):
    help = "Clean up expired OAuth authorization codes and requests"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without actually deleting",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]

        if dry_run:
            self.stdout.write(
                self.style.WARNING("DRY RUN MODE - Nothing will be deleted")
            )

        # Clean up expired authorization codes
        expired_codes = MCPOAuthAuthorizationCode.objects.filter(
            expires_at__lt=timezone.now()
        )
        code_count = expired_codes.count()

        if dry_run:
            self.stdout.write(f"Would delete {code_count} expired authorization codes")
        else:
            expired_codes.delete()
            self.stdout.write(
                self.style.SUCCESS(f"Deleted {code_count} expired authorization codes")
            )

        # Clean up expired authorization requests
        expired_requests = MCPOAuthAuthorizationRequest.objects.filter(
            expires_at__lt=timezone.now()
        )
        request_count = expired_requests.count()

        if dry_run:
            self.stdout.write(
                f"Would delete {request_count} expired authorization requests"
            )
        else:
            expired_requests.delete()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Deleted {request_count} expired authorization requests"
                )
            )

        total_count = code_count + request_count
        if dry_run:
            self.stdout.write(f"Total items that would be cleaned up: {total_count}")
        else:
            self.stdout.write(f"Total items cleaned up: {total_count}")

        if total_count == 0:
            self.stdout.write("No expired items found to clean up")
