import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings


class Command(BaseCommand):
    help = "Display timezone configuration and current time"

    def handle(self, *args, **options):
        # Get Django timezone setting
        django_tz = settings.TIME_ZONE

        # Get current time in various formats
        system_time = datetime.datetime.now()
        django_time = timezone.now()

        self.stdout.write(self.style.SUCCESS(f"Django TIME_ZONE setting: {django_tz}"))
        self.stdout.write(self.style.SUCCESS(f"System local time: {system_time}"))
        self.stdout.write(
            self.style.SUCCESS(f"Django timezone-aware time: {django_time}")
        )

        # Show timezone info
        if hasattr(django_time.tzinfo, "zone"):
            self.stdout.write(
                self.style.SUCCESS(f"Django timezone zone: {django_time.tzinfo.zone}")
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"System timezone offset: {system_time.astimezone().strftime('%z %Z')}"
            )
        )
