from django.core.management.base import BaseCommand
from django.conf import settings
from apps.accounts.services import secret_manager
import uuid


class Command(BaseCommand):
    help = "Test the secret management system with environment-prefixed paths"

    def add_arguments(self, parser):
        parser.add_argument(
            "--cleanup",
            action="store_true",
            help="Clean up test secrets after testing",
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS(
                f"Testing secret management in {settings.ENVIRONMENT} environment..."
            )
        )

        # Generate a test secret path
        test_org_id = str(uuid.uuid4())
        test_secret_name = (
            f"/ragstar/{settings.ENVIRONMENT}/org-{test_org_id}/test/api-key"
        )
        test_secret_value = f"test-secret-value-{uuid.uuid4()}"

        try:
            # Test 1: Store a secret
            self.stdout.write("1. Testing secret storage...")
            success = secret_manager.put_secret(
                test_secret_name,
                test_secret_value,
                description="Test secret for secret management validation",
            )

            if success:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"   ✓ Successfully stored secret: {test_secret_name}"
                    )
                )
            else:
                self.stdout.write(self.style.ERROR("   ✗ Failed to store secret"))
                return

            # Test 2: Retrieve the secret
            self.stdout.write("2. Testing secret retrieval...")
            retrieved_value = secret_manager.get_secret(test_secret_name)

            if retrieved_value == test_secret_value:
                self.stdout.write(
                    self.style.SUCCESS(f"   ✓ Successfully retrieved secret")
                )
            else:
                self.stdout.write(
                    self.style.ERROR(
                        f"   ✗ Retrieved value doesn't match. Expected: {test_secret_value}, Got: {retrieved_value}"
                    )
                )
                return

            # Test 3: Update the secret
            self.stdout.write("3. Testing secret update...")
            updated_value = f"updated-{test_secret_value}"
            success = secret_manager.put_secret(
                test_secret_name, updated_value, description="Updated test secret"
            )

            if success:
                retrieved_updated = secret_manager.get_secret(test_secret_name)
                if retrieved_updated == updated_value:
                    self.stdout.write(
                        self.style.SUCCESS(
                            "   ✓ Successfully updated and retrieved secret"
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.ERROR("   ✗ Updated value doesn't match")
                    )
                    return
            else:
                self.stdout.write(self.style.ERROR("   ✗ Failed to update secret"))
                return

            # Test 4: Clean up (delete the secret)
            if options["cleanup"]:
                self.stdout.write("4. Testing secret deletion...")
                success = secret_manager.delete_secret(test_secret_name)

                if success:
                    # Verify it's deleted
                    deleted_value = secret_manager.get_secret(test_secret_name)
                    if deleted_value is None:
                        self.stdout.write(
                            self.style.SUCCESS("   ✓ Successfully deleted secret")
                        )
                    else:
                        self.stdout.write(
                            self.style.WARNING(
                                "   ⚠ Secret still exists after deletion"
                            )
                        )
                else:
                    self.stdout.write(self.style.ERROR("   ✗ Failed to delete secret"))
            else:
                self.stdout.write(
                    "4. Skipping cleanup (use --cleanup to delete test secret)"
                )
                self.stdout.write(f"   Test secret path: {test_secret_name}")

            self.stdout.write(
                self.style.SUCCESS(
                    "\n✅ Secret management system test completed successfully!"
                )
            )
            self.stdout.write(f"Environment: {settings.ENVIRONMENT}")
            self.stdout.write(f"LocalStack mode: {settings.IS_LOCAL}")

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Secret management test failed: {str(e)}")
            )
            raise
