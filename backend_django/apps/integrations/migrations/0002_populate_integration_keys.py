# Generated migration for populating integration keys

from django.db import migrations


def populate_integration_keys(apps, schema_editor):
    """
    Populate integration_key field from existing integration foreign key relationships.
    """
    OrganisationIntegration = apps.get_model("integrations", "OrganisationIntegration")
    Integration = apps.get_model("integrations", "Integration")

    # Update all existing OrganisationIntegration records
    for org_integration in OrganisationIntegration.objects.all():
        if hasattr(org_integration, "integration") and org_integration.integration:
            org_integration.integration_key = org_integration.integration.key
            org_integration.save()


def reverse_populate_integration_keys(apps, schema_editor):
    """
    Reverse operation - this would be complex since we'd need to recreate
    the Integration records, so we'll just clear the integration_key field.
    """
    OrganisationIntegration = apps.get_model("integrations", "OrganisationIntegration")
    OrganisationIntegration.objects.update(integration_key="unknown")


class Migration(migrations.Migration):

    dependencies = [
        ("integrations", "0001_initial"),  # Update this to your latest migration
    ]

    operations = [
        migrations.RunPython(
            populate_integration_keys,
            reverse_populate_integration_keys,
        ),
    ]
