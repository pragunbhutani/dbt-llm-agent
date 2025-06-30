"""Move dbt Cloud API key storage from plaintext to Parameter Store.

This migration adds 'credentials_path' to DbtProject and removes the
now-redundant 'dbt_cloud_api_key' field that previously stored secrets in
plaintext.
"""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("data_sources", "0003_changed_account_id_to_bigint"),
    ]

    operations = [
        # Add the new credentials_path column (nullable for back-compat).
        migrations.AddField(
            model_name="dbtproject",
            name="credentials_path",
            field=models.CharField(
                max_length=255,
                blank=True,
                null=True,
                help_text=(
                    "Path to credentials JSON in Parameter Store, e.g. "
                    "'/ragstar/{env}/org-{org_id}/dbt-projects/{project_id}/credentials'"
                ),
            ),
        ),
        # Remove the plaintext API-key column.
        migrations.RemoveField(
            model_name="dbtproject",
            name="dbt_cloud_api_key",
        ),
    ]
