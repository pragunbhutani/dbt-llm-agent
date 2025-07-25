# Generated by Django 5.2 on 2025-06-25 07:48

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0010_fix_model_uniqueness_constraints"),
        ("data_sources", "0003_changed_account_id_to_bigint"),
        ("knowledge_base", "0003_model_dbt_project"),
    ]

    operations = [
        migrations.AlterField(
            model_name="model",
            name="name",
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name="model",
            name="unique_id",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddConstraint(
            model_name="model",
            constraint=models.UniqueConstraint(
                fields=("organisation", "dbt_project", "name"),
                name="unique_model_per_org_project",
            ),
        ),
        migrations.AddConstraint(
            model_name="model",
            constraint=models.UniqueConstraint(
                condition=models.Q(("unique_id__isnull", False)),
                fields=("organisation", "unique_id"),
                name="unique_model_id_per_org",
            ),
        ),
    ]
