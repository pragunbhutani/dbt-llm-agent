# Generated by Django 5.2 on 2025-06-12 05:11

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("data_sources", "0002_added_dbtproject_model"),
    ]

    operations = [
        migrations.AlterField(
            model_name="dbtproject",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True),
        ),
        migrations.AlterField(
            model_name="dbtproject",
            name="dbt_cloud_account_id",
            field=models.BigIntegerField(blank=True, null=True),
        ),
    ]
