# Generated by Django 5.2 on 2025-06-05 13:19

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0001_initial"),
        ("workflows", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="question",
            name="organisation",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="%(app_label)s_%(class)s_org_related",
                to="accounts.organisation",
            ),
        ),
    ]
