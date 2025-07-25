# Generated by Django 5.2 on 2025-06-25 08:48

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0010_fix_model_uniqueness_constraints"),
        ("workflows", "0005_add_actor_message_type_fields"),
    ]

    operations = [
        migrations.AddIndex(
            model_name="conversation",
            index=models.Index(
                fields=["organisation", "external_id"],
                name="conversatio_organis_390fd3_idx",
            ),
        ),
        migrations.AddConstraint(
            model_name="conversation",
            constraint=models.UniqueConstraint(
                condition=models.Q(("external_id__isnull", False)),
                fields=("organisation", "external_id"),
                name="unique_conversation_per_external_id",
            ),
        ),
    ]
