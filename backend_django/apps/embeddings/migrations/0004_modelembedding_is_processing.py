# Generated by Django 5.2 on 2025-06-16 13:23

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("embeddings", "0003_remove_modelembedding_model_name_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="modelembedding",
            name="is_processing",
            field=models.BooleanField(
                default=False,
                help_text="Whether this model is currently being processed/trained",
            ),
        ),
    ]
