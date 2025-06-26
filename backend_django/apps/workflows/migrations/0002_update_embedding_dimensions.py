# Generated migration to update embedding dimensions for text-embedding-3-large

import pgvector.django
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("workflows", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="question",
            name="question_embedding",
            field=pgvector.django.VectorField(
                blank=True,
                dimensions=3072,
                help_text="Embedding vector for the question text",
                null=True,
            ),
        ),
        migrations.AlterField(
            model_name="question",
            name="feedback_embedding",
            field=pgvector.django.VectorField(
                blank=True,
                dimensions=3072,
                help_text="Embedding vector for the feedback text",
                null=True,
            ),
        ),
        migrations.AlterField(
            model_name="question",
            name="original_message_embedding",
            field=pgvector.django.VectorField(
                blank=True,
                dimensions=3072,
                help_text="Embedding vector for the original message text",
                null=True,
            ),
        ),
    ]
