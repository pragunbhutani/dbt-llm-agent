# Generated migration to update embedding dimensions for text-embedding-3-large

import pgvector.django
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("embeddings", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="modelembedding",
            name="embedding",
            field=pgvector.django.VectorField(
                dimensions=3072,
                help_text="Embedding based on model documentation",
            ),
        ),
    ]
