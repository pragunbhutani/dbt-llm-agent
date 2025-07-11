# Generated by Django 5.2 on 2025-06-30 13:35

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0011_convert_secrets_to_parameter_paths"),
    ]

    operations = [
        migrations.AlterField(
            model_name="organisationsettings",
            name="llm_anthropic_api_key_path",
            field=models.CharField(
                blank=True,
                help_text="Path to Anthropic API key in Parameter Store, e.g., '/ragstar/{environment}/org-{org_id}/llm/anthropic-api-key'",
                max_length=255,
                null=True,
            ),
        ),
        migrations.AlterField(
            model_name="organisationsettings",
            name="llm_google_api_key_path",
            field=models.CharField(
                blank=True,
                help_text="Path to Google API key in Parameter Store, e.g., '/ragstar/{environment}/org-{org_id}/llm/google-api-key'",
                max_length=255,
                null=True,
            ),
        ),
        migrations.AlterField(
            model_name="organisationsettings",
            name="llm_openai_api_key_path",
            field=models.CharField(
                blank=True,
                help_text="Path to OpenAI API key in Parameter Store, e.g., '/ragstar/{environment}/org-{org_id}/llm/openai-api-key'",
                max_length=255,
                null=True,
            ),
        ),
    ]
