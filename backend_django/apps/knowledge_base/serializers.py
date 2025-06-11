from rest_framework import serializers

# Import from local models
from .models import Model
from apps.accounts.serializers import (
    OrganisationSerializer,
)  # Import if you want to nest it


class ModelSerializer(serializers.ModelSerializer):
    # To make organisation readable in responses, but not writable directly by API client
    organisation = OrganisationSerializer(read_only=True)

    class Meta:
        model = Model
        # Explicitly list fields. Add 'organisation' here if you want it in output.
        # If 'organisation' is in read_only_fields, it doesn't need to be in fields list for output.
        fields = [
            "id",
            "name",
            "path",
            "schema_name",
            "database",
            "materialization",
            "tags",
            "depends_on",
            "tests",
            "all_upstream_models",
            "meta",
            "raw_sql",
            "compiled_sql",
            "created_at",
            "updated_at",
            "yml_description",
            "yml_columns",
            "interpreted_columns",
            "interpreted_description",
            "interpretation_details",
            "unique_id",
            "organisation",  # Include organisation in the output fields
        ]
        read_only_fields = ["id", "created_at", "updated_at", "organisation"]
        # Alternatively, if you don't want to expose organisation details directly
        # via a nested serializer but just its ID:
        # fields = [..., 'organisation'] # Add 'organisation' to fields
        # read_only_fields = [..., 'organisation'] # And make it read-only
