from rest_framework import serializers

# Import from local models
from .models import Model
from apps.accounts.serializers import (
    OrganisationSerializer,
)  # Import if you want to nest it
from apps.embeddings.models import ModelEmbedding


class ModelSerializer(serializers.ModelSerializer):
    # To make organisation readable in responses, but not writable directly by API client
    organisation = OrganisationSerializer(read_only=True)
    answering_status = serializers.SerializerMethodField()

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
            "answering_status",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "organisation"]
        # Alternatively, if you don't want to expose organisation details directly
        # via a nested serializer but just its ID:
        # fields = [..., 'organisation'] # Add 'organisation' to fields
        # read_only_fields = [..., 'organisation'] # And make it read-only

    def get_answering_status(self, obj):
        embedding = ModelEmbedding.objects.filter(model=obj).first()
        if embedding:
            # Check processing status first - if it's being processed, always return "Training"
            if embedding.is_processing:
                return "Training"

            # Check if embedding is valid (not a placeholder)
            # Convert document to string if it's not already and check if empty
            document_str = (
                str(embedding.document) if embedding.document is not None else ""
            )

            # Check if embedding vector is a placeholder (all zeros)
            embedding_vector = embedding.embedding
            is_zero_embedding = False
            if embedding_vector is not None:
                try:
                    # Convert to list for comparison if it's a numpy array or similar
                    if hasattr(embedding_vector, "tolist"):
                        embedding_list = embedding_vector.tolist()
                    elif isinstance(embedding_vector, (list, tuple)):
                        embedding_list = list(embedding_vector)
                    else:
                        embedding_list = []

                    # Check if it's all zeros (either int or float)
                    is_zero_embedding = embedding_list == [0] * len(
                        embedding_list
                    ) or embedding_list == [0.0] * len(embedding_list)
                except (AttributeError, TypeError):
                    # If we can't convert it, assume it's not a zero embedding
                    is_zero_embedding = False

            is_placeholder = document_str == "" or is_zero_embedding

            if is_placeholder:
                # Treat placeholder embeddings as if no embedding exists
                return "No"
            elif embedding.can_be_used_for_answers:
                return "Yes"
            else:
                return "No"
        return "No"
