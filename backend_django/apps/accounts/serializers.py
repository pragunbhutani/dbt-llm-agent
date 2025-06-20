from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.db import transaction
from .models import Organisation, OrganisationSettings
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .llm_constants import CHAT_MODELS, EMBEDDING_MODELS

User = get_user_model()


class OrganisationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organisation
        fields = ["id", "name", "owner", "created_at", "updated_at"]
        read_only_fields = [
            "id",
            "owner",
            "created_at",
            "updated_at",
        ]  # Owner is set internally


class UserSerializer(serializers.ModelSerializer):
    organisation = OrganisationSerializer(read_only=True)  # Nested representation

    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name", "organisation"]
        read_only_fields = ["id", "organisation"]


class UserRegistrationSerializer(serializers.ModelSerializer):
    organisation_name = serializers.CharField(write_only=True, max_length=255)
    email = serializers.EmailField()  # Ensure email is required and validated
    first_name = serializers.CharField(max_length=150, required=False, allow_blank=True)
    last_name = serializers.CharField(max_length=150, required=False, allow_blank=True)

    class Meta:
        model = User
        fields = ["email", "password", "first_name", "last_name", "organisation_name"]
        extra_kwargs = {
            "password": {"write_only": True, "style": {"input_type": "password"}},
        }

    def validate_email(self, value):
        # Ensure email is unique
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value

    def create(self, validated_data):
        organisation_name = validated_data.pop("organisation_name")

        # Use email as username by default if not provided elsewhere
        if "username" not in validated_data or not validated_data["username"]:
            validated_data["username"] = validated_data["email"]

        user = User.objects.create_user(
            **validated_data
        )  # create_user handles password hashing

        try:
            with transaction.atomic():
                # Create the organisation
                organisation = Organisation.objects.create(
                    name=organisation_name, owner=user
                )
                # Assign the user to this organisation
                user.organisation = organisation
                user.save(update_fields=["organisation"])
        except Exception as e:
            # If organisation creation or user update fails, delete the created user
            user.delete()
            raise serializers.ValidationError(
                f"Could not create organisation or assign user: {str(e)}"
            )

        return user


class OrganisationSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = OrganisationSettings
        fields = "__all__"
        read_only_fields = ("organisation",)

    def validate(self, data):
        chat_provider = data.get("llm_chat_provider")
        chat_model = data.get("llm_chat_model")
        embeddings_provider = data.get("llm_embeddings_provider")
        embeddings_model = data.get("llm_embeddings_model")

        if chat_provider and chat_model:
            allowed_models = CHAT_MODELS.get(chat_provider)
            if allowed_models is None:
                raise serializers.ValidationError(
                    {"llm_chat_provider": f"Invalid provider: {chat_provider}"}
                )
            if chat_model not in allowed_models:
                raise serializers.ValidationError(
                    {
                        "llm_chat_model": f"Model {chat_model} is not available for provider {chat_provider}."
                    }
                )

        if embeddings_provider and embeddings_model:
            allowed_models = EMBEDDING_MODELS.get(embeddings_provider)
            if allowed_models is None:
                raise serializers.ValidationError(
                    {
                        "llm_embeddings_provider": f"Invalid provider: {embeddings_provider}"
                    }
                )
            if embeddings_model not in allowed_models:
                raise serializers.ValidationError(
                    {
                        "llm_embeddings_model": f"Model {embeddings_model} is not available for provider {embeddings_provider}."
                    }
                )

        # Auto-detect Slack team ID if bot token is provided
        slack_bot_token = data.get("slack_bot_token")
        if slack_bot_token:
            from apps.integrations.slack.handlers import get_team_info_from_token

            team_info = get_team_info_from_token(slack_bot_token)
            if team_info:
                data["slack_team_id"] = team_info["team_id"]
            else:
                raise serializers.ValidationError(
                    {
                        "slack_bot_token": "Invalid Slack bot token. Could not retrieve team information."
                    }
                )

        return data


# Serializers previously in this file moved to:
# - ModelSerializer -> apps.knowledge_base.serializers
# - QuestionSerializer -> apps.agents.serializers
# - ModelEmbeddingSerializer -> apps.embeddings.serializers
