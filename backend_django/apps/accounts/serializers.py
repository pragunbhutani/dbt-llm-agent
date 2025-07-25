from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.db import transaction
from .models import Organisation, OrganisationSettings
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .llm_constants import CHAT_MODELS, EMBEDDING_MODELS
from apps.whitelist.models import SignupWhitelist

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
        # Check against the whitelist if it's not empty
        if (
            SignupWhitelist.objects.exists()
            and not SignupWhitelist.objects.filter(email__iexact=value).exists()
        ):
            raise serializers.ValidationError(
                "This email address is not yet whitelisted for signup."
            )

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
    # Add the API key fields as write-only for backward compatibility
    llm_openai_api_key = serializers.CharField(
        write_only=True, required=False, allow_blank=True, allow_null=True
    )
    llm_google_api_key = serializers.CharField(
        write_only=True, required=False, allow_blank=True, allow_null=True
    )
    llm_anthropic_api_key = serializers.CharField(
        write_only=True, required=False, allow_blank=True, allow_null=True
    )

    class Meta:
        model = OrganisationSettings
        fields = [
            "organisation",
            "llm_chat_provider",
            "llm_chat_model",
            "llm_embeddings_provider",
            "llm_embeddings_model",
            "llm_openai_api_key_path",
            "llm_google_api_key_path",
            "llm_anthropic_api_key_path",
            "created_at",
            "updated_at",
            # Write-only fields for API keys
            "llm_openai_api_key",
            "llm_google_api_key",
            "llm_anthropic_api_key",
        ]
        read_only_fields = (
            "organisation",
            "created_at",
            "updated_at",
            "llm_openai_api_key_path",
            "llm_google_api_key_path",
            "llm_anthropic_api_key_path",
        )

    def to_representation(self, instance):
        """Custom representation to include actual API keys for frontend display."""
        data = super().to_representation(instance)

        # Add the actual API keys for display (masked for security)
        openai_key = instance.get_llm_openai_api_key()
        google_key = instance.get_llm_google_api_key()
        anthropic_key = instance.get_llm_anthropic_api_key()

        # Show masked versions or None
        data["llm_openai_api_key"] = (
            self._mask_api_key(openai_key) if openai_key else None
        )
        data["llm_google_api_key"] = (
            self._mask_api_key(google_key) if google_key else None
        )
        data["llm_anthropic_api_key"] = (
            self._mask_api_key(anthropic_key) if anthropic_key else None
        )

        return data

    def _mask_api_key(self, api_key: str) -> str:
        """Mask API key for display purposes."""
        if not api_key or len(api_key) < 8:
            return "****"
        return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"

    def _is_unmasked_key(self, key: str | None) -> bool:
        """Determine if the supplied key looks like a real (new) key versus a masked placeholder."""
        if not key:
            return False
        stripped = key.strip()
        # Common placeholder patterns – completely asterisks or starts/ends with asterisks after prefix
        if set(stripped) == {"*"}:  # e.g., "********"
            return False
        # Patterns like "sk-****" or "****abcd" or "abcd****efgh" where majority is *
        stars = stripped.count("*")
        if (
            stars and stars / len(stripped) > 0.3
        ):  # heuristic: >30% stars considered masked
            return False
        return True

    def update(self, instance, validated_data):
        """Custom update method to handle API keys using secret management."""
        # Extract API keys from validated data
        openai_key = validated_data.pop("llm_openai_api_key", None)
        google_key = validated_data.pop("llm_google_api_key", None)
        anthropic_key = validated_data.pop("llm_anthropic_api_key", None)

        # Update the regular fields first
        instance = super().update(instance, validated_data)

        # Handle API keys using secret management if a *new* key was supplied
        if self._is_unmasked_key(openai_key):
            success = instance.set_llm_openai_api_key(openai_key)
            if not success:
                raise serializers.ValidationError(
                    {"llm_openai_api_key": "Failed to store OpenAI API key securely."}
                )

        if self._is_unmasked_key(google_key):
            success = instance.set_llm_google_api_key(google_key)
            if not success:
                raise serializers.ValidationError(
                    {"llm_google_api_key": "Failed to store Google API key securely."}
                )

        if self._is_unmasked_key(anthropic_key):
            success = instance.set_llm_anthropic_api_key(anthropic_key)
            if not success:
                raise serializers.ValidationError(
                    {
                        "llm_anthropic_api_key": "Failed to store Anthropic API key securely."
                    }
                )

        return instance

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

        return data


# Serializers previously in this file moved to:
# - ModelSerializer -> apps.knowledge_base.serializers
# - QuestionSerializer -> apps.agents.serializers
# - ModelEmbeddingSerializer -> apps.embeddings.serializers
