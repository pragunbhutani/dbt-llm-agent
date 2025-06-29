import logging
import boto3
from botocore.exceptions import ClientError
from django.conf import settings
import os

logger = logging.getLogger(__name__)

# Core services can be defined here if needed later.

# Services previously in this file moved to:
# - EmbeddingService, ChatService, get_openai_api_key -> apps.llm_providers.services

# --- End of file --- Ensure everything below this line is removed ---


class SecretManager:
    """
    Centralized service for managing secrets via AWS Parameter Store.
    Uses LocalStack in local environment and real AWS in production.
    """

    def __init__(self):
        # Configure boto3 client based on environment
        if settings.IS_LOCAL:
            # Use LocalStack for local development
            # Check if we're running inside Docker or outside
            localstack_host = os.environ.get("LOCALSTACK_HOST", "localhost")
            endpoint_url = f"http://{localstack_host}:4566"

            logger.info(
                f"Initializing SecretManager for local environment with endpoint: {endpoint_url}"
            )

            self.ssm_client = boto3.client(
                "ssm",
                endpoint_url=endpoint_url,
                region_name="us-east-1",
                aws_access_key_id="test",  # LocalStack dummy credentials
                aws_secret_access_key="test",
            )
        else:
            # Use real AWS for development/production
            logger.info(
                f"Initializing SecretManager for {settings.ENVIRONMENT} environment with real AWS"
            )
            self.ssm_client = boto3.client("ssm", region_name="us-east-1")

    def get_secret(
        self, parameter_name: str, with_decryption: bool = True
    ) -> str | None:
        """
        Retrieve a secret from Parameter Store.

        Args:
            parameter_name: The name/path of the parameter
            with_decryption: Whether to decrypt SecureString parameters

        Returns:
            The parameter value or None if not found
        """
        try:
            response = self.ssm_client.get_parameter(
                Name=parameter_name, WithDecryption=with_decryption
            )
            return response["Parameter"]["Value"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "ParameterNotFound":
                logger.warning(f"Parameter not found: {parameter_name}")
                return None
            else:
                logger.error(f"Error retrieving parameter {parameter_name}: {e}")
                raise

    def put_secret(
        self,
        parameter_name: str,
        value: str,
        parameter_type: str = "SecureString",
        description: str = "",
    ) -> bool:
        """
        Store a secret in Parameter Store.

        Args:
            parameter_name: The name/path of the parameter
            value: The secret value to store
            parameter_type: Type of parameter (String, StringList, SecureString)
            description: Optional description

        Returns:
            True if successful, False otherwise
        """
        try:
            self.ssm_client.put_parameter(
                Name=parameter_name,
                Value=value,
                Type=parameter_type,
                Description=description,
                Overwrite=True,
            )
            logger.info(f"Successfully stored parameter: {parameter_name}")
            return True
        except ClientError as e:
            logger.error(f"Error storing parameter {parameter_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error storing parameter {parameter_name}: {e}")
            if settings.IS_LOCAL:
                logger.error(
                    "This might be a LocalStack connection issue. Make sure LocalStack is running and accessible."
                )
            return False

    def delete_secret(self, parameter_name: str) -> bool:
        """
        Delete a secret from Parameter Store.

        Args:
            parameter_name: The name/path of the parameter to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.ssm_client.delete_parameter(Name=parameter_name)
            logger.info(f"Successfully deleted parameter: {parameter_name}")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ParameterNotFound":
                logger.warning(f"Parameter not found for deletion: {parameter_name}")
                return True  # Consider it successful if already doesn't exist
            else:
                logger.error(f"Error deleting parameter {parameter_name}: {e}")
                return False


# Global instance for easy access
secret_manager = SecretManager()
