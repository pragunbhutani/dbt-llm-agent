# Secret Management System

## Overview

Ragstar uses AWS Parameter Store for secure secret management with environment-based path separation. This approach provides:

- **🔒 Security**: No plaintext secrets in database
- **🏗️ Environment Isolation**: Separate paths for local/development/production
- **🧪 Local Development**: LocalStack for AWS Parameter Store mocking
- **🔄 Centralized Management**: Single service for all secret operations

## Architecture

### Single AWS Account with Path Prefixes

We use one AWS account with environment-based path prefixes:

```
/ragstar/{environment}/org-{org_id}/{service}/{credential_name}
```

**Examples:**

- Local: `/ragstar/local/org-123/llm/openai-api-key`
- Development: `/ragstar/development/org-123/llm/openai-api-key`
- Production: `/ragstar/production/org-123/llm/openai-api-key`

### Environment Configuration

Set the `ENVIRONMENT` variable to control path prefixes:

- `ENVIRONMENT=local` → Uses LocalStack (default)
- `ENVIRONMENT=development` → Uses real AWS
- `ENVIRONMENT=production` → Uses real AWS

## Components

### 1. SecretManager Service (`apps/accounts/services.py`)

Central service for all secret operations:

```python
from apps.accounts.services import secret_manager

# Store a secret
secret_manager.put_secret(path, value, description)

# Retrieve a secret
value = secret_manager.get_secret(path)

# Delete a secret
secret_manager.delete_secret(path)
```

### 2. OrganisationSettings Model

LLM API keys stored in Parameter Store:

```python
# Set API keys (stores in Parameter Store)
org_settings.set_llm_openai_api_key("sk-...")
org_settings.set_llm_google_api_key("AIza...")
org_settings.set_llm_anthropic_api_key("sk-ant-...")

# Get API keys (retrieves from Parameter Store)
openai_key = org_settings.get_llm_openai_api_key()
google_key = org_settings.get_llm_google_api_key()
anthropic_key = org_settings.get_llm_anthropic_api_key()
```

### 3. OrganisationIntegration Model

Integration credentials stored as JSON in Parameter Store:

```python
# Set credentials (stores JSON in Parameter Store)
integration.set_credentials({
    "bot_token": "xoxb-...",
    "signing_secret": "abc123...",
    "app_token": "xapp-..."
})

# Get credentials (retrieves JSON from Parameter Store)
credentials = integration.credentials  # Dict[str, Any]
bot_token = integration.get_credential("bot_token")

# Update specific credentials
integration.update_credentials({"bot_token": "new-token"})
```

## Local Development Setup

### 1. LocalStack Configuration

LocalStack is configured in `docker-compose.yml`:

```yaml
localstack:
  image: localstack/localstack:latest
  ports:
    - "4566:4566"
  environment:
    - SERVICES=ssm # Systems Manager (Parameter Store)
```

### 2. Environment Variables

For local development:

```bash
ENVIRONMENT=local  # Uses LocalStack
```

## Production Setup

### 1. AWS Configuration

Set these environment variables in production:

```bash
ENVIRONMENT=production
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
```

### 2. IAM Permissions

Required IAM permissions for the service:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["ssm:GetParameter", "ssm:PutParameter", "ssm:DeleteParameter"],
      "Resource": "arn:aws:ssm:*:*:parameter/ragstar/*"
    }
  ]
}
```

## Testing

Use the management command to test the secret management system:

```bash
# Test secret operations
uv run python manage.py test_secrets

# Test and cleanup
uv run python manage.py test_secrets --cleanup
```

## Migration Strategy

Since we're in pre-production:

1. **Drop existing plaintext secrets** from database
2. **Run migrations** to update model structure
3. **Re-configure integrations** through the UI (they'll use Parameter Store)

## Security Benefits

1. **No Database Exposure**: Secrets never stored in plaintext in database
2. **Environment Isolation**: Each environment has separate secret paths
3. **Audit Trail**: AWS CloudTrail logs all Parameter Store access
4. **Encryption**: AWS handles encryption at rest and in transit
5. **Access Control**: IAM policies control who can access secrets

## Future Considerations

- **Multi-Account Setup**: Can migrate to separate AWS accounts per environment later
- **Secret Rotation**: Implement automatic secret rotation workflows
- **Monitoring**: Set up CloudWatch alerts for secret access patterns
