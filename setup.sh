#!/bin/bash

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Install dependencies
echo "Installing dependencies..."
poetry install

# Setup instructions
echo "
Setup complete! You can now run the dbt-llm-agent:

# Configure the agent
poetry run dbt-llm-agent setup --openai-api-key YOUR_API_KEY --postgres-uri postgresql://user:pass@localhost:5432/dbname

# Parse a dbt project
poetry run dbt-llm-agent parse /path/to/your/dbt/project
" 