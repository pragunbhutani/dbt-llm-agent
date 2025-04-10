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
Setup complete! You can now run the ragstar:

# Configure your environment
# Copy .env.example to .env and edit the values
cp .env.example .env
# Edit the .env file with your settings
# Make sure to set:
# - OPENAI_API_KEY
# - POSTGRES_URI

# Parse a dbt project
poetry run ragstar parse /path/to/your/dbt/project

Examples:
--------
Initialize the database:
poetry run ragstar init-db

Then ask questions:
poetry run ragstar ask \"What does the orders model do?\"" 