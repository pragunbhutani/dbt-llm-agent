# dbt-llm-agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](https://github.com/python/mypy)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)

An LLM-powered agent for interacting with dbt projects.

## Features

- **Question Answering**: Ask questions about your dbt project in natural language
- **Documentation Generation**: Automatically generate documentation for missing models
- **Slack Integration**: Ask questions and receive answers directly in Slack
- **FastAPI Server**: Interact with the agent programmatically via REST API
- **Postgres with pgvector**: Store model embeddings in Postgres using pgvector (supports Supabase)
- **dbt Model Selection**: Use dbt's model selection syntax to specify which models to work with
- **Question Tracking**: Track questions, answers, and feedback for continuous improvement

## Architecture

The agent uses a combination of:

- **dbt Project Parsing**: Extract information from your dbt project including models, sources, and documentation
- **PostgreSQL with pgvector**: Store both structured metadata and vector embeddings for semantic search
- **Model Selection**: Selectively parse and embed models using dbt's selection syntax
- **LLM Integration**: Use large language models (like GPT-4) to generate responses and documentation
- **Question Tracking**: Store a history of questions, answers, and user feedback

## Installation

### Prerequisites

- **Python 3.10+**
- **Poetry** for dependency management (install from [Poetry's documentation](https://python-poetry.org/docs/#installation))
- **PostgreSQL 13+** with the [pgvector](https://github.com/pgvector/pgvector) extension
- **OpenAI API key** (or compatible API)
- **dbt project**

### Key Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key
POSTGRES_URI=postgresql://user:password@localhost:5432/dbt_llm_agent
DBT_PROJECT_PATH=/path/to/your/dbt/project

# Optional - for Slack integration
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_SIGNING_SECRET=your_slack_signing_secret
```

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/dbt-llm-agent.git
   cd dbt-llm-agent
   ```

2. Install dependencies with Poetry:

   ```bash
   poetry install
   ```

3. Create a `.env` file with your configuration (see environment variables above)

4. Initialize the database:

   ```bash
   poetry run dbt-llm-agent migrate
   ```

## Usage

### Command Line

```bash
# Parse a dbt project (without embedding)
poetry run dbt-llm-agent parse /path/to/your/dbt/project

# Parse specific models using dbt selection syntax
poetry run dbt-llm-agent parse /path/to/your/dbt/project --select "tag:marketing,+downstream_model"

# Embed specific models in vector database
poetry run dbt-llm-agent embed --select "tag:marketing,+downstream_model"

# Interpret a single model and generate documentation
poetry run dbt-llm-agent interpret customer_orders

# Interpret multiple models using dbt selector syntax
poetry run dbt-llm-agent interpret --select "tag:marketing"

# Interpret and embed in one step
poetry run dbt-llm-agent interpret customer_orders --embed

# Run database migrations
poetry run dbt-llm-agent migrate --verbose

# Drop old columns after migration (use with caution)
poetry run dbt-llm-agent migrate --drop-old-columns

# Ask a question
poetry run dbt-llm-agent ask "What does the model customer_orders do?"

# Provide feedback on an answer
poetry run dbt-llm-agent feedback 123 --useful=true --feedback="The answer was clear and helpful"

# List past questions and answers
poetry run dbt-llm-agent questions --limit=20 --useful=true

# Start the API server
poetry run dbt-llm-agent api

# Start the Slack bot
poetry run dbt-llm-agent slack
```

### Model Selection Syntax

The agent supports dbt's model selection syntax:

- `*` - Select all models
- `model_name` - Select a specific model
- `+model_name` - Select a model and all its children (downstream dependencies)
- `@model_name` - Select a model and all its parents (upstream dependencies)
- `tag:marketing` - Select all models with the tag "marketing"
- `config.materialized:table` - Select all models materialized as tables
- `path/to/models` - Select models in a specific path
- `!model_name` - Exclude a specific model

You can combine selectors with commas, e.g. `tag:marketing,+downstream_model`.

### API Usage

The agent provides a REST API for programmatic usage:

```bash
# Start the API server
poetry run dbt-llm-agent api
```

#### Endpoints:

- `POST /ask` - Ask a question
- `POST /embed` - Embed specific models
- `POST /questions/{question_id}/feedback` - Provide feedback on an answer
- `GET /questions` - List past questions and answers

### Database Schema Migrations

The database schema may evolve with new releases. To update your existing database:

```bash
# Run migrations to add new columns
poetry run dbt-llm-agent migrate

# Use verbose mode to see detailed migration logs
poetry run dbt-llm-agent migrate --verbose

# Optionally drop old columns after migration (use with caution)
poetry run dbt-llm-agent migrate --drop-old-columns
```

Migrations will automatically run when the application starts, but you can also run them manually if needed.

## Development

### Project Structure

```
dbt_llm_agent/
├── api/                   # API server implementation
├── commands/              # CLI command implementations
├── core/                  # Core functionality
├── integrations/          # External integrations (Slack, etc.)
├── storage/              # Storage implementations
├── utils/                # Utility functions
├── cli.py                # Command line interface
└── __init__.py           # Package initialization
```

### Testing

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=dbt_llm_agent
```

### Code Quality

The project uses several tools for code quality:

```bash
# Format code
poetry run black .
poetry run isort .

# Type checking
poetry run mypy .

# Linting
poetry run ruff check .
```

## License

MIT
