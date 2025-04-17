# Meet Ragstar: Your AI Data Analyst for dbt Projects.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](https://github.com/python/mypy)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![Beta Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/pragunbhutani/ragstar)

** Previously known as dbt-llm-agent **

Ragstar is an AI agent that learns the context of your dbt project to answer questions, generate documentation, and bring data insights closer to your users. Interact via Slack or CLI, and watch it improve over time with feedback.

> **BETA NOTICE**: Ragstar is currently in beta. Core features include agentic model interpretation and semantic question answering about your dbt project.

## Table of Contents

- [Key Value](#key-value)
- [Key Features](#key-features)
- [Use Cases](#use-cases)
- [Architecture](#architecture)
- [Setup](#setup)
  - [Option 1: Docker Compose (Recommended)](#option-1-docker-compose-recommended)
  - [Option 2: Local Python Environment (Advanced)](#option-2-local-python-environment-advanced)
- [Usage](#usage)
  - [Using Docker Compose](#using-docker-compose)
  - [Using Local Python Environment](#using-local-python-environment)
  - [Core Commands](#core-commands)
- [Contributing](#contributing)
- [License](#license)

## Key Value

- **Democratize Data Access:** Allow anyone to ask questions about your dbt project in natural language via Slack or CLI.
- **Automate Documentation:** Generate model and column descriptions where they're missing, improving data catalog quality.
- **Enhance Data Discovery:** Quickly find relevant models and understand their logic without digging through code.
- **Continuous Learning:** Ragstar learns from feedback to provide increasingly accurate and helpful answers.

## Key Features

- **Natural Language Q&A**: Ask about models, sources, metrics, lineage, etc.
- **Agentic Interpretation**: Intelligently analyzes dbt models, understanding logic and context.
- **Automated Documentation Generation**: Fills documentation gaps using LLMs.
- **Semantic Search**: Finds relevant assets based on meaning, not just keywords.
- **dbt Integration**: Parses metadata from dbt Cloud, local runs (`manifest.json`), or source code.
- **Postgres + pgvector Backend**: Stores metadata and embeddings efficiently.
- **Feedback Loop**: Tracks questions and feedback for improvement.
- **Slack Integration**: Built-in Slackbot for easy interaction.

## Use Cases

- **Accelerate Data Discovery**: Quickly find relevant dbt models and understand their purpose without digging through code.
- **Improve Onboarding**: Help new team members understand the dbt project structure and logic faster.
- **Maintain Data Documentation**: Keep dbt documentation up-to-date with automated generation and suggestions.
- **Enhance Data Governance**: Gain better visibility into data lineage and model dependencies.
- **Debug dbt Models**: Ask clarifying questions about model logic and calculations.

## Architecture

Ragstar combines several technologies to provide its capabilities:

- **dbt Project Parsing**: Extracts comprehensive metadata from dbt artifacts (`manifest.json`) or source files (`.sql`, `.yml`), including models, sources, exposures, metrics, tests, columns, descriptions, and lineage.
- **PostgreSQL Database with pgvector**: Serves as the central knowledge store. It holds structured metadata parsed from the dbt project, generated documentation, question/answer history, and vector embeddings of model and column descriptions for semantic search.
- **Vector Embeddings**: Creates numerical representations (embeddings) of model and column documentation using sentence-transformer models. These embeddings capture semantic meaning, enabling powerful search capabilities.
- **Large Language Models (LLMs)**: Integrates with LLMs (e.g., OpenAI's GPT models) via APIs to:
  - Understand natural language questions.
  - Generate human-readable answers based on retrieved context from the database and embeddings.
  - Interpret model logic and generate documentation.
- **Agentic Reasoning**: Employs a step-by-step reasoning process, especially for model interpretation, where it breaks down the task, gathers evidence (e.g., upstream model definitions), and synthesizes an interpretation, similar to how a human analyst would approach it.
- **CLI Interface**: Provides command-line tools (`ragstar ...`) for initialization, embedding generation, asking questions, providing feedback, and managing the system.

## Setup

Setting up Ragstar involves configuring environment variables and initializing the application with your dbt project data. The recommended method is using Docker Compose, which bundles the application and a PostgreSQL database with the required `pgvector` extension.

### Option 1: Docker Compose (Recommended)

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/pragunbhutani/ragstar.git
    cd ragstar
    ```

2.  **Configure Environment Variables:**
    Copy the example environment file:

    ```bash
    cp .env.example .env
    ```

    Edit the `.env` file:

    - **Required:** Set your `OPENAI_API_KEY`.
    - **dbt Connection:** Choose **one** method:
      - **dbt Cloud:** Fill in `DBT_CLOUD_URL`, `DBT_CLOUD_ACCOUNT_ID`, `DBT_CLOUD_API_KEY`.
      - **Local dbt:** Fill in `DBT_PROJECT_PATH` (absolute path to your local dbt project).
    - **Slack (Optional):** If using the Slackbot, set `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET`.
    - **Database (Optional):** By default, Docker Compose sets up an internal PostgreSQL database. If you want to use an **external** PostgreSQL database (must have `pgvector` enabled), uncomment and set `EXTERNAL_POSTGRES_URL`. This will override the internal database settings.
    - **Other:** Review other variables like `APP_PORT`, `RAGSTAR_LOG_LEVEL`, etc., and adjust if needed.

3.  **Build and Run with Docker Compose:**
    From the `ragstar` directory, run:

    ```bash
    docker compose up --build -d
    ```

    This command builds the Docker image (if it doesn't exist), starts the application container (`ragstar_app`) and the database container (`ragstar_db`), and runs them in the background (`-d`). The database will be automatically initialized.

4.  **Initialize with dbt Project Data:**
    Once the containers are running, you need to load your dbt project's metadata. Run the `init` command **inside the running `app` container**:

    ```bash
    # If using dbt Cloud configuration (credentials in .env):
    docker compose exec app ragstar init cloud

    # If using local dbt project path (DBT_PROJECT_PATH in .env):
    docker compose exec app ragstar init local

    # If using local dbt source code (DBT_PROJECT_PATH in .env):
    docker compose exec app ragstar init source
    ```

    - Use `cloud`, `local`, or `source` depending on the dbt configuration you set in `.env`.
    - The `local` and `source` modes require `DBT_PROJECT_PATH` to point to your dbt project _as mounted within the container_. See `docker-compose.yml` volumes.

5.  **Build the Knowledge Base (Interpret & Embed):**
    After initializing, you need to process your models so the agent can answer questions about them.

    - **Interpret (Optional but Recommended):** Generate descriptions for models lacking them.
    - **Embed (Required for Q&A):** Create vector embeddings for semantic search.

    ```bash
    # Example: Interpret models tagged 'knowledge', save, and then embed ALL models
    docker compose exec app ragstar interpret --select "tag:knowledge" --save
    docker compose exec app ragstar embed --select "*"

    # Example: Embed all models directly (if documentation exists)
    docker compose exec ragstar_app ragstar embed --select "*"
    ```

    Choose the appropriate `--select` argument based on which models you want to process. See `Core Commands` below for more details.

6.  **Verify:**
    Check the logs to ensure everything started correctly:
    ```bash
    docker compose logs -f ragstar_app
    ```
    You should see logs indicating successful initialization and the API server starting.

### Option 2: Local Python Environment (Advanced)

If you prefer not to use Docker, you can set up a local Python environment.

1.  **Prerequisites:**

    - Python 3.10 or higher.
    - [Poetry](https://python-poetry.org/) for dependency management.
    - A running PostgreSQL server (version 11+) with the `pgvector` extension enabled.

2.  **Check Python Version:**

    ```bash
    python --version # or python3 --version
    ```

3.  **Clone Repository:**

    ```bash
    git clone https://github.com/pragunbhutani/ragstar.git
    cd ragstar
    ```

4.  **Install Dependencies:**

    ```bash
    # Install Poetry if needed: curl -sSL https://install.python-poetry.org | python3 -
    poetry install
    ```

5.  **Set up PostgreSQL:**

    - Install PostgreSQL and `pgvector`.
    - Create a database (e.g., `ragstar`).
    - Enable the `pgvector` extension in the database:
      ```sql
      -- Run in psql connected to your database
      CREATE EXTENSION IF NOT EXISTS vector;
      ```

6.  **Configure Environment Variables:**
    Copy the example environment file and fill in your details:

    ```bash
    cp .env.example .env
    ```

    Edit `.env`:

    - **Required:** Set `OPENAI_API_KEY`.
    - **Required:** Set `EXTERNAL_POSTGRES_URL` to your database connection string (e.g., `postgresql://user:password@host:port/dbname`). The Docker Compose variables (`POSTGRES_DB`, `POSTGRES_USER`, etc.) are ignored in this setup.
    - **dbt Connection:** Choose **one** method and configure the relevant variables (`DBT_CLOUD_...` or `DBT_PROJECT_PATH`).
    - **Slack (Optional):** Configure Slack variables if needed.
    - **Other:** Adjust `RAGSTAR_LOG_LEVEL`, etc.

7.  **Initialize the Database Schema:**
    Run the database initialization command:

    ```bash
    poetry run ragstar init-db
    ```

8.  **Initialize with dbt Project Data:**
    Load your dbt project metadata using the appropriate `init` mode:

    ```bash
    # Example using dbt Cloud (ensure DBT_CLOUD_* vars are set)
    poetry run ragstar init cloud

    # Example using local dbt project (ensure DBT_PROJECT_PATH is set or use --project-path)
    poetry run ragstar init local --project-path /path/to/your/dbt/project

    # Example using source code
    poetry run ragstar init source /path/to/your/dbt/project
    ```

9.  **Build the Knowledge Base (Interpret & Embed):**
    After initializing, process your models so the agent can answer questions.

    - **Interpret (Optional but Recommended):** Generate descriptions for models lacking them.
    - **Embed (Required for Q&A):** Create vector embeddings for semantic search.

    ```bash
    # Example: Interpret models tagged 'staging', save, and then embed ALL models
    poetry run ragstar interpret --select "tag:staging" --save
    poetry run ragstar embed --select "*"

    # Example: Embed all models directly (if documentation exists)
    poetry run ragstar embed --select "*"
    ```

    Choose the appropriate `--select` argument. See `Core Commands` below.

## Usage

After setup and initialization, you can interact with Ragstar.

### Using Docker Compose:

Most commands should be run **inside the `app` container** using `docker compose exec`:

```bash
# Example: Ask a question
docker compose exec app ragstar ask "Explain the fct_orders model"

# Example: Interpret and embed a model
docker compose exec app ragstar interpret --select "fct_orders" --save --embed

# Example: Run the Slackbot (will run in the foreground of the exec command)
# Note: The API server runs automatically via the Dockerfile's CMD.
# To run the Slackbot *instead* of the API server, you would need to
# modify the command in docker-compose.yml or the Dockerfile CMD.
# For typical use, run the Slackbot as a separate process if needed,
# perhaps in a different container or locally if developing.
# If you *only* want the Slackbot, change the docker-compose command:
# command: poetry run ragstar serve
```

The API server (for potential future UI or direct API calls) is started automatically by the `docker compose up` command.

### Using Local Python Environment:

Run commands directly using `poetry run`:

```bash
# Example: Ask a question
poetry run ragstar ask "Explain the fct_orders model"

# Example: Interpret and embed a model
poetry run ragstar interpret --select "fct_orders" --save --embed

# Example: Run the API server
poetry run uvicorn ragstar.api.server:app --host 0.0.0.0 --port 8000 --reload

# Example: Run the Slackbot
poetry run ragstar serve
```

### Core Commands

- **`ragstar init <cloud|local|source> [options]`**: Loads dbt project metadata into Ragstar. Run this first.
- **`ragstar interpret --select <selector> [--save] [--embed]`**: (Optional) Uses LLM to analyze and generate documentation for selected models. `--save` writes to DB, `--embed` runs embedding after.
- **`ragstar embed --select <selector>`**: (Required for Q&A) Creates vector embeddings for selected models with documentation, enabling semantic search.
- **`ragstar ask "<question>"`**: Ask a natural language question about your dbt project.
- **`ragstar questions`**: List previous questions asked.
- **`ragstar feedback <question_id> [--useful | --not-useful] [--text "<feedback>"]`**: Provide feedback on an answer to improve the agent.
- **`ragstar serve`**: Starts the Slackbot (requires Slack env vars).
- **`ragstar list`**: List models found in the database.
- **`ragstar model-details <model_name>`**: Show detailed information about a specific model.
- **`ragstar init-db`**: (Local Setup Only) Initializes the database schema.

## Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflow.

## License

[MIT License](https://opensource.org/licenses/MIT)
