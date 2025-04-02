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
- **Agentic Model Interpretation**: Intelligently interpret models using a step-by-step approach that verifies interpretations against upstream models
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

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/pragunbhutani/dbt-llm-agent.git
    cd dbt-llm-agent
    ```

2.  **Install dependencies:**
    This project uses [Poetry](https://python-poetry.org/) for dependency management.

    ```bash
    poetry install
    ```

3.  **Set up PostgreSQL:**
    You need a PostgreSQL database (version 11+) with the `pgvector` extension enabled. This database will store model metadata, embeddings, and question history.

    - Install PostgreSQL if you haven't already.
    - Install `pgvector`. Follow the instructions at [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
    - Create a database for the agent (e.g., `dbt_llm_agent`).

4.  **Configure environment variables:**
    Copy the example environment file and fill in your details:

    ```bash
    cp .env.example .env
    ```

    Edit the `.env` file with your:

    - `OPENAI_API_KEY`
    - `POSTGRES_URI` (database connection string)
    - dbt Cloud credentials (`DBT_CLOUD_...`) if using `init cloud`.
    - `DBT_PROJECT_PATH` if using `init local` or `init source` and not providing the path as an argument.

5.  **Initialize the database schema:**
    Run the following command. This creates the necessary tables and enables the `pgvector` extension if needed.
    ```bash
    poetry run dbt-llm-agent init-db
    ```

## Initializing Your dbt Project

To use the agent, you first need to load your dbt project's metadata into the database. Use the `init` command:

```bash
poetry run dbt-llm-agent init <mode> [options]
```

There are three modes available:

### 1. Cloud Mode (Recommended)

Fetches the `manifest.json` from the latest successful run in your dbt Cloud account. This provides the richest metadata, including compiled SQL.

- **Command:** `poetry run dbt-llm-agent init cloud`
- **Prerequisites:**
  - dbt Cloud account with successful job runs that generate artifacts.
  - Environment variables set in `.env`:
    - `DBT_CLOUD_URL`
    - `DBT_CLOUD_ACCOUNT_ID`
    - `DBT_CLOUD_API_KEY` (User Token or Service Token)
- **Example:**
  ```bash
  # Ensure DBT_CLOUD_URL, DBT_CLOUD_ACCOUNT_ID, DBT_CLOUD_API_KEY are in .env
  poetry run dbt-llm-agent init cloud
  ```

### 2. Local Mode

Runs `dbt compile` on your local dbt project and parses the generated `manifest.json` from the `target/` directory. Also provides rich metadata including compiled SQL.

- **Command:** `poetry run dbt-llm-agent init local --project-path /path/to/your/dbt/project`
- **Prerequisites:**
  - dbt project configured locally (`dbt_project.yml`, `profiles.yml` etc.).
  - Ability to run `dbt compile` successfully in the project directory.
  - The dbt project path can be provided via the `--project-path` argument or the `DBT_PROJECT_PATH` environment variable.
- **Example:**

  ```bash
  # Using argument
  poetry run dbt-llm-agent init local --project-path /Users/me/code/my_dbt_project

  # Using environment variable (set DBT_PROJECT_PATH in .env)
  poetry run dbt-llm-agent init local
  ```

### 3. Source Code Mode (Fallback)

Parses your dbt project directly from the source `.sql` and `.yml` files. This mode does _not_ capture compiled SQL or reliably determine data types.

- **Command:** `poetry run dbt-llm-agent init source /path/to/your/dbt/project`
- **Prerequisites:**
  - Access to the dbt project source code.
  - The dbt project path can be provided via the argument or the `DBT_PROJECT_PATH` environment variable.
- **Example:**

  ```bash
  # Using argument
  poetry run dbt-llm-agent init source /Users/me/code/my_dbt_project

  # Using environment variable
  poetry run dbt-llm-agent init source
  ```

**Note:** The `init` command replaces the older `parse` command for loading project metadata.

You only need to run `init` once initially, or again if your dbt project structure changes significantly. Use the `--force` flag with `init` to overwrite existing models in the database.

## Usage

Once initialized, you can use the agent:

1.  **Embed Models:** Generate vector embeddings for semantic search.

    ```bash
    # Embed all models
    poetry run dbt-llm-agent embed --select "*"

    # Embed specific models or tags
    poetry run dbt-llm-agent embed --select "+tag:marts"
    poetry run dbt-llm-agent embed --select "my_model"
    ```

2.  **Interpret Models (Generate Documentation):** Use LLM to generate descriptions for models and columns.

    ```bash
    # Interpret a specific model and save the results
    poetry run dbt-llm-agent interpret --select "fct_orders" --save

    # Interpret all models in the staging layer, save, and embed
    poetry run dbt-llm-agent interpret --select "tag:staging" --save --embed
    ```

    See `poetry run dbt-llm-agent interpret --help` for more options like `--recursive` and `--iterations`.

3.  **Ask Questions:** Interact with the agent to ask questions about your dbt project.

    ```bash
    poetry run dbt-llm-agent ask "What models are tagged as finance?"
    poetry run dbt-llm-agent ask "Show me the columns in the customers model"
    poetry run dbt-llm-agent ask "Explain the fct_orders model"
    ```

4.  **List Models & Details:**

    ```bash
    poetry run dbt-llm-agent list
    poetry run dbt-llm-agent model-details my_model_name
    ```

5.  **Manage Questions & Feedback:**
    ```bash
    poetry run dbt-llm-agent questions
    poetry run dbt-llm-agent feedback --question-id 1 --score 1 # Thumbs up
    poetry run dbt-llm-agent feedback --question-id 2 --score -1 --text "Incorrect model suggested"
    ```

## Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflow.

## License

[Specify your license, e.g., MIT License]
