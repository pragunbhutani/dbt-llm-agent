# ragstar (previously dbt-llm-agent)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](https://github.com/python/mypy)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![Beta Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/pragunbhutani/ragstar)

**Ragstar is an advanced LLM-based AI Agent designed specifically for interacting with and understanding your dbt (data build tool) projects.** It helps data teams streamline documentation, improve data discovery, and gain deeper insights into their dbt models through natural language interaction. Leverage the power of Large Language Models to make your dbt project more accessible and maintainable.

> **BETA NOTICE**: Ragstar is currently in beta. Key features include agentic model interpretation and semantic question answering about your dbt project. Slack integration and other enhancements are actively under development.

## Key Features

- **Natural Language Question Answering**: Ask questions about your dbt project structure, models, sources, metrics, and lineage in plain English (e.g., "Explain the `fct_orders` model", "Which models use the `stg_customers` source?").
- **Agentic Model Interpretation**: Ragstar intelligently analyzes dbt models, understanding their logic, upstream dependencies, and business context. It verifies interpretations by examining related models, ensuring accuracy.
- **Automated Documentation Generation**: Automatically generate descriptions and column-level documentation for dbt models lacking documentation, reducing manual effort and improving data catalog quality.
- **Semantic Search**: Find relevant models, columns, and documentation based on the meaning of your query, not just keywords, powered by vector embeddings.
- **dbt Integration**: Seamlessly integrates with dbt projects by parsing `manifest.json` artifacts from dbt Cloud runs or local `dbt compile` output, or directly from source code. Supports dbt's model selection syntax (`--select`, `--exclude`) for targeted operations.
- **Postgres with pgvector**: Uses PostgreSQL and the `pgvector` extension to efficiently store dbt metadata, generated documentation, and vector embeddings for fast retrieval and semantic search. Compatible with managed services like Supabase Postgres.
- **Question & Feedback Tracking**: Logs all questions, answers, and user feedback (useful/not useful, corrections) to enable continuous learning and improvement of the agent's performance.
- **Extensible Architecture**: Built with modular components for easy extension and customization.
- **Coming Soon: Slack Integration**: Interact with Ragstar directly within your Slack workspace.

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

1.  **Check Python Version:**
    This project requires Python 3.10 or higher. You can check your Python version with:

    ```bash
    python --version
    # or
    python3 --version
    ```

    If you need to upgrade or install Python 3.10+, visit [python.org/downloads](https://www.python.org/downloads/).

2.  **Clone the repository:**

    ```bash
    git clone https://github.com/pragunbhutani/ragstar.git
    cd ragstar
    ```

3.  **Install dependencies:**
    This project uses [Poetry](https://python-poetry.org/) for dependency management.

    ```bash
    # Install Poetry if you don't have it
    curl -sSL https://install.python-poetry.org | python3 -

    # Install dependencies
    poetry install
    ```

4.  **Set up PostgreSQL:**
    You need a PostgreSQL database (version 11+) with the `pgvector` extension enabled. This database will store model metadata, embeddings, and question history.

    - Install PostgreSQL if you haven't already.
    - Install `pgvector`. Follow the instructions at [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector).
    - Create a database for the agent (e.g., `ragstar`).

    Quick setup commands for local PostgreSQL:

    ```bash
    # Create database
    createdb ragstar

    # Enable pgvector extension (run this in psql)
    psql -d ragstar -c 'CREATE EXTENSION IF NOT EXISTS vector;'
    ```

5.  **Configure environment variables:**
    Copy the example environment file and fill in your details:

    ```bash
    cp .env.example .env
    ```

    Edit the `.env` file with your:

    - `OPENAI_API_KEY`
    - `POSTGRES_URI` (database connection string)
    - dbt Cloud credentials (`DBT_CLOUD_...`) if using `init cloud`.
    - `DBT_PROJECT_PATH` if using `init local` or `init source` and not providing the path as an argument.

6.  **Initialize the database schema:**
    Run the following command. This creates the necessary tables and enables the `pgvector` extension if needed.

    ```bash
    poetry run ragstar init-db
    ```

## Initializing Your dbt Project with Ragstar

To use the agent, you first need to load your dbt project's metadata into the database. Use the `init` command:

```bash
poetry run ragstar init <mode> [options]
```

There are three modes available:

### 1. Cloud Mode (Recommended)

Fetches the `manifest.json` from the latest successful run in your dbt Cloud account. This provides the richest metadata, including compiled SQL.

- **Command:** `poetry run ragstar init cloud`
- **Prerequisites:**
  - dbt Cloud account with successful job runs that generate artifacts.
  - Environment variables set in `.env`:
    - `DBT_CLOUD_URL`
    - `DBT_CLOUD_ACCOUNT_ID`
    - `DBT_CLOUD_API_KEY` (User Token or Service Token)
- **Example:**

  ```bash
  # Ensure DBT_CLOUD_URL, DBT_CLOUD_ACCOUNT_ID, DBT_CLOUD_API_KEY are in .env
  poetry run ragstar init cloud
  ```

### 2. Local Mode

Runs `dbt compile` on your local dbt project and parses the generated `manifest.json` from the `target/` directory. Also provides rich metadata including compiled SQL.

- **Command:** `poetry run ragstar init local --project-path /path/to/your/dbt/project`
- **Prerequisites:**
  - dbt project configured locally (`dbt_project.yml`, `profiles.yml` etc.).
  - Ability to run `dbt compile` successfully in the project directory.
  - The dbt project path can be provided via the `--project-path` argument or the `DBT_PROJECT_PATH` environment variable.
- **Example:**

  ```bash
  # Using argument
  poetry run ragstar init local --project-path /Users/me/code/my_dbt_project

  # Using environment variable (set DBT_PROJECT_PATH in .env)
  poetry run ragstar init local
  ```

### 3. Source Code Mode (Fallback)

Parses your dbt project directly from the source `.sql` and `.yml` files. This mode does _not_ capture compiled SQL or reliably determine data types.

- **Command:** `poetry run ragstar init source /path/to/your/dbt/project`
- **Prerequisites:**
  - Access to the dbt project source code.
  - The dbt project path can be provided via the argument or the `DBT_PROJECT_PATH` environment variable.
- **Example:**

  ```bash
  # Using argument
  poetry run ragstar init source /Users/me/code/my_dbt_project

  # Using environment variable
  poetry run ragstar init source
  ```

**Note:** The `init` command replaces the older `parse` command for loading project metadata.

You only need to run `init` once initially, or again if your dbt project structure changes significantly. Use the `--force` flag with `init` to overwrite existing models in the database.

## Usage

Once you've completed the setup and initialization, you've got the basics sorted! Now you can start using the agent's main features:

### 1. Working with Model Documentation

There are two main paths depending on whether your models already have documentation:

#### If Your Models Already Have Documentation:

Generate vector embeddings for semantic search to enable question answering:

```bash
# Embed all models
poetry run ragstar embed --select "*"

# Or embed specific models or tags
poetry run ragstar embed --select "+tag:marts"
poetry run ragstar embed --select "my_model"
```

#### If Your Models Need Documentation:

First, use the LLM to interpret and generate descriptions for models and columns:

```bash
# Interpret a specific model and save the results
poetry run ragstar interpret --select "fct_orders" --save

# Interpret all models in the staging layer, save, and embed
poetry run ragstar interpret --select "tag:staging" --save --embed
```

The `--save` flag stores the interpretations in the database, and `--embed` automatically generates embeddings after interpretation.

### 2. Asking Questions

Now that your models are embedded, you can ask questions about your dbt project:

```bash
poetry run ragstar ask "What models are tagged as finance?"
poetry run ragstar ask "Show me the columns in the customers model"
poetry run ragstar ask "Explain the fct_orders model"
poetry run ragstar ask "How is discount_amount calculated in the orders model?"
```

### 3. Providing Feedback

Help improve the agent by providing feedback on answers:

```bash
# List previous questions
poetry run ragstar questions

# Provide positive feedback
poetry run ragstar feedback 1 --useful

# Provide negative feedback with explanation
poetry run ragstar feedback 2 --not-useful --text "Use this_other_model instead"

# Just provide text feedback without marking useful/not useful
poetry run ragstar feedback 3 --text "This answer is correct but too verbose."
```

This feedback helps the agent improve its answers over time.

### 4. Additional Commands

```bash
# List all models in your project
poetry run ragstar list

# Get detailed information about a specific model
poetry run ragstar model-details my_model_name
```

## Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflow.

## License

[MIT License](https://opensource.org/licenses/MIT)
