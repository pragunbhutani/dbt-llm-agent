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

2.  **Set up environment variables:**
    Rename `.env.example` to `.env` and populate it with your specific configurations, such as your OpenAI API key and the `APP_HOST` (e.g., `localhost` or your server's IP address).

    ```bash
    cp .env.example .env
    # Open .env and fill in your values
    ```

3.  **Configure Ragstar rules:**
    Rename `.ragstarrules.example.yml` to `.ragstarrules.yml`. This file allows you to define custom instructions and behaviors for your RAG application.

    ```bash
    cp .ragstarrules.example.yml .ragstarrules.yml
    # Open .ragstarrules.yml and customize if needed
    ```

4.  **Build and run with Docker Compose:**
    This command will build the Docker images and start the application in detached mode.

    ```bash
    docker compose up --build -d
    ```

5.  **Run initial Django commands:**
    Execute these commands in the `app` container to set up the database and create an admin user.

    ```bash
    docker compose exec app uv run python manage.py migrate
    docker compose exec app uv run python manage.py createsuperuser # Follow prompts to create your admin user
    ```

6.  **Initialize your project:**
    This command sets up the necessary project configurations. You can choose between `cloud`, `core`, or `local` methods.

    ```bash
    docker compose exec app uv run python manage.py init_project --method cloud
    # Or --method core, or --method local
    ```

7.  **Access the Django Admin:**
    Open your web browser and navigate to `http://<your_APP_HOST_value>/admin` (e.g., `http://localhost/admin` if `APP_HOST=localhost`).
    Log in with the superuser credentials you created.

8.  **Embed Models:**
    In the Django admin interface, you can:
    - Navigate to "Models".
    - Click on "Interpret".
    - Select and embed the models you want to use for answering questions.

### Option 2: Local Python Environment (Advanced)

If you prefer not to use Docker, you can set up a local Python environment.

1.  **Prerequisites:**

    - Python 3.10 or higher.
    - `uv` (Python package installer and virtual environment manager). You can install it from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).
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

4.  **Create a virtual environment and install dependencies:**

    ```bash
    # Create a virtual environment (e.g., named .venv)
    python -m venv .venv
    # Or using uv: uv venv

    # Activate the virtual environment
    source .venv/bin/activate # On Windows: .venv\Scripts\activate

    # Install dependencies using uv
    uv pip install -r requirements.txt
    # Or if you have a pyproject.toml configured for uv:
    # uv pip install -e .
    ```

5.  **Set up PostgreSQL:**

    - Install PostgreSQL and `pgvector`.
    - Create a database (e.g., `ragstar_local_dev`).
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

    - **Required:** Set your `OPENAI_API_KEY`.
    - **Required:** Set `DATABASE_URL` to your local PostgreSQL connection string (e.g., `postgresql://user:password@host:port/dbname`). Ensure this matches the database you created. (The `.env.example` might have fallback variables like `DB_NAME_FALLBACK`, etc., which are used if `DATABASE_URL` is not set; for local setup, explicitly setting `DATABASE_URL` is clearer).
    - **Required:** Set `APP_HOST` (e.g., `localhost` or `127.0.0.1`).
    - **Ragstar Rules:** Rename `.ragstarrules.example.yml` to `.ragstarrules.yml` and customize if needed.
    - **Slack (Optional):** Configure `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET` if you plan to use the Slack integration.
    - **Other:** Review other variables like `RAGSTAR_LOG_LEVEL`, etc., and adjust if needed.

7.  **Run Database Migrations:**
    Apply database schema changes:

    ```bash
    uv run python manage.py migrate
    ```

8.  **Create a Superuser:**
    Create an admin account to access the Django admin interface:

    ```bash
    uv run python manage.py createsuperuser
    # Follow the prompts
    ```

9.  **Initialize your project:**
    This command sets up the necessary project configurations.

    ```bash
    uv run python manage.py init_project --method cloud
    # Or --method core, or --method local, depending on your dbt project setup.
    ```

10. **Run the Development Server:**

    ```bash
    uv run python manage.py runserver
    ```

    The application will typically be available at `http://<your_APP_HOST_value>:8000` (e.g., `http://localhost:8000`).

11. **Access the Django Admin & Embed Models:**
    Follow the same steps as in the Docker setup (steps 7 and 8) to access the admin interface (`http://<your_APP_HOST_value>:8000/admin`) and embed your models.

## Usage

After setup and initialization, you can interact with Ragstar.

### Using Docker Compose:

Most Django `manage.py` commands should be run **inside the `app` container** using `docker compose exec`:

```bash
# Example: Run database migrations (if not already done by entrypoint)
docker compose exec app uv run python manage.py migrate

# Example: Create a superuser (if not done during initial setup)
docker compose exec app uv run python manage.py createsuperuser

# Example: Initialize project
docker compose exec app uv run python manage.py init_project --method cloud
```

The application server is started automatically by `docker compose up`. Access it via `http://<your_APP_HOST_value>/admin`.

### Using Local Python Environment:

Run Django `manage.py` commands directly using `uv run` from your activated virtual environment:

```bash
# Example: Run the development server
uv run python manage.py runserver

# Example: Create a superuser
uv run python manage.py createsuperuser

# Example: Initialize project
uv run python manage.py init_project --method cloud
```

Access the application at `http://<your_APP_HOST_value>:8000` and the admin interface at `http://<your_APP_HOST_value>:8000/admin`.

### Core Django Management Commands

The primary way to manage and interact with the application (outside of the web interface) is through Django's `manage.py` script. Here are some key commands:

- **`uv run python manage.py migrate`**: Applies database migrations.
- **`uv run python manage.py createsuperuser`**: Creates an administrator account.
- **`uv run python manage.py init_project --method <cloud|core|local>`**: Initializes Ragstar with your project data (e.g., from dbt). This is crucial for setting up the knowledge base.
- **`uv run python manage.py runserver [host:port]`**: Starts the Django development web server.

Other functionalities, such as interpreting and embedding models, are primarily handled through the Django admin interface after logging in.

## Slack Integration (Optional)

Ragstar provides a Slack manifest to easily integrate its functionalities into your Slack workspace.

1.  Ensure `SLACK_SIGNING_SECRET` and `SLACK_BOT_TOKEN` are set in your `.env` file.
2.  Use the `.slack_manifest.example.json` file as a template to create a new Slack app.
3.  Follow Slack's documentation for creating an app from a manifest.
4.  This will enable features like asking questions and receiving answers directly within Slack (assuming the Slack integration is running as part of the Django application).

## Contributing

Contributions are welcome! Please follow standard fork-and-pull-request workflow.

## License

[MIT License](https://opensource.org/licenses/MIT)
