# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy only dependency files to leverage Docker layer caching
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry install --no-dev --no-root

# Copy the rest of the application code
COPY . .

# Install the project itself
RUN poetry install --no-dev

# Expose the port the app runs on (replace 8000 if your app uses a different port)
EXPOSE 8000

# Specify the command to run on container start (replace with your actual run command)
# Example: CMD ["poetry", "run", "uvicorn", "ragstar.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
# You might need to adjust this based on how your API server is started.
CMD ["tail", "-f", "/dev/null"] # Placeholder command, replace this! 