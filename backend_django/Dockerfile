# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set work directory for the application
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the latest uv (e.g., 0.7.x line)
RUN pip install -U uv

# Copy project definition and lock file
COPY pyproject.toml uv.lock ./

# Copy the rest of the application code
COPY . .

# Install project dependencies (this will create /app/.venv/)
RUN uv sync --no-cache

# Expose the port the app runs on
EXPOSE 8000

# Specify the command to run on container start using the Python from the .venv
CMD ["uv", "run", "python", "-m", "uvicorn", "ragstar.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
