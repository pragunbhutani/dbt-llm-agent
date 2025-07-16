# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install -U uv

# Copy project definition and lock file
COPY pyproject.toml uv.lock ./

# Copy source code
COPY . .

# Install project dependencies (this will create /app/.venv/)
RUN uv sync --no-cache

EXPOSE 8080

# Start server
CMD ["uv", "run", "python", "main.py"] 