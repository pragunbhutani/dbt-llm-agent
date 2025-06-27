# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install -U uv

# Copy pyproject and lockfile
COPY pyproject.toml uv.lock ./

# Copy application code
COPY . .

# Install Python dependencies
RUN uv sync --no-cache

# Expose MCP port
EXPOSE 9000

# Run the MCP FastAPI app
CMD ["uv", "run", "python", "-m", "uvicorn", "mcp_server.main:app", "--host", "0.0.0.0", "--port", "9000"] 