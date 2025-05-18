# Use an official Python runtime as a parent image
# ARG PYTHON_VERSION=3.10
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project definition and lock file (if using)
COPY pyproject.toml ./
# If you have a uv.lock file and want to use it, uncomment the next line
# COPY uv.lock ./

# Copy the rest of the application code BEFORE installing dependencies
# This ensures README.md and other project files are available for the build
COPY . .

# Install project dependencies using uv from pyproject.toml
# This installs the current project defined in pyproject.toml
RUN uv pip install --system --no-cache .
# If you prefer to use uv.lock, comment the line above and uncomment the line below
# RUN uv sync --system --no-cache uv.lock

# Expose the port the app runs on
EXPOSE 8000

# Specify the command to run on container start using Uvicorn
CMD ["uvicorn", "ragstar.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
