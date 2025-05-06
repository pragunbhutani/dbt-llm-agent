#!/bin/bash
set -e

# Run migrations
echo "Running migrations: ragstar init-db"
ragstar init-db

# Start the server
echo "Starting server: ragstar serve"
exec ragstar serve