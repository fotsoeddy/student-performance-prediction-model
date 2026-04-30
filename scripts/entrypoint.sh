#!/bin/bash
set -e

# ==============================================================================
# DEFAULT ENTRYPOINT SCRIPT
# ==============================================================================
# This script executes before the main application starts.
# You can use it to initialize tasks that should run inside the container, such as:
# - Running database migrations
# - Seeding initial data
# - Verifying external services are available
# ==============================================================================

echo "[ INFO ] Starting Student Performance Prediction Model initialization..."

# Example pre-start logic:
# echo "[ INFO ] Checking environment variables..."

echo "[ INFO ] Initialization complete. Starting the application server..."

# "exec" replaces the current shell process with the command specified in $1..$n.
# This passes signal handling directly to uvicorn.
exec "$@"
