#!/bin/bash
set -e

echo "[INFO] Starting Student Performance Prediction Model API..."

mkdir -p /app/models /app/reports /app/data/processed

echo "[INFO] Starting server on port ${PORT:-8006}..."

exec "$@"