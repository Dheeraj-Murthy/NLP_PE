#!/bin/bash
# Start RAG backend API server
# Usage: ./start_api.sh [port]

set -e

PORT="${1:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "Starting Legal RAG API on $HOST:$PORT"

cd "$(dirname "$0")/../.."

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-legal_rag}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"

DB_CONNECTION="host=$DB_HOST port=$DB_PORT dbname=$DB_NAME user=$DB_USER password=$DB_PASSWORD"

export DB_CONNECTION

echo "DB: $DB_HOST:$DB_PORT/$DB_NAME"

cd backend
python api.py --host "$HOST" --port "$PORT"
