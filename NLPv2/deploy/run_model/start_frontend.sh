#!/bin/bash
# Start Streamlit frontend
# Usage: ./start_frontend.sh [port]

set -e

PORT="${1:-8501}"
API_URL="${RAG_API_URL:-http://localhost:8000}"

echo "Starting frontend on port $PORT"
echo "API URL: $API_URL"

cd "$(dirname "$0")/../.."

cd frontend/streamlit

export RAG_API_URL="$API_URL"

streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0
