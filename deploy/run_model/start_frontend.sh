#!/bin/bash
# Start Next.js frontend
# Usage: ./start_frontend.sh [port]

set -e

PORT="${1:-3000}"
API_URL="${API_URL:-http://localhost:8000}"

echo "Starting frontend on port $PORT"
echo "API URL: $API_URL"

cd "$(dirname "$0")/../.."

cd frontend/frontend_next/chatbot-ui

export NEXT_PUBLIC_API_URL="$API_URL"

npm run dev -- -p "$PORT"
