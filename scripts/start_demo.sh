#!/usr/bin/env bash
# Run API + Streamlit in one process tree (for container / single-service hosts).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:${PORT}}"
export BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
export BACKEND_PORT="${PORT}"

poetry run uvicorn app.api:app --host 0.0.0.0 --port "${PORT}" &
API_PID=$!

cleanup() {
  kill "${API_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Give the API a moment before Streamlit starts health-checking it.
sleep 2
poetry run streamlit run ui/app.py --server.port "${STREAMLIT_PORT}" --server.address 0.0.0.0
