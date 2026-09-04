#!/usr/bin/env bash
# Single public service: internal API on 8000, Streamlit on $PORT.
set -euo pipefail
cd "$(dirname "$0")/.."

export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8000}"
PUBLIC_PORT="${PORT:-8501}"

python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

cleanup() {
  kill "${API_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1)" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

exec streamlit run ui/app.py --server.port "${PUBLIC_PORT}" --server.address 0.0.0.0 --browser.gatherUsageStats false
