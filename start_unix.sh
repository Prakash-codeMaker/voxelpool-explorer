#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python3 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
if [ ! -d frontend/node_modules ]; then
  (cd frontend && npm install)
fi
echo "Starting API in the background..."
(cd backend && ../backend/.venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000) &
API_PID=$!
cleanup() { kill "$API_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
sleep 2
echo "Starting Vite. Open http://localhost:5173"
cd frontend
npm run dev
