#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv-visualizer/bin/python ]; then
  python3 -m venv .venv-visualizer
  .venv-visualizer/bin/python -m pip install -r visualizer/requirements.txt
fi
exec .venv-visualizer/bin/python -m uvicorn visualizer.app:app --host 127.0.0.1 --port "${PORT:-8765}"
