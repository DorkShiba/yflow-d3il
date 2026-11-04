#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VIEWER=$(CDPATH= cd -- "$HERE/../.." && pwd)
export VISUALIZER_STORE="$HERE/viewer_store"
exec "$VIEWER/.venv-visualizer/bin/python" -m uvicorn visualizer.app:app --app-dir "$VIEWER" --host 127.0.0.1 --port "${PORT:-8765}"
