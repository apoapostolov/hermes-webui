#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Load .env if present (same logic as ctl.sh)
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env || true
    set +a
fi

HOST="${HERMES_WEBUI_HOST:-127.0.0.1}"
PORT="${HERMES_WEBUI_PORT:-8788}"

exec python3 "$REPO_ROOT/bootstrap.py" \
    --no-browser \
    --foreground \
    --host "$HOST" \
    "$PORT" \
    "$@"
