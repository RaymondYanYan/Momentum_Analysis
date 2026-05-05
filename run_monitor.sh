#!/usr/bin/env bash
# Run the signal monitor with Telegram alerts.
# Usage: ./run_monitor.sh [--interval MINUTES] [--once]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR"

exec "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/alerts/monitor.py" "$@"
