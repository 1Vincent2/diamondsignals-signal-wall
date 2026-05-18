#!/usr/bin/env bash
set -euo pipefail

REPORT="${1:-}"

case "$REPORT" in
  waiver-wire|waiver)
    PYTHONPATH=. python3 dashboard/lib/build_operational_status_feed.py
    PYTHONPATH=. python3 dashboard/build_waiver_wire.py
    ;;
  all)
    PYTHONPATH=. python3 dashboard/build_all.py
    ;;
  *)
    echo "Usage: scripts/build-report.sh waiver-wire|all"
    exit 1
    ;;
esac
