#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
  echo "Usage: dashboard/player_forensics.sh player name"
  exit 1
fi

python3 dashboard/build_player_signal_index.py >/dev/null
python3 dashboard/search_player_signal_index.py "$@"
