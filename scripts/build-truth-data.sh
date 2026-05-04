#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo ""
echo "== DiamondSignals Signal Wall Truth Build =="
echo "ROOT: $ROOT"

echo ""
echo "1/2 Building canonical scout universe..."
python3 dashboard/build_dashboard.py

echo ""
echo "2/2 Building player signal index..."
python3 dashboard/build_player_signal_index.py

echo ""
echo "Validating required truth files..."
python3 - <<'PY'
import json
from pathlib import Path

required = [
    ("dist/dossier_canon.json", "Canonical Scout Dossiers", 500),
    ("dist/scout_metrics.json", "Scout Metrics", 500),
    ("dist/player_index.json", "Player Index", 500),
    ("dist/admin/player_signal_index.json", "Player Signal Index", 25),
]

for file, label, min_count in required:
    p = Path(file)
    if not p.exists():
        raise SystemExit(f"❌ {label} missing: {file}")

    payload = json.loads(p.read_text())
    players = payload.get("players")

    if isinstance(players, dict):
        count = len(players)
    elif isinstance(players, list):
        count = len(players)
    else:
        count = 0

    generated_at = payload.get("generated_at", "UNKNOWN")

    if count < min_count:
        raise SystemExit(f"❌ {label} too small: {count} players in {file}")

    print(f"✅ {label}: {count} players // generated_at={generated_at}")

print("")
print("✅ Signal Wall truth build complete.")
PY
