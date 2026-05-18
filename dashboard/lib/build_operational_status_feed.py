from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

OUTPUT_PATH = Path("dashboard/data/status/mlb_operational_status_feed.json")

# TEMP STATIC FEED
# Replace later with MLB/Supabase/API ingestion.
PLAYERS = [
    {
        "player_name": "Luis L. Ortiz",
        "raw_status": "NON-DISCIPLINARY LEAVE",
        "source": "temporary_seed"
    }
]

def main() -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "player_count": len(PLAYERS),
        "players": PLAYERS,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote operational status feed -> {OUTPUT_PATH}")
    print(f"Players: {len(PLAYERS)}")

if __name__ == "__main__":
    main()
