#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import date
import sys

OUT_PATH = Path("dist/aaa_hitter_refresh.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.build_call_up_v2 import load_source_frame

def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df, week_starts = load_source_frame()

    players = []
    if df is not None and not df.empty and "pa" in df.columns:
        hitters = df[df["pa"].notna()].copy()

        for _, row in hitters.iterrows():
            players.append(
                {
                    "snapshot_date": str(row.get("week_start") or date.today().isoformat()),
                    "player_name": row.get("player_name"),
                    "player_id": row.get("player_id"),
                    "org": row.get("org"),
                    "level": row.get("level", "AAA"),
                    "pa": float(row.get("pa") or 0),
                    "bb": float(row.get("bb") or 0),
                    "so": float(row.get("so") or 0),
                    "hr": float(row.get("hr") or 0),
                    "iso": float(row.get("iso") or 0),
                }
            )

    unique_weeks = sorted({str(w) for w in week_starts if w}, reverse=True)
    latest_snapshot = unique_weeks[0] if unique_weeks else None
    is_single_snapshot = len(unique_weeks) <= 1

    payload = {
        "generated_at": date.today().isoformat(),
        "status": "current_aaa_source_exported",
        "freshness": {
            "latest_snapshot": latest_snapshot,
            "unique_weeks": unique_weeks,
            "is_single_snapshot": is_single_snapshot,
            "delta_ready": not is_single_snapshot,
            "blocking_reason": "Only one AAA snapshot is available; delta engine cannot run yet." if is_single_snapshot else None,
        },
        "week_starts": week_starts,
        "player_count": len(players),
        "players": players,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"player_count={len(players)}")
    print(f"latest_snapshot={latest_snapshot}")
    print(f"unique_weeks={unique_weeks}")
    print(f"delta_ready={not is_single_snapshot}")

if __name__ == "__main__":
    main()
