from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "canonical_player_universe.json"


def load_json(path: Path, fallback):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[UNIVERSE] failed to load {path}: {exc}")
    return fallback


def norm_id(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def upsert_player(players, row, source):
    pid = norm_id(
        row.get("player_id")
        or row.get("mlb_id")
        or row.get("batter")
        or row.get("pitcher")
        or row.get("id")
    )
    if not pid or not pid.isdigit():
        return

    existing = players.setdefault(pid, {
        "player_id": pid,
        "player_name": "",
        "team": "",
        "current_team": "",
        "position": "",
        "level": "",
        "status": "",
        "headshot_url": f"https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/{pid}/headshot/67/current",
        "sources": [],
    })

    name = (
        row.get("player_name")
        or row.get("full_name")
        or row.get("name")
        or row.get("display_name")
        or ""
    )
    team = row.get("team") or row.get("current_team") or row.get("org") or row.get("org_mlb_team") or ""
    position = row.get("position") or row.get("position_group") or row.get("primary_pos") or ""
    level = row.get("level") or ""
    status = row.get("status") or row.get("availability") or ""

    if name and not existing["player_name"]:
        existing["player_name"] = str(name).strip()
    if team and not existing["team"]:
        existing["team"] = str(team).strip()
        existing["current_team"] = str(team).strip()
    if position and not existing["position"]:
        existing["position"] = str(position).strip()
    if level and not existing["level"]:
        existing["level"] = str(level).strip()
    if status and not existing["status"]:
        existing["status"] = str(status).strip()

    if source not in existing["sources"]:
        existing["sources"].append(source)


def ingest_payload(players, path, source):
    payload = load_json(path, None)
    if not payload:
        return

    rows = []
    if isinstance(payload, dict):
        for key in ["players", "all_assets", "assets", "hitters", "pitchers", "cards", "signals"]:
            value = payload.get(key)
            if isinstance(value, dict):
                for pid, row in value.items():
                    if isinstance(row, dict):
                        row = {"player_id": pid, **row}
                        rows.append(row)
            elif isinstance(value, list):
                rows.extend([r for r in value if isinstance(r, dict)])
    elif isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]

    for row in rows:
        upsert_player(players, row, source)

    print(f"[UNIVERSE] {source}: considered {len(rows)} rows")


def build():
    players = {}

    sources = [
        (DIST / "dossier_canon.json", "dossier_canon"),
        (DIST / "waiver_wire.json", "waiver_wire"),
        (DIST / "signals.json", "signal_wall"),
        (DIST / "hidden-gems" / "mlb_extraction_ledger.json", "mlb_extraction_ledger"),
        (DIST / "apex-extraction" / "apex_extraction.json", "apex_extraction"),
        (DIST / "stuff_disruption_feed.json", "stuff_disruption"),
        (DIST / "velocity_decay_monitor.json", "velocity_decay"),
        (DIST / "typical-call-up" / "promotion_watch.json", "promotion_watch"),
        (DIST / "admin" / "player_signal_index.json", "player_signal_index"),
    ]

    for path, source in sources:
        ingest_payload(players, path, source)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "player_count": len(players),
        "players": dict(sorted(players.items(), key=lambda kv: kv[1].get("player_name") or kv[0])),
    }

    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[UNIVERSE] wrote {OUT} with {len(players)} players")


if __name__ == "__main__":
    build()
