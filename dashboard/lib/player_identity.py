from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = ROOT / "dist"
CANONICAL_UNIVERSE_PATH = DIST_DIR / "canonical_player_universe.json"


def normalize_name(value: Any) -> str:
    """Normalize a player name for loose lookup matching."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def build_headshot_url(player_id: int | str | None) -> str:
    """Return the standard MLB headshot URL for a canonical MLBAM player id."""
    if not player_id:
        return ""

    try:
        pid = int(str(player_id).strip())
    except Exception:
        return ""

    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"w_180,q_100/v1/people/{pid}/headshot/67/current"
    )


def build_scout_url(player_id: int | str | None) -> str:
    """Return the local Scout Dossier URL for a canonical player id."""
    if not player_id:
        return "#"

    pid = str(player_id).strip()
    return f"/scout/{pid}/" if pid else "#"


def load_canonical_player_universe(path: Path | None = None) -> dict[str, dict]:
    """
    Load canonical_player_universe.json as a player_id keyed dictionary.

    Supports both:
    - {"players": {"123": {...}}}
    - {"players": [{...}, {...}]}
    """
    path = path or CANONICAL_UNIVERSE_PATH

    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    players = payload.get("players", {}) if isinstance(payload, dict) else {}

    if isinstance(players, dict):
        return {
            str(pid).strip(): dict(row)
            for pid, row in players.items()
            if str(pid).strip() and isinstance(row, dict)
        }

    if isinstance(players, list):
        out: dict[str, dict] = {}
        for row in players:
            if not isinstance(row, dict):
                continue
            pid = (
                row.get("player_id")
                or row.get("mlb_id")
                or row.get("mlbam_id")
                or row.get("id")
            )
            pid = str(pid or "").strip()
            if pid:
                out[pid] = dict(row)
        return out

    return {}


def build_name_lookup(players: dict[str, dict]) -> dict[str, str]:
    """Build normalized player_name/full_name/name -> player_id lookup."""
    lookup: dict[str, str] = {}

    for pid, row in players.items():
        for key in ["player_name", "full_name", "name"]:
            name = normalize_name(row.get(key))
            if name and name not in lookup:
                lookup[name] = pid

    return lookup


def resolve_player_identity(
    player_id: int | str | None = None,
    player_name: str | None = None,
    team: str | None = None,
    players: dict[str, dict] | None = None,
    name_lookup: dict[str, str] | None = None,
) -> dict:
    """
    Resolve a player identity against canonical_player_universe.

    Returns a stable identity object even when canonical data is unavailable.
    This function does not generate metrics or signals; it only owns identity.
    """
    players = players if players is not None else load_canonical_player_universe()
    name_lookup = name_lookup if name_lookup is not None else build_name_lookup(players)

    pid = str(player_id or "").strip()
    row = players.get(pid) if pid else None

    if row is None and player_name:
        resolved_pid = name_lookup.get(normalize_name(player_name))
        if resolved_pid:
            pid = resolved_pid
            row = players.get(pid)

    row = row or {}

    resolved_name = (
        row.get("player_name")
        or row.get("full_name")
        or row.get("name")
        or player_name
        or "Unknown Player"
    )

    resolved_team = (
        row.get("team")
        or row.get("current_team")
        or row.get("team_name")
        or team
        or ""
    )

    resolved_position = row.get("position") or row.get("primary_position") or ""

    resolved_id = str(
        row.get("player_id")
        or row.get("mlb_id")
        or row.get("mlbam_id")
        or pid
        or ""
    ).strip()

    return {
        "player_id": resolved_id,
        "player_name": str(resolved_name or "Unknown Player").strip(),
        "team": str(resolved_team or "").strip(),
        "current_team": str(resolved_team or "").strip(),
        "position": str(resolved_position or "").strip(),
        "headshot_url": row.get("headshot_url") or build_headshot_url(resolved_id),
        "scout_url": build_scout_url(resolved_id),
        "identity_source": "canonical_player_universe" if row else "fallback_input",
    }


def canonicalize_player_row(
    row: dict,
    players: dict[str, dict] | None = None,
    name_lookup: dict[str, str] | None = None,
) -> dict:
    """Return row plus canonical identity fields."""
    identity = resolve_player_identity(
        player_id=(
            row.get("player_id")
            or row.get("resolved_player_id")
            or row.get("mlb_id")
            or row.get("mlbam_id")
            or row.get("batter")
            or row.get("pitcher")
            or row.get("id")
        ),
        player_name=row.get("player_name") or row.get("full_name") or row.get("name"),
        team=row.get("team") or row.get("current_team") or row.get("team_name"),
        players=players,
        name_lookup=name_lookup,
    )

    out = dict(row)
    out.update(identity)
    out["resolved_player_id"] = identity.get("player_id", "")
    return out
