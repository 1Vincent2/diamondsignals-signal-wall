from __future__ import annotations

from typing import Dict, Any


OPERATIONAL_OVERRIDES: dict[str, dict[str, Any]] = {
    "luis l. ortiz": {
        "deployment_state": "DEPLOYMENT_LOCKED",
        "deployment_label": "SIGNAL PRESENT // DEPLOYMENT LOCKED",
        "operational_status": "NON-DISCIPLINARY LEAVE",
        "status_reason": "Blocked from primary waiver deployment. Keep as surveillance-only until MLB status clears.",
        "card_action": "SURVEILLANCE ONLY",
        "visibility_state": "deployment_locked",
    },
}


def normalize_player_key(player_name: str) -> str:
    return str(player_name or "").lower().strip()


def apply_operational_status(row: dict[str, Any]) -> dict[str, Any]:
    key = normalize_player_key(row.get("player_name", ""))
    override = OPERATIONAL_OVERRIDES.get(key)

    if override:
        row.update(override)

    row["search_blob"] = " ".join([
        str(row.get("player_name", "")),
        str(row.get("team", "")),
        str(row.get("position", "")),
        str(row.get("command", "")),
        str(row.get("market_status", "")),
        str(row.get("deployment_label", "")),
        str(row.get("operational_status", "")),
    ]).lower()

    return row
