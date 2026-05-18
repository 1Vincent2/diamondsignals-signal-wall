from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_FEED_PATH = REPO_ROOT / "dashboard" / "data" / "player_operational_status_overrides.json"

BLOCKED_OPERATIONAL_STATUSES = {
    "10-DAY IL",
    "15-DAY IL",
    "60-DAY IL",
    "7-DAY IL",
    "INJURED LIST",
    "SUSPENDED",
    "RESTRICTED LIST",
    "ADMINISTRATIVE LEAVE",
    "NON-DISCIPLINARY LEAVE",
    "DFA",
    "RELEASED",
    "RETIRED",
}

WATCHLIST_ONLY_STATUSES = {
    "MINORS",
    "OPTIONED",
    "AAA",
    "AA",
}


FALLBACK_OPERATIONAL_OVERRIDES: dict[str, dict[str, Any]] = {
    "luis l. ortiz": {
        "raw_status": "NON-DISCIPLINARY LEAVE",
        "status_reason": "Blocked from primary waiver deployment. Keep as surveillance-only until MLB status clears.",
        "status_source": "manual_fallback",
    },
}


def normalize_player_key(player_name: str) -> str:
    return str(player_name or "").lower().strip()


def normalize_status(raw_status: str) -> str:
    return str(raw_status or "ACTIVE").upper().strip()


def load_status_feed() -> dict[str, dict[str, Any]]:
    if not STATUS_FEED_PATH.exists():
        return {}

    try:
        data = json.loads(STATUS_FEED_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {
            normalize_player_key(key): value
            for key, value in data.items()
            if isinstance(value, dict)
        }
    except Exception:
        return {}


def get_operational_override(player_name: str) -> dict[str, Any]:
    key = normalize_player_key(player_name)
    feed = load_status_feed()

    if key in feed:
        return feed[key]

    return FALLBACK_OPERATIONAL_OVERRIDES.get(key, {})


def classify_status(raw_status: str) -> dict[str, str]:
    status = normalize_status(raw_status)

    if status in BLOCKED_OPERATIONAL_STATUSES:
        return {
            "deployment_state": "DEPLOYMENT_LOCKED",
            "deployment_label": "SIGNAL PRESENT // DEPLOYMENT LOCKED",
            "operational_status": status,
            "card_action": "SURVEILLANCE ONLY",
            "visibility_state": "deployment_locked",
        }

    if status in WATCHLIST_ONLY_STATUSES:
        return {
            "deployment_state": "WATCHLIST_ONLY",
            "deployment_label": "WATCHLIST ONLY",
            "operational_status": status,
            "card_action": "TRACK ONLY",
            "visibility_state": "primary",
        }

    if status in {"UNKNOWN", "UNVERIFIED", "ELIGIBILITY UNKNOWN"}:
        return {
            "deployment_state": "ELIGIBILITY_UNVERIFIED",
            "deployment_label": "ELIGIBILITY UNVERIFIED",
            "operational_status": status,
            "card_action": "VERIFY BEFORE DEPLOYMENT",
            "visibility_state": "deployment_locked",
        }

    return {
        "deployment_state": "DEPLOYMENT_CLEAR",
        "deployment_label": "DEPLOYMENT CLEAR",
        "operational_status": status,
        "card_action": "OPEN PERFORMANCE AUDIT",
        "visibility_state": "primary",
    }


def rebuild_search_blob(row: dict[str, Any]) -> None:
    row["search_blob"] = " ".join([
        str(row.get("player_name", "")),
        str(row.get("team", "")),
        str(row.get("position", "")),
        str(row.get("command", "")),
        str(row.get("market_status", "")),
        str(row.get("deployment_label", "")),
        str(row.get("operational_status", "")),
    ]).lower()


def apply_operational_status(row: dict[str, Any]) -> dict[str, Any]:
    override = get_operational_override(row.get("player_name", ""))

    raw_status = override.get(
        "raw_status",
        row.get("raw_status", row.get("operational_status", "ACTIVE")),
    )
    classified = classify_status(raw_status)

    row.update(classified)
    row["status_reason"] = override.get(
        "status_reason",
        row.get("status_reason", "Status clear at build time."),
    )
    row["status_source"] = override.get(
        "status_source",
        row.get("status_source", "default_active"),
    )

    # Keep rendered metric tiles synchronized with the operational-status layer.
    for metric in row.get("metrics", []):
        if metric.get("label") == "Status Source":
            metric["value"] = row["status_source"]

    rebuild_search_blob(row)
    return row
