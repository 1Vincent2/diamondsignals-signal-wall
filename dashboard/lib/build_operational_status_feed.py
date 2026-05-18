from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "dashboard" / "data" / "status" / "mlb_operational_status_feed.json"

MLB_API = "https://statsapi.mlb.com/api/v1"
ROSTER_TYPES = ["active", "40Man", "fullSeason"]
ACTIVE_STATUSES = {"ACTIVE", "A", ""}


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_key(name: str) -> str:
    return str(name or "").lower().strip()


def normalize_status(value: str) -> str:
    return str(value or "").upper().strip()


def merge_player(feed: dict[str, dict[str, Any]], row: dict[str, Any]) -> None:
    name = row.get("player_name")
    if not name:
        return

    key = normalize_key(name)
    incoming_priority = int(row.get("priority", 50))
    existing_priority = int(feed.get(key, {}).get("priority", 999))

    if key not in feed or incoming_priority < existing_priority:
        feed[key] = row


def extract_status(entry: dict[str, Any]) -> str:
    status = entry.get("status") or {}
    return (
        status.get("description")
        or status.get("code")
        or entry.get("statusDescription")
        or entry.get("statusCode")
        or "ACTIVE"
    )


def ingest_mlb_rosters(feed: dict[str, dict[str, Any]], diagnostics: list[str]) -> None:
    teams_payload = fetch_json(f"{MLB_API}/teams?sportId=1")
    teams = teams_payload.get("teams", [])

    for team in teams:
        team_id = team.get("id")
        team_name = team.get("name", "")
        if not team_id:
            continue

        for roster_type in ROSTER_TYPES:
            try:
                roster_payload = fetch_json(f"{MLB_API}/teams/{team_id}/roster?rosterType={roster_type}")
            except Exception as exc:
                diagnostics.append(f"mlb_roster_fetch_failed team={team_id} rosterType={roster_type}: {exc}")
                continue

            for entry in roster_payload.get("roster", []):
                person = entry.get("person") or {}
                player_name = person.get("fullName")
                if not player_name:
                    continue

                raw_status = extract_status(entry)
                normalized = normalize_status(raw_status)

                if normalized in ACTIVE_STATUSES:
                    continue

                merge_player(feed, {
                    "player_name": player_name,
                    "team": team_name,
                    "team_id": team_id,
                    "raw_status": raw_status,
                    "source": "mlb_statsapi_roster",
                    "status_reason": f"MLB roster feed reports {raw_status}. Verify before deployment.",
                    "roster_type": roster_type,
                    "priority": 10,
                })


def ingest_supabase_truth(feed: dict[str, dict[str, Any]], diagnostics: list[str]) -> None:
    """
    Optional pipeline. Activates only when SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY exist.

    Expected future table/view options:
    - player_operational_status
    - player_truth_status
    - mlb_player_status
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        diagnostics.append("supabase_status_feed_skipped_missing_env")
        return

    try:
        from supabase import create_client
    except Exception as exc:
        diagnostics.append(f"supabase_status_feed_skipped_missing_client: {exc}")
        return

    candidate_tables = [
        "player_operational_status",
        "player_truth_status",
        "mlb_player_status",
    ]

    try:
        client = create_client(url, key)
    except Exception as exc:
        diagnostics.append(f"supabase_status_client_failed: {exc}")
        return

    for table in candidate_tables:
        try:
            response = client.table(table).select("*").execute()
            rows = response.data or []
        except Exception as exc:
            diagnostics.append(f"supabase_status_table_unavailable {table}: {exc}")
            continue

        for row in rows:
            player_name = row.get("player_name") or row.get("name") or row.get("full_name")
            raw_status = row.get("raw_status") or row.get("status") or row.get("injury_status") or "UNKNOWN"

            if not player_name:
                continue

            if normalize_status(raw_status) in ACTIVE_STATUSES:
                continue

            merge_player(feed, {
                "player_name": player_name,
                "team": row.get("team", ""),
                "raw_status": raw_status,
                "source": f"supabase_{table}",
                "status_reason": row.get("status_reason") or f"Supabase truth table {table} reports {raw_status}.",
                "priority": 5,
            })

        diagnostics.append(f"supabase_status_table_loaded {table}: {len(rows)} rows")
        break



def ingest_mlb_transactions(feed: dict[str, dict[str, Any]], diagnostics: list[str]) -> None:
    """
    Transaction layer catches restrictions/leave designations that may not appear cleanly
    in rosterType payloads.
    """
    today = datetime.now(timezone.utc).date()
    start_date = today.replace(month=1, day=1).isoformat()
    end_date = today.isoformat()

    try:
        tx_payload = fetch_json(f"{MLB_API}/transactions?sportId=1&startDate={start_date}&endDate={end_date}")
    except Exception as exc:
        diagnostics.append(f"mlb_transactions_fetch_failed: {exc}")
        return

    transactions = tx_payload.get("transactions", []) or []
    diagnostics.append(f"mlb_transactions_loaded: {len(transactions)} rows")

    lock_terms = [
        "restricted list",
        "administrative leave",
        "non-disciplinary",
        "suspended",
        "disciplinary",
        "bereavement",
        "family medical emergency",
    ]

    watch_terms = [
        "rehab assignment",
        "optioned",
        "assigned to",
        "minor league",
    ]

    for tx in transactions:
        person = tx.get("person") or {}
        player_name = person.get("fullName") or tx.get("playerName")
        if not player_name:
            continue

        description = " ".join([
            str(tx.get("description", "")),
            str(tx.get("typeDesc", "")),
            str(tx.get("typeCode", "")),
        ]).strip()

        lowered = description.lower()
        if not lowered:
            continue

        if any(term in lowered for term in lock_terms):
            raw_status = "DEPLOYMENT_LOCKED_TRANSACTION"
            priority = 6
        elif any(term in lowered for term in watch_terms):
            raw_status = "WATCHLIST_ONLY_TRANSACTION"
            priority = 20
        else:
            continue

        merge_player(feed, {
            "player_name": player_name,
            "team": (tx.get("toTeam") or tx.get("fromTeam") or {}).get("name", ""),
            "raw_status": raw_status,
            "source": "mlb_statsapi_transactions",
            "status_reason": description or f"MLB transaction feed reports {raw_status}.",
            "transaction_date": tx.get("date"),
            "priority": priority,
        })

def main() -> None:
    diagnostics: list[str] = []
    feed: dict[str, dict[str, Any]] = {}

    ingest_supabase_truth(feed, diagnostics)
    ingest_mlb_transactions(feed, diagnostics)
    ingest_mlb_rosters(feed, diagnostics)

    players = sorted(
        [{k: v for k, v in row.items() if k != "priority"} for row in feed.values()],
        key=lambda row: row["player_name"],
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "merged_operational_status_feed",
        "source_priority": [
            "supabase_truth_tables_if_available",
            "mlb_statsapi_transactions",
            "mlb_statsapi_roster",
        ],
        "player_count": len(players),
        "players": players,
        "diagnostics": diagnostics,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote operational status feed -> {OUTPUT_PATH}")
    print(f"Players: {len(players)}")
    print("Diagnostics:")
    for item in diagnostics:
        print(f"- {item}")


if __name__ == "__main__":
    main()
