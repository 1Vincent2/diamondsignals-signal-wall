#!/usr/bin/env python3
# AAA transactions probe for DiamondSignals SCOUT
# Purpose:
# 1) Pull recent AAA-related MLB transactions from the MLB Stats API
# 2) Classify moves into arrival_to_mlb / returned_to_aaa / rehab_assignment
# 3) Enrich with MLB people-search metadata
# 4) Produce a scout-relevant filtered list for future Recent Arrivals / SCOUT pages
#
# Output files:
# - dist/aaa_transactions_probe.json
# - dist/aaa_transactions_scout_only.json
# - dist/aaa_transactions_scout_names.txt

import json
from pathlib import Path

import requests

AAA_KEYWORDS = [
    "tides", "bees", "storm chasers", "indians", "isotopes", "railriders",
    "red wings", "stripers", "sounds", "comets", "bats", "saints", "express",
    "aviators", "space cowboys", "el paso", "mud hens", "clippers", "ironpigs",
    "jumbo shrimp", "knights", "round rock", "sugar land", "durham", "reno",
    "buffalo", "syracuse", "scranton", "omaha", "norfolk", "st. paul", "louisville",
    "nashville", "jacksonville", "charlotte", "toledo", "columbus", "lehigh valley",
    "gwinnett", "albuquerque", "las vegas", "tacoma", "salt lake", "memphis", "rochester",
    "worcester", "oklahoma city"
]

PROMOTION_TYPES = {
    "Selected",
    "Recalled",
    "Purchased",
    "Promoted",
}

RETURN_TYPES = {
    "Optioned",
    "Outrighted",
    "Assigned",
}

from datetime import datetime, timedelta

LOOKBACK_DAYS = 14

def build_date_window(days: int) -> list[str]:
    today = datetime.utcnow().date()
    start = today - timedelta(days=days - 1)
    return [
        (start + timedelta(days=i)).isoformat()
        for i in range(days)
    ]

OUTPUT_PATH = Path("dist/aaa_transactions_probe.json")


def fetch_transactions(date_str: str) -> list[dict]:
    url = "https://statsapi.mlb.com/api/v1/transactions"
    params = {"sportId": 1, "date": date_str}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("transactions", [])


def is_aaa_related(tx: dict) -> bool:
    from_team = str(tx.get("fromTeam", {}).get("name", ""))
    to_team = str(tx.get("toTeam", {}).get("name", ""))
    blob = f"{from_team} {to_team}".lower()
    return any(k in blob for k in AAA_KEYWORDS)


def classify_transaction(tx: dict) -> str:
    t = str(tx.get("typeDesc", "")).strip()
    desc = str(tx.get("description", "")).lower()

    if "rehab assignment" in desc:
        return "rehab_assignment"

    if t in PROMOTION_TYPES:
        return "arrival_to_mlb"

    if t in RETURN_TYPES:
        return "returned_to_aaa"

    return "other"


def simplify(tx: dict) -> dict:
    return {
        "person": tx.get("person", {}).get("fullName"),
        "typeDesc": tx.get("typeDesc"),
        "classification": classify_transaction(tx),
        "fromTeam": tx.get("fromTeam", {}).get("name"),
        "toTeam": tx.get("toTeam", {}).get("name"),
        "description": tx.get("description"),
        "date": tx.get("date"),
    }


def fetch_person_meta(name: str) -> dict:
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/people/search",
            params={"names": name},
            timeout=20,
        )
        resp.raise_for_status()
        people = resp.json().get("people", []) or []

        if not people:
            return {
                "mlb_id": None,
                "currentAge": None,
                "draftYear": None,
                "mlbDebutDate": None,
                "active": None,
            }

        p = people[0]
        return {
            "mlb_id": p.get("id"),
            "currentAge": p.get("currentAge"),
            "draftYear": p.get("draftYear"),
            "mlbDebutDate": p.get("mlbDebutDate"),
            "active": p.get("active"),
        }
    except Exception:
        return {
            "mlb_id": None,
            "currentAge": None,
            "draftYear": None,
            "mlbDebutDate": None,
            "active": None,
        }


def is_scout_relevant(move: dict) -> bool:
    classification = move.get("classification")
    age = move.get("currentAge")
    debut = move.get("mlbDebutDate")
    draft_year = move.get("draftYear")

    if classification == "rehab_assignment":
        return False

    if classification == "arrival_to_mlb":
        if debut is None:
            return True
        if isinstance(debut, str) and debut >= "2025-01-01":
            return True
        if age is not None and age <= 24:
            return True
        if draft_year is not None and draft_year >= 2021:
            return True
        return False

    if classification == "returned_to_aaa":
        if debut is None:
            return True
        if age is not None and age <= 24:
            return True
        if draft_year is not None and draft_year >= 2021:
            return True
        return False

    return False


def build_summary(all_moves: list[dict], scout_moves: list[dict]) -> dict:
    return {
        "arrival_to_mlb": sum(1 for m in all_moves if m["classification"] == "arrival_to_mlb"),
        "returned_to_aaa": sum(1 for m in all_moves if m["classification"] == "returned_to_aaa"),
        "rehab_assignment": sum(1 for m in all_moves if m["classification"] == "rehab_assignment"),
        "other": sum(1 for m in all_moves if m["classification"] == "other"),
        "total": len(all_moves),
        "scout_relevant": len(scout_moves),
    }


def main() -> None:
    all_moves: list[dict] = []

    for date_str in build_date_window(LOOKBACK_DAYS):
        transactions = fetch_transactions(date_str)
        aaa_moves = [simplify(t) for t in transactions if is_aaa_related(t)]
        all_moves.extend(aaa_moves)

    seen: dict[str, dict] = {}
    for move in all_moves:
        name = move.get("person")
        if not name:
            continue

        if name not in seen:
            seen[name] = fetch_person_meta(name)

        move.update(seen[name])

    scout_moves = [m for m in all_moves if is_scout_relevant(m)]
    summary = build_summary(all_moves, scout_moves)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "summary": summary,
                "moves": all_moves,
                "scout_relevant_moves": scout_moves,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scout_only_path = Path("dist/aaa_transactions_scout_only.json")
    scout_only_path.write_text(
        json.dumps(
            {
                "summary": {
                    "scout_relevant": len(scout_moves),
                },
                "scout_relevant_moves": scout_moves,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scout_names_path = Path("dist/aaa_transactions_scout_names.txt")
    scout_names = [
        f"{m['date']} | {m['person']} | {m['classification']} | {m['fromTeam']} -> {m['toTeam']}"
        for m in scout_moves
    ]
    scout_names_path.write_text("\n".join(scout_names), encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {scout_only_path}")
    print(f"Wrote {scout_names_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()