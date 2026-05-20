#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
SIGNALS_PATH = DIST / "signals.json"
MLB_EXTRACTION_PATH = DIST / "hidden-gems" / "mlb_extraction_ledger.json"
APEX_PATH = DIST / "apex-extraction" / "apex_extraction.json"
OUT_PATH = DIST / "waiver_candidates.json"

MAX_CANDIDATES = 12
MAX_ROSTERED_PCT = 35

BLOCKED_WAIVER_PLAYER_IDS = {
    # Obvious rostered/market-priced stars should never enter Waiver/Open Market
    # without an explicit verified low-rostered market feed.
    "677951",  # Bobby Witt Jr.
    "656941",  # Kyle Schwarber
    "691406",  # Junior Caminero
    "695578",  # James Wood
}

BLOCKED_WAIVER_NAME_KEYS = {
    "bobby witt",
    "bobby witt jr.",
    "kyle schwarber",
    "junior caminero",
    "james wood",
}


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def as_list(value):
    return value if isinstance(value, list) else []


def player_key(row: dict) -> str:
    pid = str(row.get("player_id") or row.get("mlbam_id") or "").strip()
    if pid:
        return f"id:{pid}"
    name = str(row.get("player_name") or row.get("name") or "").strip().lower()
    team = str(row.get("team") or "").strip().lower()
    return f"name:{name}:{team}"


def pct(value, default=None):
    if value in (None, "", "—", "N/A", "NA"):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def has_verified_market_pct(row: dict) -> bool:
    return any(
        row.get(key) not in (None, "", "—", "N/A", "NA")
        for key in ("rostered_pct", "roster_pct", "ownership_pct", "ownership")
    )


def is_blocked_waiver_candidate(row: dict, player_name: str) -> bool:
    player_id = str(row.get("player_id") or row.get("mlbam_id") or "").strip()
    if player_id in BLOCKED_WAIVER_PLAYER_IDS:
        return True

    name_key = str(player_name or "").strip().lower()
    return name_key in BLOCKED_WAIVER_NAME_KEYS


def candidate_from_row(row: dict, source: str, rank: int) -> dict | None:
    player_name = row.get("player_name") or row.get("name") or row.get("full_name")
    if not player_name:
        return None

    if is_blocked_waiver_candidate(row, player_name):
        return None

    market_pct_verified = has_verified_market_pct(row)
    rostered_pct = pct(
        row.get("rostered_pct")
        or row.get("roster_pct")
        or row.get("ownership_pct")
        or row.get("ownership"),
        default=None,
    )

    # Missing market data is not waiver eligibility. It is unverified.
    # Do not let upstream signal surfaces imply open-market availability.
    if rostered_pct is None:
        return None

    if rostered_pct > MAX_ROSTERED_PCT:
        return None

    position = row.get("position") or row.get("pos") or ""
    team = row.get("team") or row.get("mlb_team") or ""

    source_label = source.replace("_", " ").upper()

    return {
        "player_id": str(row.get("player_id") or row.get("mlbam_id") or "").strip(),
        "player_name": str(player_name).strip(),
        "team": str(team).strip(),
        "position": str(position).strip(),
        "rostered_pct": rostered_pct,
        "market_pct_verified": market_pct_verified,
        "command": row.get("command") or "WATCHLIST PROVISION",
        "command_class": row.get("command_class") or "monitor",
        "market_status": row.get("market_status") or f"{source_label} CANDIDATE",
        "surface_profile": row.get("surface_profile") or f"Dynamic Waiver candidate sourced from {source_label}.",
        "forensic_trigger": row.get("forensic_trigger") or row.get("signal_thesis") or row.get("thesis") or "Fresh upstream signal entered the open-market candidate pool.",
        "verdict": row.get("verdict") or "Track until ownership, role, and deployment status confirm actionability.",
        "ownership_gate": row.get("ownership_gate") or f"≤{MAX_ROSTERED_PCT}%",
        "signal_window": row.get("signal_window") or "72H",
        "asset_type": row.get("asset_type") or ("Arm" if "P" in str(position).upper() else "Bat"),
        "market_defect": row.get("market_defect") or "Market Lag",
        "command_metric": row.get("command_metric") or "Track",
        "risk": row.get("risk") or "Med",
        "candidate_source": source,
        "source_rank": rank,
    }


def collect_signal_wall() -> list[dict]:
    data = read_json(SIGNALS_PATH) or {}
    rows = []
    for key in ("signals", "top_signals", "players", "pitchers", "hitters"):
        rows.extend(as_list(data.get(key)))
    return rows


def collect_mlb_extraction() -> list[dict]:
    data = read_json(MLB_EXTRACTION_PATH) or {}
    rows = []
    for key in ("top_signals", "pitcher_extractions", "hitter_extractions", "assets", "players"):
        rows.extend(as_list(data.get(key)))
    return rows


def collect_apex() -> list[dict]:
    data = read_json(APEX_PATH) or {}
    rows = []
    for key in ("apex_bats", "apex_arms", "assets", "players"):
        rows.extend(as_list(data.get(key)))
    return rows


def build_candidates() -> dict:
    source_rows = [
        ("signal_wall", collect_signal_wall()),
        ("mlb_extraction", collect_mlb_extraction()),
        ("apex_extraction", collect_apex()),
    ]

    candidates = []
    seen = set()

    for source, rows in source_rows:
        for rank, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            candidate = candidate_from_row(row, source, rank)
            if not candidate:
                continue

            key = player_key(candidate)
            if key in seen:
                continue

            seen.add(key)
            candidates.append(candidate)

            if len(candidates) >= MAX_CANDIDATES:
                break

        if len(candidates) >= MAX_CANDIDATES:
            break

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dynamic_candidate_file_v1",
        "source_files": [
            str(SIGNALS_PATH.relative_to(ROOT)),
            str(MLB_EXTRACTION_PATH.relative_to(ROOT)),
            str(APEX_PATH.relative_to(ROOT)),
        ],
        "candidate_count": len(candidates),
        "max_rostered_pct": MAX_ROSTERED_PCT,
        "eligibility_policy": "verified_market_pct_required",
        "blocked_player_count": len(BLOCKED_WAIVER_PLAYER_IDS),
        "candidates": candidates,
    }


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    payload = build_candidates()
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote Waiver candidate file -> {OUT_PATH}")
    print(f"Candidate count: {payload['candidate_count']}")


if __name__ == "__main__":
    main()
