#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REPORT_PAYLOADS = {
    "promotion_watch": {
        "path": "dist/typical-call-up/promotion_watch.json",
        "required_lists": [
            "pitchers_72hr",
            "hitters_72hr",
            "pitchers_14day",
            "hitters_14day",
            "recent_arrivals",
            "depth_radar",
        ],
        "critical_fields": ["name", "player_name", "display_name", "team", "level", "signal", "status"],
    },
    "ivb_heat_map": {
        "path": "dist/ivb_heat_map.json",
        "required_lists": ["heat_cards", "climbers", "fallers", "entered_apex", "zone_shift"],
        "critical_fields": ["name", "player_name", "pitcher_name", "team", "status"],
    },
    "velocity_decay": {
        "path": "dist/velocity_decay_monitor.json",
        "required_lists": ["cards"],
        "critical_fields": ["name", "player_name", "pitcher_name", "team", "risk", "status"],
    },
    "stuff_disruption": {
        "path": "dist/stuff_disruption_feed.json",
        "required_lists": ["cards"],
        "critical_fields": ["name", "player_name", "pitcher_name", "team", "signal", "status"],
    },
    "mlb_extraction": {
        "path": "dist/hidden-gems/mlb_extraction_ledger.json",
        "required_lists": ["top_signals", "top_pitchers", "top_hitters"],
        "critical_fields": ["name", "player_name", "team", "status", "signal"],
    },
    "apex_extraction": {
        "path": "dist/apex-extraction/apex_extraction.json",
        "required_lists": ["top_signals", "apex_bats", "apex_arms"],
        "critical_fields": ["name", "player_name", "team", "status", "signal"],
    },
    "waiver_wire": {
        "path": "dist/waiver_wire.json",
        "required_lists": ["assets", "all_assets"],
        "critical_fields": ["name", "player_name", "team", "status", "verdict", "command"],
    },
    "depth_radar": {
        "path": "dist/depth_radar_refresh.json",
        "required_lists": ["top_rows", "hitters", "pitchers"],
        "critical_fields": ["name", "player_name", "team", "level", "status", "signal"],
    },
    "kinetic_drift": {
        "path": "dist/admin/kinetic_drift_signals.json",
        "required_lists": ["signals"],
        "critical_fields": ["name", "player_name", "pitcher_name", "team", "status", "signal"],
    },
}

ALLOWED_EMPTY_LISTS = {
    ("waiver_wire", "frozen_assets"),
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def is_bad_number(value: Any) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def walk_bad_values(value: Any, path: str, problems: list[str]) -> None:
    if value is None:
        problems.append(f"{path}: null value")
        return

    if is_bad_number(value):
        problems.append(f"{path}: non-finite number {value}")
        return

    if isinstance(value, dict):
        for key, child in value.items():
            walk_bad_values(child, f"{path}.{key}", problems)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_bad_values(child, f"{path}[{index}]", problems)


def count_blank_critical_fields(rows: list[Any], fields: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for field in fields:
            if field not in row:
                continue
            value = row.get(field)
            if isinstance(value, str) and not value.strip():
                counts[field] = counts.get(field, 0) + 1
    return counts


def main() -> None:
    print("--- DiamondSignals report payload null-safety audit ---")

    problems: list[str] = []

    for report_id, cfg in REPORT_PAYLOADS.items():
        payload_path = ROOT / cfg["path"]
        print(f"\n--- {report_id} ---")
        print(f"payload: {cfg['path']}")

        if not payload_path.exists():
            problems.append(f"{report_id}: missing payload {cfg['path']}")
            print("FAIL: missing payload")
            continue

        try:
            data = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"{report_id}: invalid JSON: {exc}")
            print(f"FAIL: invalid JSON: {exc}")
            continue

        payload_problems: list[str] = []
        walk_bad_values(data, report_id, payload_problems)

        if payload_problems:
            problems.extend(payload_problems[:25])
            print(f"FAIL: null/non-finite values found: {len(payload_problems)}")
            for item in payload_problems[:10]:
                print(f"  {item}")
        else:
            print("OK: no null or non-finite values")

        if not isinstance(data, dict):
            problems.append(f"{report_id}: payload root must be object")
            print("FAIL: payload root is not object")
            continue

        for list_key in cfg["required_lists"]:
            value = data.get(list_key)
            if not isinstance(value, list):
                problems.append(f"{report_id}: {list_key} must be a list")
                print(f"FAIL: {list_key} is not a list")
                continue

            print(f"{list_key}: {len(value)} rows")

            if len(value) == 0 and (report_id, list_key) not in ALLOWED_EMPTY_LISTS:
                problems.append(f"{report_id}: {list_key} is empty")
                print(f"FAIL: {list_key} is empty")

            blank_counts = count_blank_critical_fields(value, cfg["critical_fields"])
            if blank_counts:
                problems.append(f"{report_id}: blank critical fields in {list_key}: {blank_counts}")
                print(f"FAIL: blank critical fields in {list_key}: {blank_counts}")

        print("OK: payload shape inspected")

    print("\n--- summary ---")
    print(f"payloads_checked: {len(REPORT_PAYLOADS)}")
    print(f"payload_null_safety_issues: {len(problems)}")

    if problems:
        for problem in problems[:50]:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_PAYLOAD_NULL_SAFETY")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_PAYLOAD_NULL_SAFETY")


if __name__ == "__main__":
    main()
