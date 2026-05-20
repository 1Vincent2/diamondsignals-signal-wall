#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPORTS = [
    {
        "name": "Signal Wall",
        "status": "dist/status/signal-wall.json",
        "payloads": ["dist/signals.json"],
        "builders": ["dashboard/build_dashboard.py", "dashboard/build_signal_wall_v2.py"],
    },
    {
        "name": "Promotion Watch",
        "status": "dist/status/promotion-watch.json",
        "payloads": ["dist/typical-call-up/promotion_watch.json"],
        "builders": ["dashboard/build_call_up_live.py", "dashboard/build_call_up_v2.py", "dashboard/build_call_up_v3_preview.py"],
    },
    {
        "name": "Velocity Decay",
        "status": "dist/status/velocity-decay.json",
        "payloads": ["dist/velocity_decay_monitor.json"],
        "builders": ["dashboard/build_velocity_decay.py"],
    },
    {
        "name": "Stuff Disruption",
        "status": "dist/status/stuff-disruption.json",
        "payloads": ["dist/stuff_disruption_feed.json"],
        "builders": ["dashboard/build_stuff_disruption.py"],
    },
    {
        "name": "MLB Extraction",
        "status": "dist/status/mlb-extraction.json",
        "payloads": ["dist/hidden-gems/mlb_extraction_ledger.json"],
        "builders": ["dashboard/build_mlb_extraction.py", "dashboard/build_hidden_gems.py"],
    },
    {
        "name": "Apex Extraction",
        "status": "dist/status/apex-extraction.json",
        "payloads": ["dist/apex-extraction/apex_extraction.json"],
        "builders": ["dashboard/build_apex_extraction.py"],
    },
    {
        "name": "IVB Heat Map",
        "status": "dist/status/ivb-heat-map.json",
        "payloads": [],
        "builders": ["dashboard/build_ivb_heat_map.py"],
    },
    {
        "name": "Waiver Wire",
        "status": "dist/status/waiver-wire.json",
        "payloads": ["dist/waiver_wire.json"],
        "builders": ["dashboard/build_waiver_wire.py"],
    },
]

STATIC_PATTERNS = [
    r"static_seed",
    r"static editable",
    r"editable static",
    r"placeholder",
    r"manual .*override",
    r"hardcoded",
]

DYNAMIC_PATTERNS = [
    r"statcast",
    r"pybaseball",
    r"statsapi",
    r"supabase",
    r"canonical_player_universe",
    r"player_signal_index",
    r"real_data",
    r"fresh",
]

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")

def classify(status: dict, payloads: list[dict], builder_text: str) -> tuple[str, list[str]]:
    notes = []

    mode_blob = " ".join(
        str(x or "")
        for x in [
            status.get("mode"),
            *(p.get("mode") for p in payloads if isinstance(p, dict)),
            *(str(p.get("pipeline_layers")) for p in payloads if isinstance(p, dict)),
            str(status.get("pipeline_layers")),
        ]
    ).lower()

    source_age = status.get("source_age_minutes")
    state = status.get("state")
    build_success = status.get("build_success")
    degraded = status.get("degraded")
    used_fallback = status.get("used_fallback")

    if state != "fresh" or build_success is not True:
        notes.append("status_not_fresh_or_build_failed")
    if degraded:
        notes.append("degraded_true")
    if used_fallback:
        notes.append("used_fallback_true")

    static_hits = sorted(set(
        p for p in STATIC_PATTERNS
        if re.search(p, mode_blob, re.I) or re.search(p, builder_text, re.I)
    ))
    dynamic_hits = sorted(set(
        p for p in DYNAMIC_PATTERNS
        if re.search(p, mode_blob, re.I) or re.search(p, builder_text, re.I)
    ))

    section_counts = status.get("section_counts") or {}
    waiver_asset_count = int(section_counts.get("waiver_assets") or 0)

    verified_waiver_mode = (
        status.get("mode") == "verified_dynamic_candidates_only_v1"
        and "no_static_seed_fallback" in mode_blob
        and "verified_market_eligibility_required" in mode_blob
        and build_success is True
        and used_fallback is not True
    )

    if verified_waiver_mode and waiver_asset_count == 0:
        notes.append("verified_empty_waiver_mode:true")
        notes.append("no_static_seed_fallback:true")
        notes.append("verified_market_eligibility_required:true")
        return "LIVE_DYNAMIC_VERIFIED_EMPTY", notes

    if verified_waiver_mode and waiver_asset_count > 0 and state == "fresh" and not degraded:
        notes.append("verified_live_waiver_mode:true")
        notes.append(f"verified_waiver_assets:{waiver_asset_count}")
        notes.append("no_static_seed_fallback:true")
        notes.append("verified_market_eligibility_required:true")
        return "LIVE_DYNAMIC_VERIFIED_MARKET", notes

    hardened_mlb_extraction_mode = (
        status.get("mode") == "real_data_v0.1_hardened"
        and state == "fresh"
        and build_success is True
        and used_fallback is not True
        and not degraded
        and "real_statcast_source_window" in mode_blob
        and "canonical_player_universe" in mode_blob
        and "independent_mlb_extraction_scoring" in mode_blob
    )

    if hardened_mlb_extraction_mode:
        notes.append("hardened_mlb_extraction_mode:true")
        notes.append("real_statcast_source_window:true")
        notes.append("canonical_player_universe:true")
        notes.append("independent_mlb_extraction_scoring:true")
        return "LIVE_DYNAMIC_HARDENED", notes

    if static_hits:
        notes.append("static_or_placeholder_terms:" + ",".join(static_hits))
    if dynamic_hits:
        notes.append("dynamic_terms:" + ",".join(dynamic_hits))

    if "real_data" in mode_blob and not static_hits and not degraded and not used_fallback:
        return "LIVE_DYNAMIC_HARDENED", notes

    if source_age is not None and state == "fresh" and build_success is True and dynamic_hits and not static_hits:
        return "FRESH_DYNAMIC_UNLABELED", notes

    if static_hits and dynamic_hits:
        return "MIXED_DYNAMIC_WITH_STATIC_SCAFFOLD", notes

    if static_hits:
        return "STATIC_OR_PLACEHOLDER_RISK", notes

    if state == "fresh" and build_success is True:
        return "FRESH_BUT_DATA_MODE_UNKNOWN", notes

    return "UNKNOWN_OR_UNHEALTHY", notes

def main() -> int:
    failures = []
    print("--- DiamondSignals report data-mode audit ---")

    for report in REPORTS:
        status_path = ROOT / report["status"]
        status = read_json(status_path)

        payloads = [read_json(ROOT / p) for p in report["payloads"]]
        builder_text = "\n".join(read_text(ROOT / b) for b in report["builders"])

        classification, notes = classify(status, payloads, builder_text)

        print(f"\n--- {report['name']} ---")
        print(f"status_file: {report['status']}")
        print(f"state: {status.get('state')}")
        print(f"build_success: {status.get('build_success')}")
        print(f"source_age_minutes: {status.get('source_age_minutes')}")
        print(f"mode: {status.get('mode')}")
        print(f"classification: {classification}")
        print("notes:")
        for note in notes:
            print(f"  - {note}")

        if classification in {
            "STATIC_OR_PLACEHOLDER_RISK",
            "MIXED_DYNAMIC_WITH_STATIC_SCAFFOLD",
            "FRESH_BUT_DATA_MODE_UNKNOWN",
            "UNKNOWN_OR_UNHEALTHY",
        }:
            failures.append((report["name"], classification))

    print("\n--- summary ---")
    print(f"reports_checked: {len(REPORTS)}")
    print(f"reports_needing_hardening: {len(failures)}")
    for name, classification in failures:
        print(f"- {name}: {classification}")

    print("\nFINAL_STATUS: PASS_INSPECTION_ONLY")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
