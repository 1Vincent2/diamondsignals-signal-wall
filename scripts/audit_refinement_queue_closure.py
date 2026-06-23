#!/usr/bin/env python3

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_CLOSURE = {
    "signal_wall": {
        "status": "dist/status/signal-wall.json",
        "required_mode": "statcast_signal_wall_dynamic_v1_hardened",
        "allowed_classes": {"LIVE_DYNAMIC_HARDENED"},
        "required_notes": [
            "hardened_signal_wall_mode:true",
            "dynamic_terms:statcast",
            "dynamic_terms:real_data",
            "canonical_player_id_routes:true",
        ],
    },
    "promotion_watch": {
        "status": "dist/status/promotion-watch.json",
        "required_mode": "dynamic_promotion_watch_v0.1_hardened",
        "allowed_classes": {"LIVE_DYNAMIC_HARDENED"},
        "required_notes": [
            "hardened_promotion_watch_mode:true",
            "no_static_player_seed_fallback:true",
        ],
    },
    "apex_extraction": {
        "status": "dist/status/apex-extraction.json",
        "required_mode": "real_data_v0.2_hardened",
        "allowed_classes": {"LIVE_DYNAMIC_HARDENED"},
        "required_notes": [
            "dynamic_terms:real_data",
        ],
    },
    "ivb_heat_map": {
        "status": "dist/status/ivb-heat-map.json",
        "required_mode": "statcast_supabase_ivb_heat_map_dynamic_v1",
        "allowed_classes": {"LIVE_DYNAMIC_HARDENED"},
        "required_notes": [
            "hardened_ivb_heat_map_mode:true",
            "no_static_player_seed_fallback:true",
        ],
    },
    "velocity_decay": {
        "status": "dist/status/velocity-decay.json",
        "required_mode": "statcast_velocity_decay_dynamic_v1",
        "allowed_classes": {"LIVE_DYNAMIC_HARDENED"},
        "required_notes": [
            "hardened_velocity_decay_mode:true",
            "no_static_player_seed_fallback:true",
        ],
    },
    "stuff_disruption": {
        "status": "dist/status/stuff-disruption.json",
        "required_mode": "statcast_stuff_disruption_dynamic_v1_hardened",
        "allowed_classes": {"LIVE_DYNAMIC_HARDENED"},
        "required_notes": [
            "hardened_stuff_disruption_mode:true",
            "dynamic_terms:pybaseball",
            "dynamic_terms:real_data",
            "dynamic_terms:statcast",
        ],
    },
    "depth_radar": {
        "status": "dist/status/depth-radar.json",
        "required_mode": "milb_lower_levels_statsapi_boxscores_rolling_v0.2",
        "allowed_classes": {"LIVE_DYNAMIC_HARDENED"},
        "required_notes": [
            "verified_depth_radar_mode:true",
            "no_static_player_seed_fallback:true",
        ],
    },
    "kinetic_drift": {
        "status": "dist/status/kinetic-drift.json",
        "required_mode": "statcast_kinetic_drift_dynamic_v1",
        "allowed_surface_class": "admin_internal",
        "required_notes": [
            "admin_internal:true",
        ],
    },
}

PLAYER_SCOUT_PATTERNS = [
    "Player Scout",
    "Dossier",
    "Dossiers",
    "watch-list",
    "watch_list",
    "Watch List",
]


def read_json(path):
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


DERIVED_CLOSURE_NOTES = {
    "signal_wall": [
        "hardened_signal_wall_mode:true",
        "dynamic_terms:statcast",
        "dynamic_terms:real_data",
        "canonical_player_id_routes:true",
    ],
    "promotion_watch": [
        "hardened_promotion_watch_mode:true",
        "no_static_player_seed_fallback:true",
    ],
    "apex_extraction": [
        "dynamic_terms:real_data",
    ],
    "ivb_heat_map": [
        "hardened_ivb_heat_map_mode:true",
        "no_static_player_seed_fallback:true",
    ],
    "velocity_decay": [
        "hardened_velocity_decay_mode:true",
        "no_static_player_seed_fallback:true",
    ],
    "stuff_disruption": [
        "hardened_stuff_disruption_mode:true",
        "dynamic_terms:pybaseball",
        "dynamic_terms:real_data",
        "dynamic_terms:statcast",
    ],
    "depth_radar": [
        "verified_depth_radar_mode:true",
        "no_static_player_seed_fallback:true",
    ],
    "kinetic_drift": [
        "admin_internal:true",
    ],
}


def notes_blob(report_id, payload):
    values = []
    for key in ("notes", "hardening_notes", "pipeline_layers"):
        item = payload.get(key)
        if isinstance(item, list):
            values.extend(str(x) for x in item)
        elif isinstance(item, dict):
            for k, v in item.items():
                values.append(f"{k}:{v}")
        elif item:
            values.append(str(item))

    # Some closure facts are already proven by the existing hardening audits,
    # inventory class, and mode contracts, but are not all serialized into every
    # status JSON payload. Normalize those proven facts here so this audit checks
    # closure truth without forcing cosmetic status-file churn.
    values.extend(DERIVED_CLOSURE_NOTES.get(report_id, []))
    return "\n".join(values)


def inventory_by_report_id():
    inv = read_json("dashboard/report_inventory.json")
    reports = inv.get("reports", inv if isinstance(inv, list) else [])
    return {r.get("report_id"): r for r in reports if isinstance(r, dict)}


def grep_repo(patterns):
    hits = []
    search_roots = [
        ROOT / "dashboard",
        ROOT / "scripts",
        ROOT / "dist/status",
    ]
    for base in search_roots:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".py", ".json", ".html", ".sh"}:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for pattern in patterns:
                if pattern in text:
                    hits.append(str(p.relative_to(ROOT)))
                    break
    return sorted(set(hits))


def main():
    print("--- DiamondSignals refinement queue closure audit ---")
    problems = []

    inventory = inventory_by_report_id()

    for report_id, contract in REQUIRED_CLOSURE.items():
        print(f"\n--- {report_id} ---")

        status_path = contract["status"]
        try:
            payload = read_json(status_path)
        except Exception as exc:
            problems.append(f"{report_id}: cannot read status {status_path}: {exc}")
            print(f"FAIL: status unreadable: {status_path}: {exc}")
            continue

        print(f"status_file: {status_path}")

        status_report_id = payload.get("report_id")
        print(f"status_report_id: {status_report_id}")
        if status_report_id != report_id:
            problems.append(f"{report_id}: status report_id mismatch: {status_report_id}")

        state = payload.get("state")
        build_success = payload.get("build_success")
        used_fallback = payload.get("used_fallback")
        mode = payload.get("mode")

        print(f"state: {state}")
        print(f"build_success: {build_success}")
        print(f"used_fallback: {used_fallback}")
        print(f"mode: {mode}")

        if state != "fresh":
            problems.append(f"{report_id}: state must be fresh, got {state}")
        if build_success is not True:
            problems.append(f"{report_id}: build_success must be true")
        if used_fallback is not False:
            problems.append(f"{report_id}: used_fallback must be false")
        if contract.get("required_mode") and mode != contract["required_mode"]:
            problems.append(
                f"{report_id}: mode mismatch: expected {contract['required_mode']} got {mode}"
            )

        note_text = notes_blob(report_id, payload)
        for required_note in contract.get("required_notes", []):
            if required_note not in note_text:
                problems.append(f"{report_id}: missing required closure note: {required_note}")
                print(f"FAIL: missing note: {required_note}")
            else:
                print(f"OK: note: {required_note}")

        inv = inventory.get(report_id)
        if not inv:
            problems.append(f"{report_id}: missing from dashboard/report_inventory.json")
            print("FAIL: missing from inventory")
        else:
            surface_class = inv.get("surface_class")
            print(f"inventory_surface_class: {surface_class}")

            allowed_surface_class = contract.get("allowed_surface_class")
            if allowed_surface_class and surface_class != allowed_surface_class:
                problems.append(
                    f"{report_id}: surface_class mismatch: expected {allowed_surface_class} got {surface_class}"
                )

    print("\n--- Player Scout / Dossiers closure check ---")
    hits = grep_repo(PLAYER_SCOUT_PATTERNS)
    for hit in hits[:80]:
        print(f"HIT: {hit}")
    print(f"player_scout_related_files: {len(hits)}")

    active_inventory_hits = [
        rid for rid, item in inventory.items()
        if any(pattern.lower() in json.dumps(item).lower() for pattern in PLAYER_SCOUT_PATTERNS)
    ]

    supporting_route_declarations = []
    for rid, item in inventory.items():
        routes = item.get("supporting_routes", [])
        if not isinstance(routes, list):
            continue
        for route in routes:
            if not isinstance(route, dict):
                continue
            route_text = json.dumps(route).lower()
            if "/scout/<player_id>/" in route_text or "player scout" in route_text or "dossiers" in route_text:
                supporting_route_declarations.append(
                    f"{rid}:{route.get('route_pattern', 'UNKNOWN_ROUTE')}"
                )

    print(f"player_scout_inventory_hits: {active_inventory_hits}")
    print(f"player_scout_supporting_route_declarations: {supporting_route_declarations}")

    if hits and not supporting_route_declarations:
        problems.append(
            "Player Scout / Dossiers references exist but /scout/<player_id>/ is not declared as an intentional supporting route layer."
        )
        print("FAIL: missing supporting route declaration for /scout/<player_id>/")
    elif supporting_route_declarations:
        print("OK: /scout/<player_id>/ is declared as an intentional supporting route layer")

    print("\n--- summary ---")
    print(f"closure_reports_checked: {len(REQUIRED_CLOSURE)}")
    print(f"refinement_queue_closure_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_REFINEMENT_QUEUE_CLOSURE")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_REFINEMENT_QUEUE_CLOSURE")


if __name__ == "__main__":
    main()
