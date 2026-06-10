#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Status files that are intentionally not report surfaces.
# These are system/admin manifests consumed by future admin tooling and should
# not be forced into dashboard/report_inventory.json as public report entries.
ALLOWED_NON_REPORT_STATUS_IDS = {
    "admin_audit_manifest",
}
INVENTORY = ROOT / "dashboard/report_inventory.json"
STATUS_DIR = ROOT / "dist/status"

STATUS_TO_REPORT_ID = {
    "signal-wall.json": "signal_wall",
    "promotion-watch.json": "promotion_watch",
    "velocity-decay.json": "velocity_decay",
    "stuff-disruption.json": "stuff_disruption",
    "mlb-extraction.json": "mlb_extraction",
    "apex-extraction.json": "apex_extraction",
    "ivb-heat-map.json": "ivb_heat_map",
    "waiver-wire.json": "waiver_wire",
    "depth-radar.json": "depth_radar",
    "kinetic-drift.json": "kinetic_drift",
}

ALLOWED_SURFACE_CLASSES = {
    "public_surface",
    "compatibility_surface",
    "embedded_surface",
    "admin_internal",
    "stale_no_deploy",
}

REQUIRED_KEYS = {
    "report_id",
    "surface_class",
    "builder_script",
    "output_html",
    "output_json",
    "status_output",
    "freshness_threshold_minutes",
    "required_sections",
    "source_dependencies",
}

NOTE_REQUIRED_CLASSES = {
    "embedded_surface",
    "compatibility_surface",
    "admin_internal",
}

def main() -> None:
    problems: list[str] = []

    print("--- DiamondSignals report inventory truth audit ---")

    inventory_data = json.loads(INVENTORY.read_text())
    reports = inventory_data.get("reports", [])
    by_id = {report.get("report_id"): report for report in reports}

    status_ids = {
        STATUS_TO_REPORT_ID.get(path.name, path.stem.replace("-", "_"))
        for path in sorted(STATUS_DIR.glob("*.json"))
    }

    inventory_ids = set(by_id)

    missing_from_inventory = sorted((status_ids - inventory_ids) - ALLOWED_NON_REPORT_STATUS_IDS)
    missing_status_for_inventory = sorted(
        rid for rid in inventory_ids - status_ids
        if by_id[rid].get("surface_class") != "stale_no_deploy"
    )

    if missing_from_inventory:
        problems.append(f"status files missing from report_inventory.json: {missing_from_inventory}")

    if missing_status_for_inventory:
        problems.append(f"inventory records missing status files: {missing_status_for_inventory}")

    for rid, report in sorted(by_id.items()):
        print(f"\n--- {rid} ---")

        missing_keys = sorted(REQUIRED_KEYS - set(report))
        if missing_keys:
            problems.append(f"{rid} missing required keys: {missing_keys}")

        surface_class = report.get("surface_class")
        if surface_class not in ALLOWED_SURFACE_CLASSES:
            problems.append(f"{rid} has invalid surface_class: {surface_class}")

        for key in ["builder_script", "output_html", "output_json", "status_output"]:
            value = report.get(key)
            if value is None:
                continue

            path = ROOT / value
            if path.exists():
                print(f"OK: {key}: {value}")
            else:
                problems.append(f"{rid} {key} does not exist: {value}")

        fallback = report.get("fallback_artifact")
        if fallback:
            fallback_path = ROOT / fallback
            if fallback_path.exists():
                print(f"OK: fallback_artifact: {fallback}")
            else:
                print(f"WARN: fallback_artifact not present yet: {fallback}")

        if surface_class in NOTE_REQUIRED_CLASSES and not report.get("route_note"):
            problems.append(f"{rid} must include route_note explaining {surface_class}")

    print("\n--- summary ---")
    print(f"inventory_reports: {len(inventory_ids)}")
    print(f"status_reports: {len(status_ids)}")
    print(f"inventory_truth_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_REPORT_INVENTORY_TRUTH")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_REPORT_INVENTORY_TRUTH")

if __name__ == "__main__":
    main()
