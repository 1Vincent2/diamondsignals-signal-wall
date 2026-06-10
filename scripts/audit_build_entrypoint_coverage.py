#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "dashboard/report_inventory.json"
BUILD_ALL = ROOT / "dashboard/build_all.py"
BUILD_REPORT = ROOT / "scripts/build-report.sh"

SCRIPT_TO_MODULE = {
    "dashboard/build_dashboard.py": "dashboard.build_dashboard",
    "dashboard/build_signal_wall_v2.py": "dashboard.build_signal_wall_v2",
    "dashboard/build_call_up_live.py": "dashboard.build_call_up_live",
    "dashboard/build_kinetic_drift.py": "dashboard.build_kinetic_drift",
    "dashboard/build_mlb_extraction.py": "dashboard.build_mlb_extraction",
    "dashboard/build_apex_extraction.py": "dashboard.build_apex_extraction",
    "dashboard/build_ivb_heat_map.py": "dashboard.build_ivb_heat_map",
    "dashboard/build_velocity_decay.py": "dashboard.build_velocity_decay",
    "dashboard/build_stuff_disruption.py": "dashboard.build_stuff_disruption",
    "dashboard/build_market_eligibility.py": "dashboard.build_market_eligibility",
    "dashboard/build_waiver_candidates.py": "dashboard.build_waiver_candidates",
    "dashboard/build_waiver_wire.py": "dashboard.build_waiver_wire",
    "dashboard/build_watch_list.py": "dashboard.build_watch_list",
    "dashboard/build_player_signal_index.py": "dashboard.build_player_signal_index",
    "dashboard/build_canonical_player_universe.py": "dashboard.build_canonical_player_universe",
    "scripts/build_depth_radar_refresh.py": "scripts/build_depth_radar_refresh.py",
}

EXPLICIT_SUPPORTING_BUILDERS = {
    "scripts/build_depth_radar_refresh.py",
}

def main() -> None:
    problems = []

    print("--- DiamondSignals build entrypoint coverage audit ---")

    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    reports = inventory.get("reports", [])

    build_all_text = BUILD_ALL.read_text(encoding="utf-8")
    build_report_text = BUILD_REPORT.read_text(encoding="utf-8")

    for report in reports:
        report_id = report.get("report_id")
        builder_script = report.get("builder_script")
        surface_class = report.get("surface_class")

        print(f"\n--- {report_id} ---")
        print(f"builder_script: {builder_script}")
        print(f"surface_class: {surface_class}")

        module_name = SCRIPT_TO_MODULE.get(builder_script)

        if not module_name:
            problems.append(f"{report_id} builder has no SCRIPT_TO_MODULE mapping: {builder_script}")
            continue

        in_build_all = module_name in build_all_text or builder_script in build_all_text
        in_build_report = builder_script in build_report_text or module_name in build_report_text
        explicit_supporting = builder_script in EXPLICIT_SUPPORTING_BUILDERS

        print(f"in_build_all: {in_build_all}")
        print(f"in_build_report: {in_build_report}")
        print(f"explicit_supporting: {explicit_supporting}")

        if not (in_build_all or in_build_report or explicit_supporting):
            problems.append(
                f"{report_id} builder is not covered by build_all.py, build-report.sh, or explicit supporting classification: {builder_script}"
            )

    required_build_all_modules = [
        "dashboard.build_dashboard",
        "dashboard.build_call_up_live",
        "dashboard.build_kinetic_drift",
        "dashboard.build_mlb_extraction",
        "dashboard.build_apex_extraction",
        "dashboard.build_ivb_heat_map",
        "dashboard.build_velocity_decay",
        "dashboard.build_stuff_disruption",
        "dashboard.build_waiver_wire",
    ]

    print("\n--- required build_all module coverage ---")
    for module in required_build_all_modules:
        present = module in build_all_text
        print(f"{module}: {present}")
        if not present:
            problems.append(f"build_all.py missing required module: {module}")

    print("\n--- layout mutation safety ---")
    print("scope: build entrypoint coverage only")
    print("layout_files_modified: False")

    print("\n--- summary ---")
    print(f"reports_checked: {len(reports)}")
    print(f"build_entrypoint_coverage_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_BUILD_ENTRYPOINT_COVERAGE")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_BUILD_ENTRYPOINT_COVERAGE")

if __name__ == "__main__":
    main()
