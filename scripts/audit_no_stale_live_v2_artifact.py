#!/usr/bin/env python3
from pathlib import Path
import sys

issues = []

stale_paths = [
    Path("dist/live-v2"),
    Path("dist/live-v2/index.html"),
]

for path in stale_paths:
    if path.exists():
        issues.append(f"stale production artifact still exists: {path}")

inventory = Path("dashboard/report_inventory.json").read_text(encoding="utf-8")
if "dist/live-v2/index.html" in inventory or '"live_v2"' in inventory or '"signal_wall_v2"' in inventory:
    issues.append("live-v2 is unexpectedly declared in report inventory")

build_all = Path("dashboard/build_all.py").read_text(encoding="utf-8")
if "build_signal_wall_v2" in build_all:
    issues.append("build_all still invokes build_signal_wall_v2")

if issues:
    print("--- DiamondSignals stale live-v2 artifact audit ---")
    print(f"stale_live_v2_issues: {len(issues)}")
    for issue in issues:
        print(f" - {issue}")
    print("\nFINAL_STATUS: FAIL_NO_STALE_LIVE_V2_ARTIFACT")
    sys.exit(1)

print("--- DiamondSignals stale live-v2 artifact audit ---")
print("stale_live_v2_issues: 0")
print("live_v2_removed_from_public_dist: true")
print("\nFINAL_STATUS: PASS_NO_STALE_LIVE_V2_ARTIFACT")
