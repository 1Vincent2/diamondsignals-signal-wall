#!/usr/bin/env python3
from pathlib import Path
import sys

print("--- DiamondSignals Signal Wall tracking regression lock audit ---")

targets = [
    Path("src/js/player-card-actions.js"),
    Path("dist/player-card-actions.js"),
]

forbidden = [
    "OPENING PASSPORT",
    "Opening Passport",
    "opening passport",
    "data-opening-passport",
    "queued for Passport Tracking",
    "Passport Tracking",
]

required = [
    "https://app.diamondsignals.ai/auth",
    "/watchlist",
    "ADDING TO TRACKING",
    "TRACKING REQUEST SENT",
    "ASSET TRACKED",
    "tracking request sent to Tracking Radar",
]

issues = []

for target in targets:
    if not target.exists():
        issues.append(f"MISSING_TARGET: {target}")
        continue

    text = target.read_text()

    for token in forbidden:
        if token in text:
            issues.append(f"FORBIDDEN_LEGACY_TRACKING_COPY: {target}: {token}")

    for token in required:
        if token not in text:
            issues.append(f"MISSING_REQUIRED_TRACKING_CONTRACT: {target}: {token}")

for issue in issues:
    print(issue)

print("")
print("--- summary ---")
print(f"signal_wall_tracking_regression_issues: {len(issues)}")

if issues:
    print("")
    print("FINAL_STATUS: FAIL_SIGNAL_WALL_TRACKING_REGRESSION_LOCK")
    sys.exit(1)

print("")
print("FINAL_STATUS: PASS_SIGNAL_WALL_TRACKING_REGRESSION_LOCK")
