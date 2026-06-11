from pathlib import Path
import sys

source_path = Path("dashboard/build_call_up_live.py")
html_path = Path("dist/typical-call-up/index.html")

source = source_path.read_text(encoding="utf-8")
html = html_path.read_text(encoding="utf-8")

failures = []

required_source = [
    "AAA_DESKTOP_RAIL_LOCK_V6_SPECIFICITY_OVERRIDE",
    "AAA_DESKTOP_FINAL_HERO_CONTROL_POLISH_V6",
    "AAA_DESKTOP_RAIL_LOCK_CLEAN_V5_NAV_MATCH",
    "AAA_FINAL_DESKTOP_POLISH_V1_SPACING_STATUS",
    "aaa-board-controls",
    "aaa-audit-layer-strip",
    "aaa-control-deck",
    "aaa-live-feed-status",
    "pw-14d-two-column-band",
    "Final AAA Slate",
]

required_html = [
    "AAA_DESKTOP_RAIL_LOCK_V6_SPECIFICITY_OVERRIDE",
    "AAA_DESKTOP_FINAL_HERO_CONTROL_POLISH_V6",
    "AAA_DESKTOP_RAIL_LOCK_CLEAN_V5_NAV_MATCH",
    "AAA_FINAL_DESKTOP_POLISH_V1_SPACING_STATUS",
    "aaa-board-controls",
    "aaa-audit-layer-strip",
    "aaa-control-deck",
    "aaa-live-feed-status",
    "pw-14d-two-column-band",
    "Final AAA Slate",
]

for needle in required_source:
    if needle not in source:
        failures.append(f"Missing required source marker/class: {needle}")

for needle in required_html:
    if needle not in html:
        failures.append(f"Missing required rendered marker/class: {needle}")

stale_markers = [
    "AAA_DESKTOP_BOARD_RECOMPOSITION_V1",
    "AAA_DESKTOP_BOARD_RECOMPOSITION_V2_POLISH",
    "AAA_DESKTOP_RAIL_LOCK_V3_REMOVE_PULSE",
    "AAA_DESKTOP_RAIL_ALIGN_V4_HERO_AND_STATUS",
    "AAA_HERO_TITLE_TABS_COMPRESSION_V1",
    "PROMOTION_WATCH_ALL_TABS_TWO_COLUMN_DESKTOP_V1",
]

for stale in stale_markers:
    if stale in source:
        failures.append(f"Stale stacked override marker still in source: {stale}")
    if stale in html:
        failures.append(f"Stale stacked override marker still rendered: {stale}")

if 'width: min(1180px, calc(100% - 48px))' not in html:
    failures.append("Rendered AAA page no longer matches desktop nav rail width.")

if ".aaa-control-deck .system-pulse-bar" not in html:
    failures.append("Desktop pulse hide rule missing from rendered HTML.")

if "<button type=\"button\" class=\"guide-btn\"" in html:
    failures.append("Old visible guide button still rendered inside board controls.")

if html.count('id="tab-btn-72h"') != 1:
    failures.append("72 HR tab button rendered more or less than once.")

if html.count('id="tab-btn-14d"') != 1:
    failures.append("14 DAY tab button rendered more or less than once.")

if html.count('id="tab-btn-aaa-gems"') != 1:
    failures.append("AAA GEMS tab button rendered more or less than once.")

if html.count('class="tabs tabs-aaa"') != 1:
    failures.append("AAA tab cluster rendered more or less than once.")

if "Fresh AAA hitter board active" not in html:
    failures.append("Live feed status copy missing.")

if "LIVE_ENGINE_PULSE" in html and "display: none !important;" not in html:
    failures.append("LIVE_ENGINE_PULSE exists but no hard hide rule is rendered.")

if failures:
    print("AAA UI hardening audit failed:")
    for f in failures:
        print(f" - {f}")
    sys.exit(1)

print("OK: AAA UI hardening audit passed")
