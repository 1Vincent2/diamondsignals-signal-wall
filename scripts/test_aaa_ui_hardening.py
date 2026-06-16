#!/usr/bin/env python3
from pathlib import Path
import sys

SOURCE = Path("dashboard/build_call_up_live.py")
HTML = Path("dist/typical-call-up/index.html")

failures = []

if not SOURCE.exists():
    failures.append(f"Missing source file: {SOURCE}")

if not HTML.exists():
    failures.append(f"Missing rendered HTML file: {HTML}")

source = SOURCE.read_text(encoding="utf-8") if SOURCE.exists() else ""
html = HTML.read_text(encoding="utf-8") if HTML.exists() else ""

required_source_terms = [
    "AAA_DESKTOP_RAIL_LOCK_CLEAN_V5_NAV_MATCH",
    "aaa-board-controls",
    "aaa-audit-layer-strip",
    "aaa-control-deck",
    "aaa-status-chip",
    "aaa-live-feed-status",
    "pw-14d-two-column-band",
    "tabs tabs-aaa",
]

required_rendered_terms = [
    "AAA_DESKTOP_RAIL_LOCK_CLEAN_V5_NAV_MATCH",
    "aaa-board-controls",
    "aaa-audit-layer-strip",
    "aaa-control-deck",
    "aaa-status-chip",
    "aaa-live-feed-status",
    "pw-14d-two-column-band",
    "tabs tabs-aaa",
    "Fresh AAA hitter board active",
    "Active Audit Layer",
]

for term in required_source_terms:
    if term not in source:
        failures.append(f"Missing required source marker/class: {term}")

for term in required_rendered_terms:
    if term not in html:
        failures.append(f"Missing required rendered marker/class: {term}")

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
    failures.append("Desktop system-pulse hide rule missing from rendered HTML.")

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

if "LIVE_ENGINE_PULSE" in html and "display: none !important;" not in html:
    failures.append("LIVE_ENGINE_PULSE exists but no hard hide rule is rendered.")

# The Field Guide/drawer may still use guide language.
# The hardening target is only the old visible guide button inside board controls.
if '<button type="button" class="guide-btn"' in html:
    failures.append("Old visible guide button still rendered inside board controls.")

if "OPEN DOSSIER" in html:
    failures.append("Promotion Watch must not render OPEN DOSSIER until scout routes are deploy-verified.")

if "/scout/" in html:
    failures.append("Promotion Watch must not render live /scout/ links until deployed scout pages are verified.")

if "Click any player card to inspect the full performance audit." in html:
    failures.append("Promotion Watch must not promise card-click audit routing until dossier routes are verified.")

if "Use INITIATE TRACKING from each card while live dossier routing is being verified." not in html:
    failures.append("Promotion Watch missing safe tracking-first audit copy.")

js_source = Path("src/js/player-card-actions.js").read_text(encoding="utf-8")
js_dist = Path("dist/player-card-actions.js").read_text(encoding="utf-8")

if 'profileUrl === "#"' not in js_source:
    failures.append("Source player-card-actions.js does not guard against hash profile URLs.")

if js_source != js_dist:
    failures.append("dist/player-card-actions.js is not in sync with src/js/player-card-actions.js.")

if failures:
    print("AAA UI hardening audit failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("OK: AAA UI hardening audit passed")
