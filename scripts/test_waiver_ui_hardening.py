#!/usr/bin/env python3
from pathlib import Path
import sys

SOURCE = Path("dashboard/templates/waiver_wire.html")
HTML = Path("dist/waiver-wire/index.html")

failures = []

source = SOURCE.read_text(encoding="utf-8") if SOURCE.exists() else ""
html = HTML.read_text(encoding="utf-8") if HTML.exists() else ""

required_source_terms = [
    "WAIVER_UI_HARDENING_V1_LIVE_GUIDE_WINDOW",
    "Pitcher-first rolling 72-hour claim surface",
    "<b>Rolling 72H</b>",
    "waiver-summary-guide",
    "body.waiver-page .topbar .live-label",
    "body.waiver-page .waiver-summary-guide",
    "position: fixed !important",
]

required_rendered_terms = [
    "WAIVER_UI_HARDENING_V1_LIVE_GUIDE_WINDOW",
    "Pitcher-first rolling 72-hour claim surface",
    "<b>Rolling 72H</b>",
    "waiver-summary-guide",
    "body.waiver-page .topbar .live-label",
    "body.waiver-page .waiver-summary-guide",
    "position: fixed !important",
]

for term in required_source_terms:
    if term not in source:
        failures.append(f"Missing required source term: {term}")

for term in required_rendered_terms:
    if term not in html:
        failures.append(f"Missing required rendered term: {term}")

stale_terms = [
    "Pitcher-first mid-May claim surface",
    "<b>Mid-May</b>",
]

for term in stale_terms:
    if term in source:
        failures.append(f"Stale Waiver copy still in source: {term}")
    if term in html:
        failures.append(f"Stale Waiver copy still rendered: {term}")

if "gap: 10px !important" not in html:
    failures.append("LIVE dot spacing hardening gap is missing.")

if "main.shell > .waiver-field-guide-trigger:not(.waiver-summary-guide)" not in html:
    failures.append("Duplicate standalone desktop Field Guide suppression rule missing.")

if "body.waiver-field-guide-open .waiver-summary-guide" not in html:
    failures.append("Field Guide open-state protection missing.")

if failures:
    print("Waiver UI hardening audit failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("OK: Waiver UI hardening audit passed")
