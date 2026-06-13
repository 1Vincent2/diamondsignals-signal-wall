#!/usr/bin/env python3
from pathlib import Path
import sys

SOURCE_PATH = Path("dashboard/build_velocity_decay.py")
HTML_PATH = Path("dist/velocity-decay-monitor/index.html")

failures = []

def require(label: str, condition: bool) -> None:
    if not condition:
        failures.append(label)

source = SOURCE_PATH.read_text(encoding="utf-8")
rendered = HTML_PATH.read_text(encoding="utf-8", errors="replace") if HTML_PATH.exists() else ""

for term in [
    "VELOCITY_DECAY_DESKTOP_STICKY_GUIDE_TWO_COLUMN_V1",
    "@media screen and (min-width: 981px)",
    "position: fixed !important;",
    "top: 118px !important;",
    "right: max(26px, calc((100vw - 1180px) / 2 + 18px)) !important;",
    "body .cards",
    "grid-template-columns: repeat(2, minmax(0, 1fr)) !important;",
    "body .guide-body",
    "body .guide-section:first-child",
    "grid-column: 1 / -1 !important;",
]:
    require(f"Missing Velocity desktop UI source term: {term}", term in source)
    require(f"Missing Velocity desktop UI rendered term: {term}", term in rendered)

require("Velocity Field Guide trigger missing", 'class="info-trigger"' in rendered and "Field Guide" in rendered)
require("Velocity guide drawer missing", 'class="guide-drawer"' in rendered and "Velocity Decay Field Guide" in rendered)
require("Velocity tracking source tag missing", 'data-source-tag="VELOCITY_DECAY"' in rendered)
require("Shared mobile menu missing", "ds-mobile-menu-trigger" in rendered and "ds-mobile-menu-drawer" in rendered)
require("Compact desktop nav missing", "ds-pro-desktop-nav" in rendered and "[ RISK ]" in rendered)

if failures:
    print("Velocity Decay desktop UI polish audit failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("OK: Velocity Decay desktop sticky Field Guide / 2-column audit passed")
