#!/usr/bin/env python3
from pathlib import Path
import sys

SOURCE = Path("dashboard/build_velocity_decay.py")
HTML = Path("dist/velocity-decay-monitor/index.html")

failures = []

def require(message, condition):
    if not condition:
        failures.append(message)

source = SOURCE.read_text(encoding="utf-8")
rendered = HTML.read_text(encoding="utf-8") if HTML.exists() else ""

require("Missing shared nav marker", "VELOCITY_DECAY_SHARED_NAV_PATH_V1" in source)
require("Missing desktop nav template load", "shell_nav_v2.html" in source and "DESKTOP_NAV_TEMPLATE" in source)
require("Missing mobile nav template load", "shell_nav.html" in source and "MOBILE_NAV_TEMPLATE" in source)
source_lines = source.splitlines()
legacy_nav_lines = [
    line for line in source_lines
    if line.strip().startswith("NAV_TEMPLATE =")
]
require("Old single NAV_TEMPLATE path still present", not legacy_nav_lines)

require("Rendered compact desktop nav missing", "ds-pro-desktop-nav" in rendered)
require("Rendered old topnav still missing for mobile fallback", "topnav ds-shell-nav" in rendered or 'class="topnav' in rendered)
require("Rendered mobile menu trigger missing", "ds-mobile-menu-trigger" in rendered)
require("Rendered mobile menu drawer missing", "ds-mobile-menu-drawer" in rendered)
require("Velocity Decay desktop active nav missing", "[ RISK ]" in rendered and "Velocity Decay" in rendered)

require("Tracking contract missing after nav swap", "/player-card-actions.js" in rendered and "js-add-to-roster" in rendered and "VELOCITY_DECAY" in rendered)

if failures:
    print("Velocity Decay desktop nav parity audit failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("OK: Velocity Decay desktop nav parity audit passed")
