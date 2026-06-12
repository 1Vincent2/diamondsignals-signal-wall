#!/usr/bin/env python3
from pathlib import Path
import sys

SOURCE = Path("dashboard/build_ivb_heat_map.py")
HTML = Path("dist/ivb-heat-map/index.html")

source = SOURCE.read_text(encoding="utf-8", errors="replace") if SOURCE.exists() else ""
html = HTML.read_text(encoding="utf-8", errors="replace") if HTML.exists() else ""

errors = []

def require(label, condition):
    if not condition:
        errors.append(label)

for term in [
    "IVB_HEAT_MAP_FIELD_GUIDE_DESKTOP_PILL_V1",
    "body .field-guide-trigger",
    "position: fixed !important;",
    "top: 336px !important;",
    "visibility: visible !important;",
    "opacity: 1 !important;",
    "pointer-events: auto !important;",
]:
    require(f"Missing source term: {term}", term in source)

for term in [
    "IVB_HEAT_MAP_FIELD_GUIDE_DESKTOP_PILL_V1",
    "body .field-guide-trigger",
    "position: fixed !important;",
    "top: 336px !important;",
    "visibility: visible !important;",
    "opacity: 1 !important;",
    "pointer-events: auto !important;",
    '<button class="field-guide-trigger" type="button" onclick="openFieldGuide()">Field Guide</button>',
    "fieldGuideModal",
    "openFieldGuide()",
    "closeFieldGuide()",
]:
    require(f"Missing rendered term: {term}", term in html)

# Preserve core behavior.
for term in [
    "ds-mobile-menu-trigger",
    "ds-mobile-menu-drawer",
    "/player-card-actions.js",
    "js-player-card",
    "IVB_HEAT_MAP",
]:
    require(f"Regression: missing preserved IVB term {term}", term in html)

if errors:
    print("IVB Field Guide visibility audit failed:")
    for err in errors:
        print(f" - {err}")
    sys.exit(1)

print("OK: IVB Field Guide visibility audit passed")
