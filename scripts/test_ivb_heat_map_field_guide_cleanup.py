#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dashboard/build_ivb_heat_map.py"
HTML = ROOT / "dist/ivb-heat-map/index.html"

source = SOURCE.read_text(encoding="utf-8", errors="replace")
html = HTML.read_text(encoding="utf-8", errors="replace") if HTML.exists() else ""

errors = []

def require(label, condition):
    if not condition:
        errors.append(label)

for term in [
    "IVB_HEAT_MAP_FIELD_GUIDE_CLEANUP_V1",
    "IVB RAW",
    "IVB VS AVG",
    "APEX RISE",
    "THE DEAD ZONE",
    "WHIFF PROB",
    "CLIMBERS",
    "STATUS KEY",
    "VAA is reserved for the next upstream pitch-plane layer",
]:
    require(f"Missing source term: {term}", term in source)
    require(f"Missing rendered term: {term}", term in html)

require("Old VAA scaffold wording still exists in source", "VAA is scaffolded" not in source)
require("Old VAA scaffold wording still exists in rendered HTML", "VAA is scaffolded" not in html)

require("Duplicate The Dead Zone card still exists in source", '<h3 class="field-guide-term dead">The Dead Zone</h3>' not in source)
require("Duplicate The Dead Zone card still exists in rendered HTML", '<h3 class="field-guide-term dead">The Dead Zone</h3>' not in html)
require("Duplicate Whiff Prob card still exists in source", '<h3 class="field-guide-term whiff">Whiff Prob</h3>' not in source)
require("Duplicate Whiff Prob card still exists in rendered HTML", '<h3 class="field-guide-term whiff">Whiff Prob</h3>' not in html)

require("Field Guide trigger missing", "openFieldGuide()" in html and "Field Guide" in html)
require("Tracking source tag missing", "IVB_HEAT_MAP" in html)
require("Player card class missing", "js-player-card" in html)
require("Mobile menu contract missing", "ds-mobile-menu-trigger" in html and "ds-mobile-menu-drawer" in html)

if errors:
    print("IVB Field Guide cleanup audit failed:")
    for err in errors:
        print(f" - {err}")
    sys.exit(1)

print("OK: IVB Field Guide cleanup audit passed")
