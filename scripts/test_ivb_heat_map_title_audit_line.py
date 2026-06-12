from pathlib import Path
import sys

SOURCE = Path("dashboard/build_ivb_heat_map.py")
RENDERED = Path("dist/ivb-heat-map/index.html")

failures = []

source = SOURCE.read_text(encoding="utf-8") if SOURCE.exists() else ""
rendered = RENDERED.read_text(encoding="utf-8") if RENDERED.exists() else ""

def require(label, condition):
    if not condition:
        failures.append(label)

require("Missing IVB source builder", SOURCE.exists())
require("Missing rendered IVB page", RENDERED.exists())

for term in [
    "IVB_HEAT_MAP_TITLE_AUDIT_LINE_V1",
    ".ivb-active-audit-line",
    '<p class="ivb-active-audit-line">',
    "Active Audit Layer:",
    "Click any player card to inspect the full performance audit.",
]:
    require(f"Missing source term: {term}", term in source)

for term in [
    "IVB_HEAT_MAP_TITLE_AUDIT_LINE_V1",
    ".ivb-active-audit-line",
    '<p class="ivb-active-audit-line">',
    "Active Audit Layer:",
    "Click any player card to inspect the full performance audit.",
]:
    require(f"Missing rendered term: {term}", term in rendered)

hero_idx = rendered.find('<h1 class="hero-title">IVB Heat Map</h1>')
line_idx = rendered.find('<p class="ivb-active-audit-line">')
metrics_idx = rendered.find('<section class="top-metrics">')

require("Could not find IVB hero title", hero_idx != -1)
require("Could not find IVB audit line", line_idx != -1)
require("Could not find top metrics section", metrics_idx != -1)

if hero_idx != -1 and line_idx != -1 and metrics_idx != -1:
    require("IVB audit line is not inside title section before top metrics", hero_idx < line_idx < metrics_idx)

# Mobile/global contracts must remain present.
require("Rendered IVB missing mobile menu trigger", "ds-mobile-menu-trigger" in rendered)
require("Rendered IVB missing mobile menu drawer", "ds-mobile-menu-drawer" in rendered)
require("Rendered IVB missing Field Guide trigger", "openFieldGuide()" in rendered and "Field Guide" in rendered)
require("Rendered IVB missing player-card tracking", "js-player-card" in rendered and "/player-card-actions.js" in rendered)

if failures:
    print("IVB title audit-line audit failed:")
    for item in failures:
        print(f" - {item}")
    sys.exit(1)

print("OK: IVB title audit-line audit passed")
