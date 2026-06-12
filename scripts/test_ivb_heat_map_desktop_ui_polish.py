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
    "IVB_HEAT_MAP_DESKTOP_UI_POLISH_V1",
    ".hero-card",
    ".top-metrics",
    ".metric-card",
    ".section-head",
    "body .field-guide-trigger",
    "height: 38px !important;",
    "border-radius: 999px !important;",
    "text-transform: uppercase !important;",
]:
    require(f"Missing source UI polish term: {term}", term in source)

for term in [
    "IVB_HEAT_MAP_DESKTOP_UI_POLISH_V1",
    ".hero-card",
    ".top-metrics",
    ".metric-card",
    ".section-head",
    "body .field-guide-trigger",
    "height: 38px !important;",
    "border-radius: 999px !important;",
    "text-transform: uppercase !important;",
]:
    require(f"Missing rendered UI polish term: {term}", term in rendered)

require("Desktop-only media scope missing", "@media screen and (min-width: 981px)" in source and "@media screen and (min-width: 981px)" in rendered)

# Preserve hardened IVB contracts.
for term in [
    "IVB_HEAT_MAP_SHARED_NAV_PATH_V1",
    "IVB_HEAT_MAP_TITLE_AUDIT_LINE_V1",
    "IVB_HEAT_MAP_FIELD_GUIDE_CLEANUP_V1",
    "IVB_HEAT_MAP_FIELD_GUIDE_DESKTOP_PILL_V1",
]:
    require(f"Missing existing hardened marker: {term}", term in source or term in rendered)

require("Rendered IVB missing compact desktop nav", '<nav class="ds-pro-desktop-nav"' in rendered)
require("Rendered IVB missing mobile shell nav", '<div class="topnav ds-shell-nav">' in rendered)
require("Rendered IVB missing mobile menu contract", "ds-mobile-menu-trigger" in rendered and "ds-mobile-menu-drawer" in rendered)
require("Rendered IVB missing Field Guide behavior", "openFieldGuide()" in rendered and "fieldGuideModal" in rendered)
require("Rendered IVB missing tracking contract", "/player-card-actions.js" in rendered and "js-player-card" in rendered and "IVB_HEAT_MAP" in rendered)


require("Missing transparent tracking pill background", "background: rgba(2,6,23,0.18) !important;" in source)
require("Missing white tracking pill type", "color: rgba(255,255,255,0.92) !important;" in source)
require("Missing tracking pill white border", "border: 1px solid rgba(255,255,255,0.22) !important;" in source)
require("Missing card-bottom tracking row spacing", "margin-top: auto !important;" in source)
require("Missing card flex-column placement guard", "flex-direction: column !important;" in source)
require("Rendered IVB tracking button missing", "heat-provision-btn js-add-to-roster" in rendered)

if failures:
    print("IVB desktop UI polish audit failed:")
    for item in failures:
        print(f" - {item}")
    sys.exit(1)

print("OK: IVB desktop UI polish audit passed")
