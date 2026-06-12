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
    "IVB_HEAT_MAP_MOBILE_FIELD_GUIDE_BOTTOM_PILL_V1",
    "@media screen and (max-width: 980px)",
    "bottom: max(18px, env(safe-area-inset-bottom)) !important;",
    "right: max(28px, env(safe-area-inset-right)) !important;",
    "z-index: 420 !important;",
    "width: min(206px, calc(100vw - 150px)) !important;",
    "body .field-guide-trigger::before",
    'content: "I" !important;',
    "padding-bottom: calc(92px + env(safe-area-inset-bottom)) !important;",
]:
    require(f"Missing IVB mobile Field Guide source term: {term}", term in source)
    require(f"Missing IVB mobile Field Guide rendered term: {term}", term in rendered)

require("IVB modal behavior missing", "openFieldGuide()" in rendered and "closeFieldGuide()" in rendered and "fieldGuideModal" in rendered)
require("IVB Field Guide button missing", 'class="field-guide-trigger"' in rendered and "Field Guide" in rendered)

require("Expected exactly one IVB mobile close X in source", source.count('class="field-guide-close"') == 1)
require("Expected exactly one IVB mobile close X in rendered page", rendered.count('class="field-guide-close"') == 1)

for term in [
    "IVB_HEAT_MAP_MOBILE_FIELD_GUIDE_CLOSE_X_V1",
    "field-guide-close",
    'aria-label="Close Field Guide"',
    'onclick="closeFieldGuide()"',
    "top: calc(92px + env(safe-area-inset-top)) !important;",
    "z-index: 930 !important;",
]:
    require(f"Missing IVB mobile close-X source term: {term}", term in source)
    require(f"Missing IVB mobile close-X rendered term: {term}", term in rendered)
require("Shared mobile menu contract missing", "ds-mobile-menu-trigger" in rendered and "ds-mobile-menu-drawer" in rendered)
require("Tracking contract missing", "/player-card-actions.js" in rendered and "js-add-to-roster" in rendered and "IVB_HEAT_MAP" in rendered)

# Ensure this is local IVB-only source coverage, not a shared shell edit.
require("Unexpected shared shell nav edit marker in mobile Field Guide audit", "IVB_HEAT_MAP_MOBILE_FIELD_GUIDE_BOTTOM_PILL_V1" in source)

if failures:
    print("IVB mobile Field Guide audit failed:")
    for item in failures:
        print(f" - {item}")
    sys.exit(1)

print("OK: IVB mobile Field Guide bottom sticky audit passed")
