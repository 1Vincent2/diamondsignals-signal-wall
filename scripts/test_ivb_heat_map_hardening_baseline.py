from pathlib import Path
import sys

SOURCE = Path("dashboard/build_ivb_heat_map.py")
RENDERED = Path("dist/ivb-heat-map/index.html")
STATUS = Path("dist/status/ivb-heat-map.json")

failures = []

source = SOURCE.read_text(encoding="utf-8") if SOURCE.exists() else ""
rendered = RENDERED.read_text(encoding="utf-8") if RENDERED.exists() else ""
status = STATUS.read_text(encoding="utf-8") if STATUS.exists() else ""

def require(label, condition):
    if not condition:
        failures.append(label)

require("Missing IVB source builder", SOURCE.exists())
require("Missing rendered IVB page", RENDERED.exists())
require("Missing IVB status file", STATUS.exists())

require("IVB status mode missing dynamic marker", "statcast_supabase_ivb_heat_map_dynamic_v1" in source or "statcast_supabase_ivb_heat_map_dynamic_v1" in status)
require("IVB rendered page missing tracking script", "/player-card-actions.js" in rendered)
require("IVB rendered page missing player card tracking/action class", "js-player-card" in rendered)
require("IVB rendered page missing source tag", "IVB_HEAT_MAP" in rendered)
require("IVB rendered page missing Field Guide trigger", "Field Guide" in rendered and "openFieldGuide()" in rendered)
require("IVB rendered page missing Field Guide modal/drawer", "fieldGuideModal" in rendered or "field-guide-drawer" in rendered)
require("IVB rendered page missing mobile menu contract", "ds-mobile-menu-trigger" in rendered and "ds-mobile-menu-drawer" in rendered)

# Current known pre-hardening state. This is allowed for baseline, but should change in later hardening.
require("IVB baseline should still show old shell nav before nav hardening", '<div class="topnav ds-shell-nav">' in rendered)

# Guard against accidentally losing scientific copy while hardening.
for term in [
    "IVB RAW",
    "IVB VS AVG",
    "APEX RISE",
    "THE DEAD ZONE",
    "WHIFF PROB",
    "CLIMBERS",
]:
    require(f"Missing IVB Field Guide term: {term}", term in rendered)

# Guard that VAA language is intentional and no longer uses old scaffold wording.
require("Old VAA scaffold language should be removed", "VAA is scaffolded" not in rendered and "VAA is scaffolded" not in source)
require("Updated VAA reserved-layer language missing", "VAA is reserved for the next upstream pitch-plane layer" in rendered or "VAA is reserved for the next upstream pitch-plane layer" in source)

if failures:
    print("IVB Heat Map hardening baseline audit failed:")
    for item in failures:
        print(f" - {item}")
    sys.exit(1)

print("OK: IVB Heat Map hardening baseline audit passed")
