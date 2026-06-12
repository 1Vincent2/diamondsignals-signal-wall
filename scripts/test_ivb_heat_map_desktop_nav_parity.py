#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dashboard/build_ivb_heat_map.py"
HTML = ROOT / "dist/ivb-heat-map/index.html"

source = SOURCE.read_text(encoding="utf-8", errors="replace")
html = HTML.read_text(encoding="utf-8", errors="replace") if HTML.exists() else ""

errors = []

for term in [
    "IVB_HEAT_MAP_SHARED_NAV_PATH_V1",
    "DESKTOP_NAV_TEMPLATE",
    "MOBILE_NAV_TEMPLATE",
    "shell_nav_v2.html",
    "shell_nav.html",
]:
    if term not in source:
        errors.append(f"Missing source term: {term}")

for term in [
    '<nav class="ds-pro-desktop-nav"',
    '<div class="topnav ds-shell-nav">',
    '[ LAB ]</span> IVB',
    '[ LAB ]</span> IVB Heat Map',
]:
    if term not in html:
        errors.append(f"Missing rendered term: {term}")

desktop_pos = html.find('<nav class="ds-pro-desktop-nav"')
shell_pos = html.find('<div class="topnav ds-shell-nav">')
if desktop_pos < 0 or shell_pos < 0:
    errors.append("Could not verify both compact desktop nav and shell/mobile nav rendered.")
elif desktop_pos > shell_pos:
    errors.append("Compact desktop nav must render before preserved shell/mobile nav.")

if errors:
    print("IVB Heat Map desktop nav parity audit failed:")
    for err in errors:
        print(f" - {err}")
    raise SystemExit(1)

print("OK: IVB Heat Map desktop nav parity audit passed")
