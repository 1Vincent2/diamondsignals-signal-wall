#!/usr/bin/env python3
from pathlib import Path
import sys

builder = Path("dashboard/build_mlb_extraction.py").read_text(encoding="utf-8")
html = Path("dist/mlb-extraction/index.html").read_text(encoding="utf-8")

failures = []

required_builder_terms = [
    "MLB_EXTRACTION_SHARED_NAV_PATH_V1",
    "DESKTOP_NAV_TEMPLATE",
    "shell_nav_v2.html",
    "MOBILE_NAV_TEMPLATE",
    "shell_nav.html",
    "compact_desktop_nav_html",
    "mobile_shell_nav_html",
    'nav_html = compact_desktop_nav_html + "\\n" + mobile_shell_nav_html',
]

for term in required_builder_terms:
    if term not in builder:
        failures.append(f"Missing builder term: {term}")

if '<nav class="ds-pro-desktop-nav"' not in html:
    failures.append("Rendered MLB Extraction is missing compact desktop pro nav.")

if '<div class="topnav ds-shell-nav">' not in html:
    failures.append("Rendered MLB Extraction is missing shell nav/mobile menu contract.")

pro_pos = html.find('<nav class="ds-pro-desktop-nav"')
shell_pos = html.find('<div class="topnav ds-shell-nav">')

if pro_pos == -1 or shell_pos == -1:
    failures.append("Could not verify compact desktop nav plus shell nav order.")
elif pro_pos > shell_pos:
    failures.append("Compact desktop nav must render before shell nav.")

if ".ds-pro-desktop-nav + .topnav.ds-shell-nav" not in html:
    failures.append("Rendered CSS is missing desktop rule that hides shell nav behind compact pro nav.")

if "ds-mobile-menu-trigger" not in html or "ds-mobile-menu-drawer" not in html:
    failures.append("Mobile menu trigger/drawer missing from rendered MLB Extraction.")

if failures:
    print("MLB Extraction desktop nav parity audit failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("OK: MLB Extraction desktop nav parity audit passed")
