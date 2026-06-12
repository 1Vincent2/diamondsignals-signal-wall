#!/usr/bin/env python3
from pathlib import Path
import sys

source_path = Path("dashboard/build_apex_extraction.py")
html_path = Path("dist/apex-extraction/index.html")

source = source_path.read_text(encoding="utf-8")
html = html_path.read_text(encoding="utf-8") if html_path.exists() else ""

failures = []

required_source_terms = [
    "APEX_DESKTOP_REFINEMENT_V1_AUDIT_LINE_STICKY_GUIDE",
    ".apex-active-audit-line",
    "display: none !important;",
    "@media screen and (min-width: 981px)",
    "position: fixed !important;",
    "top: 336px !important;",
    "bottom: auto !important;",
    "<p class=\"apex-active-audit-line\">",
    "Active Audit Layer:",
    "Click any player card to inspect the full performance audit.",
]

for term in required_source_terms:
    if term not in source:
        failures.append(f"Missing source term: {term}")

required_rendered_terms = [
    "APEX_DESKTOP_REFINEMENT_V1_AUDIT_LINE_STICKY_GUIDE",
    "<p class=\"apex-active-audit-line\">",
    "Active Audit Layer:",
    "Click any player card to inspect the full performance audit.",
    "position: fixed !important;",
    "top: 336px !important;",
    "bottom: auto !important;",
    "Field Guide",
]

for term in required_rendered_terms:
    if term not in html:
        failures.append(f"Missing rendered term: {term}")

if html:
    audit_pos = html.find("<p class=\"apex-active-audit-line\">")
    summary_pos = html.find("<div class=\"summary-card")
    if audit_pos == -1:
        failures.append("Could not find rendered Apex audit line.")
    elif summary_pos != -1 and audit_pos > summary_pos:
        failures.append("Apex audit line should appear inside the title card before the summary card.")

if "body.apex-field-guide-open" in source or "body.apex-field-guide-open" in html:
    failures.append("Unexpected Apex body field-guide-open class added; mobile drawer contract should remain untouched.")

if failures:
    print("Apex UI refinement audit failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("OK: Apex UI refinement audit passed")
