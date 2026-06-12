from pathlib import Path
import sys

source = Path("dashboard/templates/hidden_gems/mlb_extraction.html")
rendered = Path("dist/mlb-extraction/index.html")

failures = []

src = source.read_text(encoding="utf-8")
html = rendered.read_text(encoding="utf-8") if rendered.exists() else ""

required_source_terms = [
    "MLB_EXTRACTION_DESKTOP_REFINEMENT_V1_AUDIT_LINE_STICKY_GUIDE",
    ".mlb-active-audit-line",
    "display: none !important;",
    "@media screen and (min-width: 981px)",
    "position: fixed !important;",
    "body .field-guide-btn.mlb-extraction-summary-guide",
    "<p class=\"mlb-active-audit-line\">",
    "Active Audit Layer:",
    "Click any player card to inspect the full performance audit.",
]

for term in required_source_terms:
    if term not in src:
        failures.append(f"Missing source term: {term}")

required_rendered_terms = [
    "MLB_EXTRACTION_DESKTOP_REFINEMENT_V1_AUDIT_LINE_STICKY_GUIDE",
    "<p class=\"mlb-active-audit-line\">",
    "Active Audit Layer:",
    "Click any player card to inspect the full performance audit.",
    "field-guide-btn mlb-extraction-summary-guide",
]

for term in required_rendered_terms:
    if term not in html:
        failures.append(f"Missing rendered term: {term}")

if html and html.find("<p class=\"mlb-active-audit-line\">") > html.find("<div class=\"summary-card\">"):
    failures.append("Audit line is not inside the title card before the summary card.")

if "body.mlb-field-guide-open" in src or "body.mlb-field-guide-open" in html:
    failures.append("Unexpected body field-guide-open class added; mobile drawer contract should remain untouched.")

if failures:
    print("MLB Extraction UI refinement audit failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("OK: MLB Extraction UI refinement audit passed")
