#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
HEADERS = DIST / "_headers"
WRITER = ROOT / "scripts" / "write_netlify_headers.py"

issues = []

REQUIRED_HEADER_RULES = {
    "/*": "Cache-Control: public,max-age=0,must-revalidate",
    "/*.html": "Cache-Control: public,max-age=0,must-revalidate",
    "/live/*": "Cache-Control: public,max-age=0,must-revalidate",
    "/status/*": "Cache-Control: public,max-age=0,must-revalidate",
    "/*.json": "Cache-Control: public,max-age=0,must-revalidate",
    "/*.js": "Cache-Control: public,max-age=0,must-revalidate",
    "/*.css": "Cache-Control: public,max-age=0,must-revalidate",
}

CORS_JSON_RULES = [
    "/dossier_canon.json",
    "/scout_metrics.json",
    "/player_index.json",
    "/admin/player_signal_index.json",
]

REPORT_HTML = [
    DIST / "index.html",
    DIST / "live" / "index.html",
    DIST / "typical-call-up" / "index.html",
    DIST / "velocity-decay-monitor" / "index.html",
    DIST / "stuff-disruption-feed" / "index.html",
    DIST / "ivb-heat-map" / "index.html",
    DIST / "waiver-wire" / "index.html",
    DIST / "apex-extraction" / "index.html",
    DIST / "mlb-extraction" / "index.html",
    DIST / "hidden-gems" / "index.html",
    DIST / "admin" / "kinetic-drift" / "index.html",
]

def read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

def section_for(headers_text: str, rule: str) -> str:
    lines = headers_text.splitlines()
    out = []
    capture = False
    for line in lines:
        if line.strip() == rule:
            capture = True
            out = [line]
            continue
        if capture:
            if line and not line.startswith(" ") and not line.startswith("\t"):
                break
            out.append(line)
    return "\n".join(out)

if not WRITER.exists():
    issues.append("missing scripts/write_netlify_headers.py")
else:
    writer_text = read(WRITER)
    for rule, cache_line in REQUIRED_HEADER_RULES.items():
        if rule not in writer_text or cache_line not in writer_text:
            issues.append(f"headers writer missing cache policy rule: {rule} -> {cache_line}")

if not HEADERS.exists():
    issues.append("missing generated dist/_headers")
else:
    headers_text = read(HEADERS)
    for rule, cache_line in REQUIRED_HEADER_RULES.items():
        sec = section_for(headers_text, rule)
        if not sec:
            issues.append(f"generated _headers missing rule: {rule}")
        elif cache_line not in sec:
            issues.append(f"generated _headers missing cache line for {rule}: {cache_line}")

    for rule in CORS_JSON_RULES:
        sec = section_for(headers_text, rule)
        if not sec:
            issues.append(f"generated _headers missing CORS JSON rule: {rule}")
        else:
            for required in [
                "Access-Control-Allow-Origin: https://app.diamondsignals.ai",
                "Access-Control-Allow-Methods: GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers: Content-Type",
                "Cache-Control: public,max-age=0,must-revalidate",
            ]:
                if required not in sec:
                    issues.append(f"generated _headers missing {required} for {rule}")

asset_refs = set()
for html in REPORT_HTML:
    if not html.exists():
        continue
    text = read(html)
    refs = re.findall(r'''(?:src|href)=["']([^"']+\.(?:js|css)(?:\?[^"']*)?)["']''', text)
    for ref in refs:
        asset_refs.add(ref)

unfingerprinted = sorted(
    ref for ref in asset_refs
    if not re.search(r"[.-][0-9a-f]{8,}\.(?:js|css)(?:\?|$)", ref)
)
if unfingerprinted:
    headers_text = read(HEADERS)
    for ext_rule in ["/*.js", "/*.css"]:
        sec = section_for(headers_text, ext_rule)
        if "Cache-Control: public,max-age=0,must-revalidate" not in sec:
            issues.append(f"unfingerprinted assets exist but {ext_rule} is not no-stale")
            break

print("--- DiamondSignals cache busting / CDN contract audit ---")
print(f"html_reports_checked: {sum(1 for p in REPORT_HTML if p.exists())}")
print(f"asset_refs_seen: {len(asset_refs)}")
print(f"unfingerprinted_asset_refs_seen: {len(unfingerprinted)}")
for ref in unfingerprinted:
    print(f"unfingerprinted_asset_ref_no_stale_required: {ref}")

print(f"cache_busting_cdn_issues: {len(issues)}")
for issue in issues:
    print(f"FAIL: {issue}")

if issues:
    print()
    print("FINAL_STATUS: FAIL_CACHE_BUSTING_CDN_CONTRACT")
    sys.exit(1)

print("html_cache_revalidation_enforced: true")
print("status_json_cache_revalidation_enforced: true")
print("unfingerprinted_js_css_no_stale_enforced: true")
print("cors_json_cache_revalidation_preserved: true")
print("netlify_headers_generated: true")
print()
print("FINAL_STATUS: PASS_CACHE_BUSTING_CDN_CONTRACT")
