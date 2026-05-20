#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request

CACHE_BUSTER = int(time.time())
URL = f"https://signals.diamondsignals.ai/waiver-wire/?v={CACHE_BUSTER}"
STATUS_URL = f"https://signals.diamondsignals.ai/status/waiver-wire.json?v={CACHE_BUSTER}"

def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")

html = fetch(URL)

try:
    status_payload = json.loads(fetch(STATUS_URL))
except Exception:
    status_payload = {}

section_counts = status_payload.get("section_counts") or {}
mode = status_payload.get("mode")
pipeline_layers = status_payload.get("pipeline_layers") or []
waiver_asset_count = int(section_counts.get("waiver_assets") or 0)

zero_verified_candidate_mode = (
    waiver_asset_count == 0
    and mode == "verified_dynamic_candidates_only_v1"
    and "no_static_seed_fallback" in pipeline_layers
)

audit_links = re.findall(
    r'class="performance-audit-button" href="([^"]+)"',
    html,
)

bad_links = [
    link for link in audit_links
    if link == "#" or "watchlist?player_id=" in link or not link.startswith("/scout/")
]

print("--- Waiver Wire audit-link hardening check ---")
print(f"waiver_wire_url: {URL}")
print(f"audit_link_count: {len(audit_links)}")
print(f"scout_link_count: {sum(1 for link in audit_links if link.startswith('/scout/'))}")
print(f"bad_link_count: {len(bad_links)}")
print(f"status_url: {STATUS_URL}")
print(f"status_mode: {mode}")
print(f"status_waiver_assets: {waiver_asset_count}")
print(f"zero_verified_candidate_mode: {zero_verified_candidate_mode}")

if bad_links:
    print("\n--- bad links ---")
    for link in bad_links:
        print(link)
    print("\nFINAL_STATUS: FAIL")
    sys.exit(1)

if zero_verified_candidate_mode and len(audit_links) == 0:
    print("\nFINAL_STATUS: PASS")
    print("Verified zero-candidate Waiver mode: no static/pre-seeded audit links expected.")
    sys.exit(0)

if len(audit_links) != waiver_asset_count:
    print("\nFINAL_STATUS: FAIL")
    print(f"Expected {waiver_asset_count} Waiver Wire Performance Audit links from live status.")
    sys.exit(1)

print("\nFINAL_STATUS: PASS")
