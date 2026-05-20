#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import time
import urllib.request

URL = f"https://signals.diamondsignals.ai/waiver-wire/?v={int(time.time())}"

def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")

html = fetch(URL)

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

if bad_links:
    print("\n--- bad links ---")
    for link in bad_links:
        print(link)
    print("\nFINAL_STATUS: FAIL")
    sys.exit(1)

if len(audit_links) != 12:
    print("\nFINAL_STATUS: FAIL")
    print("Expected 12 Waiver Wire Performance Audit links.")
    sys.exit(1)

print("\nFINAL_STATUS: PASS")
