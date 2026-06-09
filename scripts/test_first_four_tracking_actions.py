#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(".")
JS_PATH = ROOT / "dist/player-card-actions.js"

FIRST_FOUR = [
    ROOT / "dist/live/index.html",
    ROOT / "dist/waiver-wire/index.html",
    ROOT / "dist/apex-extraction/index.html",
    ROOT / "dist/mlb-extraction/index.html",
]

failures = 0

def check(path):
    global failures
    html = path.read_text(encoding="utf-8", errors="ignore")
    rel = path.as_posix()

    card_count = html.count("js-player-card")
    button_count = html.count("js-add-to-roster")
    player_id_count = html.count("data-player-id=")
    profile_url_count = html.count("data-profile-url=")

    if "js-player-card" not in html:
        print(f"FAIL: {rel} has no js-player-card")
        failures += 1

    if "js-add-to-roster" not in html:
        print(f"FAIL: {rel} has no js-add-to-roster")
        failures += 1

    if "/player-card-actions.js" not in html:
        print(f"FAIL: {rel} does not load /player-card-actions.js")
        failures += 1

    if button_count > card_count * 2:
        print(f"FAIL: {rel} suspicious button/card ratio: {button_count} buttons vs {card_count} cards")
        failures += 1

    if player_id_count < button_count:
        print(f"FAIL: {rel} has fewer data-player-id attrs than tracking buttons")
        failures += 1

    if profile_url_count < card_count:
        print(f"FAIL: {rel} has fewer data-profile-url attrs than tracking cards")
        failures += 1

    if re.search(r'app\.diamondsignals\.ai/watchlist|auth\?next=/watchlist|/watchlist', html):
        print(f"FAIL: {rel} contains stale watchlist route")
        failures += 1

    print(f"OK: {rel} first-four tracking contract is wired")

if not JS_PATH.exists():
    print("FAIL: missing dist/player-card-actions.js")
    sys.exit(1)

for path in FIRST_FOUR:
    if not path.exists():
        print(f"FAIL: missing {path}")
        failures += 1
    else:
        check(path)

if failures:
    print(f"\nFirst-four audit failed with {failures} issue(s).")
    sys.exit(1)

print("\nFirst-four audit passed.")
