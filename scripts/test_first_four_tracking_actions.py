#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(".")
JS_PATH = ROOT / "dist/player-card-actions.js"

FIRST_FOUR = [
    ROOT / "dist/live/index.html",
    ROOT / "dist/waiver-wire/index.html",
    ROOT / "dist/apex-extraction/index.html",
    ROOT / "dist/mlb-extraction/index.html",
]

REQUIRED_SHARED_JS = [
    'const APP_AUTH_URL = "https://app.diamondsignals.ai/auth"',
    'const APP_WATCHLIST_PATH = "/watchlist"',
    "function buildAppTrackingUrl(player)",
    'params.set("add_player_id", playerId)',
    'authUrl.searchParams.set("next", nextPath)',
    'window.location.href = buildAppTrackingUrl(player)',
]

LEGACY_EXACT_REDIRECT = 'window.location.href = "/watch-list/"'

failures = 0


def fail(message):
    global failures
    print(f"FAIL: {message}")
    failures += 1


def ok(message):
    print(f"OK: {message}")


def check_shared_js():
    if not JS_PATH.exists():
        fail("missing dist/player-card-actions.js")
        return

    js = JS_PATH.read_text(encoding="utf-8", errors="ignore")

    for marker in REQUIRED_SHARED_JS:
        if marker not in js:
            fail(f"shared JS missing authenticated tracking marker: {marker}")

    if LEGACY_EXACT_REDIRECT in js:
        fail("shared JS contains stale local watch-list redirect")


def check_surface(path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    rel = path.as_posix()

    card_count = html.count("js-player-card")
    button_count = html.count("js-add-to-roster")
    player_id_count = html.count("data-player-id=")
    profile_url_count = html.count("data-profile-url=")

    if "js-player-card" not in html:
        fail(f"{rel} has no js-player-card")

    if "js-add-to-roster" not in html:
        fail(f"{rel} has no js-add-to-roster")

    if "/player-card-actions.js" not in html:
        fail(f"{rel} does not load /player-card-actions.js")

    if button_count > card_count * 2:
        fail(f"{rel} suspicious button/card ratio: {button_count} buttons vs {card_count} cards")

    if player_id_count < button_count:
        fail(f"{rel} has fewer data-player-id attrs than tracking buttons")

    if profile_url_count < card_count:
        fail(f"{rel} has fewer data-profile-url attrs than tracking cards")

    # Public nav links to /watch-list/ are allowed for the old freemium staging page.
    # The stale production blocker is only the old imperative JS redirect.
    if LEGACY_EXACT_REDIRECT in html:
        fail(f"{rel} contains stale local watch-list redirect")

    ok(f"{rel} first-four tracking contract is wired")


check_shared_js()

for path in FIRST_FOUR:
    if not path.exists():
        fail(f"missing {path}")
    else:
        check_surface(path)

if failures:
    print(f"\nFirst-four audit failed with {failures} issue(s).")
    sys.exit(1)

print("\nFirst-four audit passed.")
