#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
JS_PATH = DIST / "player-card-actions.js"

SURFACE_PATHS = [
    DIST / "index.html",
    DIST / "live" / "index.html",
    DIST / "waiver-wire" / "index.html",
    DIST / "watch-list" / "index.html",
    DIST / "apex-extraction" / "index.html",
    DIST / "mlb-extraction" / "index.html",
    DIST / "typical-call-up" / "index.html",
    DIST / "velocity-decay-monitor" / "index.html",
    DIST / "stuff-disruption-feed" / "index.html",
    DIST / "ivb-heat-map" / "index.html",
]

FORBIDDEN_ROUTE_PATTERNS = [
    'window.location.href = "/watch-list/"',
]

REQUIRED_JS_PATTERNS = [
    'const STORAGE_KEY = "diamondsignals_watch_list_v1"',
    'const APP_AUTH_URL = "https://app.diamondsignals.ai/auth"',
    'const APP_WATCHLIST_PATH = "/watchlist"',
    "function upsertWatchListPlayer(player)",
    "function buildAppTrackingUrl(player)",
    "function syncCardProvisionStates()",
    "function scheduleProvisionSync()",
    'params.set("add_player_id", playerId)',
    'params.set("player_name", player.playerName)',
    'params.set("player_team", player.team)',
    'params.set("signal_source", player.sourceTag || "Signal Wall")',
    'authUrl.searchParams.set("next", nextPath)',
    'window.location.href = buildAppTrackingUrl(player)',
    'window.addEventListener("pageshow", scheduleProvisionSync)',
    'window.addEventListener("resize", scheduleProvisionSync)',
    'window.addEventListener("orientationchange", scheduleProvisionSync)',
    'card.classList.toggle("tracking-active"',
    'card.setAttribute("data-tracking-state"',
]

def fail(message):
    print(f"FAIL: {message}")
    return 1

def ok(message):
    print(f"OK: {message}")

def read(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8", errors="ignore")

def audit_shared_js():
    js = read(JS_PATH)

    failures = 0
    for pattern in REQUIRED_JS_PATTERNS:
        if pattern not in js:
            print(f"FAIL: shared JS missing required pattern: {pattern}")
            failures += 1

    for pattern in FORBIDDEN_ROUTE_PATTERNS:
        if re.search(pattern, js):
            print(f"FAIL: shared JS contains forbidden route/pattern: {pattern}")
            failures += 1

    if failures == 0:
        ok("shared player-card-actions.js tracking contract is intact")

    return failures

def audit_surface(path):
    html = read(path)
    rel = path.relative_to(ROOT)

    failures = 0

    has_cards = "js-player-card" in html
    has_buttons = "js-add-to-roster" in html
    has_script = "/player-card-actions.js" in html

    if has_cards or has_buttons:
        if not has_cards:
            print(f"FAIL: {rel} has tracking buttons but no js-player-card")
            failures += 1
        if not has_buttons:
            print(f"FAIL: {rel} has player cards but no js-add-to-roster")
            failures += 1
        if not has_script:
            print(f"FAIL: {rel} has tracking UI but does not load /player-card-actions.js")
            failures += 1

        card_count = html.count("js-player-card")
        button_count = html.count("js-add-to-roster")
        player_id_count = html.count("data-player-id=")
        profile_url_count = html.count("data-profile-url=")

        if button_count > card_count * 2:
            print(f"FAIL: {rel} suspicious button/card ratio: {button_count} buttons vs {card_count} cards")
            failures += 1

        if player_id_count < button_count:
            print(f"FAIL: {rel} has fewer data-player-id attrs than tracking buttons")
            failures += 1

        if profile_url_count < card_count:
            print(f"FAIL: {rel} has fewer data-profile-url attrs than tracking cards")
            failures += 1

    for pattern in FORBIDDEN_ROUTE_PATTERNS:
        if re.search(pattern, html):
            print(f"FAIL: {rel} contains forbidden route/pattern: {pattern}")
            failures += 1

    if rel.as_posix() != "dist/watch-list/index.html" and 'href="/watch-list/"' in html:
        print(f"FAIL: {rel} contains legacy Signal Wall watch-list nav link")
        failures += 1

    if failures == 0:
        if has_cards or has_buttons:
            ok(f"{rel} tracking markup is wired")
        else:
            ok(f"{rel} has no tracking UI to audit")

    return failures

def main():
    failures = 0

    if not JS_PATH.exists():
        return fail(f"missing shared JS: {JS_PATH}")

    failures += audit_shared_js()

    for path in SURFACE_PATHS:
        if path.exists():
            failures += audit_surface(path)
        else:
            print(f"WARN: skipped missing route: {path.relative_to(ROOT)}")

    if failures:
        print(f"\nTracking regression audit failed with {failures} issue(s).")
        return 1

    print("\nTracking regression audit passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
