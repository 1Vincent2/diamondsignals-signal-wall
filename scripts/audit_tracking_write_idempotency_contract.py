#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_JS = ROOT / "src" / "js" / "player-card-actions.js"
DIST_JS = ROOT / "dist" / "player-card-actions.js"

issues = []

REQUIRED_PATTERNS = [
    'const APP_WATCHLIST_STATUS_URL = "https://app.diamondsignals.ai/api/watchlist/status"',
    "const appWatchlistStatusCache = new Map()",
    "const appWatchlistStatusPending = new Set()",
    "function getPlayerStatusKey(player)",
    'const playerId = String(player?.playerId || "").trim()',
    'url.searchParams.set("player_id", playerId)',
    'credentials: "include"',
    'cache: "no-store"',
    "appWatchlistStatusPending.add(playerId)",
    "appWatchlistStatusPending.delete(playerId)",
    "appWatchlistStatusCache.set(playerId, isTracked)",
    "payload.tracked || payload.in_watchlist || payload.exists",
    "function buildAppTrackingUrl(player)",
    'const playerId = String(player.playerId || "").trim()',
    'params.set("add_player_id", playerId)',
    'params.set("player_name", player.playerName)',
    'params.set("player_team", player.team)',
    'params.set("signal_source", player.sourceTag || "Signal Wall")',
    'authUrl.searchParams.set("next", nextPath)',
    'watchButton.disabled = true',
    'watchButton.getAttribute("data-provisioned") === "true"',
    'window.location.href = buildAppTrackingUrl(player)',
    'card.setAttribute("data-tracking-state", isProvisioned ? "active" : "idle")',
    'button.disabled = !!isProvisioned',
]

FORBIDDEN_PATTERNS = [
    "upsertWatchListPlayer(player);",
    "window.localStorage.setItem(STORAGE_KEY",
    "localStorage.setItem(STORAGE_KEY",
    "localStorage.setItem(LEGACY_STORAGE_KEY",
    "diamondsignals_roster_v1",
    'window.location.href = "/watch-list/"',
    'href="/watch-list/"',
]

def read(path: Path) -> str:
    if not path.exists():
        issues.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")

src = read(SRC_JS)
dist = read(DIST_JS)

print("--- DiamondSignals tracking write idempotency contract audit ---")

if src and dist and src != dist:
    issues.append("src/js/player-card-actions.js and dist/player-card-actions.js are not identical")

for label, text in [
    ("source", src),
    ("dist", dist),
]:
    if not text:
        continue

    for pattern in REQUIRED_PATTERNS:
        if pattern not in text:
            issues.append(f"{label} tracking JS missing idempotency pattern: {pattern}")

    for pattern in FORBIDDEN_PATTERNS:
        if pattern in text and pattern != "diamondsignals_roster_v1":
            issues.append(f"{label} tracking JS contains forbidden duplicate/local write pattern: {pattern}")

    # Legacy key may exist only as a retired key marker, never as a write target.
    if "diamondsignals_roster_v1" in text and "localStorage.setItem(LEGACY_STORAGE_KEY" in text:
        issues.append(f"{label} tracking JS writes retired legacy roster key")

    if text.count('params.set("add_player_id", playerId)') != 1:
        issues.append(f"{label} tracking JS should set add_player_id exactly once")

    if text.count('url.searchParams.set("player_id", playerId)') != 1:
        issues.append(f"{label} tracking JS should status-check player_id exactly once")

    if text.count("watchButton.disabled = true") != 1:
        issues.append(f"{label} tracking JS should disable clicked button exactly once before handoff")

print(f"source_dist_tracking_js_sync: {src == dist and bool(src) and bool(dist)}")
print(f"required_idempotency_patterns_checked: {len(REQUIRED_PATTERNS)}")
print(f"forbidden_duplicate_write_patterns_checked: {len(FORBIDDEN_PATTERNS)}")
print(f"tracking_write_idempotency_issues: {len(issues)}")

for issue in issues:
    print(f"FAIL: {issue}")

if issues:
    print()
    print("FINAL_STATUS: FAIL_TRACKING_WRITE_IDEMPOTENCY_CONTRACT")
    sys.exit(1)

print("stable_player_id_handoff_enforced: true")
print("app_status_player_id_recheck_enforced: true")
print("duplicate_localstorage_write_blocked: true")
print("double_click_handoff_guard_enforced: true")
print("tracked_state_reflected_from_app_status: true")
print()
print("FINAL_STATUS: PASS_TRACKING_WRITE_IDEMPOTENCY_CONTRACT")
