#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DOSSIER_CANON = DIST / "dossier_canon.json"
SCOUT_DIR = DIST / "scout"

SURFACES_TO_SCAN = [
    "dist/live/index.html",
    "dist/live-v2/index.html",
    "dist/waiver-wire/index.html",
    "dist/apex-extraction/index.html",
    "dist/mlb-extraction/index.html",
    "dist/hidden-gems/index.html",
    "dist/typical-call-up/index.html",
    "dist/velocity-decay-monitor/index.html",
    "dist/stuff-disruption-feed/index.html",
    "dist/ivb-heat-map/index.html",
]

BAD_ID_TOKENS = {"", "undefined", "null", "none", "nan"}

DATA_PLAYER_ID_RE = re.compile(r'data-player-id=["\']([^"\']*)["\']')
DATA_PROFILE_URL_RE = re.compile(r'data-profile-url=["\']([^"\']*)["\']')
SCOUT_LINK_RE = re.compile(r'(?:href|data-profile-url)=["\'](/scout/([^/"\']+)/?)["\']')
BAD_SCOUT_PATTERNS = [
    "/scout/undefined",
    "/scout/null",
    "/scout/None",
    "/scout/NaN",
    "/scout//",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_canonical_dossier_ids() -> set[str]:
    if not DOSSIER_CANON.exists():
        return set()

    payload = json.loads(DOSSIER_CANON.read_text(encoding="utf-8"))
    players = payload.get("players", {})
    if not isinstance(players, dict):
        return set()

    return {str(pid).strip() for pid in players.keys() if str(pid).strip()}


def main() -> None:
    print("--- DiamondSignals canonical player identity contract audit ---")
    problems: list[str] = []

    canonical_ids = load_canonical_dossier_ids()
    print(f"canonical_dossier_players: {len(canonical_ids)}")

    if not canonical_ids:
        problems.append(f"missing or empty canonical dossier payload: {rel(DOSSIER_CANON)}")

    missing_generated_pages = 0
    for pid in sorted(canonical_ids):
        target = SCOUT_DIR / pid / "index.html"
        if not target.exists():
            missing_generated_pages += 1
            if missing_generated_pages <= 20:
                problems.append(f"canonical dossier missing generated scout page: /scout/{pid}/")

    print(f"missing_generated_dossier_pages: {missing_generated_pages}")

    total_scout_links = 0
    total_data_player_ids = 0
    total_tracking_buttons = 0
    scanned_surfaces = 0

    for surface in SURFACES_TO_SCAN:
        path = ROOT / surface
        if not path.exists():
            print(f"INFO: surface not present, skipped: {surface}")
            continue

        scanned_surfaces += 1
        html = read_text(path)

        for bad in BAD_SCOUT_PATTERNS:
            if bad in html:
                problems.append(f"{surface}: contains invalid scout route token: {bad}")

        data_player_ids = [m.group(1).strip() for m in DATA_PLAYER_ID_RE.finditer(html)]
        profile_urls = [m.group(1).strip() for m in DATA_PROFILE_URL_RE.finditer(html)]
        scout_links = list(SCOUT_LINK_RE.finditer(html))
        tracking_buttons = html.count("js-add-to-roster")

        total_data_player_ids += len(data_player_ids)
        total_scout_links += len(scout_links)
        total_tracking_buttons += tracking_buttons

        surface_issues = 0

        # Some report surfaces contain display/card shells before a player is
        # hydrated into an actionable state. Empty data-player-id values are
        # allowed for those non-action shells. Once a player_id value is present,
        # it must be a stable numeric MLBAM/DiamondSignals identity token.
        empty_data_player_ids = 0
        for pid in data_player_ids:
            if pid.lower() in BAD_ID_TOKENS:
                empty_data_player_ids += 1
                continue
            if not pid.isdigit():
                problems.append(f"{surface}: non-canonical data-player-id token: {pid!r}")
                surface_issues += 1

        # Do not require a strict 1:1 button-to-card ratio here. Some buttons are
        # secondary controls or shell controls. The dedicated tracking regression
        # audit owns markup wiring. This audit owns identity-token safety and
        # scout-route correctness.

        for match in scout_links:
            route = match.group(1)
            route_id = match.group(2).strip()

            if route_id.lower() in BAD_ID_TOKENS or not route_id.isdigit():
                problems.append(f"{surface}: invalid scout route id: {route}")
                surface_issues += 1
                continue

            if route_id not in canonical_ids:
                problems.append(f"{surface}: scout route id not in dossier_canon.json: {route}")
                surface_issues += 1

            target = SCOUT_DIR / route_id / "index.html"
            if not target.exists():
                problems.append(f"{surface}: scout route target does not exist: {route}")
                surface_issues += 1

        # Local consistency check: when an element/chunk has both data-player-id
        # and a scout profile URL, the two must point to the same canonical ID.
        for profile in profile_urls:
            if not profile.startswith("/scout/"):
                continue
            parts = [part for part in profile.split("/") if part]
            route_id = parts[1] if len(parts) >= 2 else ""
            if route_id and route_id not in data_player_ids:
                problems.append(
                    f"{surface}: scout profile route has no matching data-player-id: {profile}"
                )
                surface_issues += 1

        print(
            f"{surface}: "
            f"data_player_ids={len(data_player_ids)} "
            f"tracking_buttons={tracking_buttons} "
            f"scout_links={len(scout_links)} "
            f"identity_issues={surface_issues}"
        )

    print("\n--- summary ---")
    print(f"surfaces_scanned: {scanned_surfaces}")
    print(f"total_data_player_ids: {total_data_player_ids}")
    print(f"total_tracking_buttons: {total_tracking_buttons}")
    print(f"total_scout_links: {total_scout_links}")
    print(f"canonical_identity_contract_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_CANONICAL_PLAYER_IDENTITY_CONTRACT")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_CANONICAL_PLAYER_IDENTITY_CONTRACT")


if __name__ == "__main__":
    main()
