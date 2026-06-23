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

BAD_SCOUT_PATTERNS = [
    "/scout/undefined",
    "/scout/null",
    "/scout/None",
    "/scout/NaN",
    "/scout//",
]

SCOUT_LINK_RE = re.compile(r'(?:href|data-profile-url)=["\'](/scout/([^/"\']+)/?)["\']')


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    print("--- DiamondSignals scout dossier route integrity audit ---")
    problems: list[str] = []

    if not DOSSIER_CANON.exists():
        problems.append(f"missing canonical dossier payload: {rel(DOSSIER_CANON)}")
        players = {}
    else:
        try:
            payload = json.loads(DOSSIER_CANON.read_text(encoding="utf-8"))
            players = payload.get("players", {})
            if not isinstance(players, dict):
                problems.append("dossier_canon.json players must be an object")
                players = {}
        except Exception as exc:
            problems.append(f"dossier_canon.json is not valid JSON: {exc}")
            players = {}

    canonical_ids = {str(pid).strip() for pid in players.keys() if str(pid).strip()}
    print(f"canonical_dossier_players: {len(canonical_ids)}")

    if not SCOUT_DIR.exists():
        problems.append(f"missing scout output directory: {rel(SCOUT_DIR)}")
    else:
        scout_index = SCOUT_DIR / "index.html"
        if scout_index.exists():
            print(f"OK: scout shell: {rel(scout_index)}")
        else:
            problems.append(f"missing scout shell: {rel(scout_index)}")

    missing_generated_pages = []
    for pid in sorted(canonical_ids):
        target = SCOUT_DIR / pid / "index.html"
        if not target.exists():
            missing_generated_pages.append(pid)

    print(f"missing_generated_dossier_pages: {len(missing_generated_pages)}")
    if missing_generated_pages:
        problems.append(
            "canonical players missing generated /scout/<player_id>/ pages: "
            + ", ".join(missing_generated_pages[:40])
        )

    total_links = 0
    scanned_surfaces = 0
    linked_ids: set[str] = set()

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

        matches = list(SCOUT_LINK_RE.finditer(html))
        print(f"{surface}: scout_links={len(matches)}")
        total_links += len(matches)

        for match in matches:
            route = match.group(1)
            pid = match.group(2).strip()
            linked_ids.add(pid)

            if not pid.isdigit():
                problems.append(f"{surface}: non-numeric scout id in route {route}")
                continue

            if pid not in canonical_ids:
                problems.append(f"{surface}: scout route id not in dossier_canon.json: {route}")

            target = SCOUT_DIR / pid / "index.html"
            if not target.exists():
                problems.append(f"{surface}: scout route target does not exist: {route}")

    print(f"surfaces_scanned: {scanned_surfaces}")
    print(f"total_scout_links: {total_links}")
    print(f"unique_linked_scout_ids: {len(linked_ids)}")

    print("\n--- summary ---")
    print(f"scout_dossier_route_integrity_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_SCOUT_DOSSIER_ROUTE_INTEGRITY")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_SCOUT_DOSSIER_ROUTE_INTEGRITY")


if __name__ == "__main__":
    main()
