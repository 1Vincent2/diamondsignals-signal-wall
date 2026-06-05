#!/usr/bin/env python3
"""
Promotion Watch integrity audit.

Validates:
- Promotion Watch JSON exists and has populated 72HR sections.
- Rendered Promotion Watch scout links do not point to missing dossier pages.
- Every current Promotion Watch candidate with a resolved/player id has a generated scout page.
- Every current Promotion Watch candidate has promotion_watch_context in dossier_canon.json.
- AAA GEMS label remains rendered.
- Old Depth Radar language does not return.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

PROMOTION_JSON = DIST / "typical-call-up" / "promotion_watch.json"
PROMOTION_HTML = DIST / "typical-call-up" / "index.html"
DOSSIER_CANON = DIST / "dossier_canon.json"
SCOUT_ROOT = DIST / "scout"


SECTIONS = [
    "pitchers_72hr",
    "hitters_72hr",
    "pitchers_14day",
    "hitters_14day",
    "recent_arrivals",
    "depth_radar",
]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing required artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"unable to parse JSON {path}: {exc}")


def candidate_id(row: dict) -> str:
    return str(
        row.get("resolved_player_id")
        or row.get("player_id")
        or row.get("mlbam_id")
        or row.get("id")
        or ""
    ).strip()


def main() -> None:
    promotion = load_json(PROMOTION_JSON)
    canon = load_json(DOSSIER_CANON)

    if not PROMOTION_HTML.exists():
        fail(f"missing rendered Promotion Watch HTML: {PROMOTION_HTML}")

    html = PROMOTION_HTML.read_text(encoding="utf-8", errors="ignore")
    players = canon.get("players") or {}
    top = promotion.get("top_signals") or {}

    if not isinstance(top, dict):
        fail("promotion_watch.json top_signals is missing or invalid")

    section_counts = {
        section: len(top.get(section) or [])
        for section in SECTIONS
    }

    print("--- section counts ---")
    for section, count in section_counts.items():
        print(f"{section}: {count}")

    if section_counts["pitchers_72hr"] == 0:
        fail("pitchers_72hr is empty")
    if section_counts["hitters_72hr"] == 0:
        fail("hitters_72hr is empty")

    if "No 72 HR pitching prospect signals available." in html:
        fail("72HR pitching placeholder rendered")
    if "No 72 HR hitting prospect signals available." in html:
        fail("72HR hitting placeholder rendered")

    if "AAA GEMS" not in html:
        fail("AAA GEMS label missing")

    if "Depth Radar" in html or "DEPTH RADAR" in html:
        fail("old Depth Radar language rendered")

    existing_scout_ids = {
        path.parent.name
        for path in SCOUT_ROOT.glob("*/index.html")
        if path.parent.name.isdigit()
    }

    rendered_profile_urls = sorted(set(re.findall(r'data-profile-url="(/scout/([^"/]+)/?)"', html)))
    rendered_dead_links = [
        (pid, url)
        for url, pid in rendered_profile_urls
        if pid not in existing_scout_ids
    ]

    print()
    print("--- rendered link health ---")
    print("rendered_scout_profile_urls:", len(rendered_profile_urls))
    print("dead_rendered_scout_links:", len(rendered_dead_links))
    print("disabled_profile_urls:", html.count('data-profile-url="#"'))

    if rendered_dead_links:
        for pid, url in rendered_dead_links[:50]:
            print("-", pid, url)
        fail("rendered Promotion Watch contains dead scout links")

    candidates: dict[str, dict] = {}

    for section in SECTIONS:
        rows = top.get(section) or []
        if not isinstance(rows, list):
            fail(f"{section} is not a list")

        for row in rows:
            if not isinstance(row, dict):
                continue

            pid = candidate_id(row)
            if not pid:
                continue

            if pid not in candidates:
                candidates[pid] = {
                    "player_id": pid,
                    "player_name": row.get("player_name") or row.get("name") or f"Player {pid}",
                    "sections": [],
                }

            candidates[pid]["sections"].append(section)

    missing_pages = []
    missing_context = []
    missing_render_marker = []

    for pid, info in candidates.items():
        scout_page = SCOUT_ROOT / pid / "index.html"

        if not scout_page.exists():
            missing_pages.append(info)
            continue

        player = players.get(pid)
        if not player or not player.get("promotion_watch_context"):
            missing_context.append(info)

        scout_html = scout_page.read_text(encoding="utf-8", errors="ignore")
        if "Promotion Watch // Current Signal Context" not in scout_html or "renderPromotionWatchContext" not in scout_html:
            missing_render_marker.append(info)

    print()
    print("--- candidate dossier health ---")
    print("unique_promotion_candidates:", len(candidates))
    print("missing_scout_pages:", len(missing_pages))
    print("missing_promotion_watch_context:", len(missing_context))
    print("missing_render_marker:", len(missing_render_marker))

    if missing_pages:
        for item in missing_pages[:50]:
            print("-", item["player_id"], item["player_name"], item["sections"])
        fail("current Promotion Watch candidates missing scout pages")

    if missing_context:
        for item in missing_context[:50]:
            print("-", item["player_id"], item["player_name"], item["sections"])
        fail("current Promotion Watch candidates missing promotion_watch_context")

    if missing_render_marker:
        for item in missing_render_marker[:50]:
            print("-", item["player_id"], item["player_name"], item["sections"])
        fail("current Promotion Watch scout pages missing context render marker")

    section_presence = Counter()
    for item in candidates.values():
        for section in item["sections"]:
            section_presence[section] += 1

    print()
    print("--- unique candidate section presence ---")
    for section, count in section_presence.most_common():
        print(f"{section}: {count}")

    print()
    print("PASS: Promotion Watch integrity audit clean.")


if __name__ == "__main__":
    main()
