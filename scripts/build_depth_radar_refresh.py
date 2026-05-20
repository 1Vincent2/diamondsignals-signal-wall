#!/usr/bin/env python3

from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests

OUT_JSON = Path("dist/depth_radar_refresh.json")
OUT_RAW = Path("dist/depth_radar_schedule_probe.json")
STATUS_JSON = Path("dist/status/depth-radar.json")

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore"

DEPTH_LEVELS = {
    "AA": 12,
    "HIGH_A": 13,
    "LOW_A": 14,
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def fetch_final_games(level: str, sport_id: int, probe_date: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resp = requests.get(
        SCHEDULE_URL,
        params={"sportId": sport_id, "date": probe_date},
        timeout=30,
        headers=HEADERS,
    )
    resp.raise_for_status()
    payload = resp.json()

    dates = payload.get("dates", []) or []
    games = dates[0].get("games", []) if dates else []
    finals = [
        g for g in games
        if ((g.get("status", {}) or {}).get("detailedState")) == "Final"
    ]

    raw_summary = {
        "level": level,
        "sport_id": sport_id,
        "probe_date": probe_date,
        "game_count": len(games),
        "final_count": len(finals),
        "sample_games": [
            {
                "gamePk": g.get("gamePk"),
                "away": (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name"),
                "home": (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name"),
                "state": ((g.get("status") or {}).get("detailedState")),
            }
            for g in finals[:8]
        ],
    }

    return finals, raw_summary

def fetch_boxscore(game_pk: int) -> dict[str, Any]:
    resp = requests.get(
        BOXSCORE_URL.format(gamePk=game_pk),
        timeout=30,
        headers=HEADERS,
    )
    resp.raise_for_status()
    return resp.json()

def extract_hitters(payload: dict[str, Any], game: dict[str, Any], level: str, probe_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    game_pk = game.get("gamePk")

    for side in ["away", "home"]:
        team = ((payload.get("teams") or {}).get(side) or {})
        team_name = ((team.get("team") or {}).get("name"))
        players = team.get("players") or {}

        for pdata in players.values():
            person = pdata.get("person") or {}
            stats = ((pdata.get("stats") or {}).get("batting") or {})

            if stats.get("atBats") is None and stats.get("baseOnBalls") is None:
                continue

            ab = int(stats.get("atBats") or 0)
            hits = int(stats.get("hits") or 0)
            doubles = int(stats.get("doubles") or 0)
            triples = int(stats.get("triples") or 0)
            hr = int(stats.get("homeRuns") or 0)
            bb = int(stats.get("baseOnBalls") or 0)
            so = int(stats.get("strikeOuts") or 0)

            singles = max(hits - doubles - triples - hr, 0)
            total_bases = singles + 2 * doubles + 3 * triples + 4 * hr
            iso = ((total_bases - hits) / ab) if ab else 0.0
            pa_proxy = ab + bb

            depth_score = (
                (hits * 2.0)
                + (hr * 5.0)
                + (bb * 1.25)
                + (iso * 8.0)
                - (so * 0.35)
                + (pa_proxy * 0.15)
            )

            rows.append({
                "player_name": person.get("fullName"),
                "player_id": person.get("id"),
                "signal_type": "Hitter",
                "level": level,
                "org": team_name,
                "snapshot_date": game.get("gameDate", "")[:10] or probe_date,
                "gamePk": game_pk,
                "pa_proxy": pa_proxy,
                "ab": ab,
                "h": hits,
                "bb": bb,
                "so": so,
                "hr": hr,
                "iso": round(iso, 3),
                "depth_score": round(depth_score, 2),
                "source_badge": f"SRC: {level}_STATAPI_BOX_v1",
                "score_version": "DEPTH_v0.1",
                "why": f"{level} final-box signal: {hits} H, {bb} BB, {hr} HR, ISO {iso:.3f}.",
            })

    return rows

def extract_pitchers(payload: dict[str, Any], game: dict[str, Any], level: str, probe_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    game_pk = game.get("gamePk")

    for side in ["away", "home"]:
        team = ((payload.get("teams") or {}).get(side) or {})
        team_name = ((team.get("team") or {}).get("name"))
        players = team.get("players") or {}

        for pdata in players.values():
            person = pdata.get("person") or {}
            stats = ((pdata.get("stats") or {}).get("pitching") or {})

            if not stats:
                continue

            ip = str(stats.get("inningsPitched") or "0.0")
            hits = int(stats.get("hits") or 0)
            bb = int(stats.get("baseOnBalls") or 0)
            so = int(stats.get("strikeOuts") or 0)
            hr = int(stats.get("homeRuns") or 0)

            depth_score = (
                (so * 2.25)
                - (bb * 1.25)
                - (hits * 0.45)
                - (hr * 2.0)
            )

            rows.append({
                "player_name": person.get("fullName"),
                "player_id": person.get("id"),
                "signal_type": "Pitcher",
                "level": level,
                "org": team_name,
                "snapshot_date": probe_date,
                "gamePk": game_pk,
                "ip": ip,
                "h": hits,
                "bb": bb,
                "so": so,
                "hr": hr,
                "depth_score": round(depth_score, 2),
                "source_badge": f"SRC: {level}_STATAPI_BOX_v1",
                "score_version": "DEPTH_v0.1",
                "why": f"{level} final-box signal: {ip} IP, {so} K, {bb} BB, {hits} H allowed.",
            })

    return rows

def normalize_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    max_score = max(float(r.get("depth_score") or 0) for r in rows) or 0.0

    for row in rows:
        raw = float(row.get("depth_score") or 0)
        row["edge_score"] = round((raw / max_score) * 95, 1) if max_score > 0 else 0.0
        row["sample_note"] = f"{row.get('level')} final slate"

    return sorted(
        rows,
        key=lambda r: (
            float(r.get("edge_score") or 0),
            float(r.get("depth_score") or 0),
        ),
        reverse=True,
    )

def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)

    probe_date = (date.today() - timedelta(days=1)).isoformat()
    generated_at = utc_now_iso()

    all_hitters: list[dict[str, Any]] = []
    all_pitchers: list[dict[str, Any]] = []
    schedule_summaries: list[dict[str, Any]] = []
    errors: list[str] = []

    for level, sport_id in DEPTH_LEVELS.items():
        try:
            final_games, summary = fetch_final_games(level, sport_id, probe_date)
            schedule_summaries.append(summary)

            for game in final_games:
                game_pk = game.get("gamePk")
                if not game_pk:
                    continue

                try:
                    boxscore = fetch_boxscore(int(game_pk))
                    all_hitters.extend(extract_hitters(boxscore, game, level, probe_date))
                    all_pitchers.extend(extract_pitchers(boxscore, game, level, probe_date))
                except Exception as game_error:
                    errors.append(f"{level} gamePk={game_pk}: {game_error}")
        except Exception as level_error:
            errors.append(f"{level}: {level_error}")

    hitters_ranked = normalize_scores(all_hitters)
    pitchers_ranked = normalize_scores(all_pitchers)

    rows = normalize_scores(hitters_ranked[:20] + pitchers_ranked[:20])

    payload = {
        "report": "Depth Radar",
        "version": "depth_radar_v0.1",
        "generated_at": generated_at,
        "status": "ok" if rows else "empty",
        "mode": "milb_lower_levels_statsapi_boxscores_v0.1",
        "source": "MLB StatsAPI MiLB schedule + boxscore",
        "probe_date": probe_date,
        "levels": list(DEPTH_LEVELS.keys()),
        "source_locked_college": True,
        "section_counts": {
            "hitters": len(hitters_ranked),
            "pitchers": len(pitchers_ranked),
            "top_rows": len(rows),
        },
        "top_rows": rows[:24],
        "hitters": hitters_ranked[:50],
        "pitchers": pitchers_ranked[:50],
        "errors": errors,
    }

    status_payload = {
        "report_id": "depth_radar",
        "state": "fresh" if rows else "empty",
        "build_success": True,
        "degraded": bool(errors),
        "used_fallback": False,
        "generated_at": generated_at,
        "source_updated_at": generated_at,
        "source_age_minutes": 0.0,
        "mode": "milb_lower_levels_statsapi_boxscores_v0.1",
        "pipeline_layers": [
            "statsapi_milb_schedule",
            "statsapi_milb_boxscore",
            "aa_high_a_low_a",
            "d1_college_source_locked",
            "no_static_player_seed_fallback",
        ],
        "section_counts": payload["section_counts"],
        "errors": errors,
        "notes": [
            "Depth Radar V1 uses AA, High-A, and Low-A final boxscore rows from MLB StatsAPI.",
            "D1 college remains source-locked until a verified live college pipeline is wired.",
            "No static player seeds are rendered.",
        ],
    }

    OUT_RAW.write_text(json.dumps({"generated_at": generated_at, "schedules": schedule_summaries}, indent=2), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    STATUS_JSON.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {STATUS_JSON}")
    print(f"status={payload['status']}")
    print(f"top_rows={len(rows)}")
    print(f"hitters={len(hitters_ranked)}")
    print(f"pitchers={len(pitchers_ranked)}")
    print(f"errors={len(errors)}")

if __name__ == "__main__":
    main()
