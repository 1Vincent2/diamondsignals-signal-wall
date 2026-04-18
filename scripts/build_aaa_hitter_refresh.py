#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import date, timedelta
import requests

OUT_JSON = Path("dist/aaa_hitter_refresh.json")
OUT_RAW = Path("dist/aaa_schedule_probe_yesterday.json")

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore"

def fetch_yesterday_final_games(probe_date: str) -> list[dict]:
    resp = requests.get(
        SCHEDULE_URL,
        params={"sportId": 11, "date": probe_date},
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    payload = resp.json()
    OUT_RAW.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    dates = payload.get("dates", []) or []
    games = dates[0].get("games", []) if dates else []

    final_games = []
    for g in games:
        status = ((g.get("status", {}) or {}).get("detailedState"))
        if status == "Final":
            final_games.append(g)
    return final_games

def extract_hitters_from_boxscore(game: dict) -> list[dict]:
    game_pk = game["gamePk"]
    resp = requests.get(
        BOXSCORE_URL.format(gamePk=game_pk),
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    for side in ["away", "home"]:
        team = ((payload.get("teams") or {}).get(side) or {})
        team_name = ((team.get("team") or {}).get("name"))
        players = team.get("players") or {}

        for pdata in players.values():
            person = pdata.get("person") or {}
            stats = ((pdata.get("stats") or {}).get("batting") or {})
            if stats.get("atBats") is None and stats.get("baseOnBalls") is None:
                continue

            ab = stats.get("atBats") or 0
            hits = stats.get("hits") or 0
            doubles = stats.get("doubles") or 0
            triples = stats.get("triples") or 0
            hr = stats.get("homeRuns") or 0
            bb = stats.get("baseOnBalls") or 0
            so = stats.get("strikeOuts") or 0

            singles = hits - doubles - triples - hr
            total_bases = singles + 2 * doubles + 3 * triples + 4 * hr
            iso = ((total_bases - hits) / ab) if ab else 0.0

            rows.append(
                {
                    "snapshot_date": game.get("gameDate", "")[:10],
                    "gamePk": game_pk,
                    "player_name": person.get("fullName"),
                    "player_id": person.get("id"),
                    "org": team_name,
                    "level": "AAA",
                    "pa_proxy": ab + bb,
                    "ab": ab,
                    "h": hits,
                    "bb": bb,
                    "so": so,
                    "hr": hr,
                    "iso": round(iso, 3),
                }
            )
    return rows

def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    probe_date = (date.today() - timedelta(days=1)).isoformat()
    status = "mlb_statsapi_aaa_boxscores_failed"
    error = None
    players = []
    final_game_count = 0

    try:
        final_games = fetch_yesterday_final_games(probe_date)
        final_game_count = len(final_games)

        for game in final_games:
            players.extend(extract_hitters_from_boxscore(game))

        status = "mlb_statsapi_aaa_boxscores_ok"
    except Exception as e:
        error = str(e)

    payload = {
        "generated_at": date.today().isoformat(),
        "status": status,
        "source": "MLB StatsAPI AAA Final Boxscores",
        "probe_date": probe_date,
        "final_game_count": final_game_count,
        "player_count": len(players),
        "error": error,
        "players": players,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"status={status}")
    print(f"final_game_count={final_game_count}")
    print(f"player_count={len(players)}")
    print(f"error={error}")

if __name__ == "__main__":
    main()
