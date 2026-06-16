#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import date, timedelta
import requests

OUT_JSON = Path("dist/aaa_hitter_refresh.json")
OUT_PITCHERS_JSON = Path("dist/aaa_pitcher_refresh_probe.json")
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

    return [
        g for g in games
        if ((g.get("status", {}) or {}).get("detailedState")) == "Final"
    ]

LOOKBACK_DAYS = 7

def fetch_latest_final_games() -> tuple[str, list[dict], list[dict]]:
    attempts = []
    for days_back in range(1, LOOKBACK_DAYS + 1):
        probe_date = (date.today() - timedelta(days=days_back)).isoformat()
        try:
            final_games = fetch_yesterday_final_games(probe_date)
            attempts.append({
                "probe_date": probe_date,
                "final_game_count": len(final_games),
                "error": None,
            })
            if final_games:
                return probe_date, final_games, attempts
        except Exception as e:
            attempts.append({
                "probe_date": probe_date,
                "final_game_count": 0,
                "error": str(e),
            })
    return (date.today() - timedelta(days=1)).isoformat(), [], attempts


def extract_pitchers_from_boxscore(game: dict, probe_date: str) -> list[dict]:
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
            stats = ((pdata.get("stats") or {}).get("pitching") or {})
            if not stats:
                continue

            ip = str(stats.get("inningsPitched") or "0.0")
            hits = int(stats.get("hits") or 0)
            bb = int(stats.get("baseOnBalls") or 0)
            so = int(stats.get("strikeOuts") or 0)
            hr = int(stats.get("homeRuns") or 0)

            live_score = (so * 2.0) - (bb * 1.0) - (hits * 0.5) - (hr * 2.0)

            rows.append(
                {
                    "snapshot_date": probe_date,
                    "gamePk": game_pk,
                    "player_name": person.get("fullName"),
                    "player_id": person.get("id"),
                    "org": team_name,
                    "level": "AAA",
                    "ip": ip,
                    "h": hits,
                    "bb": bb,
                    "so": so,
                    "hr": hr,
                    "live_score": round(live_score, 2),
                }
            )
    return rows


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

            live_score = (
                (hits * 2.0)
                + (hr * 4.0)
                + (bb * 1.0)
                + (iso * 3.0)
                - (so * 0.5)
            )

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
                    "live_score": round(live_score, 2),
                }
            )
    return rows

def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    status = "mlb_statsapi_aaa_boxscores_failed"
    error = None
    players = []
    pitcher_rows = []
    final_game_count = 0
    probe_attempts = []

    try:
        probe_date, final_games, probe_attempts = fetch_latest_final_games()
        final_game_count = len(final_games)

        for game in final_games:
            players.extend(extract_hitters_from_boxscore(game))
            pitcher_rows.extend(extract_pitchers_from_boxscore(game, probe_date))

        players = sorted(
            players,
            key=lambda r: (
                float(r.get("live_score", 0)),
                float(r.get("hr", 0)),
                float(r.get("iso", 0)),
                float(r.get("h", 0)),
            ),
            reverse=True,
        )

        status = "mlb_statsapi_aaa_boxscores_ranked_ok"
    except Exception as e:
        error = str(e)

    payload = {
        "generated_at": date.today().isoformat(),
        "status": status,
        "source": "MLB StatsAPI AAA Final Boxscores",
        "probe_date": probe_date,
        "final_game_count": final_game_count,
        "lookback_days": LOOKBACK_DAYS,
        "probe_attempts": probe_attempts,
        "player_count": len(players),
        "top_20": players[:20],
        "error": error,
        "players": players,
    }

    pitcher_rows = sorted(
        pitcher_rows,
        key=lambda r: (
            float(r.get("live_score", 0)),
            float(r.get("so", 0)),
            -float(r.get("bb", 0)),
            -float(r.get("h", 0)),
        ),
        reverse=True,
    )

    pitcher_payload = {
        "generated_at": date.today().isoformat(),
        "status": status,
        "source": "MLB StatsAPI AAA Final Boxscores",
        "probe_date": probe_date,
        "final_game_count": final_game_count,
        "lookback_days": LOOKBACK_DAYS,
        "probe_attempts": probe_attempts,
        "player_count": len(pitcher_rows),
        "top_20": pitcher_rows[:20],
        "error": error,
        "players": pitcher_rows,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_PITCHERS_JSON.write_text(json.dumps(pitcher_payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_PITCHERS_JSON}")
    print(f"status={status}")
    print(f"final_game_count={final_game_count}")
    print(f"hitter_player_count={len(players)}")
    print(f"pitcher_player_count={len(pitcher_rows)}")
    print(f"error={error}")

if __name__ == "__main__":
    main()
