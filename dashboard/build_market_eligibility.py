#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
MARKET_DIR = DIST / "market"
OUT_PATH = MARKET_DIR / "waiver_market_eligibility.json"

# V2 design:
# - No manual CSV
# - No exported file
# - No static/pre-seeded players
# - This file is generated only from live platform/API ownership feeds
#
# Next activation target:
# - Yahoo Fantasy Sports API OAuth + percent_owned / ownership endpoint
# - Optional later: ESPN/Fantrax/NFBC/rotowire-style market feeds if licensed/available

MAX_ROSTERED_PCT = int(os.getenv("WAIVER_MAX_ROSTERED_PCT", "35"))

REQUIRED_ENV = [
    "YAHOO_CLIENT_ID",
    "YAHOO_CLIENT_SECRET",
    "YAHOO_REFRESH_TOKEN",
]


def load_local_env() -> None:
    """
    Local development helper only.
    Loads .env into os.environ without printing secrets.
    Netlify/prod should still use real environment variables.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_ready() -> bool:
    return all(bool(os.getenv(key)) for key in REQUIRED_ENV)


def get_yahoo_access_token() -> tuple[str | None, dict]:
    client_id = os.getenv("YAHOO_CLIENT_ID")
    client_secret = os.getenv("YAHOO_CLIENT_SECRET")
    refresh_token = os.getenv("YAHOO_REFRESH_TOKEN")

    if not client_id or not client_secret or not refresh_token:
        return None, {"error": "missing_yahoo_oauth_env"}

    body = urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }).encode("utf-8")

    basic = base64.b64encode(
        f"{client_id}:{client_secret}".encode("utf-8")
    ).decode("utf-8")

    req = Request(
        "https://api.login.yahoo.com/oauth2/get_token",
        data=body,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return None, {"error": "refresh_token_exchange_failed", "detail": str(exc)}

    return payload.get("access_token"), payload


def yahoo_api_get(path: str, access_token: str) -> dict:
    url = f"https://fantasysports.yahooapis.com/fantasy/v2/{path}"
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}format=json"

    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        method="GET",
    )

    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def extract_percent_owned(player_blob) -> int | None:
    for node in walk_json(player_blob):
        percent_owned = node.get("percent_owned")
        if not isinstance(percent_owned, list):
            continue

        for item in percent_owned:
            if isinstance(item, dict) and "value" in item:
                try:
                    return int(float(item.get("value")))
                except Exception:
                    return None

    return None


def extract_player_meta(player_blob: dict) -> dict | None:
    player_items = player_blob.get("player")
    if not isinstance(player_items, list) or not player_items:
        return None

    flat = []
    for item in player_items:
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, dict):
            flat.append(item)

    meta = {
        "player_id": "",
        "player_key": "",
        "player_name": "",
        "team": "",
        "position": "",
    }

    for item in flat:
        if not isinstance(item, dict):
            continue

        if "player_id" in item:
            meta["player_id"] = str(item.get("player_id") or "").strip()

        if "player_key" in item:
            meta["player_key"] = str(item.get("player_key") or "").strip()

        name = item.get("name")
        if isinstance(name, dict) and name.get("full"):
            meta["player_name"] = str(name.get("full") or "").strip()

        if "editorial_team_abbr" in item:
            meta["team"] = str(item.get("editorial_team_abbr") or "").strip()

        if "display_position" in item:
            meta["position"] = str(item.get("display_position") or "").strip()

    if not meta["player_name"]:
        return None

    return meta


def extract_players_from_yahoo_payload(payload: dict) -> list[dict]:
    players = []

    for node in walk_json(payload):
        if not isinstance(node, dict):
            continue

        player_blob = node.get("player")
        if not isinstance(player_blob, list):
            continue

        wrapper = {"player": player_blob}
        meta = extract_player_meta(wrapper)
        if not meta:
            continue

        rostered_pct = extract_percent_owned(wrapper)
        if rostered_pct is None:
            continue

        players.append({
            **meta,
            "rostered_pct": rostered_pct,
            "market_pct_verified": True,
            "market_provider": "yahoo_fantasy_api",
            "market_source": "platform_api",
            "eligible": rostered_pct <= MAX_ROSTERED_PCT,
        })

    return players


def fetch_yahoo_market_players(access_token: str) -> list[dict]:
    game_key = os.getenv("YAHOO_GAME_KEY", "469")
    all_players = []
    seen = set()

    # Yahoo pages are limited. Start conservative for build safety.
    # 0..950 at count 25 gives up to 1,000 players.
    for start in range(0, 1000, 25):
        path = f"game/{game_key}/players;start={start};count=25/percent_owned"

        try:
            payload = yahoo_api_get(path, access_token)
        except Exception as exc:
            print(f"Yahoo page failed at start={start}: {exc}")
            break

        page_players = extract_players_from_yahoo_payload(payload)
        if not page_players:
            break

        for player in page_players:
            key = player.get("player_key") or player.get("player_id") or player.get("player_name")
            if key in seen:
                continue
            seen.add(key)
            all_players.append(player)

    return all_players


def build_empty_payload(reason: str) -> dict:
    return {
        "generated_at": utc_now(),
        "mode": "market_eligibility_v2",
        "source": "platform_api",
        "provider": "yahoo_fantasy_api",
        "feed_state": "not_connected",
        "build_success": True,
        "degraded": True,
        "max_rostered_pct": MAX_ROSTERED_PCT,
        "eligibility_policy": "platform_api_verified_rostered_pct_required",
        "players": [],
        "player_count": 0,
        "errors": [],
        "notes": [
            reason,
            "No manual/exported/pre-seeded Waiver eligibility data is used.",
            "Waiver players render only after verified platform/API rostered percentage data is ingested.",
        ],
        "provider_auth_configured": False,
        "provider_auth_note": "Yahoo Fantasy API authentication was not fully configured for this refresh. Private credential names are intentionally not exposed in deployed output.",
    }


def build_market_eligibility() -> dict:
    load_local_env()

    if not env_ready():
        return build_empty_payload(
            "Yahoo Fantasy API credentials are not connected yet."
        )

    access_token, token_payload = get_yahoo_access_token()
    if not access_token:
        payload = build_empty_payload(
            "Yahoo Fantasy API refresh-token exchange failed."
        )
        payload["feed_state"] = "token_refresh_failed"
        payload["errors"] = [token_payload]
        return payload

    yahoo_players = fetch_yahoo_market_players(access_token)
    eligible_players = [
        player for player in yahoo_players
        if player.get("market_pct_verified") is True
        and player.get("rostered_pct") is not None
        and player.get("rostered_pct") <= MAX_ROSTERED_PCT
    ]

    return {
        "generated_at": utc_now(),
        "mode": "market_eligibility_v2",
        "source": "platform_api",
        "provider": "yahoo_fantasy_api",
        "feed_state": "live",
        "build_success": True,
        "degraded": False,
        "max_rostered_pct": MAX_ROSTERED_PCT,
        "eligibility_policy": "platform_api_verified_rostered_pct_required",
        "players": eligible_players,
        "player_count": len(eligible_players),
        "raw_player_count": len(yahoo_players),
        "errors": [],
        "notes": [
            "Yahoo Fantasy API percent_owned feed ingested successfully.",
            "No manual/exported/pre-seeded Waiver eligibility data is used.",
            "Only players with verified Yahoo rostered percentage at or below the ownership gate are eligible.",
        ],
        "token_state": {
            "access_token": "FOUND",
            "expires_in": token_payload.get("expires_in"),
            "token_type": token_payload.get("token_type"),
        },
        "provider_auth_configured": True,
        "provider_auth_note": "Yahoo Fantasy API authentication was configured for this refresh. Private credential names are intentionally not exposed in deployed output.",
    }


def main() -> None:
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_market_eligibility()
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote market eligibility feed -> {OUT_PATH}")
    print(f"feed_state: {payload.get('feed_state')}")
    print(f"player_count: {payload.get('player_count')}")


if __name__ == "__main__":
    main()
