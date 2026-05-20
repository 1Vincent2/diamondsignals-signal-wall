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
        "required_env": REQUIRED_ENV,
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

    # Placeholder for the next activation patch:
    # 1. Request MLB fantasy player ownership / percent_owned.
    # 2. Normalize to DiamondSignals market eligibility rows:
    #
    # {
    #   "player_id": "mlbam_or_canonical_id",
    #   "player_name": "Name",
    #   "team": "NYY",
    #   "position": "OF",
    #   "rostered_pct": 17,
    #   "market_pct_verified": true,
    #   "market_provider": "yahoo_fantasy_api",
    #   "market_source": "platform_api",
    #   "eligible": true
    # }
    #
    # Until endpoint/auth is wired, fail closed.
    payload = build_empty_payload(
        "Yahoo Fantasy API access token refreshed successfully, but live endpoint activation is not wired yet."
    )
    payload["feed_state"] = "token_ready_endpoint_pending"
    payload["token_state"] = {
        "access_token": "FOUND",
        "expires_in": token_payload.get("expires_in"),
        "token_type": token_payload.get("token_type"),
    }
    return payload


def main() -> None:
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_market_eligibility()
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote market eligibility feed -> {OUT_PATH}")
    print(f"feed_state: {payload.get('feed_state')}")
    print(f"player_count: {payload.get('player_count')}")


if __name__ == "__main__":
    main()
