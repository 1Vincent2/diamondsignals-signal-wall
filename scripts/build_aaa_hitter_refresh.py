#!/usr/bin/env python3

import json
from pathlib import Path
from datetime import date
import requests

OUT_JSON = Path("dist/aaa_hitter_refresh.json")
OUT_HTML = Path("dist/aaa_fangraphs_aaa.html")

FG_URL = "https://www.fangraphs.com/leaders/minor-league"

def main() -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    status = "fangraphs_request_failed"
    error = None
    html_size = 0

    try:
        resp = requests.get(
            FG_URL,
            params={"level": "aaa"},
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        OUT_HTML.write_text(resp.text, encoding="utf-8")
        html_size = OUT_HTML.stat().st_size
        status = "fangraphs_html_downloaded"
    except Exception as e:
        error = str(e)

    payload = {
        "generated_at": date.today().isoformat(),
        "status": status,
        "source": "FanGraphs MiLB leaderboards",
        "level": "AAA",
        "html_path": str(OUT_HTML),
        "html_size": html_size,
        "error": error,
        "players": []
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"status={status}")
    print(f"error={error}")

if __name__ == "__main__":
    main()
