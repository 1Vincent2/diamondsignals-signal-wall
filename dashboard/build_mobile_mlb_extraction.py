#!/usr/bin/env python3
import json
from pathlib import Path
from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
T = ROOT / "dashboard" / "templates"
S = ROOT / "dashboard" / "static" / "mobile"
OUT = DIST / "mobile-mlb-extraction-canary"

def read(path):
    return path.read_text(encoding="utf-8")

def display_name(value):
    text = str(value or "").strip()
    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        if first and last:
            return f"{first} {last}"
    return text

def main():
    payload_path = DIST / "hidden-gems" / "mlb_extraction_ledger.json"
    if not payload_path.exists():
        raise SystemExit("MLB Extraction canonical payload unavailable")

    payload = json.loads(read(payload_path))
    cards = payload.get("top_signals", [])
    if not cards:
        raise SystemExit("MLB Extraction canonical payload has no top signals")

    players = []
    for card in cards:
        player = dict(card)
        player["name"] = display_name(player.get("name"))
        pid = player.get("player_id")
        player["headshot_url"] = (
            f"https://img.mlbstatic.com/mlb-photos/image/upload/w_640,q_85/v1/people/{pid}/headshot/67/current"
            if pid else ""
        )
        player["profile_url"] = f"/scout/{pid}/" if pid else "#"
        raw = player.get("raw") or {}
        player["metric_1_label"] = raw.get("metric_1_label") or "PHYSICS CORE"
        player["metric_1"] = raw.get("metric_1") or "—"
        player["metric_2_label"] = raw.get("metric_2_label") or "MARKET GAP"
        player["metric_2"] = raw.get("metric_2") or "—"
        player["metric_3_label"] = raw.get("metric_3_label") or "MARKET ATTENTION"
        player["metric_3"] = raw.get("metric_3") or "—"
        players.append(player)

    body = Template(read(T / "mobile" / "surface_reports" / "mlb_extraction_command.html")).render(players=players)
    css = read(S / "mobile_surface_base.css") + "\n" + read(S / "mobile_signal_wall_command.css")
    js = read(S / "mobile_command_experience.js") + "\n" + read(S / "mobile_signal_wall_command.js")
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<title>DiamondSignals Mobile // MLB Extraction</title>'
        f'<style>{css}</style></head><body>{body}<script>{js}</script></body></html>'
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote MLB Extraction mobile canary with {len(players)} assets")

if __name__ == "__main__":
    main()
