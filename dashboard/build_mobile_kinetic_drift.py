#!/usr/bin/env python3
"""Build the DiamondSignals Kinetic Drift mobile canary."""
import json
from pathlib import Path
from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
TEMPLATES = ROOT / "dashboard" / "templates"
STATIC = ROOT / "dashboard" / "static" / "mobile"
OUT = DIST / "mobile-kinetic-drift-canary"

def read(path):
    return path.read_text(encoding="utf-8")

def display_name(name):
    value = str(name or "").strip()
    if "," in value:
        last, first = [part.strip() for part in value.split(",", 1)]
        if first and last:
            return f"{first} {last}"
    return value

def main():
    payload_path = DIST / "admin" / "kinetic_drift_signals.json"
    if not payload_path.exists():
        raise SystemExit("Kinetic Drift canonical payload unavailable")

    players = json.loads(read(payload_path)).get("signals", [])
    if not players:
        raise SystemExit("Kinetic Drift canonical payload has no signals")

    clean = []
    for player in players:
        item = dict(player)
        item["player_name"] = display_name(item.get("player_name"))
        clean.append(item)

    body = Template(
        read(TEMPLATES / "mobile" / "surface_reports" / "kinetic_drift_command.html")
    ).render(players=clean)
    css = read(STATIC / "mobile_surface_base.css") + "\n" + read(STATIC / "mobile_signal_wall_command.css")
    js = read(STATIC / "mobile_command_experience.js") + "\n" + read(STATIC / "mobile_signal_wall_command.js")
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<title>DiamondSignals Mobile // Kinetic Drift</title>'
        f"<style>{css}</style></head><body>{body}<script>{js}</script></body></html>"
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote Kinetic Drift mobile canary with {len(clean)} arms")

if __name__ == "__main__":
    main()
