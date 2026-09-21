#!/usr/bin/env python3
import json
from pathlib import Path
from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
T = ROOT / "dashboard" / "templates"
S = ROOT / "dashboard" / "static" / "mobile"
OUT = DIST / "mobile-waiver-wire-canary"

def read(path):
    return path.read_text(encoding="utf-8")

def main():
    payload_path = DIST / "waiver_wire.json"
    market_path = DIST / "market" / "waiver_market_eligibility.json"
    if not payload_path.exists():
        raise SystemExit("Waiver Wire canonical payload unavailable")

    payload = json.loads(read(payload_path))
    cards = payload.get("all_assets") or payload.get("assets") or []
    feed_state = "unknown"
    if market_path.exists():
        try:
            market = json.loads(read(market_path))
            feed_state = str(market.get("feed_state") or market.get("state") or "unknown")
        except Exception:
            feed_state = "unknown"

    players = []
    for card in cards:
        player = dict(card)
        pid = str(player.get("player_id") or "").strip()
        player["headshot_url"] = player.get("headshot_url") or (
            f"https://img.mlbstatic.com/mlb-photos/image/upload/w_640,q_85/v1/people/{pid}/headshot/67/current"
            if pid else ""
        )
        player["profile_url"] = player.get("scout_url") or (f"/scout/{pid}/" if pid else player.get("watchlist_url") or "#")
        metrics = {str(m.get("label")): m.get("value") for m in player.get("metrics", []) if isinstance(m, dict)}
        player["ownership_gate"] = metrics.get("Ownership Gate") or "Unverified"
        player["signal_window"] = metrics.get("Signal Window") or "—"
        player["market_defect"] = metrics.get("Market Defect") or "MARKET SIGNAL"
        players.append(player)

    body = Template(read(T / "mobile" / "surface_reports" / "waiver_wire_command.html")).render(
        players=players,
        feed_state=feed_state,
    )
    css = read(S / "mobile_surface_base.css") + "\n" + read(S / "mobile_signal_wall_command.css") + """
@media(max-width:760px){
.ds-mobile-waiver-empty{margin:10px 16px 24px;padding:24px 18px;border:1px solid rgba(182,255,0,.22);border-radius:24px;background:linear-gradient(180deg,#0e151f,#070b11);box-shadow:0 22px 55px rgba(0,0,0,.35)}
.ds-mobile-empty-orbit{width:72px;height:72px;display:grid;place-items:center;margin-bottom:18px;border:1px solid rgba(182,255,0,.35);border-radius:50%;color:var(--lime);font:900 34px ui-monospace,monospace;box-shadow:inset 0 0 30px rgba(182,255,0,.05),0 0 35px rgba(182,255,0,.06)}
.ds-mobile-waiver-empty h2{margin:8px 0 10px;font-size:27px;line-height:1.02;letter-spacing:-.04em}.ds-mobile-waiver-empty>p{margin:0;color:#aeb8c6;font-size:12px;line-height:1.55}
.ds-mobile-empty-grid{display:grid;grid-template-columns:1.3fr .8fr .8fr;gap:7px;margin:18px 0}.ds-mobile-empty-grid div{min-width:0;border:1px solid var(--line);border-radius:12px;background:#0a1018;padding:10px 8px}.ds-mobile-empty-grid span{display:block;color:#7f8b9c;font:800 7px ui-monospace,monospace}.ds-mobile-empty-grid strong{display:block;margin-top:7px;color:#dce3eb;font:900 10px ui-monospace,monospace;overflow-wrap:anywhere}.ds-mobile-empty-grid div:nth-child(2) strong{color:var(--lime);font-size:18px}
}"""
    js = read(S / "mobile_command_experience.js") + "\n" + read(S / "mobile_signal_wall_command.js")
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<title>DiamondSignals Mobile // Waiver Wire</title>'
        f'<style>{css}</style></head><body>{body}<script>{js}</script></body></html>'
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote Waiver Wire mobile canary with {len(players)} verified assets; feed_state={feed_state}")

if __name__ == "__main__":
    main()
