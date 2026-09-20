#!/usr/bin/env python3
import json
from pathlib import Path
from jinja2 import Template
ROOT=Path(__file__).resolve().parents[1];DIST=ROOT/"dist";T=ROOT/"dashboard"/"templates";S=ROOT/"dashboard"/"static"/"mobile";OUT=DIST/"mobile-stuff-disruption-canary"
def read(p):return p.read_text(encoding="utf-8")
def name(v):
 s=str(v or "").strip()
 if "," in s:
  last,first=[x.strip() for x in s.split(",",1)]
  if first and last:return f"{first} {last}"
 return s
def main():
 p=DIST/"stuff_disruption_feed.json"
 if not p.exists():raise SystemExit("Stuff+ canonical payload unavailable")
 cards=json.loads(read(p)).get("cards",[])
 if not cards:raise SystemExit("Stuff+ canonical payload has no cards")
 players=[]
 for x in cards:
  y=dict(x);y["player_name"]=name(y.get("player_name"));players.append(y)
 body=Template(read(T/"mobile"/"surface_reports"/"stuff_disruption_command.html")).render(players=players)
 css=read(S/"mobile_surface_base.css")+"\n"+read(S/"mobile_signal_wall_command.css");js=read(S/"mobile_command_experience.js")+"\n"+read(S/"mobile_signal_wall_command.js")
 html=f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>DiamondSignals Mobile // Stuff+ Disruption</title><style>{css}</style></head><body>{body}<script>{js}</script></body></html>'
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"index.html").write_text(html,encoding="utf-8");print(f"Wrote Stuff+ Disruption mobile canary with {len(players)} arms")
if __name__=="__main__":main()
