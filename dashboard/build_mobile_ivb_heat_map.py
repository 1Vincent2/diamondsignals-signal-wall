#!/usr/bin/env python3
import json
from pathlib import Path
from jinja2 import Template
ROOT=Path(__file__).resolve().parents[1];DIST=ROOT/"dist";T=ROOT/"dashboard"/"templates";S=ROOT/"dashboard"/"static"/"mobile";OUT=DIST/"mobile-ivb-heat-map-canary"
def read(p):return p.read_text(encoding="utf-8")
def display_name(n):
 s=str(n or "").strip()
 if "," in s:
  last,first=[x.strip() for x in s.split(",",1)]
  if first and last:return f"{first} {last}"
 return s
def main():
 p=DIST/"ivb_heat_map.json"
 if not p.exists():raise SystemExit("IVB Heat Map canonical payload unavailable")
 cards=json.loads(read(p)).get("heat_cards",[])
 if not cards:raise SystemExit("IVB Heat Map canonical payload has no heat cards")
 players=[]
 for x in cards:
  y=dict(x);y["player_name"]=display_name(y.get("player_name"))
  pid=y.get("player_id");y["headshot_url"]=f"https://img.mlbstatic.com/mlb-photos/image/upload/w_640,q_85/v1/people/{pid}/headshot/67/current" if pid else ""
  y["profile_url"]=f"/scout/{pid}/" if pid else "#"
  players.append(y)
 body=Template(read(T/"mobile"/"surface_reports"/"ivb_heat_map_command.html")).render(players=players)
 css=read(S/"mobile_surface_base.css")+"\n"+read(S/"mobile_signal_wall_command.css");js=read(S/"mobile_command_experience.js")+"\n"+read(S/"mobile_signal_wall_command.js")
 html=f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>DiamondSignals Mobile // IVB Heat Map</title><style>{css}</style></head><body>{body}<script>{js}</script></body></html>'
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"index.html").write_text(html,encoding="utf-8");print(f"Wrote IVB Heat Map mobile canary with {len(players)} arms")
if __name__=="__main__":main()
