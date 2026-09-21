#!/usr/bin/env python3
import json
from pathlib import Path
from jinja2 import Template
ROOT=Path(__file__).resolve().parents[1];DIST=ROOT/"dist";T=ROOT/"dashboard"/"templates";S=ROOT/"dashboard"/"static"/"mobile";OUT=DIST/"mobile-apex-extraction-canary"
def read(p):return p.read_text(encoding="utf-8")
def display_name(v):
 s=str(v or "").strip()
 if "," in s:
  last,first=[x.strip() for x in s.split(",",1)]
  if first and last:return f"{first} {last}"
 return s
def main():
 p=DIST/"apex-extraction"/"apex_extraction.json"
 if not p.exists():raise SystemExit("Apex Extraction canonical payload unavailable")
 payload=json.loads(read(p));cards=payload.get("top_signals",[])
 if not cards:raise SystemExit("Apex Extraction canonical payload has no top signals")
 players=[]
 for x in cards:
  y=dict(x);y["name"]=display_name(y.get("name"))
  pid=y.get("player_id");y["headshot_url"]=f"https://img.mlbstatic.com/mlb-photos/image/upload/w_640,q_85/v1/people/{pid}/headshot/67/current" if pid else ""
  y["profile_url"]=f"/scout/{pid}/" if pid else "#"
  players.append(y)
 body=Template(read(T/"mobile"/"surface_reports"/"apex_extraction_command.html")).render(players=players)
 css=read(S/"mobile_surface_base.css")+"\n"+read(S/"mobile_signal_wall_command.css");js=read(S/"mobile_command_experience.js")+"\n"+read(S/"mobile_signal_wall_command.js")
 html=f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>DiamondSignals Mobile // Apex Extraction</title><style>{css}</style></head><body>{body}<script>{js}</script></body></html>'
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"index.html").write_text(html,encoding="utf-8");print(f"Wrote Apex Extraction mobile canary with {len(players)} assets")
if __name__=="__main__":main()
