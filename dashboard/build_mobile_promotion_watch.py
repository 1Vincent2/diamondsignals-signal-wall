#!/usr/bin/env python3
"""Build isolated Mobile Promotion Watch from the canonical live AAA pipeline."""
import json,sys
from pathlib import Path
from jinja2 import Template
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"dashboard"))
from build_call_up_live import fetch_recent_aaa_weekly_signal_base,build_aaa_hitter_promotion_watch,build_aaa_pitcher_promotion_watch,build_trend_lookup
DIST=ROOT/"dist";T=ROOT/"dashboard"/"templates";S=ROOT/"dashboard"/"static"/"mobile";OUT=DIST/"mobile-promotion-watch-canary"
def txt(p):return p.read_text(encoding="utf-8")
def cname(v):
 s=str(v or "").strip()
 if "," in s:
  a,b=[x.strip() for x in s.split(",",1)]
  if a and b:s=f"{b} {a}"
 return " ".join(s.lower().split())
def index():
 p=DIST/"player_index.json"
 if not p.exists():return {}
 data=json.loads(txt(p));out={}
 for x in data.get("players",[]):
  for n in (x.get("full_name"),f'{x.get("first_name","")} {x.get("last_name","")}'):
   if cname(n):out[cname(n)]=x
 return out
def norm(row,kind,idx):
 r=row.to_dict() if hasattr(row,"to_dict") else dict(row);pi=PLAYERS.get(cname(r.get("player_name")),{})
 pid=str(r.get("resolved_player_id") or r.get("player_id") or pi.get("player_id") or "").replace(".0","")
 badges=r.get("badges") or []
 badges=[b[0] if isinstance(b,(list,tuple)) else str(b) for b in badges]
 return {"player_name":r.get("player_name","Unknown"),"player_type":kind,"edge_score":r.get("edge_score","—"),"metric_1_label":r.get("metric_1_label","SIGNAL"),"metric_1":r.get("metric_1","—"),"metric_2_label":r.get("metric_2_label","SIGNAL"),"metric_2":r.get("metric_2","—"),"metric_3_label":r.get("metric_3_label","SIGNAL"),"metric_3":r.get("metric_3","—"),"why":r.get("why","Promotion pressure under evaluation."),"sample_note":r.get("sample_note","AAA WINDOW"),"badges":badges,"org":r.get("display_org") or r.get("display_team") or pi.get("team") or "AAA","avatar":r.get("avatar","DS"),"headshot_url":pi.get("headshot_url") or (f"https://img.mlbstatic.com/mlb-photos/image/upload/w_360,q_90/v1/people/{pid}/headshot/67/current" if pid else ""),"profile_url":f"/scout/{pid}/" if pid else "#"}
def main():
 global PLAYERS
 payload_path=DIST/"typical-call-up"/"promotion_watch.json"
 if not payload_path.exists(): raise SystemExit("Promotion Watch canonical payload unavailable")
 payload=json.loads(txt(payload_path)); PLAYERS=index(); items=[]
 sections=payload.get("top_signals",{})
 for key,kind in (("hitters_14day","hitter"),("pitchers_14day","pitcher")):
  rows=sections.get(key) or []
  if isinstance(rows,list) and rows and isinstance(rows[0],dict):
   for row in rows: items.append(norm(row,kind,len(items)+1))
 if not items:
  for row in sections.get("depth_radar",[]) or []:
   if isinstance(row,dict):
    kind="pitcher" if str(row.get("signal_type","")).lower()=="pitcher" else "hitter"
    items.append(norm(row,kind,len(items)+1))
 if not items: raise SystemExit("Promotion Watch canonical payload contains no renderable player rows")
 items.sort(key=lambda x:float(x["edge_score"] or 0),reverse=True)
 body=Template(txt(T/"mobile"/"surface_reports"/"promotion_watch_command.html")).render(players=items)
 css=txt(S/"mobile_surface_base.css")+"\n"+txt(S/"mobile_signal_wall_command.css")
 js=txt(S/"mobile_command_experience.js")+"\n"+txt(S/"mobile_signal_wall_command.js")
 html=f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>DiamondSignals Mobile // Promotion Watch</title><style>{css}</style></head><body>{body}<script>{js}</script></body></html>'
 OUT.mkdir(parents=True,exist_ok=True);(OUT/"index.html").write_text(html,encoding="utf-8");print(f"Wrote Promotion Watch mobile canary with {len(items)} players from canonical payload")
if __name__=="__main__":main()
