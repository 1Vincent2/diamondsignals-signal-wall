#!/usr/bin/env python3
"""Build the quarantined DiamondSignals Mobile Signal Wall preview from current canonical artifacts."""
import json
from datetime import datetime
from pathlib import Path
from jinja2 import Template

ROOT=Path(__file__).resolve().parents[1]
DIST=ROOT/"dist"
TEMPLATES=ROOT/"dashboard"/"templates"
MOBILE_STATIC=ROOT/"dashboard"/"static"/"mobile"
OUT=DIST/"mobile-live-canary"

def text(path): return path.read_text(encoding="utf-8")

def safe_id(row):
    for key in ("resolved_player_id","player_id","batter","pitcher","mlbam_id","id"):
        value=row.get(key)
        if value not in (None,"","nan"):
            try:return str(int(float(str(value))))
            except Exception:return str(value).strip()
    return ""

def headshot(pid):
    return f"https://img.mlbstatic.com/mlb-photos/image/upload/w_360,q_90/v1/people/{pid}/headshot/67/current" if pid else ""

def canonical_name(value):
    text=str(value or "").strip()
    if "," in text:
        last,first=[part.strip() for part in text.split(",",1)]
        if first and last: text=f"{first} {last}"
    return " ".join(text.lower().split())

def load_player_index():
    path=DIST/"player_index.json"
    if not path.exists(): return {}
    payload=json.loads(path.read_text(encoding="utf-8"))
    index={}
    for p in payload.get("players",[]) or []:
        names={p.get("full_name"),f'{p.get("first_name","")} {p.get("last_name","")}'}
        for name in names:
            key=canonical_name(name)
            if key: index[key]=p
    return index

def normalize(row,kind,rank,player_index):
    name=row.get("player_name") or row.get("name") or "Unknown Player"
    match=player_index.get(canonical_name(name),{})
    pid=safe_id(row) or safe_id(match)
    badges=row.get("badges") or []
    return {
      **row,"rank":rank,"player_id":pid,"player_name":name,"player_type":kind,
      "board_label":"PITCHER" if kind=="pitcher" else "HITTER",
      "team":row.get("team") or row.get("player_team") or match.get("team") or match.get("team_name") or "",
      "profile_url":f"/scout/{pid}/" if pid else "#",
      "headshot_url":match.get("headshot_url") or headshot(pid),
      "avatar":"".join([p[0] for p in str(name).replace(","," ").split()[:2]]).upper() or "DS",
      "edge_score":row.get("edge_score","—"),"badges":badges,
      "metric_1_label":row.get("metric_1_label","SIGNAL 1"),"metric_1":row.get("metric_1","—"),
      "metric_2_label":row.get("metric_2_label","SIGNAL 2"),"metric_2":row.get("metric_2","—"),
      "metric_3_label":row.get("metric_3_label","SIGNAL 3"),"metric_3":row.get("metric_3","—"),
      "why":row.get("why") or "DiamondSignals movement threshold crossed.",
      "sample_note":row.get("sample_note") or "LIVE WINDOW",
    }

def main():
    payload=json.loads((DIST/"signals.json").read_text(encoding="utf-8"))
    raw=[]
    for kind,key in (("pitcher","top_pitchers"),("hitter","top_hitters")):
        for row in payload.get(key,[]) or []: raw.append((kind,row))
    raw.sort(key=lambda item: float(item[1].get("edge_score") or 0),reverse=True)
    player_index=load_player_index()\n    players=[normalize(row,kind,i+1,player_index) for i,(kind,row) in enumerate(raw)]
    tpl=Template(text(TEMPLATES/"mobile"/"surface_reports"/"signal_wall_command.html"))
    body=tpl.render(players=players,updated_label=datetime.now().strftime("%-I:%M %p"))
    css=text(MOBILE_STATIC/"mobile_surface_base.css")+"\n"+text(MOBILE_STATIC/"mobile_signal_wall_command.css")
    js=text(MOBILE_STATIC/"mobile_command_experience.js")+"\n"+text(MOBILE_STATIC/"mobile_signal_wall_command.js")
    html=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>DiamondSignals Mobile // Today's Edge</title><style>{css}</style></head><body>{body}<script src="/player-card-actions.js"></script><script>{js}</script></body></html>"""
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"index.html").write_text(html,encoding="utf-8")
    print(f"Wrote {OUT/'index.html'} with {len(players)} signals")
if __name__=="__main__":main()
