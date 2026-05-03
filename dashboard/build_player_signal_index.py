#!/usr/bin/env python3
import json
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "admin" / "player_signal_index.json"

SOURCES = [
    {
        "report_key": "apex_extraction",
        "label": "Apex Extraction",
        "path": DIST / "apex-extraction" / "apex_extraction.json",
        "lists": ["top_signals", "apex_bats", "apex_arms"],
    },
    {
        "report_key": "mlb_extraction_ledger",
        "label": "MLB Extraction Ledger",
        "path": DIST / "hidden-gems" / "mlb_extraction_ledger.json",
        "lists": ["top_signals", "top_pitchers", "top_hitters"],
    },
    {
        "report_key": "ivb_heat_map",
        "label": "IVB Heat Map",
        "path": DIST / "ivb_heat_map.json",
        "lists": ["heat_cards", "climbers", "fallers", "entered_apex"],
    },
    {
        "report_key": "signal_wall",
        "label": "Signal Wall",
        "path": DIST / "signals.json",
        "lists": ["top_pitchers", "top_hitters"],
    },
    {
        "report_key": "stuff_disruption",
        "label": "Stuff+ Disruption",
        "path": DIST / "stuff_disruption_feed.json",
        "lists": ["cards"],
    },
    {
        "report_key": "velocity_decay",
        "label": "Velocity Decay",
        "path": DIST / "velocity_decay_monitor.json",
        "lists": ["cards"],
    },
]

PROMOTION_WATCH = {
    "report_key": "promotion_watch",
    "label": "Promotion Watch",
    "path": DIST / "typical-call-up" / "promotion_watch.json",
    "nested": ("top_signals", ["pitchers_72hr", "hitters_72hr", "pitchers_14day", "hitters_14day", "recent_arrivals"]),
}

NAME_FIELDS = ["player_name", "name", "displayName", "player", "full_name"]
ID_FIELDS = ["player_id", "mlbam_id", "synthetic_player_id", "id"]
TEAM_FIELDS = ["team", "org", "organization", "mlb_team"]
POSITION_FIELDS = ["position", "role", "pos"]

def norm_name(value):
    value = str(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    if "," in value:
        parts = [p.strip() for p in value.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            value = f"{parts[1]} {parts[0]}"
    return value

def search_key(name):
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()

def first(item, fields):
    for f in fields:
        v = item.get(f)
        if v not in (None, ""):
            return v
    return None

def pick_metrics(item):
    keep = {}
    keys = [
        "edge_score", "score", "risk_score", "disruption_score",
        "diagnosis", "primary_alert", "apex_tier", "risk_tier",
        "ivb_delta", "ivb_vs_avg", "ivb_raw", "vaa", "vaa_delta",
        "velo_delta", "extension_delta", "perceived_velo_delta", "decay_slope",
        "movement_delta", "active_spin_delta",
        "whiff_probability", "metric_label", "metric_value",
        "metric_1_label", "metric_1", "metric_2_label", "metric_2", "metric_3_label", "metric_3",
        "contact_risk", "dead_zone_label", "heat_class", "band_label", "heat_tag",
        "analysis", "brief", "why", "body_copy",
    ]
    for k in keys:
        if k in item and item.get(k) not in (None, ""):
            keep[k] = item.get(k)
    return keep

def add_record(index, item, source, section=None, generated_at=None):
    if not isinstance(item, dict):
        return

    name = norm_name(first(item, NAME_FIELDS))
    if not name:
        return

    pid = first(item, ID_FIELDS)
    key = f"name:{search_key(name)}"

    rec = index.setdefault(key, {
        "player_id": pid,
        "player_name": name,
        "search_name": search_key(name),
        "team": first(item, TEAM_FIELDS),
        "position": first(item, POSITION_FIELDS),
        "reports_triggered": [],
        "signals": [],
        "latest_generated_at": None,
    })

    rec["player_id"] = rec.get("player_id") or pid
    rec["team"] = rec.get("team") or first(item, TEAM_FIELDS)
    rec["position"] = rec.get("position") or first(item, POSITION_FIELDS)

    if source["label"] not in rec["reports_triggered"]:
        rec["reports_triggered"].append(source["label"])

    sig = {
        "report_key": source["report_key"],
        "report_label": source["label"],
        "section": section,
        "generated_at": generated_at,
        "metrics": pick_metrics(item),
        "source_path": str(source["path"].relative_to(ROOT)),
    }
    rec["signals"].append(sig)

    if generated_at:
        rec["latest_generated_at"] = max(filter(None, [rec.get("latest_generated_at"), generated_at]))

def load_json(path):
    if not path.exists():
        print(f"MISS {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR {path}: {e}")
        return None

def main():
    index = {}

    for source in SOURCES:
        data = load_json(source["path"])
        if not isinstance(data, dict):
            continue
        generated_at = data.get("generated_at")
        for list_key in source["lists"]:
            rows = data.get(list_key)
            if isinstance(rows, list):
                for item in rows:
                    add_record(index, item, source, section=list_key, generated_at=generated_at)

    source = PROMOTION_WATCH
    data = load_json(source["path"])
    if isinstance(data, dict):
        generated_at = data.get("generated_at")
        top_key, sections = source["nested"]
        top = data.get(top_key, {})
        if isinstance(top, dict):
            for section in sections:
                rows = top.get(section)
                if isinstance(rows, list):
                    for item in rows:
                        add_record(index, item, source, section=section, generated_at=generated_at)

    players = sorted(index.values(), key=lambda r: r["player_name"].lower())

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "player_count": len(players),
        "source_count": len(SOURCES) + 1,
        "players": players,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Players indexed: {len(players)}")

if __name__ == "__main__":
    main()
