#!/usr/bin/env python3
import json
import requests
import re
import unicodedata
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "admin" / "player_signal_index.json"
AUDIT_OUT = DIST / "admin" / "signal_identity_audit.json"
CATALOG_OVERLAY_OUT = DIST / "admin" / "player_catalog_overlay.json"
PLAYER_INDEX = DIST / "player_index.json"

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

# Manual identity resolver for Promotion Watch / minor-league residue.
# These are not replacing the canonical universe; they only fill gaps when player_index.json
# cannot resolve a signal row by MLBAM id or normalized name.
MANUAL_PLAYER_RESOLVER = {
    "allan winans": {"team": "NYY", "position": "P"},
    "blaine crim": {"team": "TEX", "position": "1B"},
    "blas castano": {"team": "UNK", "position": "P"},
    "brendan beck": {"team": "NYY", "position": "P"},
    "brennen davis": {"team": "SEA", "position": "OF"},
    "brett harris": {"team": "UNK", "position": "INF"},
    "cameron cauley": {"team": "TEX", "position": "INF"},
    "carlos lagrange": {"team": "NYY", "position": "P"},
    "carlos perez": {"team": "CHC", "position": "C"},
    "cj alexander": {"team": "HOU", "position": "INF"},
    "elieser hernandez": {"team": "ATL", "position": "P"},
    "isaac coffey": {"team": "BOS", "position": "P"},
    "jhostynxon garcia": {"team": "BOS", "position": "OF"},
    "jonah tong": {"team": "NYM", "position": "P"},
    "lucas braun": {"team": "ATL", "position": "P"},
    "miguel andujar": {"team": "UNK", "position": "OF"},
    "nick morabito": {"team": "NYM", "position": "OF"},
    "nick sogard": {"team": "BOS", "position": "INF"},
    "pedro ramirez": {"team": "CHC", "position": "INF"},
    "sawyer gipson long": {"team": "DET", "position": "P"},
    "spencer packard": {"team": "SEA", "position": "OF"},
    "sung mun song": {"team": "UNK", "position": "INF"},
    "william fleming": {"team": "SEA", "position": "P"},
}

def norm_name(value):
    value = str(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    if "," in value:
        parts = [p.strip() for p in value.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            value = f"{parts[1]} {parts[0]}"
    return value

def search_key(name):
    value = unicodedata.normalize("NFKD", str(name or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

def first(item, fields):
    for f in fields:
        v = item.get(f)
        if v not in (None, ""):
            return v
    return None

CURRENT_SEASON = datetime.now(timezone.utc).year
MLB_STATS_CACHE = {}

def safe_div(numerator, denominator):
    try:
        numerator = float(numerator or 0)
        denominator = float(denominator or 0)
        if denominator == 0:
            return None
        return numerator / denominator
    except Exception:
        return None

def fmt_pct(value):
    if value is None:
        return None
    return f"{value * 100:.1f}%"

def fmt_ratio(value):
    if value is None:
        return None
    return f"{value:.2f}"

def fetch_mlb_season_stat(player_id, group):
    pid = as_int_or_none(player_id)
    if pid is None:
        return None

    cache_key = (pid, group, CURRENT_SEASON)
    if cache_key in MLB_STATS_CACHE:
        return MLB_STATS_CACHE[cache_key]

    url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
    params = {
        "stats": "season",
        "group": group,
        "season": CURRENT_SEASON,
    }

    try:
        response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
        splits = (((data.get("stats") or [{}])[0]).get("splits") or [])
        stat = (splits[0].get("stat") if splits else None) or {}
    except Exception as exc:
        print(f"DISCIPLINE_ENRICH_MISS player_id={pid} group={group}: {exc}")
        stat = None

    MLB_STATS_CACHE[cache_key] = stat
    return stat

def infer_stat_group(player):
    position = str(player.get("position") or "").upper()
    reports = set(player.get("reports_triggered") or [])
    signal_text = json.dumps(player.get("signals") or [], default=str).lower()

    pitcher_positions = {"P", "SP", "RP", "LHP", "RHP", "PITCHER"}
    hitter_positions = {"BAT", "B", "H", "C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH", "INF", "UTIL"}

    if position in pitcher_positions:
        return "pitching"

    if position in hitter_positions:
        return "hitting"

    if any(report in reports for report in ["Velocity Decay", "Stuff+ Disruption", "IVB Heat Map"]):
        return "pitching"

    if any(term in signal_text for term in ["pitch", "velo", "ivb", "vaa", "whiff", "arsenal", "stuff"]):
        return "pitching"

    return "hitting"



def safe_num(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def pitcher_babip_fallback(stat, strikeouts):
    """
    Pitcher BABIP allowed fallback:
    (H - HR) / (AB - K - HR + SF)

    MLB Stats API does not always expose pitcher babip directly in season stat payloads,
    so calculate it when the component fields are available.
    """
    direct = stat.get("babip")
    if direct not in (None, ""):
        return direct

    hits = safe_num(stat.get("hits"))
    home_runs = safe_num(stat.get("homeRuns"))
    at_bats = safe_num(stat.get("atBats"))
    strikeouts_num = safe_num(strikeouts)
    sac_flies = safe_num(stat.get("sacFlies") or stat.get("sacrificeFlies") or 0)

    if None in (hits, home_runs, at_bats, strikeouts_num):
        return None

    denominator = at_bats - strikeouts_num - home_runs + sac_flies
    numerator = hits - home_runs

    if denominator <= 0:
        return None

    return f"{numerator / denominator:.3f}"

def discipline_context_for_player(player):
    pid = player.get("player_id")
    group = infer_stat_group(player)

    stat = fetch_mlb_season_stat(pid, group)

    # Fallback: if inferred side has no season stats, try the other side.
    if not stat:
        fallback_group = "hitting" if group == "pitching" else "pitching"
        stat = fetch_mlb_season_stat(pid, fallback_group)
        if stat:
            group = fallback_group

    if not stat:
        return {}

    is_pitcher = group == "pitching"

    strikeouts = stat.get("strikeOuts")
    walks = stat.get("baseOnBalls")
    babip = pitcher_babip_fallback(stat, strikeouts) if group == "pitching" else stat.get("babip")

    if is_pitcher:
        batters_faced = stat.get("battersFaced")
        k_pct = safe_div(strikeouts, batters_faced)
        bb_pct = safe_div(walks, batters_faced)
        k_bb = safe_div(strikeouts, walks)

        return {
            "season_context": "COMMAND_PROFILE",
            "season": CURRENT_SEASON,
            "k_pct": fmt_pct(k_pct),
            "bb_pct": fmt_pct(bb_pct),
            "k_bb_ratio": fmt_ratio(k_bb),
            "babip": babip,
            "batters_faced": batters_faced,
            "strikeouts": strikeouts,
            "walks": walks,
        }

    plate_appearances = stat.get("plateAppearances")
    k_pct = safe_div(strikeouts, plate_appearances)
    bb_pct = safe_div(walks, plate_appearances)
    bb_k = safe_div(walks, strikeouts)

    return {
        "season_context": "PLATE_DISCIPLINE",
        "season": CURRENT_SEASON,
        "bb_pct": fmt_pct(bb_pct),
        "k_pct": fmt_pct(k_pct),
        "bb_k_ratio": fmt_ratio(bb_k),
        "babip": babip,
        "plate_appearances": plate_appearances,
        "strikeouts": strikeouts,
        "walks": walks,
    }

def apply_discipline_context(players):
    for player in players:
        context = discipline_context_for_player(player)
        if not context:
            continue

        player["season_context"] = context

        for signal in player.get("signals") or []:
            metrics = signal.setdefault("metrics", {})
            for key, value in context.items():
                if value not in (None, "") and key not in metrics:
                    metrics[key] = value

    return players




# Final resolver normalization overrides.
MANUAL_PLAYER_RESOLVER.update({
    "blas castano": {"team": "UNK", "position": "P"},
    "carlos perez": {"team": "CHC", "position": "C"},
    "elieser hernandez": {"team": "ATL", "position": "P"},
    "miguel andujar": {"team": "UNK", "position": "OF"},
    "pedro ramirez": {"team": "CHC", "position": "INF"},
})

def as_int_or_none(value):
    try:
        if value in (None, ""):
            return None
        return int(str(value).strip())
    except Exception:
        return None


def load_player_universe():
    data = load_json(PLAYER_INDEX)
    players = data.get("players", []) if isinstance(data, dict) else []
    by_id = {}
    by_name = {}

    for row in players:
        if not isinstance(row, dict):
            continue

        pid = as_int_or_none(row.get("player_id"))
        name = norm_name(row.get("full_name") or row.get("player_name") or row.get("name"))

        normalized = {
            "player_id": pid,
            "player_name": name,
            "team": row.get("team") or row.get("team_name"),
            "position": row.get("position"),
            "bats": row.get("bats"),
            "throws": row.get("throws"),
            "headshot_url": row.get("headshot_url"),
            "status": row.get("status"),
        }

        if pid is not None:
            by_id[pid] = normalized

        if name:
            by_name[search_key(name)] = normalized

    return by_id, by_name


def enrich_from_universe(rec, universe_by_id, universe_by_name):
    pid = as_int_or_none(rec.get("player_id"))
    source = None

    if pid is not None and pid in universe_by_id:
        source = universe_by_id[pid]
    else:
        source = universe_by_name.get(search_key(rec.get("player_name")))

    if source:
        rec["player_id"] = rec.get("player_id") or source.get("player_id")
        rec["team"] = rec.get("team") or source.get("team")
        rec["position"] = rec.get("position") or source.get("position")
        rec["headshot_url"] = rec.get("headshot_url") or source.get("headshot_url")
        rec["bats"] = rec.get("bats") or source.get("bats")
        rec["throws"] = rec.get("throws") or source.get("throws")
        rec["universe_match"] = True

    manual = MANUAL_PLAYER_RESOLVER.get(search_key(rec.get("player_name")))
    if manual:
        rec["team"] = rec.get("team") or manual.get("team")
        rec["position"] = rec.get("position") or manual.get("position")
        rec["manual_identity_resolver"] = True

    return rec


def build_catalog_overlay(players):
    overlay = []

    for p in players:
        pid = as_int_or_none(p.get("player_id"))
        if pid is None:
            continue

        reports = p.get("reports_triggered") or []
        primary_report = reports[0] if reports else "SIGNAL OVERLAY"

        overlay.append({
            "id": pid,
            "name": p.get("player_name"),
            "org": p.get("team") or "UNK",
            "role": p.get("position") or "UNK",
            "signal": primary_report,
            "note": "Backfilled from DiamondSignals active signal overlay.",
            "signal_tier": "ACTIVE_SIGNAL",
            "decision_bias": "MONITOR",
            "risk": "Unknown",
            "trend": "Active",
        })

    return sorted(overlay, key=lambda r: str(r.get("name") or "").lower())


def build_identity_audit(players, source_count):
    missing_id = []
    missing_team = []
    missing_position = []
    duplicate_ids = {}
    seen_ids = {}

    for p in players:
        pid = as_int_or_none(p.get("player_id"))
        slim = {
            "player_id": p.get("player_id"),
            "player_name": p.get("player_name"),
            "team": p.get("team"),
            "position": p.get("position"),
            "reports_triggered": p.get("reports_triggered", []),
        }

        if pid is None:
            missing_id.append(slim)
        else:
            seen_ids.setdefault(str(pid), []).append(slim)

        if not p.get("team"):
            missing_team.append(slim)

        if not p.get("position"):
            missing_position.append(slim)

    for pid, rows in seen_ids.items():
        if len(rows) > 1:
            duplicate_ids[pid] = rows

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": source_count,
        "players_indexed": len(players),
        "missing_player_id_count": len(missing_id),
        "missing_team_count": len(missing_team),
        "missing_position_count": len(missing_position),
        "duplicate_id_count": len(duplicate_ids),
        "missing_player_id": missing_id,
        "missing_team": missing_team,
        "missing_position": missing_position,
        "duplicate_ids": duplicate_ids,
    }

def pick_metrics(item):
    keep = {}
    # PLAYER_SIGNAL_INDEX_KEEP_SIGNAL_WALL_CONTEXT_METRICS_V1
    # Preserve cross-surface context metrics from Signal Wall so downstream
    # Performance Audit cards can inherit BABIP / SEAGER when the UI needs them.
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

        # Season / plate-discipline context carried by Signal Wall + enrichment.
        "season_context", "season",
        "seager_score", "babip",
        "bb_k_ratio", "k_bb_ratio",
        "bb_pct", "k_pct",
        "plate_appearances", "batters_faced",
        "strikeouts", "walks",
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

    universe_by_id, universe_by_name = load_player_universe()

    players = [
        enrich_from_universe(rec, universe_by_id, universe_by_name)
        for rec in index.values()
    ]
    players = apply_discipline_context(players)
    players = sorted(players, key=lambda r: r["player_name"].lower())

    source_count = len(SOURCES) + 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "player_count": len(players),
        "source_count": source_count,
        "players": players,
    }

    catalog_overlay = {
        "generated_at": payload["generated_at"],
        "player_count": len(build_catalog_overlay(players)),
        "players": build_catalog_overlay(players),
    }

    audit = build_identity_audit(players, source_count)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    CATALOG_OVERLAY_OUT.write_text(json.dumps(catalog_overlay, indent=2, ensure_ascii=False), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {CATALOG_OVERLAY_OUT.relative_to(ROOT)}")
    print(f"Wrote {AUDIT_OUT.relative_to(ROOT)}")
    print(f"Players indexed: {len(players)}")
    print(f"Missing IDs after universe enrichment: {audit['missing_player_id_count']}")
    print(f"Missing teams after universe enrichment: {audit['missing_team_count']}")
    print(f"Missing positions after universe enrichment: {audit['missing_position_count']}")

if __name__ == "__main__":
    main()
