#!/usr/bin/env python3
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

INDEX_PATH = DIST / "admin" / "player_signal_index.json"

SURFACE_JSONS = [
    DIST / "signals.json",
    DIST / "hidden-gems" / "mlb_extraction_ledger.json",
    DIST / "apex-extraction" / "apex_extraction.json",
    DIST / "ivb_heat_map.json",
    DIST / "stuff_disruption_feed.json",
    DIST / "velocity_decay_monitor.json",
    DIST / "typical-call-up" / "promotion_watch.json",
]

NAME_KEYS = [
    "player_name",
    "name",
    "displayName",
    "display_name",
    "player",
    "full_name",
]

ID_KEYS = [
    "player_id",
    "resolved_player_id",
    "mlbam_id",
    "batter",
    "pitcher",
    "id",
]


def search_key(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def id_key(value):
    if value in (None, ""):
        return None
    try:
        return str(int(str(value).strip()))
    except Exception:
        value = str(value).strip()
        return value if value else None


def first_present(row, keys):
    if not isinstance(row, dict):
        return None

    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value

    raw = row.get("raw")
    if isinstance(raw, dict):
        for key in keys:
            value = raw.get(key)
            if value not in (None, ""):
                return value

    return None


def load_json(path):
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR reading {path.relative_to(ROOT)}: {exc}")
        return None


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_context_maps():
    index = load_json(INDEX_PATH)
    if not isinstance(index, dict):
        raise SystemExit(f"Missing or invalid {INDEX_PATH.relative_to(ROOT)}")

    by_id = {}
    by_name = {}

    for player in index.get("players", []):
        if not isinstance(player, dict):
            continue

        ctx = player.get("season_context")
        if not isinstance(ctx, dict) or not ctx:
            continue

        pid = id_key(player.get("player_id"))
        name = search_key(player.get("player_name"))

        if pid:
            by_id[pid] = ctx
        if name:
            by_name[name] = ctx

    return by_id, by_name


def enrich_node(node, by_id, by_name):
    touched = 0

    if isinstance(node, dict):
        pid = id_key(first_present(node, ID_KEYS))
        name = search_key(first_present(node, NAME_KEYS))

        ctx = None
        if pid and pid in by_id:
            ctx = by_id[pid]
        elif name and name in by_name:
            ctx = by_name[name]

        if ctx and not node.get("season_context"):
            node["season_context"] = ctx
            touched += 1

        raw = node.get("raw")
        if isinstance(raw, dict) and ctx and not raw.get("season_context"):
            raw["season_context"] = ctx
            touched += 1

        for value in node.values():
            touched += enrich_node(value, by_id, by_name)

    elif isinstance(node, list):
        for item in node:
            touched += enrich_node(item, by_id, by_name)

    return touched


def main():
    by_id, by_name = build_context_maps()

    print(f"Season context lookup: {len(by_id)} ids // {len(by_name)} names")

    total = 0

    for path in SURFACE_JSONS:
        data = load_json(path)
        if data is None:
            print(f"MISS {path.relative_to(ROOT)}")
            continue

        touched = enrich_node(data, by_id, by_name)
        total += touched

        if touched:
            write_json(path, data)

        print(f"{path.relative_to(ROOT)} -> {touched} season_context insertions")

    print(f"TOTAL_SURFACE_CONTEXT_INSERTIONS={total}")


if __name__ == "__main__":
    main()
