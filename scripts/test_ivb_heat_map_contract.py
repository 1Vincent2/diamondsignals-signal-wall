#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STATUS_PATH = ROOT / "dist/status/ivb-heat-map.json"
PAYLOAD_PATH = ROOT / "dist/ivb_heat_map.json"
HTML_PATH = ROOT / "dist/ivb-heat-map/index.html"

EXPECTED_MODE = "statcast_supabase_ivb_heat_map_dynamic_v1"
REQUIRED_LAYERS = {
    "pybaseball_statcast_pitch_level_feed",
    "fastball_ivb_window",
    "velocity_bucket_ivb_baseline",
    "ivb_vs_avg_scoring",
    "dead_zone_detection",
    "supabase_lab_write_readback",
    "tracking_identity_payloads",
    "no_static_player_seed_fallback",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    status = read_json(STATUS_PATH)
    payload = read_json(PAYLOAD_PATH)

    if status.get("report_id") != "ivb_heat_map":
        fail(f"unexpected report_id: {status.get('report_id')}")

    if status.get("state") != "fresh":
        fail(f"status is not fresh: {status.get('state')}")

    if status.get("build_success") is not True:
        fail("build_success is not true")

    if status.get("degraded") is True:
        fail("degraded is true")

    if status.get("used_fallback") is True:
        fail("used_fallback is true")

    if status.get("mode") != EXPECTED_MODE:
        fail(f"mode mismatch: expected {EXPECTED_MODE}, got {status.get('mode')}")

    layers = set(status.get("pipeline_layers") or [])
    missing_layers = sorted(REQUIRED_LAYERS - layers)
    if missing_layers:
        fail(f"missing pipeline layers: {missing_layers}")

    section_counts = status.get("section_counts") or {}
    tile_count = int(section_counts.get("ivb_tiles") or 0)
    raw_row_count = int(section_counts.get("raw_statcast_rows") or 0)
    grouped_pitchers = int(section_counts.get("grouped_pitchers") or 0)
    lab_rows_written = int(section_counts.get("lab_rows_written") or 0)
    lab_rows_read = int(section_counts.get("lab_rows_read") or 0)

    if tile_count <= 0:
        fail("ivb_tiles count is zero")

    if raw_row_count <= 0:
        fail("raw_statcast_rows count is zero")

    if grouped_pitchers <= 0:
        fail("grouped_pitchers count is zero")

    if lab_rows_written <= 0:
        fail("lab_rows_written count is zero")

    if lab_rows_read <= 0:
        fail("lab_rows_read count is zero")

    heat_cards = payload.get("heat_cards")
    if not isinstance(heat_cards, list) or not heat_cards:
        fail("payload heat_cards missing or empty")

    if len(heat_cards) != tile_count:
        fail(f"status ivb_tiles {tile_count} does not match payload heat_cards {len(heat_cards)}")

    first = heat_cards[0]
    for key in ["player_id", "player_name", "team", "ivb_raw", "ivb_vs_avg", "heat_class", "band_label"]:
        if first.get(key) in (None, ""):
            fail(f"first heat card missing {key}")

    if payload.get("raw_row_count", 0) <= 0:
        fail("payload raw_row_count is zero")

    if payload.get("grouped_pitcher_count", 0) <= 0:
        fail("payload grouped_pitcher_count is zero")

    if payload.get("used_lab_readback") is not True:
        fail("payload used_lab_readback is not true")

    if payload.get("used_direct_grouped_fallback") is True:
        fail("payload used_direct_grouped_fallback is true")

    if not HTML_PATH.exists():
        fail(f"missing rendered HTML: {HTML_PATH}")

    html = HTML_PATH.read_text(encoding="utf-8", errors="replace")
    html_lower = html.lower()

    for term in ["ivb", "heat", "data-player", "track"]:
        if term not in html_lower:
            fail(f"HTML missing expected term/markup: {term}")

    for forbidden in ["lorem ipsum", "todo"]:
        if forbidden in html_lower:
            fail(f"HTML contains forbidden developer term: {forbidden}")

    # Do not fail merely on the word "placeholder".
    # Safe UI placeholder/empty-state language is allowed.
    # Static seeded fallback is guarded through status mode, pipeline layers,
    # real player identity fields, Supabase lab readback, and status/payload alignment.

    print("OK: IVB Heat Map status, payload, rendered HTML, and tracking contract are aligned.")


if __name__ == "__main__":
    main()
