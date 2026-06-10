#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

STATUS_PATH = Path("dist/status/velocity-decay.json")
PAYLOAD_PATH = Path("dist/velocity_decay_monitor.json")
HTML_PATH = Path("dist/velocity-decay-monitor/index.html")

EXPECTED_MODE = "statcast_velocity_decay_dynamic_v1"
EXPECTED_PIPELINE_LAYERS = {
    "pybaseball_statcast_pitch_level_feed",
    "fastball_velocity_window",
    "recent_baseline_delta_scoring",
    "extension_decay_detection",
    "perceived_velocity_proxy",
    "risk_alert_classification",
    "tracking_identity_payloads",
    "no_static_player_seed_fallback",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing file: {path}")
    return json.loads(path.read_text())


def require_nonempty(row: dict, key: str) -> None:
    if key not in row or row.get(key) in (None, "", []):
        fail(f"Velocity Decay card missing required non-empty key: {key}")


def main() -> None:
    status = load_json(STATUS_PATH)
    payload = load_json(PAYLOAD_PATH)

    if not HTML_PATH.exists():
        fail(f"missing file: {HTML_PATH}")

    html = HTML_PATH.read_text(errors="replace")
    html_lower = html.lower()

    if status.get("report_id") != "velocity_decay":
        fail(f"unexpected report_id: {status.get('report_id')}")

    if status.get("state") != "fresh":
        fail(f"Velocity Decay state is not fresh: {status.get('state')}")

    if status.get("build_success") is not True:
        fail("Velocity Decay build_success is not true")

    if status.get("used_fallback") is not False:
        fail(f"Velocity Decay used_fallback should be false, got {status.get('used_fallback')}")

    if status.get("degraded") is not False:
        fail(f"Velocity Decay degraded should be false, got {status.get('degraded')}")

    if status.get("mode") != EXPECTED_MODE:
        fail(f"Velocity Decay mode mismatch: expected {EXPECTED_MODE}, got {status.get('mode')}")

    pipeline_layers = set(status.get("pipeline_layers") or [])
    missing_layers = sorted(EXPECTED_PIPELINE_LAYERS - pipeline_layers)
    if missing_layers:
        fail(f"Velocity Decay missing pipeline layer(s): {missing_layers}")

    section_counts = status.get("section_counts") or {}
    status_count = section_counts.get("velocity_decay_cards")
    if status_count is None:
        fail("status missing section_counts.velocity_decay_cards")

    cards = payload.get("cards")
    if not isinstance(cards, list):
        fail("payload.cards is not a list")

    if len(cards) <= 0:
        fail("payload.cards is empty")

    if int(status_count) != len(cards):
        fail(f"status/payload count mismatch: status={status_count}, payload={len(cards)}")

    first = cards[0]

    required_card_keys = [
        "player_id",
        "player_name",
        "team",
        "risk_score",
        "risk_tier",
        "primary_alert",
        "analysis",
        "velo_delta",
        "extension_delta",
        "perceived_velo_delta",
        "trend_values",
        "sample_count",
        "profile_url",
        "headshot_url",
        "risk_score_label",
        "score_class",
        "alert_class",
        "sparkline_class",
        "velo_delta_label",
        "extension_delta_label",
        "perceived_delta_label",
        "decay_slope_label",
        "trend_points",
        "sample_note",
    ]

    for key in required_card_keys:
        require_nonempty(first, key)

    if not isinstance(first.get("trend_values"), list) or len(first["trend_values"]) < 3:
        fail("first card trend_values must contain at least three values")

    if "velocity" not in html_lower:
        fail("HTML missing velocity language")

    if "decay" not in html_lower:
        fail("HTML missing decay language")

    if "data-player" not in html:
        fail("HTML missing data-player tracking identity markup")

    if "track" not in html_lower:
        fail("HTML missing track language/markup")

    for forbidden in ["lorem ipsum", "todo"]:
        if forbidden in html_lower:
            fail(f"HTML contains forbidden developer/placeholder term: {forbidden}")

    # Do not fail merely on the word "placeholder".
    # Safe UI placeholder/empty-state language is allowed.
    # Static seeded player fallback is guarded through status mode, pipeline layers,
    # real player_id/player_name fields, and status/payload count alignment.

    print("OK: Velocity Decay status, payload, rendered HTML, and tracking contract are aligned.")


if __name__ == "__main__":
    main()
