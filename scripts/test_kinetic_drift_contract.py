#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

STATUS_PATH = Path("dist/status/kinetic-drift.json")
PAYLOAD_PATH = Path("dist/admin/kinetic_drift_signals.json")
HTML_PATH = Path("dist/admin/kinetic-drift/index.html")

EXPECTED_MODE = "statcast_kinetic_drift_dynamic_v1"
EXPECTED_PIPELINE_LAYERS = {
    "pybaseball_statcast_pitch_level_feed",
    "fastball_movement_velocity_window",
    "pitcher_recent_baseline_comparison",
    "krs_kes_kis_scoring",
    "snapshot_fallback_available",
    "tracking_identity_markup",
    "no_static_player_seed_fallback",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing file: {path}")
    return json.loads(path.read_text())


def require_nonempty(row: dict, key: str) -> None:
    if key not in row or row.get(key) in (None, ""):
        fail(f"signal missing required non-empty key: {key}")


def main() -> None:
    status = load_json(STATUS_PATH)
    payload = load_json(PAYLOAD_PATH)

    if not HTML_PATH.exists():
        fail(f"missing file: {HTML_PATH}")

    html = HTML_PATH.read_text(errors="replace")

    if status.get("report_id") != "kinetic_drift":
        fail(f"unexpected report_id: {status.get('report_id')}")

    if status.get("state") != "fresh":
        fail(f"KDE state is not fresh: {status.get('state')}")

    if status.get("build_success") is not True:
        fail("KDE build_success is not true")

    if status.get("used_fallback") is not False:
        fail(f"KDE used_fallback should be false, got {status.get('used_fallback')}")

    if status.get("degraded") is not False:
        fail(f"KDE degraded should be false, got {status.get('degraded')}")

    if status.get("mode") != EXPECTED_MODE:
        fail(f"KDE mode mismatch: expected {EXPECTED_MODE}, got {status.get('mode')}")

    pipeline_layers = set(status.get("pipeline_layers") or [])
    missing_layers = sorted(EXPECTED_PIPELINE_LAYERS - pipeline_layers)
    if missing_layers:
        fail(f"KDE missing pipeline layer(s): {missing_layers}")

    section_counts = status.get("section_counts") or {}
    status_count = section_counts.get("kinetic_drift_signals")
    if status_count is None:
        fail("status missing section_counts.kinetic_drift_signals")

    signals = payload.get("signals")
    if not isinstance(signals, list):
        fail("payload.signals is not a list")

    if len(signals) <= 0:
        fail("payload.signals is empty")

    if int(status_count) != len(signals):
        fail(f"status/payload count mismatch: status={status_count}, payload={len(signals)}")

    if payload.get("module") != "KINETIC_DRIFT_ENGINE_V1":
        fail(f"unexpected payload module: {payload.get('module')}")

    first = signals[0]

    required_signal_keys = [
        "player_id",
        "player_name",
        "team",
        "latest_game_date",
        "kde_score",
        "kde_band",
        "kinetic_risk_score",
        "kinetic_emergence_score",
        "kinetic_instability_score",
        "movement_state",
        "movement_state_label",
        "trace_behavior",
        "confidence_score",
        "operator_action",
        "operator_note",
        "drift_trace",
        "metrics",
        "trend",
    ]

    for key in required_signal_keys:
        require_nonempty(first, key)

    if not isinstance(first.get("drift_trace"), list) or not first["drift_trace"]:
        fail("first signal drift_trace is empty or not a list")

    if not isinstance(first.get("metrics"), dict) or not first["metrics"]:
        fail("first signal metrics is empty or not a dict")

    if not isinstance(first.get("trend"), dict) or not first["trend"]:
        fail("first signal trend is empty or not a dict")

    html_lower = html.lower()

    for term in ["kinetic", "drift"]:
        if term not in html_lower:
            fail(f"HTML missing expected KDE language: {term}")

    if "data-player" not in html:
        fail("HTML missing data-player tracking identity markup")

    if "track" not in html_lower:
        fail("HTML missing track language/markup")

    for forbidden in ["lorem ipsum", "todo"]:
        if forbidden in html_lower:
            fail(f"HTML contains forbidden placeholder/developer term: {forbidden}")

    # Important: do not fail merely on the word "placeholder".
    # KDE currently has safe UI/empty-state placeholder language; the data-mode audit
    # should distinguish UI copy from static seeded player fallback.

    print("OK: Kinetic Drift status, payload, admin HTML, and tracking contract are aligned.")


if __name__ == "__main__":
    main()
