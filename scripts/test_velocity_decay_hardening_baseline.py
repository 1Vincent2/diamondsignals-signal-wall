#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

SOURCE = Path("dashboard/build_velocity_decay.py")
HTML = Path("dist/velocity-decay-monitor/index.html")
STATUS = Path("dist/status/velocity-decay.json")
PAYLOAD = Path("dist/velocity_decay_monitor.json")

EXPECTED_MODE = "statcast_velocity_decay_dynamic_v1"
EXPECTED_LAYERS = {
    "pybaseball_statcast_pitch_level_feed",
    "fastball_velocity_window",
    "recent_baseline_delta_scoring",
    "extension_decay_detection",
    "perceived_velocity_proxy",
    "risk_alert_classification",
    "tracking_identity_payloads",
    "no_static_player_seed_fallback",
}

failures = []

def require(label: str, condition: bool) -> None:
    if not condition:
        failures.append(label)

def load_json(path: Path):
    require(f"Missing file: {path}", path.exists())
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

source = SOURCE.read_text(encoding="utf-8") if SOURCE.exists() else ""
html = HTML.read_text(encoding="utf-8", errors="replace") if HTML.exists() else ""
html_lower = html.lower()
status = load_json(STATUS)
payload = load_json(PAYLOAD)

cards = payload.get("cards") if isinstance(payload, dict) else []
section_counts = status.get("section_counts") if isinstance(status, dict) else {}
pipeline_layers = set(status.get("pipeline_layers") or [])

require("Missing Velocity Decay builder", SOURCE.exists())
require("Missing Velocity Decay rendered HTML", HTML.exists())
require("Missing Velocity Decay status JSON", STATUS.exists())
require("Missing Velocity Decay payload JSON", PAYLOAD.exists())

require("Unexpected report_id", status.get("report_id") == "velocity_decay")
require("Velocity Decay status is not fresh", status.get("state") == "fresh")
require("Velocity Decay build_success is not true", status.get("build_success") is True)
require("Velocity Decay used_fallback should be false", status.get("used_fallback") is False)
require("Velocity Decay degraded should be false", status.get("degraded") is False)
require("Velocity Decay mode mismatch", status.get("mode") == EXPECTED_MODE)

missing_layers = sorted(EXPECTED_LAYERS - pipeline_layers)
require(f"Missing expected pipeline layers: {missing_layers}", not missing_layers)

require("Payload cards is not a non-empty list", isinstance(cards, list) and len(cards) > 0)
require(
    "Status/payload card count mismatch",
    int(section_counts.get("velocity_decay_cards") or -1) == len(cards) if isinstance(cards, list) else False,
)

require("Rendered page missing Velocity Decay title", "Velocity Decay Monitor" in html)
require("Rendered page missing Field Guide trigger", "Field Guide" in html and "openGuide()" in html)
require("Rendered page missing Field Guide drawer", "guideDrawer" in html and "closeGuide()" in html)
require("Rendered page missing tracking script", "/player-card-actions.js" in html)
require("Rendered page missing tracking source tag", 'data-source-tag="VELOCITY_DECAY"' in html)
require("Rendered page missing add-to-roster tracking hook", "js-add-to-roster" in html)

require("Builder missing status mode constant", EXPECTED_MODE in source)
require("Builder missing pipeline layer constant", "VELOCITY_DECAY_PIPELINE_LAYERS" in source)
require("Builder missing no-static-fallback layer", "no_static_player_seed_fallback" in source)
require("Builder missing Statcast fetch pathway", "statcast" in source and "fetch_statcast_window" in source)
require("Builder missing tracking source tag", 'data-source-tag="VELOCITY_DECAY"' in source)

for forbidden in ["lorem ipsum", "todo"]:
    require(f"Forbidden placeholder term present: {forbidden}", forbidden not in html_lower)

if failures:
    print("Velocity Decay hardening baseline audit failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("OK: Velocity Decay hardening baseline audit passed")
