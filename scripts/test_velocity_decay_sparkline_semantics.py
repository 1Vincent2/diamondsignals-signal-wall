#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCE_PATH = Path("dashboard/build_velocity_decay.py")
HTML_PATH = Path("dist/velocity-decay-monitor/index.html")
PAYLOAD_PATH = Path("dist/velocity_decay_monitor.json")

failures: list[str] = []

def require(label: str, condition: bool) -> None:
    if not condition:
        failures.append(label)

source = SOURCE_PATH.read_text(encoding="utf-8", errors="replace")
rendered = HTML_PATH.read_text(encoding="utf-8", errors="replace")
payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))

for term in [
    "data-trend-source=\"recent_fastball_velocity\"",
    "data-trend-window=\"recent_appearances\"",
    "data-trend-values=\"{{ row.trend_values|join(',') }}\"",
    "data-velo-delta=\"{{ row.velo_delta_label }}\"",
    "Recent FB Velo",
    "Last 5 appearances",
    "recent fastball velocity trend",
    "role=\"img\"",
]:
    require(f"Missing sparkline semantic source term: {term}", term in source)

for term in [
    "data-trend-source=\"recent_fastball_velocity\"",
    "data-trend-window=\"recent_appearances\"",
    "Recent FB Velo",
    "Last 5 appearances",
    "recent fastball velocity trend",
    "role=\"img\"",
]:
    require(f"Missing sparkline semantic rendered term: {term}", term in rendered)

require("Old vague label still rendered: Velocity Trend", "Velocity Trend" not in rendered)
require("Old vague window still rendered: Last 5 starts", "Last 5 starts" not in rendered)

cards = payload.get("cards") or []
require("Payload cards missing", isinstance(cards, list) and len(cards) > 0)

for idx, card in enumerate(cards[:12], start=1):
    trend_values = card.get("trend_values")
    trend_points = card.get("trend_points")
    velo_delta_label = card.get("velo_delta_label")

    require(f"Card {idx} missing trend_values list", isinstance(trend_values, list) and len(trend_values) >= 3)
    require(f"Card {idx} missing trend_points", isinstance(trend_points, str) and len(trend_points.strip()) > 0)
    require(f"Card {idx} missing velo_delta_label", isinstance(velo_delta_label, str) and velo_delta_label.strip() not in ("", "--"))

if failures:
    print("Velocity Decay sparkline semantics audit failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("OK: Velocity Decay sparkline semantics audit passed")
