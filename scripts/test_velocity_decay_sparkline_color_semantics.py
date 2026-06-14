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

source = SOURCE_PATH.read_text(encoding="utf-8")
rendered = HTML_PATH.read_text(encoding="utf-8", errors="replace")
payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))

allowed_classes = {
    "velocity-sparkline-rebound",
    "velocity-sparkline-stable",
    "velocity-sparkline-watch",
    "velocity-sparkline-decay",
    "velocity-sparkline-cliff",
    "velocity-sparkline-deceptive",
}

for term in [
    "VELOCITY_DECAY_SPARKLINE_SEMANTIC_COLOR_V1",
    "def classify_sparkline_semantic_class",
    '"sparkline_semantic_class"',
    "data-sparkline-color-source=\"velocity_decay_model_state\"",
    "data-sparkline-color-class=\"{{ row.sparkline_semantic_class }}\"",
    "sparkline-path {{ row.sparkline_class }} {{ row.sparkline_semantic_class }}",
]:
    require(f"Missing semantic sparkline source term: {term}", term in source)

for cls in sorted(allowed_classes):
    require(f"Missing CSS class definition for {cls}", f".sparkline-path.{cls}" in source)

cards = payload.get("cards") or []
require("Velocity Decay payload.cards is empty", bool(cards))

seen_classes = set()
for idx, card in enumerate(cards, start=1):
    cls = card.get("sparkline_semantic_class")
    seen_classes.add(cls)
    require(f"Card #{idx} missing sparkline_semantic_class", bool(cls))
    require(f"Card #{idx} has invalid sparkline semantic class: {cls}", cls in allowed_classes)

for cls in seen_classes:
    if cls:
        require(f"Rendered HTML missing semantic class {cls}", cls in rendered)

require("Rendered HTML missing semantic color source marker", 'data-sparkline-color-source="velocity_decay_model_state"' in rendered)
require("Rendered HTML missing semantic color class attribute", 'data-sparkline-color-class="' in rendered)
require("Rendered sparkline SVG missing role image", 'role="img"' in rendered)
require("Rendered sparkline title missing recent fastball language", "recent fastball velocity trend" in rendered)

# Guard against decorative-only color: the class must be emitted from card data, not hardcoded once.
require(
    "Sparkline semantic class is not template-driven",
    "{{ row.sparkline_semantic_class }}" in source,
)

if failures:
    print("Velocity Decay sparkline semantic color audit failed:")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("OK: Velocity Decay sparkline semantic color audit passed")
