#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.lib.metric_display import (
    MISSING_METRIC_DISPLAY,
    delta_badge_class,
    format_metric_number,
    metric_definition,
    metric_key,
    metric_label_attrs,
    safe_metric_value,
)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    print("--- DiamondSignals metric display contract smoke test ---")

    assert_true(metric_key("Velo Delta proxy") == "VELO_DELTA_PROXY", "VELO_DELTA proxy key normalization failed")
    assert_true("Velocity" in metric_definition("VELO_DELTA"), "VELO_DELTA definition missing")
    assert_true("isolated" in metric_definition("ISO_LIVE").lower(), "ISO_LIVE definition missing")
    assert_true("Level-adjusted" in metric_definition("LVL_ADJUST"), "LVL_ADJUST definition missing")

    assert_true(safe_metric_value(None) == MISSING_METRIC_DISPLAY, "None fallback failed")
    assert_true(safe_metric_value("") == MISSING_METRIC_DISPLAY, "blank fallback failed")
    assert_true(safe_metric_value("nan") == MISSING_METRIC_DISPLAY, "nan fallback failed")
    assert_true(safe_metric_value("12.3") == "12.3", "normal string fallback failed")

    assert_true(format_metric_number(1.234, decimals=2, signed=True) == "+1.23", "signed positive formatting failed")
    assert_true(format_metric_number(-1.234, decimals=2, signed=True) == "-1.23", "signed negative formatting failed")
    assert_true(format_metric_number(None) == MISSING_METRIC_DISPLAY, "missing number formatting failed")

    assert_true(delta_badge_class(1.0) == "metric-badge-positive", "positive badge failed")
    assert_true(delta_badge_class(-1.0) == "metric-badge-negative", "negative badge failed")
    assert_true(delta_badge_class(0.1) == "metric-badge-neutral", "neutral badge failed")

    attrs = metric_label_attrs("IVB Raw")
    assert_true(attrs["data_metric_key"] == "IVB_RAW", "metric label attrs key failed")
    assert_true("title" in attrs and "IVB Raw:" in attrs["title"], "metric label title failed")

    print("FINAL_STATUS: PASS_METRIC_DISPLAY_CONTRACT")


if __name__ == "__main__":
    main()
