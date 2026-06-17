#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Any


MISSING_METRIC_DISPLAY = "———"


METRIC_DEFINITIONS: dict[str, str] = {
    "LIVE_SCORE": "Card ranking score inside the active report board.",
    "EDGE_SCORE": "A 0 to 100 signal-strength score versus baseline.",
    "SIGNAL": "Primary movement or opportunity signal for the player.",
    "STABILITY": "Short-horizon confidence and repeatability context.",
    "CONTEXT": "Roster, level, opportunity, or market context supporting the signal.",

    "ISO": "Isolated power. Slugging percentage minus batting average.",
    "ISO_LIVE": "Current isolated-power value from the live source window.",
    "ISO_DELTA": "Change in isolated power versus recent baseline.",
    "ISO_DELTA_PROXY": "Proxy for short-window power acceleration versus recent baseline.",
    "K/BB": "Strikeout-to-walk ratio.",
    "K/BB_STABILITY": "Strikeout-to-walk stability across the active window.",
    "BB/K": "Walk-to-strikeout ratio.",
    "BB_LIVE": "Current walk count or walk signal from the active source window.",
    "HR_LIVE": "Current home-run signal from the active source window.",

    "VELO_DELTA": "Velocity change versus recent baseline.",
    "VELO_DELTA_PROXY": "Proxy for short-window velocity acceleration or deceleration versus recent baseline.",
    "WHIFF_STABILITY": "Strikeout or whiff-support stability across the active window.",
    "LVL_ADJUST": "Level-adjusted context for translating production across competition level.",
    "IP_LIVE": "Current innings-pitched value from the active source window.",
    "K_LIVE": "Current strikeout count or strikeout signal from the active source window.",

    "FIELD_TILT": "Share of tracked arms carrying the target shape or carry profile.",
    "IVB": "Induced vertical break. Fastball ride or carry measured in inches.",
    "IVB_RAW": "Raw fastball induced vertical break in inches.",
    "IVB_VS_AVG": "Fastball IVB compared with the expected value for the same velocity bucket.",
    "DEAD_ZONE_COUNT": "Count of arms in the flatter 12 to 15 inch carry band where contact risk rises.",
    "WHIFF_PROB": "Translation layer estimating bat-missing expectation from pitch-shape inputs.",

    "RISK_SCORE": "Composite risk score for decay, fatigue, or mechanical-drift exposure.",
    "VELO_DELTA_DECAY": "Raw speed change versus recent baseline.",
    "EXT_DELTA": "Release-extension change versus recent baseline.",
    "PERCEIVED_DELTA": "Estimated change in how fast the pitch plays to hitters.",
    "DECAY_SLOPE": "Short-horizon trend state for velocity or shape decay.",

    "DISRUPTION_SCORE": "Composite score for pitch-shape disruption and breakout movement.",
    "IVB_DELTA": "Change in induced vertical break versus recent baseline.",
    "VAA_DELTA": "Change in vertical approach angle versus recent baseline.",
    "MOVE_DELTA": "Change in pitch movement profile versus recent baseline.",
    "ACTIVE_SPIN": "Active-spin movement support relative to baseline.",

    "PHYSICS_CORE": "Underlying movement, power, or shape-quality support.",
    "MARKET_GAP": "Gap between underlying signal strength and public-market attention.",
    "MARKET_ATTENTION_FEED": "Public exposure and market-awareness context.",
    "SEAGER": "Hitter quality signal used by the extraction layer.",
    "BABIP": "Batting average on balls in play.",
    "SAMPLE": "Current sample-size context.",
    "PA": "Plate appearances.",
    "BF": "Batters faced.",

    "APEX_SCORE": "Composite breakout score for the Apex Extraction surface.",
    "HIGH_STAKES_DELTA": "High-leverage change signal used by the Apex layer.",
    "DIAMOND_DELTA": "DiamondSignals change signal versus baseline or prior state.",
    "FB_IVB": "Fastball induced vertical break.",
    "WHIFF%": "Whiff percentage.",
}


def metric_key(label: str | None) -> str:
    raw = str(label or "").strip().upper()
    raw = raw.replace(" ", "_").replace("-", "_")
    raw = raw.replace("__", "_")
    if raw == "VELO_DELTA_PROXY":
        return "VELO_DELTA_PROXY"
    if raw == "VELO_DELTA":
        return "VELO_DELTA"
    return raw


def metric_definition(label: str | None) -> str:
    key = metric_key(label)
    return METRIC_DEFINITIONS.get(key, f"{str(label or 'Metric').strip()} metric used by this report.")


def metric_title(label: str | None) -> str:
    clean = str(label or "Metric").strip()
    return f"{clean}: {metric_definition(clean)}"


def safe_metric_value(value: Any, placeholder: str = MISSING_METRIC_DISPLAY) -> str:
    if value is None:
        return placeholder

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return placeholder

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "inf", "infinity", "undefined", "null"}:
        return placeholder

    return text


def format_metric_number(
    value: Any,
    *,
    decimals: int = 1,
    prefix: str = "",
    suffix: str = "",
    signed: bool = False,
    placeholder: str = MISSING_METRIC_DISPLAY,
) -> str:
    try:
        number = float(value)
    except Exception:
        return placeholder

    if math.isnan(number) or math.isinf(number):
        return placeholder

    sign = "+" if signed and number > 0 else ""
    return f"{prefix}{sign}{number:.{decimals}f}{suffix}"


def delta_badge_class(value: Any, *, positive_threshold: float = 0.75, negative_threshold: float = -0.75) -> str:
    try:
        number = float(value)
    except Exception:
        return "metric-badge-neutral"

    if math.isnan(number) or math.isinf(number):
        return "metric-badge-neutral"
    if number >= positive_threshold:
        return "metric-badge-positive"
    if number <= negative_threshold:
        return "metric-badge-negative"
    return "metric-badge-neutral"


def metric_label_attrs(label: str | None) -> dict[str, str]:
    title = metric_title(label)
    return {
        "title": title,
        "aria_label": title,
        "data_metric_key": metric_key(label),
    }
