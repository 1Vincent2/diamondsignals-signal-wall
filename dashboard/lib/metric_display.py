#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Any


MISSING_METRIC_DISPLAY = "———"


METRIC_DEFINITIONS: dict[str, str] = {
    "LIVE_SCORE": "Card ranking score inside the active report board.",
    "EDGE_SCORE": "A 0 to 100 signal-strength score versus baseline.",
    "SIGNAL": "Primary movement or opportunity signal for the player.",
    "EXTRACTION_SCORE": "Top-line MLB Extraction conviction score blending hidden skill quality, market gap, and remaining public hesitation.",
    "PHYSICS_CORE": "Underlying skill signal before the box score or public market fully reflects it.",
    "MARKET_GAP": "Distance between the player signal and public-facing perception or roster-market price.",
    "MARKET_ATTENTION_FEED": "Market visibility and attention context supporting the extraction read.",
    "BALLISTICS_VS_SURFACE": "Mismatch between underlying batted-ball or pitch-shape signal and visible surface results.",
    "EXTRACTION_THESIS": "Plain-English reason the player appears on the MLB Extraction surface.",
    "CONTACT_QUALITY_CONTEXT": "Recent hitter contact-quality signal such as EV delta, average EV, or barrel support.",
    "PITCHING_PHYSICS_CONTEXT": "Recent pitcher physics signal such as velocity delta, whiff rate, or fastball quality.",
    "MARKET_ATTENTION_CONTEXT": "Fallback market or roster-attention context supporting the extraction card.",
    "SEAGER": "Hitter quality/context score used by the player signal index.",
    "BABIP": "Batting average on balls in play; used here as context for surface-result noise.",
    "BB_K": "Walk-to-strikeout ratio context for hitter approach quality.",
    "K_BB": "Strikeout-to-walk ratio context for pitcher command and dominance.",
    "PA": "Plate appearances in the supporting sample.",
    "BF": "Batters faced in the supporting sample.",
    "SAMPLE": "Current supporting sample size or availability note.",
    "SEASON": "Season context for the supporting sample.",
    "PHYSICAL": "Underlying force, pitch-shape, movement, carry, extension, or contact-authority signal.",
    "VISION": "Conversion layer showing whether the physical trait is becoming usable skill.",
    "MARKET": "Latency gap between the underlying Apex signal and public recognition.",
    "PRIMARY_SIGNAL": "Plain-English lead signal that triggered the Apex card.",
    "SUPPORTING_METRIC": "Secondary metric or metric pair supporting the Apex diagnosis.",
    "GEOMETRY": "Pitch-plane or movement-shape evidence supporting the Apex read.",
    "DECEPTION": "Movement, release, or perception evidence that may hide the true pitch quality.",
    "MOVEMENT_DEVIATION": "Observed movement change or deviation supporting the Apex arm signal.",
    "DYNAMIC_HARD_HIT": "Contact-authority proxy for flight-optimized damage.",
    "LA_CONSISTENCY": "Launch-angle consistency showing whether force is converting into flight.",
    "STATUS_SOURCE": "Source rail that produced or verified the player status.",
    "RISK": "Risk readout attached to the waiver recommendation.",
    "COMMAND": "Plain-English deployment instruction for the current waiver asset.",
    "MARKET_DEFECT": "The market inefficiency or public-pricing gap creating the waiver opportunity.",
    "ASSET_TYPE": "Player asset classification used by the Waiver Wire surface.",
    "SIGNAL_WINDOW": "Time window or source window behind the current waiver signal.",
    "OWNERSHIP_GATE": "Roster-availability gate showing whether the player is still broadly actionable in the market.",
    "SIGNAL_LAYER": "Players ranked by recent underlying signal quality, not just public-facing box-score reputation.",
    "LIVE_SIGNAL_LAYER": "Live source-window signal layer for current AAA production or movement intelligence.",
    "AAA_GEMS": "Lower-minors surveillance board for verified AA, High-A, Low-A, or source-locked depth signals.",
    "DEPTH_V0.1": "Lower-minors surveillance model/version tag used by AAA GEMS.",
    "DATE": "Transaction, movement, or source-event date.",
    "MLB_STATUS": "MLB arrival, recall, debut, active-roster, or transaction context.",
    "MOVEMENT": "Transaction or roster-movement type attached to the player signal.",
    "MOVEMENT_LAYER": "Recent arrivals, recalls, debuts, and roster movement events supporting promotion timing.",
    "TEAM_LABELS": "Minor-league affiliate context first, with MLB parent organization as secondary context.",
    "72_HR": "Short-window acceleration board prioritizing near-term promotion pressure.",
    "14_DAY": "Broader scout window confirming recent AAA production and reducing one-game noise.",
    "KDE_SCORE": "Composite Kinetic Drift Engine score ranking pitcher movement, velocity, release, and instability drift.",
    "KINETIC_RISK": "Risk layer showing decay, fatigue, or delivery-collapse pressure.",
    "KINETIC_RISK_SCORE": "Composite risk score for negative kinetic drift or delivery instability.",
    "KINETIC_EMERGENCE_SCORE": "Positive emergence score for shape, power, or movement gains.",
    "KINETIC_INSTABILITY_SCORE": "Instability score for release, extension, velocity, spin, or movement volatility.",
    "MOVEMENT_STATE": "Machine-readable KDE movement-state bucket for the current signal.",
    "MOVEMENT_STATE_LABEL": "Plain-English movement-state label shown to the operator.",
    "TRACE_BEHAVIOR": "Recent trajectory behavior across the drift trace.",
    "CONFIDENCE_SCORE": "Confidence score based on KDE strength, sample depth, and pitch-count support.",
    "OPERATOR_ACTION": "Recommended operator action generated by the KDE signal.",
    "OPERATOR_NOTE": "Plain-English note explaining how to treat the current KDE signal.",
    "DIAGNOSIS": "KDE diagnosis summarizing the inferred movement or fatigue pattern.",
    "RAW_DIAGNOSIS": "Unmodified diagnosis before final KDE state labeling.",
    "RECENT_APPEARANCES": "Recent appearance count used for the active KDE comparison window.",
    "BASELINE_APPEARANCES": "Baseline appearance count used for recent-versus-prior KDE comparison.",
    "DRIFT_TRACE": "Game-by-game trace of recent movement, velocity, release, and extension changes.",
    "LATEST_GAME_DATE": "Most recent game date included in the KDE signal.",
    "KDE_BAND": "Kinetic Drift Engine severity band assigned from the composite score.",
    "CLASSIFICATION": "Risk classification bucket derived from the KDE score.",
    "KRS": "Kinetic Risk Score, focused on decay, fatigue, and instability exposure.",
    "KES": "Kinetic Emergence Score, focused on positive shape or power gains.",
    "KIS": "Kinetic Instability Score, focused on release and movement volatility.",
    "EXTENSION_DELTA": "Release-extension change versus baseline.",
    "RELEASE_Z_DELTA": "Vertical release-point change versus baseline.",
    "RELEASE_X_DELTA": "Horizontal release-point change versus baseline.",
    "SPIN_DELTA": "Spin-rate change versus baseline.",
    "IVB_DELTA": "Change in induced vertical break versus recent baseline.",
    "VELO_DELTA": "Velocity change versus recent baseline.",
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
    "DEAD_ZONE": "Whether the fastball carry profile sits in the 12–15 inch contact-risk band.",
    "VAA": "Estimated vertical approach angle for the fastball shape window.",
    "TRACKED_ARMS": "Pitchers meeting the fastball sample threshold in the current IVB window.",
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
