#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pybaseball import cache, statcast
from jinja2 import Template

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DIST_DIR = REPO_ROOT / "dist"
OUT_DIR = DIST_DIR / "admin"
OUT_PATH = OUT_DIR / "kinetic_drift_signals.json"
HTML_DIR = OUT_DIR / "kinetic-drift"
HTML_PATH = HTML_DIR / "index.html"
STATUS_DIR = DIST_DIR / "status"
STATUS_PATH = STATUS_DIR / "kinetic-drift.json"
SNAPSHOT_DIR = DIST_DIR / "_snapshots" / "kinetic-drift"
SNAPSHOT_PATH = SNAPSHOT_DIR / "kinetic_drift_signals.json"
TEMPLATE_PATH = BASE_DIR / "templates" / "admin" / "kinetic_drift.html"
SHELL_STYLES_PATH = BASE_DIR / "templates" / "shell_styles.css"
SHELL_NAV_PATH = BASE_DIR / "templates" / "shell_nav.html"

LOOKBACK_DAYS = 45
RECENT_APPEARANCES = 3
BASELINE_APPEARANCES = 8
MIN_FASTBALLS_PER_GAME = 8
MIN_TOTAL_APPEARANCES = 6
MAX_SIGNALS = 60

FASTBALL_TYPES = {"FF", "FA", "SI", "FC"}


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def z_score(current: float | None, mean: float | None, std: float | None) -> float | None:
    current = safe_float(current)
    mean = safe_float(mean)
    std = safe_float(std)
    if current is None or mean is None or std is None or std < 0.01:
        return None
    return (current - mean) / std


def mean_or_none(series) -> float | None:
    val = pd.to_numeric(series, errors="coerce").dropna()
    if val.empty:
        return None
    return safe_float(val.mean())


def std_or_none(series) -> float | None:
    val = pd.to_numeric(series, errors="coerce").dropna()
    if len(val) < 3:
        return None
    return safe_float(val.std(ddof=0))


def fetch_statcast_window(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Fetching Statcast from {start_date} to {end_date}...")
    cache.enable()
    raw = statcast(start_dt=start_date, end_dt=end_date)
    if raw is None or raw.empty:
        return pd.DataFrame()
    return raw.copy()


def load_fastball_pitches(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["pitch_type"] = df.get("pitch_type").astype(str)
    df = df[df["pitch_type"].isin(FASTBALL_TYPES)].copy()
    if df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "release_speed",
        "release_spin_rate",
        "release_extension",
        "release_pos_x",
        "release_pos_z",
        "pfx_x",
        "pfx_z",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")
    df["game_pk"] = pd.to_numeric(df.get("game_pk"), errors="coerce").astype("Int64")
    df["pitcher"] = pd.to_numeric(df.get("pitcher"), errors="coerce").astype("Int64")
    df["player_name"] = df.get("player_name").fillna("").astype(str)
    df["home_team"] = df.get("home_team").fillna("").astype(str)
    df["away_team"] = df.get("away_team").fillna("").astype(str)

    df["ivb_inches"] = df["pfx_z"] * 12.0
    df["hb_inches"] = df["pfx_x"] * 12.0

    df = df.dropna(
        subset=[
            "pitcher",
            "game_pk",
            "game_date",
            "release_speed",
            "release_extension",
            "release_pos_x",
            "release_pos_z",
            "ivb_inches",
        ]
    ).copy()

    return df


def build_pitcher_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    if pitches.empty:
        return pd.DataFrame()

    appearances = (
        pitches.groupby(["pitcher", "game_pk"], dropna=True)
        .agg(
            game_date=("game_date", "max"),
            player_name=("player_name", "last"),
            team=("home_team", "last"),
            release_speed=("release_speed", "mean"),
            release_spin_rate=("release_spin_rate", "mean"),
            release_extension=("release_extension", "mean"),
            release_pos_x=("release_pos_x", "mean"),
            release_pos_z=("release_pos_z", "mean"),
            ivb_inches=("ivb_inches", "mean"),
            hb_inches=("hb_inches", "mean"),
            pitch_count=("release_speed", "size"),
        )
        .reset_index()
    )

    appearances = appearances[appearances["pitch_count"] >= MIN_FASTBALLS_PER_GAME].copy()
    appearances = appearances.sort_values(
        ["pitcher", "game_date", "game_pk"], ascending=[True, False, False]
    ).reset_index(drop=True)

    return appearances


def classify_kds(kds: float) -> str:
    if kds >= 90:
        return "CRITICAL KINETIC FAILURE"
    if kds >= 75:
        return "BREAKDOWN RISK"
    if kds >= 60:
        return "DRIFT DETECTED"
    if kds >= 40:
        return "WATCH"
    return "STABLE"


def infer_diagnosis(
    release_z_delta,
    release_x_abs_z,
    extension_delta,
    velo_delta,
    ivb_delta,
    spin_delta,
) -> str:
    rz = safe_float(release_z_delta) or 0.0
    rxz = abs(safe_float(release_x_abs_z) or 0.0)
    ext = safe_float(extension_delta) or 0.0
    velo = safe_float(velo_delta) or 0.0
    ivb = safe_float(ivb_delta) or 0.0
    spin = safe_float(spin_delta) or 0.0

    if rz <= -0.08 and ext <= -0.12 and velo <= -0.8:
        return "LOWER-BODY FATIGUE / DELIVERY COLLAPSE"
    if abs(velo) < 0.5 and ivb <= -1.0 and spin <= -80:
        return "GRIP / FINGER-FEEL DISRUPTION"
    if rxz >= 2.0 and abs(velo) < 0.8:
        return "ARM-SLOT INSTABILITY"
    if ext <= -0.15 and velo >= 0.0:
        return "EFFORT SPIKE / SHORTENING RELEASE"
    if ivb >= 1.0 and velo >= 0.3:
        return "EMERGING SHAPE / POWER GAIN"
    if velo <= -0.8 or ivb <= -1.0 or ext <= -0.12:
        return "EARLY KINETIC DECAY"
    return "NO ACUTE DRIFT"



def confidence_score(kde_score: float, recent_n: int, baseline_n: int, pitch_count: int) -> float:
    sample_bonus = min(20.0, recent_n * 3.0 + baseline_n * 1.25)
    pitch_bonus = min(15.0, pitch_count / 8.0)
    return round(clamp(kde_score * 0.65 + sample_bonus + pitch_bonus, 0, 100), 1)



def classify_kde_band(score: float) -> str:
    if score >= 90:
        return "EXTREME MOVEMENT ANOMALY"
    if score >= 75:
        return "MAJOR KINETIC SHIFT"
    if score >= 60:
        return "ACTIONABLE DRIFT"
    if score >= 45:
        return "EARLY MOVEMENT SIGNAL"
    return "STABLE BASELINE"


def build_drift_trace(recent: pd.DataFrame, baseline: pd.DataFrame) -> list[dict]:
    base_speed = mean_or_none(baseline["release_speed"])
    base_ext = mean_or_none(baseline["release_extension"])
    base_ivb = mean_or_none(baseline["ivb_inches"])
    base_rel_x = mean_or_none(baseline["release_pos_x"])
    base_rel_z = mean_or_none(baseline["release_pos_z"])

    trace = []
    ordered = recent.sort_values(["game_date", "game_pk"], ascending=True)

    for _, row in ordered.iterrows():
        speed = safe_float(row.get("release_speed"))
        ext = safe_float(row.get("release_extension"))
        ivb = safe_float(row.get("ivb_inches"))
        rel_x = safe_float(row.get("release_pos_x"))
        rel_z = safe_float(row.get("release_pos_z"))

        speed_delta = (speed - base_speed) if speed is not None and base_speed is not None else 0.0
        ext_delta = (ext - base_ext) if ext is not None and base_ext is not None else 0.0
        ivb_delta = (ivb - base_ivb) if ivb is not None and base_ivb is not None else 0.0
        rel_x_delta = (rel_x - base_rel_x) if rel_x is not None and base_rel_x is not None else 0.0
        rel_z_delta = (rel_z - base_rel_z) if rel_z is not None and base_rel_z is not None else 0.0

        drift_index = (
            abs(speed_delta) * 16
            + abs(ext_delta) * 65
            + abs(ivb_delta) * 8
            + abs(rel_x_delta) * 55
            + abs(rel_z_delta) * 55
        )

        trace.append({
            "game_date": str(pd.to_datetime(row.get("game_date")).date()),
            "drift_index": round(clamp(drift_index, 0, 100), 1),
            "speed_delta": round(speed_delta, 2),
            "extension_delta": round(ext_delta, 2),
            "ivb_delta": round(ivb_delta, 2),
            "release_x_delta": round(rel_x_delta, 2),
            "release_z_delta": round(rel_z_delta, 2),
        })

    return trace



def classify_trace_behavior(trace: list[dict]) -> str:
    if len(trace) < 3:
        return "INSUFFICIENT TRACE"

    values = [safe_float(point.get("drift_index")) or 0.0 for point in trace[-3:]]
    first, middle, last = values

    total_move = last - first
    leg_one = middle - first
    leg_two = last - middle

    if abs(total_move) <= 8 and max(values) - min(values) <= 15:
        return "HOLDING"

    if leg_one > 12 and leg_two < -12:
        return "CHOPPY / REVERSAL"

    if leg_one < -12 and leg_two > 12:
        return "CHOPPY / REBOUND"

    if total_move >= 18:
        return "ACCELERATING"

    if total_move <= -18:
        return "COOLING"

    return "MIXED"


def movement_state_label(movement_state: str) -> str:
    if movement_state == "BREAKDOWN_RISK":
        return "DECAY / FATIGUE RISK"
    if movement_state == "EMERGENCE":
        return "EMERGENCE / POWER GAIN"
    if movement_state == "INSTABILITY":
        return "MECHANICAL INSTABILITY"
    return movement_state


def align_diagnosis_to_state(
    movement_state: str,
    diagnosis: str,
    risk: float,
    emergence: float,
    instability: float,
    trace_behavior: str,
) -> str:
    if movement_state == "BREAKDOWN_RISK":
        if trace_behavior == "ACCELERATING":
            return "ACCELERATING DECAY / FATIGUE RISK"
        if trace_behavior == "COOLING":
            return "DECAY EVENT COOLING / STILL RISK-FAMILY"
        if instability >= 60:
            return "DECAY RISK WITH MECHANICAL VOLATILITY"
        return diagnosis if diagnosis != "NO ACUTE DRIFT" else "EARLY KINETIC DECAY"

    if movement_state == "EMERGENCE":
        if trace_behavior == "ACCELERATING":
            return "EMERGENCE SIGNAL STRENGTHENING"
        if trace_behavior == "COOLING":
            return "EMERGENCE SIGNAL COOLING / VERIFY HOLD"
        if instability >= 55:
            return "EMERGENCE WITH RELEASE VOLATILITY"
        return diagnosis if diagnosis != "NO ACUTE DRIFT" else "EMERGING SHAPE / POWER GAIN"

    if movement_state == "INSTABILITY":
        if risk >= 55:
            return "MECHANICAL VOLATILITY WITH DECAY PRESSURE"
        if emergence >= 45:
            return "MECHANICAL VOLATILITY WITH EMERGENCE PRESSURE"
        if trace_behavior in {"CHOPPY / REVERSAL", "CHOPPY / REBOUND"}:
            return "CHOPPY RELEASE / SHAPE VOLATILITY"
        if trace_behavior == "HOLDING":
            return "MECHANICAL INSTABILITY HOLDING"
        return "MECHANICAL VOLATILITY // VERIFY RELEASE WINDOW"

    return diagnosis

def classify_operator_action(movement_state: str, risk: float, emergence: float, instability: float) -> str:
    if movement_state == "BREAKDOWN_RISK":
        if risk >= 75:
            return "EXIT / BENCH IMMEDIATELY"
        if risk >= 60:
            return "REDUCE EXPOSURE"
        return "MONITOR NEXT OUTING"

    if movement_state == "EMERGENCE":
        if emergence >= 75:
            return "INITIATE TRACKING"
        if emergence >= 60:
            return "TRACK / STASH"
        return "MONITOR FOR CONFIRMATION"

    if movement_state == "INSTABILITY":
        if instability >= 80:
            return "VOLATILITY WATCH"
        if instability >= 65:
            return "HOLD / VERIFY MECHANICS"
        return "MONITOR MECHANICS"

    return "NO ACTION"


def operator_directive(operator_action: str, movement_state: str, trace_behavior: str) -> str:
    if operator_action == "EXIT / BENCH IMMEDIATELY":
        return "High-risk kinetic decay profile. Treat as roster-damage prevention until the next outing proves stabilization."
    if operator_action == "REDUCE EXPOSURE":
        return "Risk family is active but not terminal. Trim exposure, verify velocity and release window before trusting volume."
    if operator_action == "MONITOR NEXT OUTING":
        return "Early decay read. No panic move yet, but next appearance becomes the confirmation checkpoint."

    if operator_action == "INITIATE TRACKING":
        return "Power or shape gain has cleared the signal threshold. Move asset onto Tracking Radar before the market reprices."
    if operator_action == "TRACK / STASH":
        return "Emergence profile is building. Track aggressively, but wait for one more confirmation layer before full deployment."
    if operator_action == "MONITOR FOR CONFIRMATION":
        return "Emergence pressure exists, but the edge is not clean enough yet. Keep under surveillance."

    if operator_action == "VOLATILITY WATCH":
        return "Mechanical instability is the dominant read. Do not blindly add or exit; verify release-window repeatability."
    if operator_action == "HOLD / VERIFY MECHANICS":
        return "Volatility is actionable, but direction is unresolved. Hold current exposure and demand mechanical confirmation."
    if operator_action == "MONITOR MECHANICS":
        return "Low-grade instability. Watch release slot, IVB, and velocity shape before changing exposure."

    if movement_state == "STABLE":
        return "No acute command. Maintain baseline surveillance."

    return f"Operator should verify {movement_state} with trace behavior: {trace_behavior}."

def build_kinetic_signals(appearances: pd.DataFrame) -> list[dict]:
    if appearances.empty:
        return []

    signals: list[dict] = []

    for pitcher_id, group in appearances.groupby("pitcher", dropna=True):
        g = group.sort_values(["game_date", "game_pk"], ascending=[False, False]).copy()
        if len(g) < MIN_TOTAL_APPEARANCES:
            continue

        recent = g.head(RECENT_APPEARANCES)
        baseline = g.iloc[RECENT_APPEARANCES : RECENT_APPEARANCES + BASELINE_APPEARANCES]
        if len(recent) < RECENT_APPEARANCES or len(baseline) < 3:
            continue

        current = {
            "release_speed": mean_or_none(recent["release_speed"]),
            "release_spin_rate": mean_or_none(recent["release_spin_rate"]),
            "release_extension": mean_or_none(recent["release_extension"]),
            "release_pos_x": mean_or_none(recent["release_pos_x"]),
            "release_pos_z": mean_or_none(recent["release_pos_z"]),
            "ivb_inches": mean_or_none(recent["ivb_inches"]),
            "hb_inches": mean_or_none(recent["hb_inches"]),
        }

        base_mean = {col: mean_or_none(baseline[col]) for col in current}
        base_std = {col: std_or_none(baseline[col]) for col in current}

        z = {col: z_score(current[col], base_mean[col], base_std[col]) for col in current}
        delta = {
            col: (
                safe_float(current[col]) - safe_float(base_mean[col])
                if safe_float(current[col]) is not None and safe_float(base_mean[col]) is not None
                else None
            )
            for col in current
        }

        def cz(name: str) -> float:
            return clamp(abs(z.get(name) or 0.0), 0.0, 3.0)

        release_instability = max(cz("release_pos_x"), cz("release_pos_z"))
        velocity_instability = cz("release_speed")
        extension_instability = cz("release_extension")
        spin_instability = cz("release_spin_rate")
        ivb_instability = cz("ivb_inches")
        hb_instability = cz("hb_inches")

        velo_delta = safe_float(delta.get("release_speed")) or 0.0
        ext_delta = safe_float(delta.get("release_extension")) or 0.0
        ivb_delta = safe_float(delta.get("ivb_inches")) or 0.0
        spin_delta = safe_float(delta.get("release_spin_rate")) or 0.0

        # Directional scores:
        # KRS = Kinetic Risk Score: bearish decay/fatigue profile.
        # KES = Kinetic Emergence Score: bullish shape/power gain profile.
        # KIS = Kinetic Instability Score: non-directional volatility/mechanical movement.
        #
        # Important: KRS should not treat all movement as bad. Release drift matters,
        # but risk should be driven mainly by negative velocity, extension, IVB, and spin deltas.
        risk_raw = (
            max(0.0, -velo_delta) * 18
            + max(0.0, -ext_delta * 8.0) * 12
            + max(0.0, -ivb_delta) * 10
            + max(0.0, -spin_delta / 140.0) * 6
            + max(0.0, release_instability - 1.35) * 8
            + max(0.0, velocity_instability - 1.35) * 5
            + max(0.0, extension_instability - 1.35) * 4
        )

        emergence_raw = (
            max(0.0, velo_delta) * 16
            + max(0.0, ivb_delta) * 16
            + max(0.0, spin_delta / 140.0) * 6
            + max(0.0, ext_delta * 8.0) * 5
            + max(0.0, 1.35 - release_instability) * 4
        )

        instability_raw = (
            max(0.0, release_instability - 1.15) * 14
            + max(0.0, ivb_instability - 1.15) * 10
            + max(0.0, hb_instability - 1.15) * 7
            + max(0.0, velocity_instability - 1.15) * 7
            + max(0.0, extension_instability - 1.15) * 7
            + max(0.0, spin_instability - 1.15) * 5
        )

        kinetic_risk_score = round(clamp(risk_raw, 0, 100), 1)
        kinetic_emergence_score = round(clamp(emergence_raw, 0, 100), 1)
        kinetic_instability_score = round(clamp(instability_raw, 0, 100), 1)

        kde_score = max(
            kinetic_risk_score,
            kinetic_emergence_score,
            kinetic_instability_score,
        )
        diagnosis = infer_diagnosis(
            delta.get("release_pos_z"),
            z.get("release_pos_x"),
            delta.get("release_extension"),
            delta.get("release_speed"),
            delta.get("ivb_inches"),
            delta.get("release_spin_rate"),
        )

        movement_state = (
            "BREAKDOWN_RISK"
            if kinetic_risk_score >= max(kinetic_emergence_score, kinetic_instability_score) and kinetic_risk_score >= 40
            else "EMERGENCE"
            if kinetic_emergence_score >= max(kinetic_risk_score, kinetic_instability_score) and kinetic_emergence_score >= 40
            else "INSTABILITY"
            if kinetic_instability_score >= 40
            else "STABLE"
        )

        if movement_state == "INSTABILITY" and diagnosis == "NO ACUTE DRIFT":
            diagnosis = "KINETIC VARIABILITY SPIKE"

        if movement_state == "INSTABILITY" and kinetic_risk_score >= 50 and "DECAY" in diagnosis:
            diagnosis = "UNSTABLE DECAY PROFILE"

        if movement_state == "INSTABILITY" and kinetic_emergence_score >= 50 and "EMERGING" in diagnosis:
            diagnosis = "UNSTABLE EMERGENCE PROFILE"

        drift_trace = build_drift_trace(recent, baseline)
        trace_behavior = classify_trace_behavior(drift_trace)
        diagnosis_detail = align_diagnosis_to_state(
            movement_state,
            diagnosis,
            kinetic_risk_score,
            kinetic_emergence_score,
            kinetic_instability_score,
            trace_behavior,
        )

        total_recent_fastballs = int(pd.to_numeric(recent["pitch_count"], errors="coerce").fillna(0).sum())
        confidence = confidence_score(kde_score, len(recent), len(baseline), total_recent_fastballs)
        operator_action = classify_operator_action(
            movement_state,
            kinetic_risk_score,
            kinetic_emergence_score,
            kinetic_instability_score,
        )
        operator_note = operator_directive(operator_action, movement_state, trace_behavior)

        if kde_score < 45 and diagnosis == "NO ACUTE DRIFT":
            continue

        latest = g.iloc[0]
        signals.append(
            {
                "player_id": int(pitcher_id),
                "player_name": str(latest.get("player_name") or ""),
                "team": str(latest.get("team") or ""),
                "latest_game_date": str(pd.to_datetime(latest.get("game_date")).date()),
                "kde_score": kde_score,
                "kde_band": classify_kde_band(kde_score),
                "kinetic_risk_score": kinetic_risk_score,
                "kinetic_emergence_score": kinetic_emergence_score,
                "kinetic_instability_score": kinetic_instability_score,
                "classification": classify_kds(kinetic_risk_score),
                "movement_state": movement_state,
                "movement_state_label": movement_state_label(movement_state),
                "trace_behavior": trace_behavior,
                "confidence_score": confidence,
                "operator_action": operator_action,
                "operator_note": operator_note,
                "drift_trace": drift_trace,
                "diagnosis": diagnosis_detail,
                "raw_diagnosis": diagnosis,
                "recent_appearances": int(len(recent)),
                "baseline_appearances": int(len(baseline)),
                "metrics": {
                    "release_speed_delta": safe_float(delta.get("release_speed")),
                    "release_speed_z": safe_float(z.get("release_speed")),
                    "release_extension_delta": safe_float(delta.get("release_extension")),
                    "release_extension_z": safe_float(z.get("release_extension")),
                    "release_pos_x_delta": safe_float(delta.get("release_pos_x")),
                    "release_pos_x_z": safe_float(z.get("release_pos_x")),
                    "release_pos_z_delta": safe_float(delta.get("release_pos_z")),
                    "release_pos_z_z": safe_float(z.get("release_pos_z")),
                    "ivb_delta": safe_float(delta.get("ivb_inches")),
                    "ivb_z": safe_float(z.get("ivb_inches")),
                    "hb_delta": safe_float(delta.get("hb_inches")),
                    "hb_z": safe_float(z.get("hb_inches")),
                    "spin_delta": safe_float(delta.get("release_spin_rate")),
                    "spin_z": safe_float(z.get("release_spin_rate")),
                },
                "trend": {
                    "release_speed": [safe_float(v) for v in g.head(8)["release_speed"].tolist()],
                    "release_extension": [safe_float(v) for v in g.head(8)["release_extension"].tolist()],
                    "release_pos_z": [safe_float(v) for v in g.head(8)["release_pos_z"].tolist()],
                    "ivb_inches": [safe_float(v) for v in g.head(8)["ivb_inches"].tolist()],
                },
            }
        )

    signals.sort(key=lambda row: row["kde_score"], reverse=True)
    return signals[:MAX_SIGNALS]


def write_json(signals: list[dict], start_date: str, end_date: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "module": "KINETIC_DRIFT_ENGINE_V1",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "lookback_days": LOOKBACK_DAYS,
        },
        "method": {
            "recent_appearances": RECENT_APPEARANCES,
            "baseline_appearances": BASELINE_APPEARANCES,
            "min_fastballs_per_game": MIN_FASTBALLS_PER_GAME,
            "score": "KDE v1 generates KRS/KES/KIS: kinetic risk, emergence, and instability scores versus each pitcher’s own recent baseline.",
        },
        "signals": signals,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(signals)} kinetic drift signals -> {OUT_PATH}")

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    HTML_PATH.write_text(
        template.render(
            signals=signals,
            payload=payload,
            shell_styles=SHELL_STYLES_PATH.read_text(encoding="utf-8"),
            shell_nav=Template(SHELL_NAV_PATH.read_text(encoding="utf-8")).render(active_nav="kinetic_drift"),
        ),
        encoding="utf-8",
    )
    print(f"Wrote kinetic drift preview -> {HTML_PATH}")


def write_kde_status(
    *,
    build_started_at: datetime,
    build_finished_at: datetime,
    build_success: bool,
    used_fallback: bool,
    degraded: bool,
    signal_count: int,
    errors: list[str] | None = None,
    notes: list[str] | None = None,
) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "report_id": "kinetic_drift",
        "state": "fresh" if build_success and not degraded else "degraded",
        "build_success": build_success,
        "used_fallback": used_fallback,
        "degraded": degraded,
        "threshold_minutes": 1440,
        "build_started_at": build_started_at.isoformat(),
        "build_finished_at": build_finished_at.isoformat(),
        "source_updated_at": None,
        "source_age_minutes": None,
        "section_counts": {
            "kinetic_drift_signals": signal_count,
        },
        "errors": errors or [],
        "notes": notes or [],
        "generated_at": build_finished_at.isoformat(),
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote KDE status -> {STATUS_PATH}")


def save_kde_snapshot() -> None:
    if OUT_PATH.exists():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT_PATH, SNAPSHOT_PATH)
        print(f"Saved KDE snapshot -> {SNAPSHOT_PATH}")


def restore_kde_snapshot() -> bool:
    if SNAPSHOT_PATH.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SNAPSHOT_PATH, OUT_PATH)
        print(f"Restored KDE snapshot -> {OUT_PATH}")
        return True
    return False


def main() -> None:
    build_started_at = datetime.utcnow()
    end = datetime.utcnow().date()
    start = end - timedelta(days=LOOKBACK_DAYS)

    try:
        raw = fetch_statcast_window(str(start), str(end))
        pitches = load_fastball_pitches(raw)
        appearances = build_pitcher_appearances(pitches)
        signals = build_kinetic_signals(appearances)

        if not signals:
            raise RuntimeError("KDE build produced zero signals.")

        write_json(signals, str(start), str(end))
        save_kde_snapshot()

        build_finished_at = datetime.utcnow()
        write_kde_status(
            build_started_at=build_started_at,
            build_finished_at=build_finished_at,
            build_success=True,
            used_fallback=False,
            degraded=False,
            signal_count=len(signals),
            notes=[f"Fresh KDE build completed for Statcast window {start} to {end}."],
        )

    except Exception as error:
        print(f"KDE build failed: {error}")

        used_fallback = restore_kde_snapshot()
        fallback_count = 0

        if used_fallback and OUT_PATH.exists():
            try:
                payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
                fallback_count = len(payload.get("signals", []))
                write_json(payload.get("signals", []), str(start), str(end))
            except Exception as fallback_error:
                print(f"KDE fallback render failed: {fallback_error}")

        build_finished_at = datetime.utcnow()
        write_kde_status(
            build_started_at=build_started_at,
            build_finished_at=build_finished_at,
            build_success=used_fallback,
            used_fallback=used_fallback,
            degraded=True,
            signal_count=fallback_count,
            errors=[str(error)],
            notes=["KDE used the latest available snapshot." if used_fallback else "KDE failed and no snapshot was available."],
        )

        if not used_fallback:
            raise


if __name__ == "__main__":
    main()
