#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pybaseball import statcast
from jinja2 import Template

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DIST_DIR = REPO_ROOT / "dist"
OUT_DIR = DIST_DIR / "admin"
OUT_PATH = OUT_DIR / "kinetic_drift_signals.json"
HTML_DIR = OUT_DIR / "kinetic-drift"
HTML_PATH = HTML_DIR / "index.html"
TEMPLATE_PATH = BASE_DIR / "templates" / "admin" / "kinetic_drift.html"

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
    if score >= 85:
        return "EXTREME MOVEMENT ANOMALY"
    if score >= 70:
        return "MAJOR KINETIC SHIFT"
    if score >= 55:
        return "ACTIONABLE DRIFT"
    if score >= 40:
        return "EARLY MOVEMENT SIGNAL"
    return "STABLE BASELINE"


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
        return "MONITOR"

    return "NO ACTION"

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
            max(0.0, -velo_delta) * 16
            + max(0.0, -ext_delta * 8.0) * 14
            + max(0.0, -ivb_delta) * 12
            + max(0.0, -spin_delta / 120.0) * 8
            + max(0.0, release_instability - 1.25) * 10
            + max(0.0, velocity_instability - 1.25) * 5
            + max(0.0, extension_instability - 1.25) * 5
        )

        emergence_raw = (
            max(0.0, velo_delta) * 18
            + max(0.0, ivb_delta) * 18
            + max(0.0, spin_delta / 120.0) * 8
            + max(0.0, ext_delta * 8.0) * 6
            + max(0.0, 1.5 - release_instability) * 6
        )

        instability_raw = (
            max(0.0, release_instability - 1.0) * 18
            + max(0.0, ivb_instability - 1.0) * 12
            + max(0.0, hb_instability - 1.0) * 8
            + max(0.0, velocity_instability - 1.0) * 8
            + max(0.0, extension_instability - 1.0) * 8
            + max(0.0, spin_instability - 1.0) * 6
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

        total_recent_fastballs = int(pd.to_numeric(recent["pitch_count"], errors="coerce").fillna(0).sum())
        confidence = confidence_score(kde_score, len(recent), len(baseline), total_recent_fastballs)
        operator_action = classify_operator_action(
            movement_state,
            kinetic_risk_score,
            kinetic_emergence_score,
            kinetic_instability_score,
        )

        if kde_score < 40 and diagnosis == "NO ACUTE DRIFT":
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
                "confidence_score": confidence,
                "operator_action": operator_action,
                "diagnosis": diagnosis,
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
        template.render(signals=signals, payload=payload),
        encoding="utf-8",
    )
    print(f"Wrote kinetic drift preview -> {HTML_PATH}")


def main() -> None:
    end = datetime.utcnow().date()
    start = end - timedelta(days=LOOKBACK_DAYS)

    raw = fetch_statcast_window(str(start), str(end))
    pitches = load_fastball_pitches(raw)
    appearances = build_pitcher_appearances(pitches)
    signals = build_kinetic_signals(appearances)
    write_json(signals, str(start), str(end))


if __name__ == "__main__":
    main()
