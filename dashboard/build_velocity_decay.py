#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Template
from pybaseball import statcast

from dashboard.lib.report_status import build_report_status
from dashboard.lib.metric_display import metric_title, safe_metric_value

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DIST_DIR = REPO_ROOT / "dist"
RISK_DIR = DIST_DIR / "velocity-decay-monitor"
STATUS_DIR = DIST_DIR / "status"
VELOCITY_DECAY_STATUS_PATH = STATUS_DIR / "velocity-decay.json"
TEMPLATES_DIR = BASE_DIR / "templates"

# VELOCITY_DECAY_SHARED_NAV_PATH_V1
# Desktop uses compact pro nav; mobile keeps the existing shell menu contract.
DESKTOP_NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav_v2.html").read_text(encoding="utf-8")
MOBILE_NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")

LOOKBACK_DAYS = 30
MIN_FASTBALLS_PER_GAME = 8
MIN_APPEARANCES = 3
MAX_CARDS = 12

FASTBALL_TYPES = {"FF", "FA", "SI", "FC"}

VELOCITY_DECAY_STATUS_MODE = "statcast_velocity_decay_dynamic_v1"
VELOCITY_DECAY_PIPELINE_LAYERS = [
    "pybaseball_statcast_pitch_level_feed",
    "fastball_velocity_window",
    "recent_baseline_delta_scoring",
    "extension_decay_detection",
    "perceived_velocity_proxy",
    "risk_alert_classification",
    "tracking_identity_payloads",
    "no_static_player_seed_fallback",
]


def safe_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def format_signed(value: float | None, suffix: str) -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}{suffix}"


def build_headshot_url(player_id: int | str | None) -> str:
    if not player_id:
        return ""
    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"w_180,q_100/v1/people/{int(player_id)}/headshot/67/current"
    )


def values_to_polyline(values: list[float]) -> str:
    if not values:
        return "0,17 100,17"

    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return "0,17 100,17"

    lo = min(vals)
    hi = max(vals)
    span = hi - lo if hi != lo else 1.0

    points: list[str] = []
    n = len(vals)
    for idx, val in enumerate(vals):
        x = 100 * idx / (n - 1)
        y = 30 - ((val - lo) / span) * 20
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def format_decay_slope_label(value: float | None) -> str:
    if value is None:
        return "--"
    if value <= -0.45:
        return "CLIFF"
    if value <= -0.20:
        return "DECAY"
    if value < 0.05:
        return "STABLE"
    return "REBOUND"


def classify_risk_alert(
    velo_delta: float | None,
    extension_delta: float | None,
    perceived_velo_delta: float | None,
    decay_slope: float | None,
) -> tuple[str, str, str]:
    vd = velo_delta if velo_delta is not None else 0.0
    ed = extension_delta if extension_delta is not None else 0.0
    pvd = perceived_velo_delta if perceived_velo_delta is not None else 0.0
    ds = decay_slope if decay_slope is not None else 0.0

    if vd <= -1.5 and ed > -0.10:
        return (
            "SEVERE",
            "VELOCITY CLIFF",
            "Significant raw power loss detected. Arm fatigue likely. Immediate benching recommended.",
        )

    if vd <= -1.0 and ed <= -0.10:
        return (
            "CRITICAL",
            "MECHANICAL DECAY",
            "Velocity loss paired with shortening extension. Strong indicator of latent injury or mechanical breakdown risk.",
        )

    if vd >= 0.2 and ed <= -0.10:
        return (
            "WARNING",
            "EFFORT SPIKE",
            "Pitcher appears to be muscling the ball to maintain speed. High volatility and command risk next outing.",
        )

    if abs(vd) < 0.5 and pvd <= -0.5:
        return (
            "CAUTION",
            "DECEPTIVE DROP",
            "Raw speed is holding, but release characteristics have made the pitch play slower to hitters.",
        )

    if ds <= -0.45:
        return (
            "WARNING",
            "DECAY TREND",
            "Three-start velocity slope is deteriorating quickly. Performance regression risk is rising even without a full collapse signal yet.",
        )

    if vd <= -0.7 or ed <= -0.15 or pvd <= -0.8:
        return (
            "CAUTION",
            "EARLY DECAY",
            "Underlying velocity or release traits are slipping versus recent baseline. This profile needs closer monitoring.",
        )

    return (
        "STABLE",
        "NO ACUTE DECAY",
        "No major decay trigger detected in the current window.",
    )


def fetch_statcast_window(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Fetching Statcast from {start_date} to {end_date}...")
    print("This is a large query, it may take a moment to complete")
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

    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")
    df["game_pk"] = pd.to_numeric(df.get("game_pk"), errors="coerce").astype("Int64")
    df["pitcher"] = pd.to_numeric(df.get("pitcher"), errors="coerce").astype("Int64")
    df["release_speed"] = pd.to_numeric(df.get("release_speed"), errors="coerce")
    df["release_extension"] = pd.to_numeric(df.get("release_extension"), errors="coerce")
    df["player_name"] = df.get("player_name").fillna("").astype(str)
    df["home_team"] = df.get("home_team").fillna("").astype(str)
    df["away_team"] = df.get("away_team").fillna("").astype(str)

    df = df.dropna(
        subset=["pitcher", "game_pk", "game_date", "release_speed", "release_extension"]
    ).copy()

    if df.empty:
        return pd.DataFrame()

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
            release_extension=("release_extension", "mean"),
            pitch_count=("release_speed", "size"),
        )
        .reset_index()
    )

    appearances = appearances[appearances["pitch_count"] >= MIN_FASTBALLS_PER_GAME].copy()
    appearances = appearances.sort_values(
        ["pitcher", "game_date", "game_pk"], ascending=[True, False, False]
    ).reset_index(drop=True)

    if appearances.empty:
        return pd.DataFrame()

    appearances["effective_velo_proxy"] = (
        appearances["release_speed"] + (appearances["release_extension"] - 6.0) * 1.5
    )
    return appearances


def compute_decay_slope(values: list[float]) -> float | None:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if len(vals) < 3:
        return None

    recent3 = vals[:3]
    y = list(reversed(recent3))
    x = [0.0, 1.0, 2.0]

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x)
    if den == 0:
        return None
    return num / den


def compute_risk_score(
    velo_delta: float | None,
    extension_delta: float | None,
    decay_slope: float | None,
    perceived_velo_delta: float | None,
) -> float:
    vd = max(-(velo_delta or 0.0), 0.0)
    ed = max(-(extension_delta or 0.0), 0.0)
    ds = max(-(decay_slope or 0.0), 0.0)
    pvd = max(-(perceived_velo_delta or 0.0), 0.0)

    raw_score = (
        50.0
        + 14.0 * vd
        + 24.0 * ed
        + 12.0 * ds
        + 4.0 * pvd
    )
    return round(clamp(raw_score, 45.0, 96.5), 1)


def build_velocity_decay_rows(appearances: pd.DataFrame) -> list[dict]:
    if appearances.empty:
        return []

    rows: list[dict] = []

    for pitcher_id, pitcher_apps in appearances.groupby("pitcher", sort=False):
        pitcher_apps = pitcher_apps.sort_values(
            ["game_date", "game_pk"], ascending=[False, False]
        ).reset_index(drop=True)

        if len(pitcher_apps) < MIN_APPEARANCES:
            continue

        recent = pitcher_apps.iloc[0]
        baseline_pool = pitcher_apps.iloc[1:5].copy()
        if baseline_pool.empty:
            continue

        recent_velo = safe_float(recent.get("release_speed"))
        recent_ext = safe_float(recent.get("release_extension"))
        recent_eff = safe_float(recent.get("effective_velo_proxy"))

        baseline_velo = safe_float(baseline_pool["release_speed"].mean())
        baseline_ext = safe_float(baseline_pool["release_extension"].mean())
        baseline_eff = safe_float(baseline_pool["effective_velo_proxy"].mean())

        if recent_velo is None or recent_ext is None or recent_eff is None:
            continue
        if baseline_velo is None or baseline_ext is None or baseline_eff is None:
            continue

        velo_delta = recent_velo - baseline_velo
        extension_delta = recent_ext - baseline_ext
        perceived_velo_delta = recent_eff - baseline_eff

        trend_values = [
            safe_float(v)
            for v in pitcher_apps["release_speed"].head(5).tolist()
        ]
        trend_values = [v for v in trend_values if v is not None]
        decay_slope = compute_decay_slope(trend_values)

        risk_tier, primary_alert, analysis = classify_risk_alert(
            velo_delta,
            extension_delta,
            perceived_velo_delta,
            decay_slope,
        )

        risk_score = compute_risk_score(
            velo_delta,
            extension_delta,
            decay_slope,
            perceived_velo_delta,
        )

        team = str(recent.get("team") or "").strip() or "TEAM"
        player_name = str(recent.get("player_name") or "").strip() or "Unknown Pitcher"

        rows.append(
            {
                "player_id": int(pitcher_id),
                "player_name": player_name,
                "team": team,
                "risk_score": risk_score,
                "risk_tier": risk_tier,
                "primary_alert": primary_alert,
                "analysis": analysis,
                "velo_delta": round(velo_delta, 2),
                "extension_delta": round(extension_delta, 2),
                "perceived_velo_delta": round(perceived_velo_delta, 2),
                "decay_slope": None if decay_slope is None else round(decay_slope, 2),
                "trend_values": trend_values,
                "sample_count": len(trend_values),
                "profile_url": f"/scout/{int(pitcher_id)}/",
                "headshot_url": build_headshot_url(int(pitcher_id)),
            }
        )

    rows = sorted(
        rows,
        key=lambda r: (
            safe_float(r.get("risk_score")) or 0.0,
            abs(safe_float(r.get("velo_delta")) or 0.0),
        ),
        reverse=True,
    )
    return rows[:MAX_CARDS]



def classify_sparkline_semantic_class(
    velo_delta: float | None,
    extension_delta: float | None,
    perceived_velo_delta: float | None,
    decay_slope: float | None,
    primary_alert: str,
    risk_tier: str,
) -> str:
    """Map Velocity Decay model state to a semantic sparkline color class."""
    vd = velo_delta if velo_delta is not None else 0.0
    ed = extension_delta if extension_delta is not None else 0.0
    pvd = perceived_velo_delta if perceived_velo_delta is not None else 0.0
    ds = decay_slope if decay_slope is not None else 0.0
    alert = (primary_alert or "").upper()
    tier = (risk_tier or "").upper()

    if "DECEPTIVE" in alert:
        return "velocity-sparkline-deceptive"

    if vd <= -2.5 or ds <= -0.45 or "CLIFF" in alert or "EXTREME" in tier:
        return "velocity-sparkline-cliff"

    if vd <= -1.5 or ds <= -0.25 or "DECAY" in alert or "HIGH" in tier:
        return "velocity-sparkline-decay"

    if vd <= -0.8 or ds < 0.0 or ed <= -0.10 or pvd <= -0.8 or "EARLY" in alert or "TREND" in alert:
        return "velocity-sparkline-watch"

    if vd >= 0.5 and ds >= 0.05:
        return "velocity-sparkline-rebound"

    return "velocity-sparkline-stable"

def format_velocity_decay_cards(rows: list[dict]) -> list[dict]:
    cards = []

    for row in rows:
        velo_delta = safe_float(row.get("velo_delta"))
        extension_delta = safe_float(row.get("extension_delta"))
        perceived_delta = safe_float(row.get("perceived_velo_delta"))
        decay_slope = safe_float(row.get("decay_slope"))
        risk_score = safe_float(row.get("risk_score"))

        if risk_score is not None and risk_score >= 85:
            score_class = "score-critical"
        elif risk_score is not None and risk_score >= 75:
            score_class = "score-warning"
        else:
            score_class = "score-neutral"

        primary_alert = str(row.get("primary_alert") or "").upper()
        risk_tier = str(row.get("risk_tier") or "").upper()

        if primary_alert == "MECHANICAL DECAY":
            alert_class = "alert-mechanical"
        elif primary_alert == "VELOCITY CLIFF":
            alert_class = "alert-cliff"
        elif primary_alert == "EFFORT SPIKE":
            alert_class = "alert-effort"
        elif primary_alert == "DECEPTIVE DROP":
            alert_class = "alert-deceptive"
        elif primary_alert in {"EARLY DECAY", "DECAY TREND"}:
            alert_class = "alert-warning"
        else:
            alert_class = "alert-neutral"

        if risk_tier in {"CRITICAL", "SEVERE"}:
            sparkline_class = "sparkline-critical"
        elif risk_tier in {"WARNING", "CAUTION"}:
            sparkline_class = "sparkline-warning"
        else:
            sparkline_class = "sparkline-neutral"

        velo_delta_class = ""
        extension_delta_class = ""

        if velo_delta is not None and velo_delta <= -1.5:
            velo_delta_class = "metric-critical"
        elif velo_delta is not None and velo_delta <= -1.0:
            velo_delta_class = "metric-warning"

        if extension_delta is not None and extension_delta <= -0.20:
            extension_delta_class = "metric-critical"
        elif extension_delta is not None and extension_delta <= -0.10:
            extension_delta_class = "metric-warning"

        cards.append(
            {
                **row,
                "risk_score_label": safe_metric_value("--" if risk_score is None else f"{risk_score:.1f}"),
                "risk_score_title": metric_title("Risk Score"),
                "score_class": score_class,
                "alert_class": alert_class,
                "sparkline_class": sparkline_class,
            "sparkline_semantic_class": classify_sparkline_semantic_class(
                velo_delta,
                extension_delta,
                (locals().get("perceived_velo_delta") or locals().get("perceived_delta") or locals().get("perceived_velocity_delta") or 0.0),
                decay_slope,
                primary_alert,
                risk_tier,
            ),
                "velo_delta_class": velo_delta_class,
                "extension_delta_class": extension_delta_class,
                "velo_delta_label": safe_metric_value(format_signed(velo_delta, " mph")),
                "velo_delta_title": metric_title("Velo Delta"),
                "extension_delta_label": safe_metric_value(format_signed(extension_delta, " ft")),
                "extension_delta_title": metric_title("Ext Delta"),
                "perceived_delta_label": safe_metric_value(format_signed(perceived_delta, " mph")),
                "perceived_delta_title": metric_title("Perceived Delta"),
                "decay_slope_label": safe_metric_value(format_decay_slope_label(decay_slope)),
                "decay_slope_title": metric_title("Decay Slope"),
                "trend_points": values_to_polyline(row.get("trend_values") or []),
                "sample_note": f'{len(row.get("trend_values") or [])} start trend',
            }
        )

    return cards


HTML_TEMPLATE = Template(
    r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals — Velocity Decay Monitor</title>
  <style>
    :root {
      --bg: #080808;
      --card-radial: radial-gradient(circle at top left, #1a1a1a 0%, #080808 100%);
      --border: #2d2d2d;
      --text: #f0f0f0;
      --muted: #71717a;
      --soft: #a1a1aa;
      --tiny: #8a8a93;
      --red: #ef4444;
      --red-soft: #f87171;
      --blue: #6aa6ff;
      --lime-hot: #b6ff00;
      --shadow: 0 14px 34px rgba(0, 0, 0, 0.34);
      --radius: 18px;
      --mono: "JetBrains Mono","Roboto Mono","SFMono-Regular",Menlo,Consolas,monospace;
      --sans: Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--sans); }
    body {
      background:
        radial-gradient(circle at top left, rgba(106,166,255,0.06), transparent 24%),
        radial-gradient(circle at top right, rgba(239,68,68,0.06), transparent 20%),
        linear-gradient(180deg, #101010 0%, #080808 34%, #050505 100%);
      line-height: 1.35;
    }
    .app { width: min(1180px, calc(100% - 24px)); margin: 0 auto; padding: 18px 0 34px; }

    .info-trigger {
      height: 34px;
      border-radius: 999px;
      border: 1px solid rgba(255,49,49,0.22);
      background: rgba(255,255,255,0.05);
      color: var(--text);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0 12px;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      cursor: pointer;
      box-shadow: 0 0 10px rgba(255,49,49,0.08);
    }

    .guide-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.52);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.22s ease;
      z-index: 80;
    }

    .guide-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }

    .guide-drawer {
      position: fixed;
      top: 0;
      right: 0;
      width: min(560px,100vw);
      height: 100vh;
      background: linear-gradient(180deg, #101010 0%, #080808 100%);
      border-left: 1px solid rgba(255,255,255,0.08);
      box-shadow: -12px 0 40px rgba(0,0,0,0.42);
      transform: translateX(100%);
      transition: transform 0.24s ease;
      z-index: 90;
      display: flex;
      flex-direction: column;
    }

    .guide-drawer.open {
      transform: translateX(0);
    }

    .guide-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 18px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.02);
    }

    .guide-kicker {
      font-size: 10px;
      line-height: 1;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #FF3131;
      font-weight: 800;
      margin-bottom: 8px;
    }

    .guide-title {
      margin: 0;
      font-size: 20px;
      line-height: 1.05;
      letter-spacing: -0.03em;
      text-transform: uppercase;
      font-weight: 900;
      color: var(--text);
    }

    .guide-close {
      width: 34px;
      height: 34px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.03);
      color: var(--text);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      cursor: pointer;
    }

    .guide-body {
      overflow-y: auto;
      padding: 18px;
      display: grid;
      gap: 18px;
    }

    .guide-section {
      border: 1px solid rgba(255,255,255,0.05);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
      padding: 14px;
    }

    .guide-section-title {
      margin: 0 0 12px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: #FFBF00;
      font-weight: 800;
      font-family: var(--mono);
    }

    .guide-copy {
      font-size: 13px;
      line-height: 1.55;
      color: var(--soft);
    }

    .hero-card, .section, .player-card {
      background: var(--card-radial);
      border: 0.5px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }
    .hero-card::before, .section::before, .player-card::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      border-radius: inherit;
      padding: 0.5px;
      background: linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.01));
      -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.55;
    }
    .hero-card { padding: 18px; margin-bottom: 16px; }
    .eyebrow { font-size: 10px; line-height: 1; letter-spacing: 0.18em; text-transform: uppercase; color: #ff6b6b; font-weight: 800; margin-bottom: 10px; }
    .hero-title { margin: 0 0 10px; font-size: clamp(32px, 6vw, 56px); line-height: 0.95; letter-spacing: -0.04em; font-weight: 900; text-transform: uppercase; }
    .hero-copy { margin: 0; max-width: 760px; color: var(--soft); font-size: 14px; }

    .section-head {
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      padding: 16px 16px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      background: linear-gradient(180deg, rgba(255,255,255,0.022), rgba(255,255,255,0.008));
    }
    .section-kicker { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 800; }
    .section-title { margin: 0; font-size: 18px; font-weight: 800; letter-spacing: -0.02em; text-transform: uppercase; }
    .section-badge {
      font-family: var(--mono); font-size: 11px; color: var(--soft); border: 1px solid rgba(255,255,255,0.08);
      border-radius: 999px; padding: 7px 10px; background: rgba(255,255,255,0.02);
    }

    .cards { display: grid; gap: 10px; padding: 10px; }
    .player-card { padding: 14px; }
    .player-card.top-risk { box-shadow: var(--shadow), 0 0 10px rgba(255,49,49,0.05); }
    .player-top { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px; align-items: start; margin-bottom: 10px; }
    .player-head { display: flex; align-items: start; gap: 12px; min-width: 0; }
    .avatar {
      width: 42px; height: 42px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.10);
      display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.03);
      color: var(--text); font-size: 13px; font-weight: 800;
      flex: 0 0 auto;
    }
    .player-ident { min-width: 0; }
    .rankline, .score-label, .metric-label, .sparkline-label {
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 800;
    }
    .score-label {
      font-size: 9px;
      letter-spacing: 0.08em;
      color: var(--tiny);
      margin-bottom: 4px;
    }
    .player-name {
      margin: 0 0 4px; font-size: clamp(20px, 2.4vw, 30px); line-height: 1; letter-spacing: -0.03em;
      font-weight: 800; text-transform: uppercase;
    }
    .signal-line {
      font-size: 10px; color: var(--soft); font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.10em;
    }
    .scorebox {
      text-align: right;
      min-width: 148px;
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 10px;
    }
    .score-value { font-family: var(--sans); font-size: 36px; line-height: 0.92; font-weight: 900; font-style: italic; letter-spacing: -0.05em; color: var(--red-soft); }

    .analysis-stack {
      border-radius: 12px;
      background: rgba(255,255,255,0.015);
      border: 1px solid rgba(255,255,255,0.035);
      padding: 10px 10px 12px;
      margin-bottom: 12px;
    }

    .diagnosis-wrap {
      margin: 0;
    }

    .diagnosis-label {
      margin-bottom: 4px;
      color: var(--tiny);
      font-size: 9px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-weight: 800;
    }

    .diagnosis-banner {
      display: flex;
      align-items: center;
      width: 100%;
      min-height: 38px;
      padding: 0 14px;
      border-radius: 12px 12px 0 0;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      pointer-events: none;
      user-select: none;
      box-shadow: none;
    }

    .diagnosis-banner.alert-cliff {
      border: 1px solid rgba(255,49,49,0.22);
      background: rgba(255,49,49,0.14);
      color: #ffd2d2;
    }

    .diagnosis-banner.alert-mechanical {
      border: 1px solid rgba(255,191,0,0.20);
      background: rgba(255,191,0,0.10);
      color: #ffe39a;
    }

    .diagnosis-banner.alert-effort {
      border: 1px solid rgba(106,166,255,0.18);
      background: rgba(106,166,255,0.08);
      color: #cfe0ff;
    }

    .diagnosis-banner.alert-deceptive {
      border: 1px solid rgba(168,139,250,0.18);
      background: rgba(168,139,250,0.08);
      color: #ddd6fe;
    }

    .diagnosis-banner.alert-warning {
      border: 1px solid rgba(255,214,102,0.16);
      background: rgba(255,214,102,0.07);
      color: #e8d9a8;
    }

    .diagnosis-banner.alert-neutral {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      color: var(--soft);
    }

    .action-row {
      display: flex;
      justify-content: flex-end;
      width: 100%;
      align-items: flex-start;
    }
    .player-card .score-value.score-critical {
      color: #FF3131;
      text-shadow: 0 0 14px rgba(255,49,49,0.30);
    }
    .player-card .score-value.score-warning {
      color: #FFBF00;
      text-shadow: 0 0 10px rgba(255,191,0,0.18);
    }
    .player-card .score-value.score-neutral {
      color: var(--red-soft);
    }

    .alert-pill {
      display: inline-flex; align-items: center; border-radius: 999px; padding: 7px 10px;
      border: 1px solid rgba(239,68,68,0.22); background: rgba(239,68,68,0.08);
      color: #fca5a5; font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em;
      text-transform: uppercase; margin: 10px 0 12px;
    }
    .alert-pill.alert-cliff {
  border-color: rgba(255,49,49,0.55);
  background: rgba(255,49,49,0.18);
  color: #ffe1e1;
  box-shadow: 0 0 12px rgba(255,49,49,0.14);
}

.alert-pill.alert-mechanical {
  border-color: rgba(255,191,0,0.42);
  background: rgba(255,191,0,0.12);
  color: #ffe39a;
  box-shadow: 0 0 10px rgba(255,191,0,0.08);
}

.alert-pill.alert-effort {
  border-color: rgba(106,166,255,0.34);
  background: rgba(106,166,255,0.10);
  color: #cfe0ff;
}

.alert-pill.alert-deceptive {
  border-color: rgba(168,139,250,0.34);
  background: rgba(168,139,250,0.10);
  color: #ddd6fe;
}

.alert-pill.alert-warning {
  border-color: rgba(255,214,102,0.24);
  background: rgba(255,214,102,0.07);
  color: #e8d9a8;
}

.alert-pill.alert-neutral {
  border-color: rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  color: var(--soft);
}
    }
    .js-add-to-roster {
      min-height: 40px;
      padding: 0 16px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      border: 1px solid rgba(255,49,49,0.22);
      background: rgba(18,18,18,0.92);
      color: #ffffff;
      box-shadow: none;
    }
    .js-add-to-roster:hover {
      border-color: rgba(255,49,49,0.34);
      background: rgba(24,24,24,0.98);
      color: white;
      box-shadow: 0 0 12px rgba(255,49,49,0.10);
      transform: translateY(-1px);
    }

    .sparkline-wrap {
      border-radius: 0 0 12px 12px;
      background: rgba(255,255,255,0.015);
      padding: 10px 10px 6px;
      margin-top: 0;
      margin-bottom: 14px;
      border-top: 0;
    }
    .sparkline-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
    .sparkline-note { font-family: var(--mono); font-size: 10px; color: var(--tiny); text-transform: uppercase; letter-spacing: 0.08em; }
    svg.sparkline { display: block; width: 100%; height: 34px; }
    .sparkline-path { stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; fill: none; }
    .sparkline-path.sparkline-critical { stroke: #FF3131; }
    .sparkline-path.sparkline-warning { stroke: #FFBF00; }
    .sparkline-path.sparkline-neutral { stroke: #ef4444; }

    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 16px; align-items: stretch; }
    .metric {
      border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 10px 10px 9px;
      background: rgba(255,255,255,0.02); min-width: 0;
    }
    .metric.metric-warning {
      border-color: rgba(255,191,0,0.18);
      background: rgba(255,191,0,0.04);
    }
    .metric.metric-critical {
      border-color: rgba(255,49,49,0.24);
      background: rgba(255,49,49,0.05);
    }
    .metric-value {
      font-family: var(--mono); font-size: 15px; line-height: 1.1; color: var(--text);
      font-weight: 700; word-break: break-word; font-variant-numeric: tabular-nums;
    }
    .why { font-size: 12px; line-height: 1.65; color: var(--soft); margin-top: 6px; }


    .topbar {
      position: sticky;
      top: 0;
      z-index: 60;
      background: rgba(8, 8, 8, 0.90);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .topbar-inner,
    .app {
      width: min(1180px, calc(100% - 24px));
      margin: 0 auto;
    }

    .topbar-inner {
      min-height: 62px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 0;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .brand-mark {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--lime-hot);
      box-shadow:
        0 0 0 2px rgba(182,255,0,0.06),
        0 0 12px rgba(182,255,0,0.55),
        0 0 24px rgba(182,255,0,0.22);
      animation: dsPulse 1.8s ease-in-out infinite;
      flex: 0 0 auto;
    }

    @keyframes dsPulse {
      0%, 100% {
        transform: scale(1);
        box-shadow:
          0 0 0 2px rgba(182,255,0,0.06),
          0 0 12px rgba(182,255,0,0.45),
          0 0 22px rgba(182,255,0,0.18);
      }
      50% {
        transform: scale(1.18);
        box-shadow:
          0 0 0 4px rgba(182,255,0,0.10),
          0 0 18px rgba(182,255,0,0.90),
          0 0 34px rgba(182,255,0,0.34);
      }
    }

    .brand-text {
      min-width: 0;
    }

    .brand-kicker {
      font-size: 10px;
      line-height: 1;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-weight: 800;
      margin-bottom: 4px;
    }

    .brand-white { color: #f5f7fb; }
    .brand-blue  { color: #4f8cff; }

    .brand-title {
      font-size: 16px;
      line-height: 1.05;
      letter-spacing: -0.02em;
      font-weight: 800;
      color: var(--text);
    }

    .livebox {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 4px;
      flex: 0 0 auto;
    }

    .live-label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--soft);
    }

    .live-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--lime-hot);
      box-shadow:
        0 0 0 2px rgba(182,255,0,0.06),
        0 0 12px rgba(182,255,0,0.55),
        0 0 24px rgba(182,255,0,0.22);
      animation: dsPulse 1.8s ease-in-out infinite;
    }

    .live-time {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--tiny);
      letter-spacing: 0.08em;
    }


    {{ shell_styles | safe }}

    @media (max-width: 700px) {
      .js-add-to-roster {
        min-height: 32px !important;
        padding: 0 12px !important;
        border-radius: 10px !important;
        width: auto !important;
        min-width: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        font-size: 10px !important;
        letter-spacing: 0.04em !important;
      }


      .app { width: min(100%, calc(100% - 16px)); }
      .player-name { font-size: 26px; line-height: 0.96; letter-spacing: -0.04em; font-weight: 800; }
      .signal-line { font-size: 11px; line-height: 1.3; letter-spacing: 0.10em; }
      .score-label { font-size: 9px; letter-spacing: 0.08em; }
      .score-value { font-family: var(--sans); font-size: 44px; line-height: 0.88; font-weight: 900; font-style: italic; letter-spacing: -0.06em; }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }





/* VELOCITY_DECAY_SPARKLINE_SEMANTIC_COLOR_V1
   Semantic sparkline color system:
   - line color is driven by model state, not decoration
   - rebound/stable/watch/decay/cliff/deceptive states map to explicit classes
   - preserves existing sparkline geometry and card metrics
*/
.sparkline-path.velocity-sparkline-rebound {
  stroke: rgba(182,255,0,0.98) !important;
  filter: drop-shadow(0 0 10px rgba(182,255,0,0.26)) !important;
}

.sparkline-path.velocity-sparkline-stable {
  stroke: rgba(125,255,211,0.96) !important;
  filter: drop-shadow(0 0 10px rgba(125,255,211,0.18)) !important;
}

.sparkline-path.velocity-sparkline-watch {
  stroke: rgba(250,204,21,0.98) !important;
  filter: drop-shadow(0 0 10px rgba(250,204,21,0.20)) !important;
}

.sparkline-path.velocity-sparkline-decay {
  stroke: rgba(249,115,22,0.98) !important;
  filter: drop-shadow(0 0 10px rgba(249,115,22,0.24)) !important;
}

.sparkline-path.velocity-sparkline-cliff {
  stroke: rgba(255,61,81,0.98) !important;
  filter: drop-shadow(0 0 12px rgba(255,61,81,0.34)) !important;
}

.sparkline-path.velocity-sparkline-deceptive {
  stroke: rgba(56,189,248,0.98) !important;
  filter: drop-shadow(0 0 10px rgba(56,189,248,0.26)) !important;
}

/* VELOCITY_DECAY_DESKTOP_STICKY_GUIDE_TWO_COLUMN_V1
   Desktop-only refinement:
   - makes Field Guide persist as a sticky/floating command pill
   - restores a two-column analytical card grid on desktop
   - keeps mobile menu, mobile drawer, tracking, and status contract untouched
*/
@media screen and (min-width: 981px) {
  body .app {
    position: relative !important;
  }

  body .hero-card {
    position: relative !important;
    padding-right: clamp(190px, 17vw, 260px) !important;
  }

  body .info-trigger,
  body button.info-trigger {
    position: fixed !important;
    top: 118px !important;
    right: max(26px, calc((100vw - 1180px) / 2 + 18px)) !important;
    z-index: 420 !important;

    min-width: 178px !important;
    height: 46px !important;
    padding: 0 22px !important;

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;

    border-radius: 999px !important;
    border: 1px solid rgba(182,255,0,0.42) !important;
    background:
      radial-gradient(circle at 18% 50%, rgba(182,255,0,0.14), transparent 36%),
      rgba(7,10,10,0.92) !important;
    color: rgba(255,255,255,0.94) !important;
    box-shadow:
      0 0 0 1px rgba(255,255,255,0.04),
      0 16px 34px rgba(0,0,0,0.50),
      0 0 24px rgba(182,255,0,0.14) !important;

    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 900 !important;
    letter-spacing: 0.16em !important;
    line-height: 1 !important;
    text-transform: uppercase !important;

    transform: none !important;
    opacity: 1 !important;
    visibility: visible !important;
    pointer-events: auto !important;
  }

  body .info-trigger:hover,
  body button.info-trigger:hover {
    border-color: rgba(182,255,0,0.70) !important;
    color: #ffffff !important;
  }

  body .cards {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 18px !important;
    align-items: stretch !important;
  }

  body .player-card {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
  }

  body .analysis-stack {
    flex: 1 1 auto !important;
  }

  body .why {
    margin-top: auto !important;
  }

  body .guide-drawer {
    width: min(860px, calc(100vw - 64px)) !important;
  }

  body .guide-body {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 14px !important;
  }

  body .guide-section:first-child {
    grid-column: 1 / -1 !important;
  }
}

@media screen and (max-width: 980px) {
  body .cards {
    grid-template-columns: 1fr !important;
  }

  body .guide-body {
    display: block !important;
  }
}


      /* VELOCITY_DECAY_MOBILE_REFACTOR_V1 */
      @media screen and (max-width: 760px) {
        body {
          overflow-x: hidden !important;
        }

        .search-strip {
          display: none !important;
        }

        .app {
          width: min(100%, calc(100% - 28px)) !important;
          padding: 18px 0 52px !important;
        }

        .hero-card {
          padding: 18px 16px !important;
          margin: 16px 0 14px !important;
          border-radius: 22px !important;
        }

        .hero-card > div {
          display: grid !important;
          grid-template-columns: 1fr !important;
          gap: 12px !important;
        }

        .eyebrow {
          font-size: 9px !important;
          line-height: 1 !important;
          letter-spacing: 0.16em !important;
          margin-bottom: 10px !important;
        }

        .hero-title {
          font-size: 34px !important;
          line-height: 1.04 !important;
          letter-spacing: -0.045em !important;
          font-weight: 700 !important;
          text-transform: none !important;
          margin: 0 0 10px !important;
        }

        .hero-copy {
          font-size: 13px !important;
          line-height: 1.48 !important;
          color: rgba(226,232,240,0.72) !important;
          margin: 0 !important;
        }

        .info-trigger {
          width: 100% !important;
          min-height: 38px !important;
          justify-content: center !important;
          text-align: center !important;
          font-size: 9px !important;
          letter-spacing: 0.14em !important;
          border-radius: 999px !important;
        }

        .section {
          border-radius: 22px !important;
        }

        .section-head {
          flex-direction: column !important;
          align-items: stretch !important;
          gap: 10px !important;
          padding: 14px !important;
        }

        .section-kicker {
          font-size: 9px !important;
          line-height: 1 !important;
          letter-spacing: 0.16em !important;
        }

        .section-title {
          font-size: 21px !important;
          line-height: 1 !important;
          letter-spacing: -0.03em !important;
          font-weight: 760 !important;
        }

        .section-badge {
          width: 100% !important;
          min-height: 34px !important;
          display: flex !important;
          align-items: center !important;
          justify-content: center !important;
          text-align: center !important;
          font-size: 9px !important;
          line-height: 1.25 !important;
          letter-spacing: 0.12em !important;
        }

        .cards {
          gap: 14px !important;
          padding: 10px !important;
        }

        .player-card {
          padding: 12px !important;
          border-radius: 20px !important;
        }

        .player-top {
          display: grid !important;
          grid-template-columns: 1fr !important;
          gap: 10px !important;
          margin-bottom: 10px !important;
        }

        .player-head {
          display: block !important;
        }

        .avatar {
          display: none !important;
        }

        .rankline {
          font-size: 8px !important;
          line-height: 1.15 !important;
          letter-spacing: 0.14em !important;
        }

        .player-name {
          font-size: 23px !important;
          line-height: 0.98 !important;
          letter-spacing: -0.04em !important;
          font-weight: 760 !important;
          text-transform: uppercase !important;
        }

        .signal-line {
          font-size: 9px !important;
          line-height: 1.35 !important;
          letter-spacing: 0.12em !important;
        }

        .scorebox {
          align-items: flex-start !important;
          text-align: left !important;
          min-width: 0 !important;
          padding-top: 8px !important;
          border-top: 1px solid rgba(255,255,255,0.06) !important;
        }

        .score-value {
          font-size: 38px !important;
          line-height: 0.86 !important;
          letter-spacing: -0.055em !important;
        }

        .action-row {
          justify-content: stretch !important;
        }

        .alert-pill {
          width: 100% !important;
          min-height: 38px !important;
          justify-content: center !important;
          text-align: center !important;
          font-size: 9px !important;
          letter-spacing: 0.14em !important;
        }

        .analysis-stack {
          padding: 9px !important;
          border-radius: 16px !important;
        }

        .diagnosis-label,
        .sparkline-label,
        .metric-label {
          font-size: 8px !important;
          line-height: 1 !important;
          letter-spacing: 0.14em !important;
        }

        .diagnosis-banner {
          min-height: 36px !important;
          padding: 8px 10px !important;
          justify-content: center !important;
          text-align: center !important;
          font-size: 9px !important;
          line-height: 1.25 !important;
          letter-spacing: 0.12em !important;
          border-radius: 14px !important;
        }

        .sparkline-head {
          flex-direction: column !important;
          align-items: flex-start !important;
          gap: 4px !important;
        }

        .metric-grid {
          grid-template-columns: 1fr !important;
          gap: 8px !important;
        }

        .metric {
          padding: 8px 9px !important;
        }

        .metric-value {
          font-size: 16px !important;
          line-height: 18px !important;
        }

        .why {
          font-size: 12px !important;
          line-height: 1.45 !important;
        }
      }

  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="brand-mark"></div>
        <div class="brand-text">
          <div class="brand-kicker"><span class="brand-white">DIAMOND</span><span class="brand-blue">SIGNALS</span></div>
          <div class="brand-title">Risk Terminal // Velocity Decay</div>
        </div>
      </div>
      <div class="livebox">
        <div class="live-label"><span class="live-dot"></span>LIVE</div>
        <div class="live-time">{{ generated_at }}</div>
      </div>
    </div>
  </div>
  <div class="guide-overlay" id="guideOverlay" onclick="closeGuide()"></div>

  <aside class="guide-drawer" id="guideDrawer" aria-hidden="true">
    <div class="guide-head">
      <div>
        <div class="guide-kicker">Risk Terminal</div>
        <h2 class="guide-title">Velocity Decay Field Guide</h2>
      </div>
      <button class="guide-close" type="button" onclick="closeGuide()" aria-label="Close field guide">×</button>
    </div>

    <div class="guide-body">
      <section class="guide-section">
        <h3 class="guide-section-title">What this page does</h3>
        <div class="guide-copy">
          The Velocity Decay Monitor is a diagnostic alert feed, not a leaderboard. It looks for velocity loss, extension deterioration, perceived velocity changes, and short-horizon trend decay that can signal fatigue, mechanical drift, or pre-collapse performance risk.
        </div>
      </section>

      <section class="guide-section">
        <h3 class="guide-section-title">How to read the cards</h3>
        <div class="guide-copy">
          Velo Delta shows the raw speed change versus recent baseline. Ext Delta shows whether release extension is shrinking or holding. Perceived Delta estimates how the pitch is playing to hitters, not just what the radar gun says. Decay Slope summarizes the last three-start trend into a quick diagnostic state such as CLIFF, DECAY, STABLE, or REBOUND.
        </div>
      </section>

      <section class="guide-section">
        <h3 class="guide-section-title">How to use the alerts</h3>
        <div class="guide-copy">
          VELOCITY CLIFF and MECHANICAL DECAY are your strongest landmine signals. DECEPTIVE DROP catches cases where raw speed may look fine but the ball is effectively arriving worse. EARLY DECAY and DECAY TREND are softer warnings that deserve monitoring before the market reacts.
        </div>
      </section>
    </div>
  </aside>

  {{ nav_html | safe }}
  {{ search_html | safe }}

  <div class="app">
    <section class="hero-card">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap;">
        <div>
          <div class="eyebrow">Risk Terminal</div>
          <h1 class="hero-title">Velocity Decay Monitor</h1>
          <p class="hero-copy">
            Diagnostic alert feed for latent fatigue, mechanical decay, and pre-collapse velocity signals. This board tracks divergence from each pitcher’s recent truth, not just raw surface outcomes.
          </p>
        </div>
        <button class="info-trigger" type="button" onclick="openGuide()">Field Guide</button>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <div class="section-kicker">Landmine Detection</div>
          <h2 class="section-title">Top Risk Signals</h2>
        </div>
        <div class="section-badge">{{ cards|length }} flagged arms</div>
      </div>

      <div class="cards">
        {% for row in cards %}
        <article
          class="player-card js-player-card{% if loop.index <= 3 %} top-risk{% endif %}"
          data-player-id="{{ row.player_id }}"
          data-player-name="{{ row.player_name }}"
          data-player-type="pitcher"
          data-player-team="{{ row.team }}"
          data-profile-url="/scout/{{ row.player_id }}/"
        >
          <div class="player-top">
            <div class="player-head">
              <div class="avatar">{{ row.player_name[:2]|upper }}</div>
              <div class="player-ident">
                <div class="rankline">#{{ loop.index }} Risk Trigger</div>
                <h3 class="player-name">{{ row.player_name }}</h3>
                <div class="signal-line">{{ row.team }} // {{ row.risk_tier }} // {{ row.sample_note }}</div>
              </div>
            </div>

            <div class="scorebox">
              <div>
                <div class="score-label" title="{{ row.risk_score_title }}" aria-label="{{ row.risk_score_title }}">Risk Score</div>
                <div class="score-value {{ row.score_class }}">{{ row.risk_score_label }}</div>
              </div>
              <div class="action-row">
                <button
                  type="button"
                  class="alert-pill js-add-to-roster"
                  data-default-label="INITIATE TRACKING"
                  data-player-id="{{ row.player_id }}"
                  data-player-name="{{ row.player_name }}"
                  data-player-type="pitcher"
                  data-player-team="{{ row.team }}"
                  data-profile-url="/scout/{{ row.player_id }}/"
                  data-source-tag="VELOCITY_DECAY"
                >INITIATE TRACKING</button>
              </div>
            </div>
          </div>

          <div class="analysis-stack">
            <div class="diagnosis-wrap">
              <div class="diagnosis-label">Diagnosis</div>
              <div class="diagnosis-banner {{ row.alert_class }}">{{ row.primary_alert }}</div>
            </div>

            <div
              class="sparkline-wrap"
              data-trend-source="recent_fastball_velocity"
              data-trend-window="recent_appearances"
              data-trend-values="{{ row.trend_values|join(',') }}"
              data-velo-delta="{{ row.velo_delta_label }}"
              data-sparkline-color-source="velocity_decay_model_state"
              data-sparkline-color-class="{{ row.sparkline_semantic_class }}"
            >
              <div class="sparkline-head">
                <div class="sparkline-label">Recent FB Velo</div>
                <div class="sparkline-note">Last 5 appearances</div>
              </div>
              <svg class="sparkline" viewBox="0 0 100 34" preserveAspectRatio="none" role="img" aria-label="{{ row.player_name }} recent fastball velocity trend">
                <title>{{ row.player_name }} recent fastball velocity trend. Velo delta {{ row.velo_delta_label }}.</title>
                <polyline class="sparkline-path {{ row.sparkline_class }} {{ row.sparkline_semantic_class }}" points="{{ row.trend_points }}" />
              </svg>
            </div>

            <div class="metric-grid">
            <div class="metric {{ row.velo_delta_class }}">
              <div class="metric-label" title="{{ row.velo_delta_title }}" aria-label="{{ row.velo_delta_title }}">Velo Delta</div>
              <div class="metric-value">{{ row.velo_delta_label }}</div>
            </div>
            <div class="metric {{ row.extension_delta_class }}">
              <div class="metric-label" title="{{ row.extension_delta_title }}" aria-label="{{ row.extension_delta_title }}">Ext Delta</div>
              <div class="metric-value">{{ row.extension_delta_label }}</div>
            </div>
            <div class="metric">
              <div class="metric-label" title="{{ row.perceived_delta_title }}" aria-label="{{ row.perceived_delta_title }}">Perceived Delta</div>
              <div class="metric-value">{{ row.perceived_delta_label }}</div>
            </div>
            <div class="metric">
              <div class="metric-label" title="{{ row.decay_slope_title }}" aria-label="{{ row.decay_slope_title }}">Decay Slope</div>
              <div class="metric-value">{{ row.decay_slope_label }}</div>
            </div>
          </div>

          </div>

          <div class="why">{{ row.analysis }}</div>
        </article>
        {% endfor %}
      </div>
    </section>

    {{ footer_html | safe }}
  </div>

    <script src="/player-search.js"></script>
    <script src="/player-card-actions.js"></script>
    <script>
    function openGuide() {
      document.getElementById("guideOverlay").classList.add("open");
      document.getElementById("guideDrawer").classList.add("open");
      document.getElementById("guideDrawer").setAttribute("aria-hidden", "false");
    }

    function closeGuide() {
      document.getElementById("guideOverlay").classList.remove("open");
      document.getElementById("guideDrawer").classList.remove("open");
      document.getElementById("guideDrawer").setAttribute("aria-hidden", "true");
    }
  </script>
</body>
</html>
"""
)


def copy_static_assets() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    search_src = REPO_ROOT / "src" / "js" / "player-search.js"
    search_dest = DIST_DIR / "player-search.js"
    if search_src.exists():
        search_dest.write_text(search_src.read_text(encoding="utf-8"), encoding="utf-8")
        print("Wrote dist/player-search.js")

    actions_src = REPO_ROOT / "src" / "js" / "player-card-actions.js"
    actions_dest = DIST_DIR / "player-card-actions.js"
    if actions_src.exists():
        actions_dest.write_text(actions_src.read_text(encoding="utf-8"), encoding="utf-8")
        print("Wrote dist/player-card-actions.js")



def write_velocity_decay_status(
    *,
    build_started_at: str,
    build_finished_at: str,
    cards,
) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

    card_count = int(len(cards)) if cards is not None else 0
    degraded = card_count == 0

    risk_counts = {}
    alert_counts = {}
    for row in cards or []:
        risk = str(row.get("risk_tier") or "UNKNOWN").strip() or "UNKNOWN"
        alert = str(row.get("primary_alert") or "UNKNOWN").strip() or "UNKNOWN"
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        alert_counts[alert] = alert_counts.get(alert, 0) + 1

    status_payload = build_report_status(
        "velocity_decay",
        build_success=True,
        threshold_minutes=240,
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        source_updated_at=build_finished_at,
        section_counts={
            "velocity_decay_cards": card_count,
            "risk_groups": len(risk_counts),
        },
        degraded=degraded,
        notes=[
            f"Velocity Decay Monitor built with {card_count} active decay cards."
        ],
    )

    status_payload["mode"] = VELOCITY_DECAY_STATUS_MODE
    status_payload["pipeline_layers"] = VELOCITY_DECAY_PIPELINE_LAYERS
    status_payload["risk_counts"] = risk_counts
    status_payload["alert_counts"] = alert_counts

    VELOCITY_DECAY_STATUS_PATH.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote Velocity Decay status -> {VELOCITY_DECAY_STATUS_PATH}")


def write_velocity_decay_monitor() -> None:
    build_started_at = datetime.now(timezone.utc).isoformat()
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    RISK_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    raw = fetch_statcast_window(str(start_date), str(end_date))
    pitches = load_fastball_pitches(raw)
    appearances = build_pitcher_appearances(pitches)
    rows = build_velocity_decay_rows(appearances)
    cards = format_velocity_decay_cards(rows)

    html = HTML_TEMPLATE.render(
        nav_html=(
            Template(DESKTOP_NAV_TEMPLATE).render(active_nav="velocity_decay_monitor")
            + "\n"
            + Template(MOBILE_NAV_TEMPLATE).render(active_nav="velocity_decay_monitor")
        ),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
        cards=cards,
        generated_at=datetime.now().strftime("%b %d, %Y %I:%M %p"),
    )

    (RISK_DIR / "index.html").write_text(html, encoding="utf-8")
    print("Wrote dist/velocity-decay-monitor/index.html")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "cards": cards,
    }
    (DIST_DIR / "velocity_decay_monitor.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print("Wrote dist/velocity_decay_monitor.json")

    build_finished_at = datetime.now(timezone.utc).isoformat()
    write_velocity_decay_status(
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        cards=cards,
    )

    copy_static_assets()


def main() -> None:
    write_velocity_decay_monitor()


if __name__ == "__main__":
    main()