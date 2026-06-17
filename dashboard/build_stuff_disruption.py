#!/usr/bin/env python3
from __future__ import annotations

import json
import math
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
APEX_DIR = DIST_DIR / "stuff-disruption-feed"
STATUS_DIR = DIST_DIR / "status"
STUFF_DISRUPTION_STATUS_PATH = STATUS_DIR / "stuff-disruption.json"
TEMPLATES_DIR = BASE_DIR / "templates"

NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")

LOOKBACK_DAYS = 30
MIN_FASTBALLS_PER_GAME = 8
MIN_APPEARANCES = 3
MAX_CARDS = 12
FASTBALL_TYPES = {"FF", "FA", "SI", "FC"}


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


def approx_vaa(row) -> float | None:
    try:
        y0 = 50.0
        plate_y = 17.0 / 12.0

        vy0 = float(row["vy0"])
        ay = float(row["ay"])
        vz0 = float(row["vz0"])
        az = float(row["az"])

        a = 0.5 * ay
        b = vy0
        c = y0 - plate_y

        disc = b * b - 4 * a * c
        if disc < 0:
            return None

        if abs(a) < 1e-9:
            if abs(b) < 1e-9:
                return None
            t = -c / b
            if t <= 0:
                return None
        else:
            t1 = (-b - math.sqrt(disc)) / (2 * a)
            t2 = (-b + math.sqrt(disc)) / (2 * a)
            ts = [t for t in (t1, t2) if t > 0]
            if not ts:
                return None
            t = min(ts)

        vy_plate = vy0 + ay * t
        vz_plate = vz0 + az * t
        if vy_plate == 0:
            return None

        return math.degrees(math.atan(vz_plate / abs(vy_plate)))
    except Exception:
        return None


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

    numeric_cols = [
        "release_speed",
        "release_extension",
        "release_spin_rate",
        "spin_axis",
        "pfx_x",
        "pfx_z",
        "vx0",
        "vy0",
        "vz0",
        "ax",
        "ay",
        "az",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
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
    df["vaa"] = df.apply(approx_vaa, axis=1)

    df = df.dropna(
        subset=[
            "pitcher",
            "game_pk",
            "game_date",
            "release_speed",
            "release_spin_rate",
            "ivb_inches",
            "hb_inches",
        ]
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
            release_spin_rate=("release_spin_rate", "mean"),
            ivb_raw=("ivb_inches", "mean"),
            hb_raw=("hb_inches", "mean"),
            vaa=("vaa", "mean"),
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

    appearances["active_spin_proxy"] = (
        (appearances["ivb_raw"].abs() * 0.7 + appearances["hb_raw"].abs() * 0.3)
        / appearances["release_spin_rate"].replace(0, pd.NA)
        * 100.0
    )

    return appearances


def classify_apex_alert(
    ivb_delta: float | None,
    vaa_delta: float | None,
    movement_delta: float | None,
    active_spin_delta: float | None,
) -> tuple[str, str, str]:
    ivb = ivb_delta if ivb_delta is not None else 0.0
    vaa = vaa_delta if vaa_delta is not None else 0.0
    move = movement_delta if movement_delta is not None else 0.0
    spin = active_spin_delta if active_spin_delta is not None else 0.0

    if ivb >= 1.5 and vaa <= -0.3:
        return (
            "S-TIER",
            "APEX SHAPE SHIFT",
            "Elite ballistic profile detected. Ride and approach angle are both moving in the right direction, creating a high-end breakout structure.",
        )

    if ivb >= 1.0 and spin >= 0.15:
        return (
            "A-TIER",
            "ACTIVE SPIN GAIN",
            "Movement quality appears to be improving through more efficient spin conversion, not just raw effort. Breakout conditions are building.",
        )

    if move >= 3.0:
        return (
            "B-TIER",
            "SWEEP DISRUPTION",
            "Horizontal movement profile is jumping versus prior baseline. This shape change can quickly alter whiff behavior and contact quality.",
        )

    if vaa <= -0.3:
        return (
            "A-TIER",
            "DECEPTION PEAK",
            "Approach angle is flattening into a stronger deception band. This can make the fastball play more explosively even before results fully move.",
        )

    return (
        "WATCH",
        "SHAPE MONITOR",
        "Shape profile is moving, but not yet in a fully qualified disruption tier. Keep this arm on the radar for another appearance.",
    )


def compute_disruption_score(
    ivb_delta: float | None,
    vaa_delta: float | None,
    movement_delta: float | None,
    active_spin_delta: float | None,
) -> float:
    ivb = clamp(max(ivb_delta or 0.0, 0.0), 0.0, 2.0)
    vaa = clamp(max(-(vaa_delta or 0.0), 0.0), 0.0, 0.5)
    move = clamp(max(movement_delta or 0.0, 0.0), 0.0, 2.5)
    spin = clamp(max(active_spin_delta or 0.0, 0.0), 0.0, 2.0)

    raw_score = (
        46.0
        + 10.0 * ivb
        + 10.0 * vaa
        + 6.0 * move
        + 4.0 * spin
    )
    return round(clamp(raw_score, 45.0, 92.5), 1)


def build_apex_rows(appearances: pd.DataFrame) -> list[dict]:
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

        recent_ivb = safe_float(recent.get("ivb_raw"))
        recent_vaa = safe_float(recent.get("vaa"))
        recent_hb = safe_float(recent.get("hb_raw"))
        recent_spin_proxy = safe_float(recent.get("active_spin_proxy"))

        base_ivb = safe_float(baseline_pool["ivb_raw"].mean())
        base_vaa = safe_float(baseline_pool["vaa"].mean())
        base_hb = safe_float(baseline_pool["hb_raw"].mean())
        base_spin_proxy = safe_float(baseline_pool["active_spin_proxy"].mean())

        if recent_ivb is None or recent_hb is None or recent_spin_proxy is None:
            continue
        if base_ivb is None or base_hb is None or base_spin_proxy is None:
            continue

        ivb_delta = recent_ivb - base_ivb
        vaa_delta = None if recent_vaa is None or base_vaa is None else recent_vaa - base_vaa
        movement_delta = abs(recent_hb - base_hb)
        active_spin_delta = recent_spin_proxy - base_spin_proxy

        apex_tier, primary_alert, analysis = classify_apex_alert(
            ivb_delta,
            vaa_delta,
            movement_delta,
            active_spin_delta,
        )

        disruption_score = compute_disruption_score(
            ivb_delta,
            vaa_delta,
            movement_delta,
            active_spin_delta,
        )

        trend_values = [
            safe_float(v)
            for v in pitcher_apps["ivb_raw"].head(5).tolist()
        ]
        trend_values = [v for v in trend_values if v is not None]

        team = str(recent.get("team") or "").strip() or "TEAM"
        player_name = str(recent.get("player_name") or "").strip() or "Unknown Pitcher"

        rows.append(
            {
                "player_id": int(pitcher_id),
                "player_name": player_name,
                "team": team,
                "apex_tier": apex_tier,
                "primary_alert": primary_alert,
                "analysis": analysis,
                "disruption_score": disruption_score,
                "ivb_delta": round(ivb_delta, 2),
                "vaa_delta": None if vaa_delta is None else round(vaa_delta, 2),
                "movement_delta": round(movement_delta, 2),
                "active_spin_delta": round(active_spin_delta, 2),
                "trend_values": trend_values,
                "profile_url": f"/scout/{int(pitcher_id)}/",
                "headshot_url": build_headshot_url(int(pitcher_id)),
            }
        )

    rows = sorted(
        rows,
        key=lambda r: safe_float(r.get("disruption_score")) or 0.0,
        reverse=True,
    )
    return rows[:MAX_CARDS]


def format_apex_cards(rows: list[dict]) -> list[dict]:
    cards = []

    for row in rows:
        disruption_score = safe_float(row.get("disruption_score"))
        ivb_delta = safe_float(row.get("ivb_delta"))
        vaa_delta = safe_float(row.get("vaa_delta"))
        movement_delta = safe_float(row.get("movement_delta"))
        active_spin_delta = safe_float(row.get("active_spin_delta"))

        apex_tier = str(row.get("apex_tier") or "").upper()
        primary_alert = str(row.get("primary_alert") or "").upper()

        if apex_tier == "S-TIER":
            score_class = "score-s-tier"
        elif apex_tier == "A-TIER":
            score_class = "score-a-tier"
        elif apex_tier == "B-TIER":
            score_class = "score-b-tier"
        else:
            score_class = "score-neutral"

        if primary_alert == "APEX SHAPE SHIFT":
            alert_class = "alert-shape"
            sparkline_class = "sparkline-shape"
        elif primary_alert == "ACTIVE SPIN GAIN":
            alert_class = "alert-spin"
            sparkline_class = "sparkline-spin"
        elif primary_alert == "SWEEP DISRUPTION":
            alert_class = "alert-sweep"
            sparkline_class = "sparkline-sweep"
        elif primary_alert == "DECEPTION PEAK":
            alert_class = "alert-deception"
            sparkline_class = "sparkline-neutral"
        else:
            alert_class = "alert-neutral"
            sparkline_class = "sparkline-neutral"

        ivb_class = "metric-shape" if ivb_delta is not None and ivb_delta >= 1.0 else ""
        vaa_class = "metric-warning" if vaa_delta is not None and vaa_delta <= -0.2 else ""
        movement_class = "metric-sweep" if movement_delta is not None and movement_delta >= 1.5 else ""
        spin_class = "metric-spin" if active_spin_delta is not None and active_spin_delta >= 0.10 else ""
        card_class = "apex-top" if apex_tier == "S-TIER" else ""

        cards.append(
            {
                **row,
                "score_class": score_class,
                "alert_class": alert_class,
                "sparkline_class": sparkline_class,
                "ivb_class": ivb_class,
                "vaa_class": vaa_class,
                "movement_class": movement_class,
                "spin_class": spin_class,
                "card_class": card_class,
                "disruption_score_label": safe_metric_value("--" if disruption_score is None else f"{disruption_score:.1f}"),
                "disruption_score_title": metric_title("Disruption Score"),
                "ivb_delta_label": safe_metric_value(format_signed(ivb_delta, '"')),
                "ivb_delta_title": metric_title("IVB Delta"),
                "vaa_delta_label": safe_metric_value(format_signed(vaa_delta, "°")),
                "vaa_delta_title": metric_title("VAA Delta"),
                "movement_delta_label": safe_metric_value(format_signed(movement_delta, '"')),
                "movement_delta_title": metric_title("Move Delta"),
                "active_spin_delta_label": safe_metric_value(format_signed(active_spin_delta, "%")),
                "active_spin_delta_title": metric_title("Active Spin"),
                "sample_note": f'{len(row.get("trend_values") or [])} start trend',
                "trend_points": values_to_polyline(row.get("trend_values") or []),
            }
        )

    return cards


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


HTML_TEMPLATE = Template(
    r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals — Stuff+ Disruption Feed</title>
  <style>
    :root {
      --bg: #080808;
      --card-radial: radial-gradient(circle at top left, #1a1a1a 0%, #080808 100%);
      --border: #2d2d2d;
      --text: #f0f0f0;
      --muted: #71717a;
      --soft: #a1a1aa;
      --tiny: #8a8a93;
      --blue: #6aa6ff;
      --cyan: #22d3ee;
      --violet: #a78bfa;
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
        radial-gradient(circle at top right, rgba(167,139,250,0.06), transparent 20%),
        linear-gradient(180deg, #101010 0%, #080808 34%, #050505 100%);
      line-height: 1.35;
    }
    .app { width: min(1180px, calc(100% - 24px)); margin: 0 auto; padding: 18px 0 34px; }

    .info-trigger {
      height: 34px;
      border-radius: 999px;
      border: 1px solid rgba(34,211,238,0.22);
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
      box-shadow: 0 0 10px rgba(34,211,238,0.08);
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
      color: var(--cyan);
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
      color: var(--violet);
      font-weight: 800;
      font-family: var(--mono);
    }
    .guide-copy {
      font-size: 13px;
      line-height: 1.55;
      color: var(--soft);
    }

    .hero-card, .section, .player-card, .meta-card {
      background: var(--card-radial);
      border: 0.5px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }
    .hero-card::before, .section::before, .player-card::before, .meta-card::before {
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
    .eyebrow { font-size: 10px; line-height: 1; letter-spacing: 0.18em; text-transform: uppercase; color: var(--cyan); font-weight: 800; margin-bottom: 10px; }
    .hero-title { margin: 0 0 10px; font-size: clamp(32px, 6vw, 56px); line-height: 0.95; letter-spacing: -0.04em; font-weight: 900; text-transform: uppercase; }
    .hero-copy { margin: 0; max-width: 760px; color: var(--soft); font-size: 14px; }

    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 16px;
    }
    .meta-card { padding: 14px; }
    .meta-label {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      font-weight: 800;
      margin-bottom: 6px;
    }
    .meta-value {
      font-family: var(--mono);
      font-size: 13px;
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }

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
    .player-card.apex-top {
      border-color: rgba(34,211,238,0.22);
      box-shadow: var(--shadow), 0 0 14px rgba(34,211,238,0.08);
    }
    .player-top { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px; align-items: start; margin-bottom: 10px; }
    .player-head { display: flex; align-items: start; gap: 12px; min-width: 0; }
    .avatar {
      width: 42px; height: 42px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.10);
      display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.03);
      color: var(--text); font-size: 13px; font-weight: 800;
      flex: 0 0 auto;
    }
    .player-ident { min-width: 0; }
    .rankline, .score-label, .metric-label, .sparkline-label, .diagnosis-label {
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
    .score-value { font-family: var(--sans); font-size: 36px; line-height: 0.92; font-weight: 900; font-style: italic; letter-spacing: -0.05em; color: var(--cyan); }
    .score-value.score-s-tier {
      color: #22d3ee;
      text-shadow: 0 0 12px rgba(34,211,238,0.20);
    }
    .score-value.score-a-tier {
      color: #a78bfa;
      text-shadow: 0 0 10px rgba(167,139,250,0.18);
    }
    .score-value.score-b-tier {
      color: #b6ff00;
      text-shadow: 0 0 8px rgba(182,255,0,0.14);
    }
    .score-value.score-neutral {
      color: var(--soft);
    }

    .alert-pill {
      display: inline-flex; align-items: center; border-radius: 999px; padding: 7px 10px;
      border: 1px solid rgba(34,211,238,0.22); background: rgba(34,211,238,0.08);
      color: #c9fbff; font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em;
      text-transform: uppercase; margin: 10px 0 12px;
    }
    .alert-pill.alert-shape {
      border-color: rgba(34,211,238,0.38);
      background: rgba(34,211,238,0.12);
      color: #d8fdff;
      box-shadow: 0 0 10px rgba(34,211,238,0.10);
    }
    .alert-pill.alert-spin {
      border-color: rgba(167,139,250,0.34);
      background: rgba(167,139,250,0.10);
      color: #e5dcff;
    }
    .alert-pill.alert-sweep {
      border-color: rgba(182,255,0,0.30);
      background: rgba(182,255,0,0.08);
      color: #ecffc0;
    }
    .alert-pill.alert-deception {
      border-color: rgba(106,166,255,0.34);
      background: rgba(106,166,255,0.10);
      color: #d6e5ff;
    }
    .alert-pill.alert-neutral {
      border-color: rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.04);
      color: var(--soft);
    }

    .js-add-to-roster {
      min-height: 40px;
      padding: 0 16px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      border: 1px solid rgba(34,211,238,0.24);
      background: rgba(18,18,18,0.92);
      color: #ffffff;
      box-shadow: none;
    }
    .js-add-to-roster:hover {
      border-color: rgba(34,211,238,0.36);
      background: rgba(24,24,24,0.98);
      color: white;
      box-shadow: 0 0 12px rgba(34,211,238,0.12);
      transform: translateY(-1px);
    }

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
    .diagnosis-banner.alert-shape {
      border: 1px solid rgba(34,211,238,0.22);
      background: rgba(34,211,238,0.12);
      color: #d8fdff;
    }
    .diagnosis-banner.alert-spin {
      border: 1px solid rgba(167,139,250,0.22);
      background: rgba(167,139,250,0.10);
      color: #e5dcff;
    }
    .diagnosis-banner.alert-sweep {
      border: 1px solid rgba(182,255,0,0.20);
      background: rgba(182,255,0,0.08);
      color: #ecffc0;
    }
    .diagnosis-banner.alert-deception {
      border: 1px solid rgba(106,166,255,0.22);
      background: rgba(106,166,255,0.08);
      color: #d6e5ff;
    }
    .diagnosis-banner.alert-neutral {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      color: var(--soft);
    }

    .sparkline-wrap {
      border-radius: 0 0 12px 12px;
      background: rgba(255,255,255,0.015);
      padding: 10px 10px 6px;
      margin-top: 0;
      margin-bottom: 14px;
      border-top: 0;
    }

    .action-row {
      display: flex;
      justify-content: flex-end;
      width: 100%;
      align-items: flex-start;
    }
    .sparkline-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
    .sparkline-note { font-family: var(--mono); font-size: 10px; color: var(--tiny); text-transform: uppercase; letter-spacing: 0.08em; }
    svg.sparkline { display: block; width: 100%; height: 34px; }
    .sparkline-path { stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; fill: none; }
    .sparkline-path.sparkline-shape { stroke: #22d3ee; }
    .sparkline-path.sparkline-spin { stroke: #a78bfa; }
    .sparkline-path.sparkline-sweep { stroke: #b6ff00; }
    .sparkline-path.sparkline-neutral { stroke: #6aa6ff; }

    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 16px; align-items: stretch; }
    .metric {
      border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 10px 10px 9px;
      background: rgba(255,255,255,0.02); min-width: 0;
    }
    .metric.metric-shape {
      border-color: rgba(34,211,238,0.20);
      background: rgba(34,211,238,0.04);
    }
    .metric.metric-spin {
      border-color: rgba(167,139,250,0.24);
      background: rgba(167,139,250,0.06);
      box-shadow: 0 0 10px rgba(167,139,250,0.05);
    }
    .metric.metric-sweep {
      border-color: rgba(182,255,0,0.18);
      background: rgba(182,255,0,0.04);
    }
    .metric.metric-warning {
      border-color: rgba(106,166,255,0.18);
      background: rgba(106,166,255,0.04);
    }
    .metric-value {
      font-family: var(--mono); font-size: 15px; line-height: 1.1; color: var(--text);
      font-weight: 700; word-break: break-word; font-variant-numeric: tabular-nums;
    }
    .why { font-size: 12px; line-height: 1.65; color: var(--soft); margin-top: 6px; }

    /* STUFF_DISRUPTION_DESKTOP_FINISH_V1
       Desktop-only refinement:
       - sticky/floating Field Guide command pill
       - two-column analytical card grid on desktop
       - wider Field Guide reading drawer
       - mobile menu, mobile drawer, mobile Field Guide behavior untouched
    */
    @media screen and (min-width: 981px) {
      body .app {
        position: relative !important;
      }

      body .hero-card {
        position: relative !important;
        padding-right: clamp(190px, 17vw, 260px) !important;
      }

      body .hero-card .info-trigger,
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
        border: 1px solid rgba(34,211,238,0.42) !important;
        background:
          radial-gradient(circle at 18% 50%, rgba(34,211,238,0.16), transparent 36%),
          rgba(7,10,10,0.92) !important;
        color: rgba(255,255,255,0.94) !important;
        box-shadow:
          0 0 0 1px rgba(255,255,255,0.04),
          0 16px 34px rgba(0,0,0,0.50),
          0 0 24px rgba(34,211,238,0.14) !important;

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
        border-color: rgba(34,211,238,0.70) !important;
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
      .app { width: min(100%, calc(100% - 16px)); }
      .player-name { font-size: 26px; line-height: 0.96; letter-spacing: -0.04em; font-weight: 800; }
      .signal-line { font-size: 11px; line-height: 1.3; letter-spacing: 0.10em; }
      .score-label { font-size: 9px; letter-spacing: 0.08em; }
      .score-value { font-family: var(--sans); font-size: 44px; line-height: 0.88; font-weight: 900; font-style: italic; letter-spacing: -0.06em; }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .meta-grid { grid-template-columns: 1fr; }

      .player-top {
        grid-template-columns: auto 1fr;
        gap: 10px;
      }

      .player-head {
        display: contents;
      }

      .scorebox {
        grid-column: 1 / -1;
        min-width: 0;
        width: 100%;
        text-align: left;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
      }

      .analysis-stack {
        padding: 10px 10px 12px;
      }

      .diagnosis-banner {
        width: 100%;
        justify-content: center;
        min-height: 36px;
        font-size: 10px;
      }

      .action-row {
        width: 100%;
        justify-content: stretch;
      }

      .action-row .js-add-to-roster {
        min-height: 34px !important;
        padding: 0 12px !important;
        border-radius: 8px !important;
        border: 1px solid rgba(34,211,238,0.30) !important;
        background: rgba(18,18,18,0.94) !important;
        color: white !important;
        box-shadow: none !important;
        font-size: 8px !important;
        letter-spacing: 0.05em !important;
        white-space: nowrap !important;
        width: 100%;
        justify-content: center;
      }

      .action-row .js-add-to-roster:hover {
        border-color: rgba(34,211,238,0.42) !important;
        background: rgba(24,24,24,0.98) !important;
        color: white !important;
        box-shadow: 0 0 16px rgba(34,211,238,0.14) !important;
        transform: translateY(-1px);
      }
    }

      /* STUFF_DISRUPTION_MOBILE_REFACTOR_V1 */
      @media screen and (max-width: 760px) {
        body {
          overflow-x: hidden !important;
        }

        .search-strip {
          display: none !important;
        }

        .topbar {
          min-height: 72px !important;
        }

        .topbar-inner,
        .app {
          width: min(100%, calc(100% - 28px)) !important;
        }

        .topbar-inner {
          min-height: 62px !important;
          padding: 10px 0 !important;
          display: flex !important;
          align-items: center !important;
          justify-content: space-between !important;
          gap: 12px !important;
        }

        .brand {
          gap: 9px !important;
          min-width: 0 !important;
        }

        .brand-mark {
          width: 14px !important;
          height: 14px !important;
          min-width: 14px !important;
          min-height: 14px !important;
        }

        .brand-kicker {
          font-size: 9px !important;
          line-height: 1 !important;
          letter-spacing: 0.16em !important;
          margin-bottom: 4px !important;
        }

        .brand-title {
          max-width: 230px !important;
          font-family: var(--sans) !important;
          font-size: 15px !important;
          line-height: 1.04 !important;
          letter-spacing: -0.035em !important;
          font-weight: 750 !important;
          text-transform: none !important;
        }

        .livebox {
          min-width: 70px !important;
          text-align: right !important;
        }

        .live-label {
          font-size: 10px !important;
          gap: 6px !important;
        }

        .live-dot {
          width: 14px !important;
          height: 14px !important;
          min-width: 14px !important;
          min-height: 14px !important;
        }

        .live-time {
          font-size: 9px !important;
          line-height: 1.12 !important;
        }

        .app {
          padding: 18px 0 52px !important;
        }

        .hero-card {
          padding: 18px 16px !important;
          margin: 16px 0 14px !important;
          border-radius: 22px !important;
        }

        .hero-card > div {
          display: block !important;
        }

        .eyebrow {
          font-size: 9px !important;
          line-height: 1.2 !important;
          letter-spacing: 0.16em !important;
          margin-bottom: 12px !important;
        }

        .hero-title {
          font-family: var(--sans) !important;
          font-size: 34px !important;
          line-height: 1.02 !important;
          letter-spacing: -0.045em !important;
          font-weight: 720 !important;
          text-transform: none !important;
          margin-bottom: 10px !important;
        }

        .hero-copy {
          font-size: 13px !important;
          line-height: 1.48 !important;
          color: rgba(226,232,240,.72) !important;
        }

        .hero-card .info-trigger {
          display: none !important;
        }

        .meta-grid {
          display: none !important;
        }

        .section {
          border-radius: 22px !important;
          margin-top: 14px !important;
        }

        .section-head {
          padding: 14px 14px 12px !important;
          align-items: flex-start !important;
          flex-direction: column !important;
          gap: 8px !important;
        }

        .section-kicker {
          font-size: 9px !important;
          line-height: 1 !important;
          letter-spacing: 0.16em !important;
        }

        .section-title {
          font-size: 22px !important;
          line-height: 1 !important;
          letter-spacing: -0.035em !important;
          font-weight: 760 !important;
        }

        .section-badge {
          font-size: 9px !important;
          padding: 6px 9px !important;
        }

        .cards {
          padding: 10px !important;
          gap: 12px !important;
        }

        .player-card {
          padding: 13px !important;
          border-radius: 20px !important;
        }

        .player-top {
          display: grid !important;
          grid-template-columns: auto 1fr !important;
          gap: 10px !important;
          align-items: start !important;
          margin-bottom: 10px !important;
        }

        .player-head {
          display: contents !important;
        }

        .avatar {
          width: 34px !important;
          height: 34px !important;
          font-size: 11px !important;
        }

        .rankline,
        .score-label,
        .metric-label,
        .sparkline-label,
        .diagnosis-label {
          font-size: 8px !important;
          line-height: 1 !important;
          letter-spacing: 0.14em !important;
        }

        .player-name {
          font-size: 24px !important;
          line-height: 0.98 !important;
          letter-spacing: -0.045em !important;
          font-weight: 760 !important;
        }

        .signal-line {
          font-size: 9px !important;
          line-height: 1.28 !important;
          letter-spacing: 0.10em !important;
        }

        .scorebox {
          grid-column: 1 / -1 !important;
          min-width: 0 !important;
          width: 100% !important;
          text-align: left !important;
          align-items: flex-start !important;
          gap: 8px !important;
        }

        .score-value {
          font-size: 42px !important;
          line-height: 0.88 !important;
          letter-spacing: -0.06em !important;
        }

        .action-row {
          width: 100% !important;
          justify-content: stretch !important;
        }

        .action-row .js-add-to-roster {
          width: 100% !important;
          min-height: 36px !important;
          padding: 0 12px !important;
          border-radius: 12px !important;
          border: 1px solid rgba(34,211,238,0.34) !important;
          background: rgba(2, 6, 23, 0.58) !important;
          color: rgba(255,255,255,0.92) !important;
          box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.06),
            0 0 14px rgba(34,211,238,0.10) !important;
          font-size: 9px !important;
          letter-spacing: 0.08em !important;
          white-space: nowrap !important;
          justify-content: center !important;
        }

        .action-row .js-add-to-roster:hover {
          background: rgba(2, 6, 23, 0.76) !important;
          border-color: rgba(34,211,238,0.54) !important;
          box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.08),
            0 0 18px rgba(34,211,238,0.16) !important;
          transform: none !important;
        }

        .analysis-stack {
          padding: 9px !important;
          border-radius: 16px !important;
        }

        .diagnosis-banner {
          min-height: 36px !important;
          padding: 0 10px !important;
          justify-content: center !important;
          font-size: 9px !important;
          line-height: 1.25 !important;
          letter-spacing: 0.12em !important;
          text-align: center !important;
        }

        .sparkline-wrap {
          border-radius: 14px !important;
        }

        .sparkline {
          height: 42px !important;
        }

        .metric-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
          gap: 8px !important;
        }

        .metric {
          padding: 8px 9px !important;
          border-radius: 14px !important;
        }

        .metric-value {
          font-size: 16px !important;
          line-height: 18px !important;
        }

        .why {
          font-size: 12px !important;
          line-height: 1.45 !important;
        }

        .field-guide-pill {
          right: 14px !important;
          bottom: 14px !important;
          padding: 10px 12px !important;
          font-size: 9px !important;
          letter-spacing: 0.14em !important;
        }
      }

      /* STUFF_DISRUPTION_MOBILE_VERTICAL_TIGHTEN_V1 */
      @media screen and (max-width: 760px) {
        .hero-card {
          margin-bottom: 10px !important;
        }

        .meta-grid {
          gap: 7px !important;
          margin-bottom: 10px !important;
        }

        .meta-card {
          padding: 9px 10px !important;
          border-radius: 18px !important;
        }

        .meta-label {
          margin-bottom: 4px !important;
          font-size: 8px !important;
          line-height: 1 !important;
          letter-spacing: 0.16em !important;
        }

        .meta-value {
          font-size: 12px !important;
          line-height: 1.2 !important;
        }

        .section-head {
          padding: 12px 12px 10px !important;
        }

        .cards {
          padding: 8px !important;
          gap: 9px !important;
        }
      }

      /* STUFF_DISRUPTION_MOBILE_META_GAP_TIGHTEN_V1 */
      @media screen and (max-width: 760px) {
        .meta-grid {
          gap: 4px !important;
          margin-bottom: 8px !important;
        }

        .meta-card {
          margin: 0 !important;
          padding-top: 8px !important;
          padding-bottom: 8px !important;
        }
      }

      /* STUFF_DISRUPTION_MOBILE_META_HEIGHT_HALF_V1 */
      @media screen and (max-width: 760px) {
        .meta-grid {
          gap: 4px !important;
          margin-bottom: 6px !important;
        }

        .meta-card {
          min-height: 0 !important;
          padding: 6px 12px !important;
          border-radius: 18px !important;
        }

        .meta-label {
          margin-bottom: 4px !important;
          font-size: 8px !important;
          line-height: 1 !important;
          letter-spacing: 0.14em !important;
        }

        .meta-value {
          font-size: 12px !important;
          line-height: 1.15 !important;
        }
      }

      /* STUFF_DISRUPTION_MOBILE_HERO_INITIAL_CAPS_V1 */
      @media screen and (max-width: 760px) {
        .hero-title {
          text-transform: none !important;
          font-size: 38px !important;
          line-height: 0.96 !important;
          letter-spacing: -0.055em !important;
          font-weight: 850 !important;
        }
      }

      /* STUFF_DISRUPTION_MOBILE_FIELD_GUIDE_AND_TITLE_WEIGHT_V1 */
      @media screen and (max-width: 760px) {
        .hero-title {
          font-weight: 680 !important;
          font-size: 36px !important;
          line-height: 1.02 !important;
          letter-spacing: -0.05em !important;
          text-transform: none !important;
        }

        .hero-card .info-trigger,
        .info-trigger {
          display: inline-flex !important;
          align-items: center !important;
          justify-content: center !important;
          width: auto !important;
          min-width: 0 !important;
          min-height: 38px !important;
          margin-top: 16px !important;
          padding: 0 18px !important;
          border-radius: 999px !important;
          border: 1px solid rgba(34, 211, 238, 0.35) !important;
          background: rgba(2, 6, 23, 0.62) !important;
          color: rgba(255, 255, 255, 0.92) !important;
          font-family: var(--mono) !important;
          font-size: 10px !important;
          font-weight: 850 !important;
          letter-spacing: 0.16em !important;
          text-transform: uppercase !important;
          box-shadow: 0 0 16px rgba(34, 211, 238, 0.12) !important;
        }

        .hero-card > div {
          align-items: flex-start !important;
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
          <div class="brand-title">Apex Terminal // Stuff+ Disruption</div>
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
        <div class="guide-kicker">Apex Terminal</div>
        <h2 class="guide-title">Stuff+ Disruption Field Guide</h2>
      </div>
      <button class="guide-close" type="button" onclick="closeGuide()" aria-label="Close field guide">×</button>
    </div>

    <div class="guide-body">
      <section class="guide-section">
        <h3 class="guide-section-title">Why IVB matters</h3>
        <div class="guide-copy">
          IVB, or induced vertical break, is your ride signal. When a fastball gains IVB, it can play above the barrel more often and support whiff growth, especially when paired with a flatter approach angle.
        </div>
      </section>

      <section class="guide-section">
        <h3 class="guide-section-title">Why VAA matters</h3>
        <div class="guide-copy">
          VAA, or vertical approach angle, tells you how the pitch enters the zone. A flatter VAA can make a fastball look more explosive up in the zone, changing how hitters perceive the ball even before the stat line fully reacts.
        </div>
      </section>

      <section class="guide-section">
        <h3 class="guide-section-title">How to use this page</h3>
        <div class="guide-copy">
          APEX SHAPE SHIFT is the premium signal. ACTIVE SPIN GAIN highlights improving efficiency. SWEEP DISRUPTION captures sharper lateral shape changes. SHAPE MONITOR means the profile is moving, but not yet qualified enough to act aggressively.
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
          <div class="eyebrow">Apex Terminal</div>
          <h1 class="hero-title">Stuff+ Disruption</h1>
          <p class="hero-copy">
            Diagnostic breakout feed for IVB spikes, VAA flattening, movement disruptions, and active spin gains. This page is built to surface shape-shifters before their strikeout surge shows up in the obvious places.
          </p>
        </div>
        <button class="info-trigger" type="button" onclick="openGuide()">Field Guide</button>
      </div>
    </section>

    <section class="meta-grid">
      <div class="meta-card">
        <div class="meta-label">Model State</div>
        <div class="meta-value">Real Data Pass v1</div>
      </div>
      <div class="meta-card">
        <div class="meta-label">Window</div>
        <div class="meta-value">Recent Appearance vs Prior Baseline</div>
      </div>
      <div class="meta-card">
        <div class="meta-label">Next Pass</div>
        <div class="meta-value">Calibration + stronger spin logic</div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <div class="section-kicker">Breakout Detection</div>
          <h2 class="section-title">Apex Candidates</h2>
        </div>
        <div class="section-badge">{{ cards|length }} tracked arms</div>
      </div>

      <div class="cards">
        {% for row in cards %}
        <article
          class="player-card js-player-card {{ row.card_class }}"
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
                <div class="rankline">#{{ loop.index }} Apex Trigger</div>
                <h3 class="player-name">{{ row.player_name }}</h3>
                <div class="signal-line">{{ row.team }} // {{ row.apex_tier }} // {{ row.sample_note }}</div>
              </div>
            </div>

            <div class="scorebox">
              <div>
                <div class="score-label" title="{{ row.disruption_score_title }}" aria-label="{{ row.disruption_score_title }}">Disruption Score</div>
                <div class="score-value {{ row.score_class }}">{{ row.disruption_score_label }}</div>
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
                  data-source-tag="STUFF_DISRUPTION"
                >INITIATE TRACKING</button>
              </div>
            </div>
          </div>

          <div class="analysis-stack">
            <div class="diagnosis-wrap">
              <div class="diagnosis-label">Diagnosis</div>
              <div class="diagnosis-banner {{ row.alert_class }}">{{ row.primary_alert }}</div>
            </div>

            <div class="sparkline-wrap">
              <div class="sparkline-head">
                <div class="sparkline-label">Shape Trend</div>
                <div class="sparkline-note">Last {{ row.sample_note }}</div>
              </div>
              <svg class="sparkline" viewBox="0 0 100 34" preserveAspectRatio="none" aria-hidden="true">
                <polyline class="sparkline-path {{ row.sparkline_class }}" points="{{ row.trend_points }}" />
              </svg>
          </div>

          <div class="metric-grid">
            <div class="metric {{ row.ivb_class }}">
              <div class="metric-label" title="{{ row.ivb_delta_title }}" aria-label="{{ row.ivb_delta_title }}">IVB Delta</div>
              <div class="metric-value">{{ row.ivb_delta_label }}</div>
            </div>
            <div class="metric {{ row.vaa_class }}">
              <div class="metric-label" title="{{ row.vaa_delta_title }}" aria-label="{{ row.vaa_delta_title }}">VAA Delta</div>
              <div class="metric-value">{{ row.vaa_delta_label }}</div>
            </div>
            <div class="metric {{ row.movement_class }}">
              <div class="metric-label" title="{{ row.movement_delta_title }}" aria-label="{{ row.movement_delta_title }}">Move Delta</div>
              <div class="metric-value">{{ row.movement_delta_label }}</div>
            </div>
            <div class="metric {{ row.spin_class }}">
              <div class="metric-label" title="{{ row.active_spin_delta_title }}" aria-label="{{ row.active_spin_delta_title }}">Active Spin</div>
              <div class="metric-value">{{ row.active_spin_delta_label }}</div>
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



def write_stuff_disruption_status(
    *,
    build_started_at: str,
    build_finished_at: str,
    cards,
) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

    card_count = int(len(cards)) if cards is not None else 0
    degraded = card_count == 0

    alert_counts = {}
    for row in cards or []:
        alert = str(row.get("primary_alert") or row.get("alert") or "UNKNOWN").strip() or "UNKNOWN"
        alert_counts[alert] = alert_counts.get(alert, 0) + 1

    status_payload = build_report_status(
        "stuff_disruption",
        build_success=True,
        threshold_minutes=240,
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        source_updated_at=build_finished_at,
        section_counts={
            "stuff_disruption_cards": card_count,
            "alert_groups": len(alert_counts),
        },
        degraded=degraded,
        notes=[
            f"Stuff+ Disruption built with {card_count} active disruption cards."
        ],
    )

    status_payload["alert_counts"] = alert_counts

    STUFF_DISRUPTION_STATUS_PATH.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote Stuff+ Disruption status -> {STUFF_DISRUPTION_STATUS_PATH}")


def write_stuff_disruption_feed() -> None:
    build_started_at = datetime.now(timezone.utc).isoformat()
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    APEX_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    raw = fetch_statcast_window(str(start_date), str(end_date))
    pitches = load_fastball_pitches(raw)
    appearances = build_pitcher_appearances(pitches)
    rows = build_apex_rows(appearances)
    cards = format_apex_cards(rows)

    html = HTML_TEMPLATE.render(
        nav_html=Template(NAV_TEMPLATE).render(active_nav="stuff_disruption_feed"),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
        cards=cards,
        generated_at=datetime.now().strftime("%b %d, %Y %I:%M %p"),
    )

    (APEX_DIR / "index.html").write_text(html, encoding="utf-8")
    print("Wrote dist/stuff-disruption-feed/index.html")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "cards": cards,
        "mode": "real_data_v1",
    }
    (DIST_DIR / "stuff_disruption_feed.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print("Wrote dist/stuff_disruption_feed.json")

    build_finished_at = datetime.now(timezone.utc).isoformat()
    write_stuff_disruption_status(
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        cards=cards,
    )

    copy_static_assets()


def main() -> None:
    write_stuff_disruption_feed()


if __name__ == "__main__":
    main()