#!/usr/bin/env python3
import os
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from jinja2 import Template
from pybaseball import statcast, playerid_reverse_lookup

DIST_DIR = Path("dist")
DIST_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = Path(__file__).parent / "templates"
NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "65"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
SITE_URL = os.getenv("SITE_URL", "").strip()
TIMEZONE_LABEL = os.getenv("TIMEZONE_LABEL", "America/New_York")

TODAY = date.today()
START_DATE = TODAY - timedelta(days=28)
END_DATE = TODAY


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / std


def safe_name(value: str) -> str:
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()
    if not text:
        return "Unknown"

    if text.lower() == "unknown":
        return "Unknown"

    return " ".join(part.capitalize() for part in text.split())


def markdown_escape(text: str) -> str:
    specials = r"_*[]()~`>#+-=|{}.!"
    out = str(text)
    for ch in specials:
        out = out.replace(ch, f"\\{ch}")
    return out


def fetch_statcast_window(start_dt: date, end_dt: date) -> pd.DataFrame:
    print(f"Fetching Statcast from {start_dt} to {end_dt}...")
    df = statcast(start_dt=start_dt.strftime("%Y-%m-%d"), end_dt=end_dt.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        raise RuntimeError("Statcast returned no data for the requested window.")
    return df


def build_batter_name_map(batter_ids) -> dict[int, str]:
    ids = []
    for value in batter_ids:
        try:
            if pd.notna(value):
                ids.append(int(value))
        except Exception:
            continue

    ids = sorted(set(ids))
    if not ids:
        return {}

    name_map = {}

    try:
        lookup = playerid_reverse_lookup(ids, key_type="mlbam")
    except Exception:
        lookup = None

    if lookup is not None and not lookup.empty:
        for _, row in lookup.iterrows():
            try:
                pid = int(row["key_mlbam"])
            except Exception:
                continue

            first = str(row.get("name_first", "")).strip()
            last = str(row.get("name_last", "")).strip()
            full_name = f"{first} {last}".strip()

            if full_name:
                name_map[pid] = full_name

    for pid in ids:
        if pid not in name_map:
            name_map[pid] = f"Player {pid}"

    return name_map


def fill_missing_batter_names_with_statsapi(name_map: dict[int, str], batter_ids) -> dict[int, str]:
    ids = []
    for value in batter_ids:
        try:
            if pd.notna(value):
                ids.append(int(value))
        except Exception:
            continue

    ids = sorted(set(ids))

    for pid in ids:
        current = str(name_map.get(pid, "")).strip()
        if current and not current.startswith("Player "):
            continue

        try:
            url = f"https://statsapi.mlb.com/api/v1/people/{pid}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            payload = response.json()

            people = payload.get("people", [])
            if people:
                full_name = str(people[0].get("fullName", "")).strip()
                if full_name:
                    name_map[pid] = full_name
        except Exception:
            continue

    return name_map


def build_hitter_signals(df: pd.DataFrame) -> pd.DataFrame:
    hitters = df.copy()

    batter_ids = hitters["batter"].dropna().unique()
    batter_name_map = build_batter_name_map(batter_ids)
    batter_name_map = fill_missing_batter_names_with_statsapi(batter_name_map, batter_ids)

    pitcher_ids = set(
        pd.to_numeric(df["pitcher"], errors="coerce").dropna().astype(int).tolist()
    )

    bbe = hitters[hitters["launch_speed"].notna()].copy()

    recent_cutoff = pd.Timestamp(TODAY - timedelta(days=7))
    recent_bbe_counts = (
        bbe[pd.to_datetime(bbe["game_date"]) >= recent_cutoff]
        .groupby("batter", dropna=False)
        .size()
        .reset_index(name="recent_bbe_count")
    )

    bbe = bbe.merge(recent_bbe_counts, on="batter", how="left")
    bbe["recent_bbe_count"] = bbe["recent_bbe_count"].fillna(0)
    bbe = bbe[bbe["recent_bbe_count"] >= 4].copy()
    bbe = bbe[
        ~pd.to_numeric(bbe["batter"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .isin(pitcher_ids)
    ].copy()

    if bbe.empty:
        return pd.DataFrame()

    bbe["game_date"] = pd.to_datetime(bbe["game_date"])
    bbe["is_recent"] = bbe["game_date"] >= pd.Timestamp(TODAY - timedelta(days=7))
    bbe["is_baseline"] = (bbe["game_date"] < pd.Timestamp(TODAY - timedelta(days=7))) & (
        bbe["game_date"] >= pd.Timestamp(TODAY - timedelta(days=28))
    )

    bbe["barrel_like"] = (
        (pd.to_numeric(bbe["launch_speed"], errors="coerce") >= 98)
        & (pd.to_numeric(bbe["launch_angle"], errors="coerce").between(26, 30, inclusive="both"))
    ).astype(int)

    recent = (
        bbe[bbe["is_recent"]]
        .groupby(["batter"], dropna=False)
        .agg(
            recent_bbe=("launch_speed", "size"),
            recent_ev=("launch_speed", "mean"),
            recent_max_ev=("launch_speed", "max"),
            recent_barrel_rate=("barrel_like", "mean"),
        )
        .reset_index()
    )

    baseline = (
        bbe[bbe["is_baseline"]]
        .groupby(["batter"], dropna=False)
        .agg(
            baseline_bbe=("launch_speed", "size"),
            baseline_ev=("launch_speed", "mean"),
            baseline_max_ev=("launch_speed", "max"),
            baseline_barrel_rate=("barrel_like", "mean"),
        )
        .reset_index()
    )

    merged = recent.merge(
        baseline,
        on=["batter"],
        how="left",
        suffixes=("", "_base")
    )

    merged = merged[
        (merged["recent_bbe"] >= 6) &
        (merged["recent_max_ev"] >= 95)
    ].copy()

    if merged.empty:
        return pd.DataFrame()

    merged["baseline_bbe"] = merged["baseline_bbe"].fillna(0)

    merged["ev_delta"] = merged["recent_ev"] - merged["baseline_ev"].fillna(merged["recent_ev"])
    merged["barrel_rate_delta"] = (
        merged["recent_barrel_rate"]
        - merged["baseline_barrel_rate"].fillna(merged["recent_barrel_rate"])
    )

    merged["quality_index"] = (
        0.50 * zscore(merged["recent_ev"])
        + 0.25 * zscore(merged["recent_max_ev"])
        + 0.25 * zscore(merged["recent_barrel_rate"])
    )

    merged["delta_index"] = (
        0.65 * zscore(merged["ev_delta"])
        + 0.35 * zscore(merged["barrel_rate_delta"])
    )

    merged["edge_score_raw"] = (
        50
        + 11 * merged["quality_index"]
        + 8 * merged["delta_index"]
        + 2 * zscore(merged["recent_bbe"])
    )

    merged["edge_score"] = (
        50 + (merged["edge_score_raw"] - 50) * 0.82
    ).clip(5, 95).round(1)

    merged["player_name"] = merged["batter"].apply(
        lambda x: batter_name_map.get(int(x), f"Player {int(x)}") if pd.notna(x) else "Unknown"
    )
    merged["player_name"] = merged["player_name"].apply(safe_name)

    merged["signal_type"] = "Hitter"
    merged["why"] = merged.apply(
        lambda r: (
            f"Avg EV {r['recent_ev']:.1f} mph "
            f"({r['ev_delta']:+.1f} vs baseline), "
            f"barrel-like rate {100 * r['recent_barrel_rate']:.1f}%."
        ),
        axis=1
    )

    merged["metric_1"] = merged["recent_ev"].round(1)
    merged["metric_1_label"] = "Avg EV"
    merged["metric_2"] = (100 * merged["recent_barrel_rate"]).round(1)
    merged["metric_2_label"] = "Barrel-like %"
    merged["metric_3"] = merged["recent_max_ev"].round(1)
    merged["metric_3_label"] = "Max EV"
    merged["sample_note"] = merged["recent_bbe"].apply(lambda x: f"{int(x)} BBE")

    def hitter_badges(row: pd.Series) -> list[str]:
        badges = []

        if pd.notna(row["ev_delta"]) and row["ev_delta"] >= 2.0:
            badges.append("EV Burst")

        if pd.notna(row["barrel_rate_delta"]) and row["barrel_rate_delta"] >= 0.08:
            badges.append("Barrel Jump")

        if pd.notna(row["recent_max_ev"]) and row["recent_max_ev"] >= 108:
            badges.append("Impact EV")

        if not badges:
            badges.append("Trend Confirming")
        elif "Trend Confirming" not in badges:
            badges.append("Trend Confirming")

        return badges

    def hitter_badge_classes(row: pd.Series) -> list[str]:
        classes = []
        for badge in row["badges"]:
            if badge in ["EV Burst", "Barrel Jump", "Impact EV"]:
                classes.append("positive")
            else:
                classes.append("neutral")
        return classes

    merged["badges"] = merged.apply(hitter_badges, axis=1)
    merged["badge_classes"] = merged.apply(hitter_badge_classes, axis=1)

    recent_daily_ev = (
        bbe[bbe["is_recent"]]
        .groupby(["batter", "game_date"], dropna=False)
        .agg(day_ev=("launch_speed", "mean"))
        .reset_index()
    )

    def build_trend_points(player_id) -> str:
        player_days = recent_daily_ev[recent_daily_ev["batter"] == player_id].copy()
        if player_days.empty:
            return "0,24 20,22 40,21 60,19 80,18 100,16 120,14"

        player_days = player_days.sort_values("game_date")
        vals = player_days["day_ev"].tolist()

        if len(vals) == 1:
            vals = vals * 7
        elif len(vals) < 7:
            vals = [vals[0]] * (7 - len(vals)) + vals
        else:
            vals = vals[-7:]

        vmin = min(vals)
        vmax = max(vals)

        if vmax == vmin:
            yvals = [17 for _ in vals]
        else:
            yvals = [26 - ((v - vmin) / (vmax - vmin)) * 16 for v in vals]

        xvals = [0, 20, 40, 60, 80, 100, 120]
        points = [f"{x},{round(y, 1)}" for x, y in zip(xvals, yvals)]
        return " ".join(points)

    merged["trend_points"] = merged["batter"].apply(build_trend_points)
    merged["trend_glow"] = merged["ev_delta"] >= 2.0

    return merged.sort_values("edge_score", ascending=False).reset_index(drop=True)


def build_pitcher_signals(df: pd.DataFrame) -> pd.DataFrame:
    pitchers = df.copy()
    pitchers["game_date"] = pd.to_datetime(pitchers["game_date"])
    pitchers["is_whiff"] = pitchers["description"].isin(
        ["swinging_strike", "swinging_strike_blocked"]
    ).astype(int)

    fastballs = {"FF", "FT", "SI", "FC"}
    pitchers["is_fastball"] = pitchers["pitch_type"].isin(fastballs).astype(int)
    pitchers["fastball_speed"] = pd.to_numeric(
        pitchers["release_speed"], errors="coerce"
    ).where(pitchers["is_fastball"] == 1)

    pitchers["is_recent"] = pitchers["game_date"] >= pd.Timestamp(TODAY - timedelta(days=7))
    pitchers["is_baseline"] = (pitchers["game_date"] < pd.Timestamp(TODAY - timedelta(days=7))) & (
        pitchers["game_date"] >= pd.Timestamp(TODAY - timedelta(days=28))
    )

    recent = (
        pitchers[pitchers["is_recent"]]
        .groupby(["pitcher", "player_name"], dropna=False)
        .agg(
            recent_pitches=("pitch_type", "size"),
            recent_whiff_rate=("is_whiff", "mean"),
            recent_fb_velo=("fastball_speed", "mean"),
            recent_extension=("release_extension", "mean"),
        )
        .reset_index()
    )

    baseline = (
        pitchers[pitchers["is_baseline"]]
        .groupby(["pitcher", "player_name"], dropna=False)
        .agg(
            baseline_pitches=("pitch_type", "size"),
            baseline_whiff_rate=("is_whiff", "mean"),
            baseline_fb_velo=("fastball_speed", "mean"),
            baseline_extension=("release_extension", "mean"),
        )
        .reset_index()
    )

    merged = recent.merge(baseline, on=["pitcher", "player_name"], how="left")
    merged = merged[
        (merged["recent_pitches"] >= 60) &
        (merged["recent_fb_velo"].fillna(0) >= 90)
    ].copy()

    if merged.empty:
        return pd.DataFrame()

    merged["velo_delta"] = merged["recent_fb_velo"] - merged["baseline_fb_velo"].fillna(merged["recent_fb_velo"])
    merged["whiff_delta"] = merged["recent_whiff_rate"] - merged["baseline_whiff_rate"].fillna(merged["recent_whiff_rate"])
    merged["extension_delta"] = merged["recent_extension"] - merged["baseline_extension"].fillna(merged["recent_extension"])

    merged["quality_index"] = (
        0.50 * zscore(merged["recent_whiff_rate"])
        + 0.30 * zscore(merged["recent_fb_velo"])
        + 0.20 * zscore(merged["recent_extension"])
    )

    merged["delta_index"] = (
        0.50 * zscore(merged["velo_delta"])
        + 0.35 * zscore(merged["whiff_delta"])
        + 0.15 * zscore(merged["extension_delta"])
    )

    merged["edge_score_raw"] = (
        50
        + 11 * merged["quality_index"]
        + 9 * merged["delta_index"]
        + 2 * zscore(merged["recent_pitches"])
    )

    merged["edge_score"] = (
        50 + (merged["edge_score_raw"] - 50) * 0.84
    ).clip(5, 95).round(1)

    merged["player_name"] = merged["player_name"].apply(safe_name)
    merged["signal_type"] = "Pitcher"
    merged["why"] = merged.apply(
        lambda r: (
            f"Whiff rate {100 * r['recent_whiff_rate']:.1f}% "
            f"({r['whiff_delta']:+.1f} pts vs baseline), "
            f"FB velo {r['recent_fb_velo']:.1f} mph ({r['velo_delta']:+.1f})."
        ),
        axis=1
    )

    merged["metric_1"] = (100 * merged["recent_whiff_rate"]).round(1)
    merged["metric_1_label"] = "Whiff %"
    merged["metric_2"] = merged["recent_fb_velo"].round(1)
    merged["metric_2_label"] = "FB Velo"
    merged["metric_3"] = merged["recent_extension"].round(2)
    merged["metric_3_label"] = "Extension"
    merged["sample_note"] = merged["recent_pitches"].apply(lambda x: f"{int(x)} P")

    def pitcher_badges(row: pd.Series) -> list[str]:
        badges = []

        if pd.notna(row["whiff_delta"]) and row["whiff_delta"] >= 0.03:
            badges.append("Whiff Lift")

        if pd.notna(row["velo_delta"]) and row["velo_delta"] >= 0.8:
            badges.append("Velo Jump")

        if pd.notna(row["extension_delta"]) and row["extension_delta"] >= 0.10:
            badges.append("Extension Gain")

        if not badges:
            badges.append("Trend Confirming")
        elif "Trend Confirming" not in badges:
            badges.append("Trend Confirming")

        return badges

    def pitcher_badge_classes(row: pd.Series) -> list[str]:
        classes = []
        for badge in row["badges"]:
            if badge in ["Whiff Lift", "Velo Jump", "Extension Gain"]:
                classes.append("positive")
            else:
                classes.append("neutral")
        return classes

    merged["badges"] = merged.apply(pitcher_badges, axis=1)
    merged["badge_classes"] = merged.apply(pitcher_badge_classes, axis=1)

    recent_daily_whiff = (
        pitchers[pitchers["is_recent"]]
        .groupby(["pitcher", "game_date"], dropna=False)
        .agg(day_whiff=("is_whiff", "mean"))
        .reset_index()
    )

    def build_pitcher_trend_points(player_id) -> str:
        player_days = recent_daily_whiff[recent_daily_whiff["pitcher"] == player_id].copy()
        if player_days.empty:
            return "0,25 20,23 40,21 60,19 80,17 100,15 120,13"

        player_days = player_days.sort_values("game_date")
        vals = player_days["day_whiff"].tolist()

        if len(vals) == 1:
            base = vals[0]
            vals = [base * 0.985, base * 0.99, base * 0.995, base, base * 1.005, base * 1.01, base * 1.015]
        elif len(vals) < 7:
            pad_vals = [vals[0]] * (7 - len(vals))
            vals = pad_vals + vals
        else:
            vals = vals[-7:]

        vmin = min(vals)
        vmax = max(vals)

        if vmax == vmin:
            yvals = [24, 22, 21, 19, 18, 16, 14]
        else:
            yvals = [26 - ((v - vmin) / (vmax - vmin)) * 16 for v in vals]

        xvals = [0, 20, 40, 60, 80, 100, 120]
        points = [f"{x},{round(y, 1)}" for x, y in zip(xvals, yvals)]
        return " ".join(points)

    merged["trend_points"] = merged["pitcher"].apply(build_pitcher_trend_points)
    merged["trend_glow"] = merged["whiff_delta"] >= 0.03

    return merged.sort_values("edge_score", ascending=False).reset_index(drop=True)


def build_telegram_message(row: pd.Series) -> str:
    title = f"{row['signal_type']} Trigger: {row['player_name']}"
    body = (
        f"*{markdown_escape(title)}*\n"
        f"Edge Score: *{markdown_escape(row['edge_score'])}*\n"
        f"{markdown_escape(row['metric_1_label'])}: {markdown_escape(row['metric_1'])}\n"
        f"{markdown_escape(row['metric_2_label'])}: {markdown_escape(row['metric_2'])}\n"
        f"{markdown_escape(row['metric_3_label'])}: {markdown_escape(row['metric_3'])}\n"
        f"{markdown_escape(row['why'])}"
    )
    if SITE_URL:
        body += f"\n[Open Signal Wall]({markdown_escape(SITE_URL)})"
    return body


def send_telegram_alerts(signals: pd.DataFrame) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing; skipping Telegram dispatch.")
        return

    alerts = signals[signals["edge_score"] >= ALERT_THRESHOLD].copy()
    if alerts.empty:
        print(f"No alerts above threshold {ALERT_THRESHOLD}.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for _, row in alerts.iterrows():
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": build_telegram_message(row),
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True,
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        print(f"Telegram alert sent: {row['player_name']} ({row['edge_score']})")


HTML_TEMPLATE = Template("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals — Signal Wall</title>
  <style>
    :root {
      --bg: #080808;
      --surface: #121212;
      --surface-deep: #080808;
      --card-radial: radial-gradient(circle at top left, #1a1a1a 0%, #080808 100%);
      --border: #2d2d2d;
      --text: #f0f0f0;
      --muted: #71717a;
      --soft: #a1a1aa;
      --tiny: #8a8a93;
      --emerald: #4ade80;
      --lime-hot: #b6ff00;
      --cyan-hot: #00e5ff;
      --ghost-grey: #444444;
      --crimson: #f87171;
      --blue: #6aa6ff;
      --shadow: 0 14px 34px rgba(0, 0, 0, 0.34);
      --radius: 18px;
      --mono: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }

    html, body {
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
    }

    body {
      background:
        radial-gradient(circle at top left, rgba(106,166,255,0.06), transparent 24%),
        radial-gradient(circle at top right, rgba(239,68,68,0.04), transparent 20%),
        linear-gradient(180deg, #101010 0%, #080808 34%, #050505 100%);
      line-height: 1.35;
    }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(8, 8, 8, 0.90);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
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
      gap: 10px;
      min-width: 0;
    }

    .brand-mark {
      width: 11px;
      height: 11px;
      border-radius: 999px;
      background: var(--lime-hot);
      box-shadow: 0 0 10px rgba(182,255,0,0.35);
      animation: heartbeatPulse 2.2s infinite ease-in-out;
      flex: 0 0 auto;
    }

    @keyframes heartbeatPulse {
      0% {
        opacity: 0.55;
        transform: scale(0.88);
        box-shadow: 0 0 0 0 rgba(182,255,0,0.00), 0 0 8px rgba(182,255,0,0.18);
      }
      35% {
        opacity: 1;
        transform: scale(1.18);
        box-shadow: 0 0 0 6px rgba(182,255,0,0.10), 0 0 14px rgba(182,255,0,0.38);
      }
      70% {
        opacity: 0.9;
        transform: scale(1.00);
        box-shadow: 0 0 0 12px rgba(182,255,0,0.00), 0 0 10px rgba(182,255,0,0.22);
      }
      100% {
        opacity: 0.55;
        transform: scale(0.88);
        box-shadow: 0 0 0 0 rgba(182,255,0,0.00), 0 0 8px rgba(182,255,0,0.18);
      }
    }

    @keyframes badgePulse {
      0%, 100% { opacity: 0.8; }
      50% { opacity: 1; }
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

    .brand-white {
      color: var(--text);
    }

    .brand-blue {
      color: var(--blue);
    }

    .brand-title {
      font-size: 16px;
      line-height: 1.05;
      letter-spacing: -0.02em;
      font-weight: 800;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex: 0 0 auto;
    }

    .info-trigger {
      width: 30px;
      height: 30px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.03);
      color: var(--text);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-family: var(--mono);
      font-size: 13px;
      font-weight: 800;
      cursor: pointer;
      transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }

    .info-trigger:hover {
      transform: translateY(-1px);
      border-color: rgba(182,255,0,0.20);
      background: rgba(255,255,255,0.05);
    }

    .livebox {
      text-align: right;
      flex: 0 0 auto;
    }

    .topnav {
      border-top: 1px solid rgba(255,255,255,0.04);
      border-bottom: 1px solid rgba(255,255,255,0.05);
      background: rgba(255,255,255,0.015);
    }

    .topnav-inner {
      width: min(1180px, calc(100% - 24px));
      margin: 0 auto;
      display: flex;
      align-items: center;
      gap: 10px;
      overflow-x: auto;
      white-space: nowrap;
      padding: 10px 0;
      scrollbar-width: none;
    }

    .topnav-inner::-webkit-scrollbar {
      display: none;
    }

    .topnav-link {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
      font-family: var(--mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--soft);
      padding: 8px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.02);
      transition: opacity 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
    }

    .topnav-link:hover {
      opacity: 1;
      transform: translateY(-1px);
      border-color: rgba(255,255,255,0.12);
    }

    .topnav-link.active {
      color: var(--text);
      border-color: rgba(182,255,0,0.20);
      box-shadow: 0 0 8px rgba(182,255,0,0.08);
    }

    .topnav-tag {
      color: var(--lime-hot);
    }

    .live-label {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--lime-hot);
      font-weight: 800;
      margin-bottom: 4px;
    }

    .live-dot {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--lime-hot);
      box-shadow: 0 0 10px rgba(182,255,0,0.35);
      animation: heartbeatPulse 2.2s infinite ease-in-out;
    }

    .live-time {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

    .glossary-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.52);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.22s ease;
      z-index: 80;
    }

    .glossary-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }

    .glossary-drawer {
      position: fixed;
      top: 0;
      right: 0;
      width: min(560px, 100vw);
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

    .glossary-drawer.open {
      transform: translateX(0);
    }

    .glossary-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 18px 18px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.02);
    }

    .glossary-kicker {
      font-size: 10px;
      line-height: 1;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--blue);
      font-weight: 800;
      margin-bottom: 8px;
    }

    .glossary-title {
      margin: 0;
      font-size: 20px;
      line-height: 1.05;
      letter-spacing: -0.03em;
      text-transform: uppercase;
      font-weight: 900;
      color: var(--text);
    }

    .glossary-close {
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
      flex: 0 0 auto;
    }

    .glossary-body {
      overflow-y: auto;
      padding: 18px;
      display: grid;
      gap: 18px;
    }

    .glossary-section {
      border: 1px solid rgba(255,255,255,0.05);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
      padding: 14px;
    }

    .glossary-section-title {
      margin: 0 0 12px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--lime-hot);
      font-weight: 800;
      font-family: var(--mono);
    }

    .glossary-item {
      margin-bottom: 12px;
    }

    .glossary-item:last-child {
      margin-bottom: 0;
    }

    .glossary-term {
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text);
      font-weight: 800;
      font-family: var(--mono);
    }

    .glossary-definition {
      font-size: 13px;
      line-height: 1.5;
      color: var(--soft);
    }

    .app {
      padding: 18px 0 34px;
    }

    .hero {
      display: grid;
      gap: 14px;
      margin-bottom: 16px;
    }

    .hero-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }

    .hero-card,
    .meta-card,
    .section,
    .player-card {
      background: var(--card-radial);
      border: 0.5px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }

    .hero-card::before,
    .meta-card::before,
    .section::before,
    .player-card::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      border-radius: inherit;
      padding: 0.5px;
      background: linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.01));
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
              mask-composite: exclude;
      opacity: 0.55;
    }

    .hero-card {
      padding: 18px;
    }

    .eyebrow {
      font-size: 10px;
      line-height: 1;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--blue);
      font-weight: 800;
      margin-bottom: 10px;
    }

    .hero-title {
      margin: 0 0 10px;
      font-size: clamp(28px, 7vw, 50px);
      line-height: 0.95;
      letter-spacing: -0.04em;
      font-weight: 900;
      text-transform: uppercase;
    }

    .hero-copy {
      margin: 0;
      max-width: 760px;
      color: var(--soft);
      font-size: 14px;
    }

    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }

    .meta-card {
      padding: 14px;
    }

    .meta-label,
    .metric-label,
    .sparkline-label,
    .section-kicker,
    .score-label,
    .rankline,
    .status-badge {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      font-weight: 800;
    }

    .meta-label {
      margin-bottom: 6px;
    }

    .meta-value {
      font-family: var(--mono);
      font-size: 13px;
      color: var(--text);
      word-break: break-word;
      font-variant-numeric: tabular-nums;
    }

    .slate-heat-card {
      padding: 14px;
    }

    .slate-heat-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }

    .slate-heat-bar {
      height: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.04);
      overflow: hidden;
    }

    .slate-heat-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #444444 0%, #b6ff00 100%);
      box-shadow: 0 0 8px rgba(182,255,0,0.16);
    }

    .slate-heat-value {
      font-family: var(--mono);
      font-size: 13px;
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }

    .board {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }

    .section {
      overflow: hidden;
    }

    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 16px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      background: linear-gradient(180deg, rgba(255,255,255,0.022), rgba(255,255,255,0.008));
      position: relative;
    }

    .section-head::after {
      content: "◆";
      position: absolute;
      right: 76px;
      top: 50%;
      transform: translateY(-50%);
      font-size: 30px;
      color: rgba(255,255,255,0.03);
      pointer-events: none;
      letter-spacing: 0;
    }

    .section-kicker {
      margin-bottom: 5px;
    }

    .section-title {
      margin: 0;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: -0.02em;
      text-transform: uppercase;
    }

    .section-badge {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--soft);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 999px;
      padding: 7px 10px;
      background: rgba(255,255,255,0.02);
      white-space: nowrap;
      position: relative;
      z-index: 1;
      font-variant-numeric: tabular-nums;
    }

    .cards {
      display: grid;
      gap: 10px;
      padding: 10px;
    }

    .player-card {
      padding: 14px;
    }

    .player-card.high-edge {
      border-color: rgba(74,222,128,0.22);
      box-shadow: var(--shadow), 0 0 8px rgba(74,222,128,0.07);
    }

    .player-card.regression {
      border-color: rgba(248,113,113,0.22);
      box-shadow: var(--shadow), 0 0 8px rgba(248,113,113,0.05);
    }

    .player-top {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: start;
      margin-bottom: 12px;
    }

    .avatar {
      width: 42px;
      height: 42px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.10);
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(255,255,255,0.03);
      color: var(--text);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.04em;
      flex: 0 0 auto;
      font-variant-numeric: tabular-nums;
    }

    .player-ident {
      min-width: 0;
    }

    .rankline {
      margin-bottom: 4px;
    }

    .player-name {
      font-size: 19px;
      line-height: 1.02;
      letter-spacing: -0.03em;
      font-weight: 800;
      margin: 0 0 4px;
      word-break: break-word;
    }

    .signal-line {
      font-size: 11px;
      color: var(--soft);
      font-family: var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-variant-numeric: tabular-nums;
    }

    .scorebox {
      text-align: right;
      min-width: 88px;
    }

    .score-label {
      margin-bottom: 4px;
    }

    .score-value {
      font-family: var(--mono);
      font-size: 28px;
      line-height: 1;
      font-weight: 800;
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }

    .score-value.edge-up {
      color: var(--emerald);
      text-shadow: 0 0 6px rgba(74,222,128,0.18);
    }

    .score-value.edge-down {
      color: var(--crimson);
      text-shadow: 0 0 6px rgba(248,113,113,0.18);
    }

    .sparkline-wrap {
      margin: 0 0 12px;
      padding: 8px 10px;
      border: 1px solid rgba(255,255,255,0.04);
      border-radius: 12px;
      background: rgba(255,255,255,0.015);
    }

    .sparkline-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
    }

    .sparkline-note {
      font-family: var(--mono);
      font-size: 10px;
      color: var(--tiny);
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-variant-numeric: tabular-nums;
    }

    svg.sparkline {
      display: block;
      width: 100%;
      height: 34px;
    }

    .sparkline-path {
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
      fill: none;
    }

    .sparkline-path.glow {
      filter: drop-shadow(0 0 2px rgba(182, 255, 0, 0.5));
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }

    .metric {
      border: 1px solid rgba(255,255,255,0.05);
      border-radius: 12px;
      padding: 10px 10px 9px;
      background: rgba(255,255,255,0.02);
      min-width: 0;
    }

    .metric-label {
      margin-bottom: 6px;
    }

    .metric-value {
      font-family: var(--mono);
      font-size: 15px;
      line-height: 1.1;
      color: var(--text);
      font-weight: 700;
      word-break: break-word;
      font-variant-numeric: tabular-nums;
    }

    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 0 0 12px;
    }

    .status-badge {
      line-height: 1;
      border-radius: 999px;
      padding: 7px 9px;
      border: 1px solid rgba(255,255,255,0.08);
      color: var(--soft);
      background: rgba(255,255,255,0.02);
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
    }

    .status-badge.positive {
      color: var(--emerald);
      border-color: rgba(74,222,128,0.18);
      box-shadow: 0 0 6px rgba(74,222,128,0.08);
      background: rgba(74,222,128,0.03);
    }

    .status-badge.negative {
      color: var(--crimson);
      border-color: rgba(248,113,113,0.18);
      box-shadow: 0 0 6px rgba(248,113,113,0.06);
      background: rgba(248,113,113,0.03);
    }

    .status-badge.neutral {
      color: var(--soft);
      border-color: rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.02);
    }

    .status-badge.active-pulse {
      animation: badgePulse 2.2s infinite ease-in-out;
    }

    .why {
      font-size: 10px;
      line-height: 1.45;
      color: var(--tiny);
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
    }

    .footer {
      padding: 16px 4px 0;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-variant-numeric: tabular-nums;
    }

    @media (min-width: 900px) {
      .hero-grid {
        grid-template-columns: 1.35fr 0.9fr;
        align-items: stretch;
      }

      .board {
        grid-template-columns: 1fr 1fr;
      }
    }

    @media (max-width: 640px) {
      .topbar-inner,
      .app,
      .topnav-inner {
        width: min(100%, calc(100% - 16px));
      }

      .topbar-inner {
        min-height: 58px;
      }

      .brand-title {
        font-size: 14px;
      }

      .livebox {
        max-width: 44%;
      }

      .meta-grid {
        grid-template-columns: 1fr;
      }

      .player-name {
        font-size: 17px;
      }

      .score-value {
        font-size: 24px;
      }

      .section-head::after {
        right: 64px;
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
          <div class="brand-title">Signal Wall // Institutional Elite</div>
        </div>
      </div>
      <div class="header-actions">
        <button class="info-trigger" type="button" onclick="openGlossary()" aria-label="Open glossary">i</button>
        <div class="livebox">
          <div class="live-label"><span class="live-dot"></span>LIVE</div>
          <div class="live-time">{{ generated_at }}</div>
        </div>
      </div>
    </div>
  </div>

  {{ nav_html | safe }}

  <div id="glossaryOverlay" class="glossary-overlay" onclick="closeGlossary()"></div>

  <aside id="glossaryDrawer" class="glossary-drawer" aria-hidden="true">
    <div class="glossary-head">
      <div>
        <div class="glossary-kicker">DiamondSignals Intelligence</div>
        <h2 class="glossary-title">Glossary</h2>
      </div>
      <button class="glossary-close" type="button" onclick="closeGlossary()" aria-label="Close glossary">×</button>
    </div>

    <div class="glossary-body">
      <section class="glossary-section">
        <h3 class="glossary-section-title">I. Global System Metrics</h3>

        <div class="glossary-item">
          <span class="glossary-term">Slate Heat</span>
          <div class="glossary-definition">
            A model-driven index of total opportunity across the day's schedule. A full Neon Green bar indicates a high density of ballistic anomalies, suggesting a target-rich day for fantasy and DFS adjustments.
          </div>
        </div>

        <div class="glossary-item">
          <span class="glossary-term">System Status</span>
          <div class="glossary-definition">
            Confirms the state of the live data pipeline. Our Python architecture monitors 750+ Ballistic Profiles in real-time, pulling direct Statcast feeds to detect skill shifts before they hit the box score.
          </div>
        </div>

        <div class="glossary-item">
          <span class="glossary-term">Edge Score</span>
          <div class="glossary-definition">
            A 0–100 proprietary ranking that quantifies the strength of a signal. Higher scores indicate high-conviction buy opportunities based on rolling ballistic data.
          </div>
        </div>
      </section>

      <section class="glossary-section">
        <h3 class="glossary-section-title">II. The Ballistic Terminal (Pitching)</h3>

        <div class="glossary-item">
          <span class="glossary-term">IVB (Induced Vertical Break)</span>
          <div class="glossary-definition">
            The secret sauce of pitching. It measures how much a pitch defies gravity and rises. Spikes in IVB are the primary indicator of increased whiff rates.
          </div>
        </div>

        <div class="glossary-item">
          <span class="glossary-term">VAA (Vertical Approach Angle)</span>
          <div class="glossary-definition">
            The angle at which a pitch enters the zone. Flat VAAs at the top of the zone create an elite look that hitters cannot adjust to in time.
          </div>
        </div>

        <div class="glossary-item">
          <span class="glossary-term">FB Velo / Extension</span>
          <div class="glossary-definition">
            We track raw velocity alongside physical extension, or how close to the plate the ball is released. Higher extension creates perceived velocity, making a 95 mph pitch feel closer to 98.
          </div>
        </div>
      </section>

      <section class="glossary-section">
        <h3 class="glossary-section-title">III. Impact & Power Metrics (Hitting)</h3>

        <div class="glossary-item">
          <span class="glossary-term">Avg / Max EV (Exit Velocity)</span>
          <div class="glossary-definition">
            The speed of the ball off the bat. Sudden jumps in Max EV indicate a physical breakthrough in a player’s raw power ceiling.
          </div>
        </div>

        <div class="glossary-item">
          <span class="glossary-term">Barrel-Like %</span>
          <div class="glossary-definition">
            Hits that fall into the optimal launch-angle-plus-velocity bucket. This is one of the strongest leading indicators for upcoming home run production.
          </div>
        </div>

        <div class="glossary-item">
          <span class="glossary-term">EV Burst</span>
          <div class="glossary-definition">
            An automated trigger that fires when a player’s hard-hit profile deviates significantly from their 30-day baseline.
          </div>
        </div>
      </section>

      <section class="glossary-section">
        <h3 class="glossary-section-title">IV. Understanding the Player Trend Cards</h3>

        <div class="glossary-item">
          <span class="glossary-term">7 Day Rolling Trend Analysis</span>
          <div class="glossary-definition">
            Each player card features a 7-day rolling trend view. Read the visual data as a signal context tool, not a full historical chart.
          </div>
        </div>

        <div class="glossary-item">
          <span class="glossary-term">The Baseline</span>
          <div class="glossary-definition">
            The dimmer or gray line represents the player’s established seasonal average. It is the market value of the player.
          </div>
        </div>

        <div class="glossary-item">
          <span class="glossary-term">The Signal</span>
          <div class="glossary-definition">
            The bright Neon Green line tracks the actual data from the last 7 days.
          </div>
        </div>

        <div class="glossary-item">
          <span class="glossary-term">The Edge</span>
          <div class="glossary-definition">
            When you see the Neon Green line pull away from the gray baseline, the system has confirmed a breakout signal. The gap between those two lines is the working edge between what the player is doing now and what the market still thinks they are doing.
          </div>
        </div>
      </section>
    </div>
  </aside>

  <div class="app">
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-card">
          <div class="eyebrow">Executive Terminal</div>
          <h1 class="hero-title">Today’s Signal Wall</h1>
          <p class="hero-copy">
            A live, mobile-first DiamondSignals board built for fast scan readability.
            Institutional dark surfaces, terminal-grade data density, and restrained signal emphasis.
          </p>
        </div>

        <div class="meta-grid">
          <div class="meta-card">
            <div class="meta-label">Last Updated</div>
            <div class="meta-value">{{ generated_at }}</div>
          </div>
          <div class="meta-card">
            <div class="meta-label">Lookback</div>
            <div class="meta-value">28D / 7D Split</div>
          </div>
          <div class="meta-card">
            <div class="meta-label">Alert Threshold</div>
            <div class="meta-value">{{ threshold }}</div>
          </div>
        </div>

        <div class="meta-card slate-heat-card">
          <div class="meta-label">Slate Heat</div>
          <div class="slate-heat-row">
            <div class="slate-heat-bar">
              <div class="slate-heat-fill" style="width: {{ slate_heat }}%;"></div>
            </div>
            <div class="slate-heat-value">{{ slate_heat }}</div>
          </div>
        </div>
      </div>
    </section>

    <section class="board">
      <div class="section">
        <div class="section-head">
          <div>
            <div class="section-kicker">Pitching Board</div>
            <h2 class="section-title">Top 5 Pitchers</h2>
          </div>
          <div class="section-badge">Live Ranked</div>
        </div>

        <div class="cards">
          {% for row in pitchers %}
          <article class="player-card {% if row.edge_score >= 65 %}high-edge{% endif %}">
            <div class="player-top">
              <div class="avatar">{{ row.player_name[:2]|upper }}</div>

              <div class="player-ident">
                <div class="rankline">#{{ loop.index }} Pitcher Trigger</div>
                <h3 class="player-name">{{ row.player_name }}</h3>
                <div class="signal-line">Pitcher // Live Edge Signal // {{ row.sample_note }}</div>
              </div>

              <div class="scorebox">
                <div class="score-label">Edge Score</div>
                <div class="score-value {% if row.edge_score >= 65 %}edge-up{% endif %}">{{ row.edge_score }}</div>
              </div>
            </div>

            <div class="sparkline-wrap">
              <div class="sparkline-head">
                <div class="sparkline-label">7 Day Trend</div>
                <div class="sparkline-note">7D Trend Analysis</div>
              </div>
              <svg class="sparkline" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden="true">
                <defs>
                  <linearGradient id="pitcherGradient{{ loop.index }}" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#444444" stop-opacity="0.65"></stop>
                    <stop offset="100%" stop-color="{% if row.edge_score >= 65 %}#b6ff00{% else %}#00e5ff{% endif %}" stop-opacity="1"></stop>
                  </linearGradient>
                </defs>
                <polyline
                  class="sparkline-path {% if row.trend_glow %}glow{% endif %}"
                  stroke="url(#pitcherGradient{{ loop.index }})"
                  points="{{ row.trend_points }}" />
              </svg>
            </div>

            <div class="badge-row">
              {% for badge in row.badges %}
              <span class="status-badge {{ row.badge_classes[loop.index0] }}">{{ badge }}</span>
              {% endfor %}
            </div>

            <div class="metric-grid">
              <div class="metric">
                <div class="metric-label">{{ row.metric_1_label }}</div>
                <div class="metric-value">{{ row.metric_1 }}</div>
              </div>
              <div class="metric">
                <div class="metric-label">{{ row.metric_2_label }}</div>
                <div class="metric-value">{{ row.metric_2 }}</div>
              </div>
              <div class="metric">
                <div class="metric-label">{{ row.metric_3_label }}</div>
                <div class="metric-value">{{ row.metric_3 }}</div>
              </div>
            </div>

            <div class="why">{{ row.why }}</div>
          </article>
          {% endfor %}
        </div>
      </div>

      <div class="section">
        <div class="section-head">
          <div>
            <div class="section-kicker">Hitting Board</div>
            <h2 class="section-title">Top 5 Hitters</h2>
          </div>
          <div class="section-badge">Live Ranked</div>
        </div>

        <div class="cards">
          {% for row in hitters %}
          <article class="player-card {% if row.edge_score >= 65 %}high-edge{% endif %}">
            <div class="player-top">
              <div class="avatar">{{ row.player_name[:2]|upper }}</div>

              <div class="player-ident">
                <div class="rankline">#{{ loop.index }} Hitter Trigger</div>
                <h3 class="player-name">{{ row.player_name }}</h3>
                <div class="signal-line">Hitter // Live Edge Signal // {{ row.sample_note }}</div>
              </div>

              <div class="scorebox">
                <div class="score-label">Edge Score</div>
                <div class="score-value {% if row.edge_score >= 65 %}edge-up{% endif %}">{{ row.edge_score }}</div>
              </div>
            </div>

            <div class="sparkline-wrap">
              <div class="sparkline-head">
                <div class="sparkline-label">7 Day Trend</div>
                <div class="sparkline-note">7D Trend Analysis</div>
              </div>
              <svg class="sparkline" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden="true">
                <defs>
                  <linearGradient id="hitterGradient{{ loop.index }}" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#444444" stop-opacity="0.65"></stop>
                    <stop offset="100%" stop-color="{% if row.edge_score >= 65 %}#b6ff00{% else %}#00e5ff{% endif %}" stop-opacity="1"></stop>
                  </linearGradient>
                </defs>
                <polyline
                  class="sparkline-path {% if row.edge_score >= 65 %}glow{% endif %}"
                  stroke="url(#hitterGradient{{ loop.index }})"
                  points="{{ row.trend_points }}" />
              </svg>
            </div>

            <div class="badge-row">
              {% for badge in row.badges %}
              <span class="status-badge {{ row.badge_classes[loop.index0] }} {% if badge in ['EV Burst', 'Barrel Jump'] %}active-pulse{% endif %}">{{ badge }}</span>
              {% endfor %}
            </div>

            <div class="metric-grid">
              <div class="metric">
                <div class="metric-label">{{ row.metric_1_label }}</div>
                <div class="metric-value">{{ row.metric_1 }}</div>
              </div>
              <div class="metric">
                <div class="metric-label">{{ row.metric_2_label }}</div>
                <div class="metric-value">{{ row.metric_2 }}</div>
              </div>
              <div class="metric">
                <div class="metric-label">{{ row.metric_3_label }}</div>
                <div class="metric-value">{{ row.metric_3 }}</div>
              </div>
            </div>

            <div class="why">{{ row.why }}</div>
          </article>
          {% endfor %}
        </div>
      </div>
    </section>

    <div class="footer">
      DiamondSignals Signal Wall // Generated During Netlify Build // {{ timezone_label }}
    </div>
  </div>

  <script>
    function openGlossary() {
      const overlay = document.getElementById("glossaryOverlay");
      const drawer = document.getElementById("glossaryDrawer");
      if (overlay) overlay.classList.add("open");
      if (drawer) {
        drawer.classList.add("open");
        drawer.setAttribute("aria-hidden", "false");
      }
      document.body.style.overflow = "hidden";
    }

    function closeGlossary() {
      const overlay = document.getElementById("glossaryOverlay");
      const drawer = document.getElementById("glossaryDrawer");
      if (overlay) overlay.classList.remove("open");
      if (drawer) {
        drawer.classList.remove("open");
        drawer.setAttribute("aria-hidden", "true");
      }
      document.body.style.overflow = "";
    }

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        closeGlossary();
      }
    });
  </script>
</body>
</html>
""")


def render_html(pitchers: pd.DataFrame, hitters: pd.DataFrame) -> str:
    combined = pd.concat([pitchers, hitters], ignore_index=True)
    slate_heat = 0
    if not combined.empty and "edge_score" in combined.columns:
        slate_heat = int(round(combined["edge_score"].head(10).mean()))

    return HTML_TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        threshold=f"{ALERT_THRESHOLD:.0f}+",
        timezone_label=TIMEZONE_LABEL,
        slate_heat=slate_heat,
        nav_html=Template(NAV_TEMPLATE).render(active_nav="signal_wall"),
        pitchers=pitchers.to_dict(orient="records"),
        hitters=hitters.to_dict(orient="records"),
    )


def main() -> None:
    raw = fetch_statcast_window(START_DATE, END_DATE)

    hitter_signals = build_hitter_signals(raw)
    pitcher_signals = build_pitcher_signals(raw)

    if hitter_signals.empty and pitcher_signals.empty:
        raise RuntimeError("No hitter or pitcher signals were produced.")

    top_hitters = hitter_signals.head(5).copy()
    top_pitchers = pitcher_signals.head(5).copy()

    combined_alerts = pd.concat([top_pitchers, top_hitters], ignore_index=True)
    combined_alerts = combined_alerts.sort_values("edge_score", ascending=False).reset_index(drop=True)

    html = render_html(top_pitchers, top_hitters)
    output_path = DIST_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "top_pitchers": top_pitchers[
            [
                "player_name",
                "edge_score",
                "metric_1_label",
                "metric_1",
                "metric_2_label",
                "metric_2",
                "metric_3_label",
                "metric_3",
                "why",
                "badges",
            ]
        ].to_dict(orient="records"),
        "top_hitters": top_hitters[
            [
                "player_name",
                "edge_score",
                "metric_1_label",
                "metric_1",
                "metric_2_label",
                "metric_2",
                "metric_3_label",
                "metric_3",
                "why",
                "badges",
            ]
        ].to_dict(orient="records"),
    }

    (DIST_DIR / "signals.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote dist/signals.json")

    send_telegram_alerts(combined_alerts)


if __name__ == "__main__":
    main()