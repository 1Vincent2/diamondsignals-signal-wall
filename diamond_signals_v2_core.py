#!/usr/bin/env python3
import os
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from jinja2 import Template
from pybaseball import statcast

DIST_DIR = Path("dist")
DIST_DIR.mkdir(parents=True, exist_ok=True)

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
    return str(value).strip()


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


def build_hitter_signals(df: pd.DataFrame) -> pd.DataFrame:
    hitters = df.copy()
    bbe = hitters[hitters["launch_speed"].notna()].copy()
    if bbe.empty:
        return pd.DataFrame()

    bbe["game_date"] = pd.to_datetime(bbe["game_date"])
    bbe["is_recent"] = bbe["game_date"] >= pd.Timestamp(TODAY - timedelta(days=7))
    bbe["is_baseline"] = (bbe["game_date"] < pd.Timestamp(TODAY - timedelta(days=7))) & (
        bbe["game_date"] >= pd.Timestamp(TODAY - timedelta(days=28))
    )

    bbe["barrel_like"] = (
        (pd.to_numeric(bbe["launch_speed"], errors="coerce") >= 98) &
        (pd.to_numeric(bbe["launch_angle"], errors="coerce").between(26, 30, inclusive="both"))
    ).astype(int)

    recent = (
        bbe[bbe["is_recent"]]
        .groupby(["batter", "player_name"], dropna=False)
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
        .groupby(["batter", "player_name"], dropna=False)
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
        on=["batter", "player_name"],
        how="left",
        suffixes=("", "_base")
    )

    merged = merged[merged["recent_bbe"] >= 4].