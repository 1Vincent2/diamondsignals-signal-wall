#!/usr/bin/env python3
import re
import json
from html import escape
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from dashboard.lib.publish_safe import write_temp_output, promote_output_if_valid, save_snapshot
from dashboard.lib.metric_display import metric_title
from dashboard.lib.report_status import build_report_status, utc_now_iso
from dashboard.lib.report_validation import build_validation_report, validate_min_rows

import pandas as pd
import requests
from jinja2 import Template
from pybaseball import playerid_reverse_lookup, statcast

DIST_DIR = Path("dist")
DIST_DIR.mkdir(parents=True, exist_ok=True)

STATUS_DIR = DIST_DIR / "status"
SNAPSHOT_DIR = DIST_DIR / "_snapshots" / "signal-wall"
SIGNAL_WALL_STATUS_PATH = STATUS_DIR / "signal-wall.json"

TEMPLATES_DIR = Path(__file__).parent / "templates"
NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
NAV_V2_TEMPLATE = (TEMPLATES_DIR / "shell_nav_v2.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")
LEDGER_STYLES_TEMPLATE = (TEMPLATES_DIR / "mobile_canary" / "ledger_styles_canary.css").read_text(encoding="utf-8")
HOME_SIGNAL_LEDGER_CARD_TEMPLATE = (TEMPLATES_DIR / "components" / "home_signal_ledger_card_canary.html").read_text(encoding="utf-8")
HOME_SIGNAL_LEDGER_CARD = Template(HOME_SIGNAL_LEDGER_CARD_TEMPLATE)

def metric_tooltip_attr(label) -> str:
    tip = metric_title(label)
    escaped = escape(tip, quote=True)
    return f'data-tooltip="{escaped}" aria-label="{escaped}"'


try:
    HOME_SIGNAL_LEDGER_CARD.globals["metric_tooltip_attr"] = metric_tooltip_attr
except Exception:
    pass

SIGNALS_FRONT_DOOR_TEMPLATE = (TEMPLATES_DIR / "signals_front_door.html").read_text(encoding="utf-8")

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "65"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
SITE_URL = os.getenv("SITE_URL", "").strip()
TIMEZONE_LABEL = os.getenv("TIMEZONE_LABEL", "America/New_York")

TODAY = date.today()
START_DATE = TODAY - timedelta(days=28)
END_DATE = TODAY
RECENT_DAYS = 7
BASELINE_DAYS = 28

SIGNAL_WALL_MODE = "statcast_signal_wall_dynamic_v1_hardened"
SIGNAL_WALL_PIPELINE_LAYERS = [
    "statcast",
    "player_lookup",
    "canonical_dossier",
    "scout_supporting_route",
]
SIGNAL_WALL_HARDENING_NOTES = [
    "hardened_signal_wall_mode:true",
    "dynamic_terms:statcast",
    "dynamic_terms:real_data",
    "canonical_player_id_routes:true",
]

_ID_RESOLUTION_CACHE: dict[str, str] = {}


def resolve_player_id_by_name(name: str) -> str:
    safe = str(name or "").strip()
    if not safe:
        return ""

    cached = _ID_RESOLUTION_CACHE.get(safe)
    if cached is not None:
        return cached

    try:
        import requests

        url = "https://statsapi.mlb.com/api/v1/people/search"
        resp = requests.get(url, params={"names": safe}, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        people = payload.get("people", []) or []

        if people:
            pid = str(people[0].get("id") or "").strip()
            _ID_RESOLUTION_CACHE[safe] = pid
            return pid
    except Exception:
        pass

    _ID_RESOLUTION_CACHE[safe] = ""
    return ""


def backfill_resolved_player_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    if "player_id" in out.columns:
        out["player_id"] = out["player_id"].fillna("").astype(str).str.strip()
    else:
        out["player_id"] = ""

    out["resolved_player_id"] = out["player_id"]

    def resolve_signal_wall_player_name(value) -> str:
        text = "" if pd.isna(value) else str(value).strip()
        if not text:
            return ""

        # Statcast player_name values often arrive as "Last, First".
        # MLB Stats API lookup is much more reliable as "First Last".
        if "," in text:
            last, first = [part.strip() for part in text.split(",", 1)]
            if first and last:
                text = f"{first} {last}"

        return str(resolve_player_id_by_name(text) or "").strip()

    # Signal Wall cards must route to the player's dossier page, not merely the
    # first numeric field present on a Statcast-derived signal row. Some pitcher
    # rows can carry a generic/wrong player_id value that resolves to another
    # player, creating /scout/<id>/ links that are either 404s or wrong dossiers.
    # Prefer normalized player_name resolution when player_name is available.
    if "player_name" in out.columns:
        name_resolved = (
            out["player_name"]
            .fillna("")
            .astype(str)
            .map(resolve_signal_wall_player_name)
            .fillna("")
            .astype(str)
            .str.strip()
        )
        has_name_resolved = name_resolved.ne("")
        out.loc[has_name_resolved, "resolved_player_id"] = name_resolved.loc[has_name_resolved]

    missing_mask = out["resolved_player_id"].fillna("").astype(str).str.strip().eq("")
    if missing_mask.any() and "player_name" in out.columns:
        out.loc[missing_mask, "resolved_player_id"] = (
            out.loc[missing_mask, "player_name"]
            .fillna("")
            .astype(str)
            .map(resolve_signal_wall_player_name)
        )

    out["resolved_player_id"] = out["resolved_player_id"].fillna("").astype(str).str.strip()
    return out

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
    if not text or text.lower() == "unknown":
        return "Unknown"
    return " ".join(part.capitalize() for part in text.split())


def markdown_escape(text: str) -> str:
    specials = r"_*[]()~`>#+-=|{}.!"
    out = str(text)
    for ch in specials:
        out = out.replace(ch, f"\\{ch}")
    return out


def safe_int(value):
    try:
        if pd.notna(value):
            return int(value)
    except Exception:
        pass
    return None


def safe_float(value):
    try:
        if pd.notna(value):
            return float(value)
    except Exception:
        pass
    return None


def build_headshot_url(player_id: int) -> str:
    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"w_180,q_100/v1/people/{player_id}/headshot/67/current"
    )


def fetch_statcast_window(start_dt: date, end_dt: date) -> pd.DataFrame:
    print(f"Fetching Statcast from {start_dt} to {end_dt}...")
    try:
        df = statcast(
            start_dt=start_dt.strftime("%Y-%m-%d"),
            end_dt=end_dt.strftime("%Y-%m-%d"),
        )
    except Exception as exc:
        print(f"Statcast fetch failed: {exc}")
        print("Falling back to existing dist/signals.json build artifacts.")
        return pd.DataFrame()

    if df is None or df.empty:
        print("Statcast returned no data; falling back to existing dist/signals.json build artifacts.")
        return pd.DataFrame()

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

    name_map: dict[int, str] = {}

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


def fill_missing_batter_names_with_statsapi(
    name_map: dict[int, str], batter_ids
) -> dict[int, str]:
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
            url = f"https://statsapi.mlb.com/api/v1/people/{pid}?hydrate=currentTeam"
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



def add_seager_score(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """
    SEAGER v1 = Z-Swing% - O-Swing%.
    Correct aggression: attack pitches in the zone, refuse pitches out of the zone.
    """
    if signals is None or signals.empty or df is None or df.empty:
        return signals

    required = {"batter", "zone", "description"}
    if not required.issubset(set(df.columns)):
        print("SEAGER_SKIPPED_MISSING_COLUMNS:", sorted(required - set(df.columns)))
        signals["seager_score"] = None
        return signals

    work = df.copy()
    work["batter"] = pd.to_numeric(work["batter"], errors="coerce")
    work["zone"] = pd.to_numeric(work["zone"], errors="coerce")

    # Statcast zone 1-9 = rulebook zone grid; 11-14 = chase/waste zones.
    work["in_zone"] = work["zone"].between(1, 9)

    swing_descriptions = {
        "swinging_strike",
        "swinging_strike_blocked",
        "foul",
        "foul_tip",
        "foul_bunt",
        "missed_bunt",
        "bunt_foul_tip",
        "hit_into_play",
        "hit_into_play_no_out",
        "hit_into_play_score",
    }

    work["is_swing"] = work["description"].astype(str).isin(swing_descriptions)

    rows = []
    for batter, g in work.dropna(subset=["batter"]).groupby("batter"):
        z = g[g["in_zone"]]
        o = g[~g["in_zone"]]

        z_swing = float(z["is_swing"].mean()) if len(z) else None
        o_swing = float(o["is_swing"].mean()) if len(o) else None

        score = None if z_swing is None or o_swing is None else round((z_swing - o_swing) * 100, 1)

        rows.append({
            "batter": int(batter),
            "seager_score": score,
            "z_swing_pct": round(z_swing * 100, 1) if z_swing is not None else None,
            "o_swing_pct": round(o_swing * 100, 1) if o_swing is not None else None,
        })

    if not rows:
        signals["seager_score"] = None
        return signals

    seager = pd.DataFrame(rows)
    out = signals.merge(seager, on="batter", how="left")
    print("SEAGER_ATTACHED_HITTERS:", int(out["seager_score"].notna().sum()))
    return out



def build_hitter_signals(df: pd.DataFrame) -> pd.DataFrame:
    hitters = df.copy()

    batter_ids = hitters["batter"].dropna().unique()
    batter_name_map = build_batter_name_map(batter_ids)
    batter_name_map = fill_missing_batter_names_with_statsapi(
        batter_name_map, batter_ids
    )

    pitcher_ids = set(
        pd.to_numeric(df["pitcher"], errors="coerce").dropna().astype(int).tolist()
    )

    bbe = hitters[hitters["launch_speed"].notna()].copy()

    recent_cutoff = pd.Timestamp(TODAY - timedelta(days=RECENT_DAYS))
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
    bbe["is_recent"] = bbe["game_date"] >= pd.Timestamp(
        TODAY - timedelta(days=RECENT_DAYS)
    )
    bbe["is_baseline"] = (
        bbe["game_date"] < pd.Timestamp(TODAY - timedelta(days=RECENT_DAYS))
    ) & (bbe["game_date"] >= pd.Timestamp(TODAY - timedelta(days=BASELINE_DAYS)))

    bbe["barrel_like"] = (
        (pd.to_numeric(bbe["launch_speed"], errors="coerce") >= 98)
        & (
            pd.to_numeric(bbe["launch_angle"], errors="coerce").between(
                26, 30, inclusive="both"
            )
        )
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

    merged = recent.merge(baseline, on=["batter"], how="left", suffixes=("", "_base"))
    merged = merged[
        (merged["recent_bbe"] >= 6) & (merged["recent_max_ev"] >= 95)
    ].copy()

    if merged.empty:
        return pd.DataFrame()

    merged["baseline_bbe"] = merged["baseline_bbe"].fillna(0)
    merged["ev_delta"] = merged["recent_ev"] - merged["baseline_ev"].fillna(
        merged["recent_ev"]
    )
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
        lambda x: batter_name_map.get(int(x), f"Player {int(x)}")
        if pd.notna(x)
        else "Unknown"
    )
    merged["player_name"] = merged["player_name"].apply(safe_name)
    merged["signal_type"] = "Hitter"
    merged["why"] = merged.apply(
        lambda r: (
            f"Avg EV {r['recent_ev']:.1f} mph "
            f"({r['ev_delta']:+.1f} vs baseline), "
            f"barrel-like rate {100 * r['recent_barrel_rate']:.1f}%."
        ),
        axis=1,
    )

    merged["metric_1"] = merged["recent_ev"].round(1)
    merged["metric_1_label"] = "Blast Path"
    merged["metric_2"] = (100 * merged["recent_barrel_rate"]).round(1)
    merged["metric_2_label"] = "Blast Rate"
    merged["metric_3"] = merged["recent_max_ev"].round(1)
    merged["metric_3_label"] = "Apex Damage"
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
        return badges

    def hitter_badge_classes(row: pd.Series) -> list[str]:
        classes = []
        for badge in row["badges"]:
            classes.append(
                "positive"
                if badge in ["EV Burst", "Barrel Jump", "Impact EV"]
                else "neutral"
            )
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
        yvals = (
            [17 for _ in vals]
            if vmax == vmin
            else [26 - ((v - vmin) / (vmax - vmin)) * 16 for v in vals]
        )
        xvals = [0, 20, 40, 60, 80, 100, 120]
        return " ".join(f"{x},{round(y, 1)}" for x, y in zip(xvals, yvals))

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

    pitchers["is_recent"] = pitchers["game_date"] >= pd.Timestamp(
        TODAY - timedelta(days=RECENT_DAYS)
    )
    pitchers["is_baseline"] = (
        pitchers["game_date"] < pd.Timestamp(TODAY - timedelta(days=RECENT_DAYS))
    ) & (pitchers["game_date"] >= pd.Timestamp(TODAY - timedelta(days=BASELINE_DAYS)))

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
        (merged["recent_pitches"] >= 60) & (merged["recent_fb_velo"].fillna(0) >= 90)
    ].copy()

    if merged.empty:
        return pd.DataFrame()

    merged["velo_delta"] = merged["recent_fb_velo"] - merged[
        "baseline_fb_velo"
    ].fillna(merged["recent_fb_velo"])
    merged["whiff_delta"] = merged["recent_whiff_rate"] - merged[
        "baseline_whiff_rate"
    ].fillna(merged["recent_whiff_rate"])
    merged["extension_delta"] = merged["recent_extension"] - merged[
        "baseline_extension"
    ].fillna(merged["recent_extension"])

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
        axis=1,
    )

    merged["metric_1"] = (100 * merged["recent_whiff_rate"]).round(1)
    merged["metric_1_label"] = "Miss Engine"
    merged["metric_2"] = merged["recent_fb_velo"].round(1)
    merged["metric_2_label"] = "Velocity Fuel"
    merged["metric_3"] = merged["recent_extension"].round(1).map(
        lambda x: f"{x:.1f} ft" if pd.notna(x) else "—"
    )
    merged["metric_3_label"] = "Release Deception"
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
        return badges

    def pitcher_badge_classes(row: pd.Series) -> list[str]:
        classes = []
        for badge in row["badges"]:
            classes.append(
                "positive"
                if badge in ["Whiff Lift", "Velo Jump", "Extension Gain"]
                else "neutral"
            )
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
        player_days = recent_daily_whiff[
            recent_daily_whiff["pitcher"] == player_id
        ].copy()
        if player_days.empty:
            return "0,25 20,23 40,21 60,19 80,17 100,15 120,13"

        player_days = player_days.sort_values("game_date")
        vals = player_days["day_whiff"].tolist()

        if len(vals) == 1:
            base = vals[0]
            vals = [
                base * 0.985,
                base * 0.99,
                base * 0.995,
                base,
                base * 1.005,
                base * 1.01,
                base * 1.015,
            ]
        elif len(vals) < 7:
            vals = [vals[0]] * (7 - len(vals)) + vals
        else:
            vals = vals[-7:]

        vmin = min(vals)
        vmax = max(vals)
        yvals = (
            [24, 22, 21, 19, 18, 16, 14]
            if vmax == vmin
            else [26 - ((v - vmin) / (vmax - vmin)) * 16 for v in vals]
        )
        xvals = [0, 20, 40, 60, 80, 100, 120]
        return " ".join(f"{x},{round(y, 1)}" for x, y in zip(xvals, yvals))

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


def extract_player_ids(df: pd.DataFrame) -> list[int]:
    ids = set()
    if "batter" in df.columns:
        for value in df["batter"].dropna().tolist():
            pid = safe_int(value)
            if pid is not None:
                ids.add(pid)
    if "pitcher" in df.columns:
        for value in df["pitcher"].dropna().tolist():
            pid = safe_int(value)
            if pid is not None:
                ids.add(pid)
    return sorted(ids)


def fetch_player_identity(player_id: int) -> Optional[dict]:
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=currentTeam"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
        people = payload.get("people", [])
        if not people:
            return None

        person = people[0]
        current_team = person.get("currentTeam") or {}
        primary_position = person.get("primaryPosition") or {}
        bat_side = person.get("batSide") or {}
        pitch_hand = person.get("pitchHand") or {}
        status = person.get("status") or {}

        return {
            "player_id": player_id,
            "full_name": str(person.get("fullName", "")).strip(),
            "first_name": str(person.get("firstName", "")).strip(),
            "last_name": str(person.get("lastName", "")).strip(),
            "team": str(current_team.get("name", "")).strip(),
            "team_name": str(current_team.get("name", "")).strip(),
            "position": str(primary_position.get("abbreviation", "")).strip(),
            "bats": str(bat_side.get("code", "")).strip(),
            "throws": str(pitch_hand.get("code", "")).strip(),
            "status": str(status.get("description", "")).strip(),
            "headshot_url": build_headshot_url(player_id),
        }
    except Exception:
        return None


def build_player_index(df: pd.DataFrame) -> dict:
    print("Gathering player lookup table. This may take a moment.")
    player_ids = extract_player_ids(df)
    players = []
    for player_id in player_ids:
        identity = fetch_player_identity(player_id)
        if identity:
            players.append(identity)

    players.sort(key=lambda row: (row.get("last_name", ""), row.get("first_name", "")))
    return {"generated_at": datetime.now().isoformat(), "players": players}


def _build_hitter_result_frame(df: pd.DataFrame) -> pd.DataFrame:
    hitters = df.copy()
    hitters["launch_speed"] = pd.to_numeric(
        hitters.get("launch_speed"), errors="coerce"
    )
    hitters["launch_angle"] = pd.to_numeric(
        hitters.get("launch_angle"), errors="coerce"
    )
    hitters["estimated_ba_using_speedangle"] = pd.to_numeric(
        hitters.get("estimated_ba_using_speedangle"), errors="coerce"
    )

    hitters["is_ab"] = hitters["events"].isin(
        [
            "single",
            "double",
            "triple",
            "home_run",
            "field_out",
            "force_out",
            "grounded_into_double_play",
            "fielders_choice_out",
            "strikeout",
            "strikeout_double_play",
            "double_play",
            "triple_play",
            "lineout",
            "flyout",
            "pop_out",
            "groundout",
        ]
    )
    hitters["is_hit"] = hitters["events"].isin(["single", "double", "triple", "home_run"])
    hitters["is_so"] = hitters["events"].isin(["strikeout", "strikeout_double_play"])
    hitters["is_bb"] = hitters["events"].isin(["walk", "intent_walk"])
    hitters["is_sf"] = hitters["events"].eq("sac_fly")

    bbe = hitters[hitters["launch_speed"].notna()].copy()
    if bbe.empty:
        return pd.DataFrame()

    bbe["hard_hit"] = (bbe["launch_speed"] >= 95).astype(int)
    bbe["sweet_spot"] = bbe["launch_angle"].between(8, 32, inclusive="both").astype(int)
    bbe["barrel_like"] = (
        (bbe["launch_speed"] >= 98)
        & (bbe["launch_angle"].between(26, 30, inclusive="both"))
    ).astype(int)

    bbe_grouped = (
        bbe.groupby("batter", dropna=False)
        .agg(
            avg_ev=("launch_speed", "mean"),
            max_ev=("launch_speed", "max"),
            hard_hit_pct=("hard_hit", "mean"),
            sweet_spot_pct=("sweet_spot", "mean"),
            barrel_pct=("barrel_like", "mean"),
            launch_angle=("launch_angle", "mean"),
            xba=("estimated_ba_using_speedangle", "mean"),
            bbe_count=("launch_speed", "size"),
        )
        .reset_index()
    )

    result_grouped = (
        hitters.groupby("batter", dropna=False)
        .agg(
            ab=("is_ab", "sum"),
            hits=("is_hit", "sum"),
            strikeouts=("is_so", "sum"),
            walks=("is_bb", "sum"),
            sac_fly=("is_sf", "sum"),
            pa=("batter", "size"),
        )
        .reset_index()
    )

    return bbe_grouped.merge(result_grouped, on="batter", how="left")


def build_scout_hitter_metrics(df: pd.DataFrame) -> dict:
    merged = _build_hitter_result_frame(df)
    if merged.empty:
        return {}

    out = {}
    for _, row in merged.iterrows():
        player_id = safe_int(row.get("batter"))
        if player_id is None:
            continue

        ab = int(row["ab"]) if pd.notna(row["ab"]) else 0
        hits = int(row["hits"]) if pd.notna(row["hits"]) else 0
        strikeouts = int(row["strikeouts"]) if pd.notna(row["strikeouts"]) else 0
        pa = int(row["pa"]) if pd.notna(row["pa"]) else 0

        avg = (hits / ab) if ab > 0 else None
        k_rate = (strikeouts / pa * 100.0) if pa > 0 else None
        xba = safe_float(row.get("xba"))
        xba_delta = (xba - avg) if (xba is not None and avg is not None) else None

        avg_ev = safe_float(row.get("avg_ev"))
        max_ev = safe_float(row.get("max_ev"))
        hard_hit_pct = safe_float(row.get("hard_hit_pct"))
        hard_hit_pct = (hard_hit_pct * 100.0) if hard_hit_pct is not None else None
        sweet_spot_pct = safe_float(row.get("sweet_spot_pct"))
        sweet_spot_pct = (sweet_spot_pct * 100.0) if sweet_spot_pct is not None else None
        barrel_pct = safe_float(row.get("barrel_pct"))
        barrel_pct = (barrel_pct * 100.0) if barrel_pct is not None else None
        launch_angle = safe_float(row.get("launch_angle"))

        if xba_delta is not None and xba_delta > 0.100 and (
            max_ev is not None and max_ev > 110
        ):
            signal_label = "PLATINUM BUY"
        elif xba_delta is not None and xba_delta > 0.050:
            signal_label = "GOLD BUY"
        elif xba_delta is not None and xba_delta < -0.050:
            signal_label = "CAUTION"
        else:
            signal_label = "NEUTRAL"

        parts = []

        if avg is not None and xba is not None and xba_delta is not None:
            if xba_delta >= 0.100:
                parts.append(
                    f"He materially outperformed his surface AVG quality this week. "
                    f"The profile shows AVG {avg:.3f} against xBA {xba:.3f}, a strong +{xba_delta:.3f} gap that points to better underlying contact than the box score suggests."
                )
            elif xba_delta >= 0.050:
                parts.append(
                    f"He showed positive separation between results and contact quality this week. "
                    f"AVG came in at {avg:.3f} with xBA at {xba:.3f}, a +{xba_delta:.3f} edge that supports a constructive buy signal."
                )
            elif xba_delta <= -0.050:
                parts.append(
                    f"His surface AVG ran ahead of the quality-of-contact profile this week. "
                    f"AVG was {avg:.3f} versus xBA {xba:.3f}, a {xba_delta:.3f} gap that suggests some caution."
                )
            else:
                parts.append(
                    f"His results and contact-quality profile were mostly in line this week, "
                    f"with AVG at {avg:.3f} and xBA at {xba:.3f}."
                )

        if max_ev is not None:
            if max_ev >= 110:
                parts.append(
                    f"The top-end damage was real, with a max exit velocity of {max_ev:.1f} mph, which is premium impact territory."
                )
            elif max_ev >= 105:
                parts.append(
                    f"He still showed legitimate impact ability, reaching {max_ev:.1f} mph at peak exit velocity."
                )
            else:
                parts.append(
                    f"The max exit velocity checked in at {max_ev:.1f} mph, which was more solid than explosive."
                )

        if barrel_pct is not None and hard_hit_pct is not None:
            parts.append(
                f"The contact mix was supported by a {barrel_pct:.1f}% barrel rate and {hard_hit_pct:.1f}% hard-hit rate over the recent window."
            )
        elif barrel_pct is not None:
            parts.append(
                f"Barrel rate registered at {barrel_pct:.1f}% over the recent window."
            )
        elif hard_hit_pct is not None:
            parts.append(
                f"Hard-hit rate registered at {hard_hit_pct:.1f}% over the recent window."
            )

        if signal_label == "PLATINUM BUY":
            parts.append(
                "Net assessment: Platinum Buy. The past week looks stronger under the hood than the surface line, with enough impact authority to support a bullish forward read."
            )
        elif signal_label == "GOLD BUY":
            parts.append(
                "Net assessment: Gold Buy. The recent profile supports a positive read, even if it was not a full top-tier explosion across every category."
            )
        elif signal_label == "CAUTION":
            parts.append(
                "Net assessment: Caution. Recent results appear less trustworthy once the underlying contact quality is weighed."
            )
        else:
            parts.append(
                "Net assessment: Neutral. There are some usable signs here, but not enough separation yet for a stronger conviction label."
            )

        out[str(player_id)] = {
            "player_type": "hitter",
            "ballistics": {
                "label_1": "Avg Exit Velo",
                "value_1": avg_ev,
                "label_2": "Max Exit Velo",
                "value_2": max_ev,
                "label_3": "Hard Hit %",
                "value_3": hard_hit_pct,
                "label_4": "Diamond Delta",
                "value_4": xba_delta,
            },
            "movement": {
                "label_1": "Sweet Spot %",
                "value_1": sweet_spot_pct,
                "label_2": "Barrel %",
                "value_2": barrel_pct,
                "label_3": "Launch Angle",
                "value_3": launch_angle,
                "label_4": "xBA",
                "value_4": xba,
            },
            "results": {
                "label_1": "Batting Avg",
                "value_1": avg,
                "label_2": "K Rate",
                "value_2": k_rate,
                "label_3": "wRC+",
                "value_3": None,
                "label_4": "Signal",
                "value_4": signal_label,
            },
            "briefing": " ".join(parts) if parts else "Live hitter profile loaded.",
        }

    return out


def build_scout_pitcher_metrics(df: pd.DataFrame) -> dict:
    pitchers = df.copy()
    if pitchers.empty or "pitcher" not in pitchers.columns:
        return {}

    pitchers["game_date"] = pd.to_datetime(pitchers["game_date"])
    pitchers["release_speed"] = pd.to_numeric(
        pitchers.get("release_speed"), errors="coerce"
    )
    pitchers["release_extension"] = pd.to_numeric(
        pitchers.get("release_extension"), errors="coerce"
    )
    pitchers["release_spin_rate"] = pd.to_numeric(
        pitchers.get("release_spin_rate"), errors="coerce"
    )
    pitchers["pfx_z"] = pd.to_numeric(pitchers.get("pfx_z"), errors="coerce")
    pitchers["is_whiff"] = pitchers["description"].isin(
        ["swinging_strike", "swinging_strike_blocked"]
    ).astype(int)
    pitchers["is_so"] = pitchers["events"].isin(
        ["strikeout", "strikeout_double_play"]
    ).astype(int)
    pitchers["is_bb"] = pitchers["events"].isin(["walk", "intent_walk"]).astype(int)
    pitchers["is_pa_event"] = pitchers["events"].notna().astype(int)

    fastballs = {"FF", "FT", "SI", "FC"}
    sliders = {"SL", "ST"}

    recent_cutoff = pd.Timestamp(TODAY - timedelta(days=RECENT_DAYS))
    recent = pitchers[pitchers["game_date"] >= recent_cutoff].copy()
    if recent.empty:
        return {}

    recent["fastball_speed"] = recent["release_speed"].where(
        recent["pitch_type"].isin(fastballs)
    )
    recent["slider_spin"] = recent["release_spin_rate"].where(
        recent["pitch_type"].isin(sliders)
    )
    recent["ivb_inches"] = (recent["pfx_z"] * 12.0).where(
        recent["pitch_type"].isin(fastballs)
    )

    grouped = (
        recent.groupby("pitcher", dropna=False)
        .agg(
            fb_avg_velo=("fastball_speed", "mean"),
            fb_max_velo=("fastball_speed", "max"),
            extension=("release_extension", "mean"),
            slider_spin=("slider_spin", "mean"),
            ivb=("ivb_inches", "mean"),
            whiff_rate=("is_whiff", "mean"),
            strikeouts=("is_so", "sum"),
            walks=("is_bb", "sum"),
            pa=("is_pa_event", "sum"),
            pitch_count=("pitch_type", "size"),
        )
        .reset_index()
    )

    out = {}
    for _, row in grouped.iterrows():
        player_id = safe_int(row.get("pitcher"))
        if player_id is None:
            continue

        fb_avg_velo = safe_float(row.get("fb_avg_velo"))
        fb_max_velo = safe_float(row.get("fb_max_velo"))
        extension = safe_float(row.get("extension"))
        slider_spin = safe_float(row.get("slider_spin"))
        ivb = safe_float(row.get("ivb"))
        whiff_rate = safe_float(row.get("whiff_rate"))
        whiff_pct = (whiff_rate * 100.0) if whiff_rate is not None else None
        pa = int(row.get("pa")) if pd.notna(row.get("pa")) else 0
        strikeouts = int(row.get("strikeouts")) if pd.notna(row.get("strikeouts")) else 0
        walks = int(row.get("walks")) if pd.notna(row.get("walks")) else 0
        k_rate = (strikeouts / pa * 100.0) if pa > 0 else None
        kbb = ((strikeouts - walks) / pa * 100.0) if pa > 0 else None

        if fb_avg_velo is not None and fb_avg_velo >= 97.5 and (
            whiff_rate is not None and whiff_rate >= 0.40
        ):
            signal_label = "APEX POWER"
        elif whiff_rate is not None and whiff_rate >= 0.32:
            signal_label = "TIER 1 ACE"
        else:
            signal_label = "LIVE PROFILE"

        parts = []

        if fb_avg_velo is not None:
            if fb_avg_velo >= 97.5:
                parts.append(
                    f"He carried premium fastball power this week, averaging {fb_avg_velo:.1f} mph."
                )
            elif fb_avg_velo >= 95:
                parts.append(
                    f"Fastball velocity held in a strong working range at {fb_avg_velo:.1f} mph."
                )
            else:
                parts.append(
                    f"Fastball velocity averaged {fb_avg_velo:.1f} mph, giving the profile more of a command-and-shape read than pure overpowering force."
                )

        if extension is not None:
            parts.append(
                f"Release extension averaged {extension:.1f} feet, which helps frame how the raw velocity may be playing to hitters."
            )

        if ivb is not None:
            parts.append(
                f"Fastball IVB checked in at {ivb:.1f} inches, adding needed context to the carry profile."
            )

        if whiff_pct is not None:
            if whiff_pct >= 40:
                parts.append(
                    f"The whiff output was dominant at {whiff_pct:.1f}%, which supports a true bat-missing profile."
                )
            elif whiff_pct >= 30:
                parts.append(
                    f"Whiff rate came in at {whiff_pct:.1f}%, strong enough to support a high-end swing-and-miss read."
                )
            else:
                parts.append(
                    f"Whiff rate was {whiff_pct:.1f}%, useful but not overwhelming."
                )

        if signal_label == "APEX POWER":
            parts.append(
                "Net assessment: Apex Power. The recent arsenal shape and bat-missing output support a premium power-pitcher read."
            )
        elif signal_label == "TIER 1 ACE":
            parts.append(
                "Net assessment: Tier 1 Ace. The recent mix supports a high-conviction starter profile with real dominance traits."
            )
        else:
            parts.append(
                "Net assessment: Live Profile. There are actionable traits here, but not enough separation yet for the top labels."
            )

        out[str(player_id)] = {
            "player_type": "pitcher",
            "ballistics": {
                "label_1": "FB Avg Velo",
                "value_1": fb_avg_velo,
                "label_2": "FB Max Velo",
                "value_2": fb_max_velo,
                "label_3": "Extension",
                "value_3": extension,
                "label_4": "Pitch Count",
                "value_4": safe_int(row.get("pitch_count")),
            },
            "movement": {
                "label_1": "Slider Spin",
                "value_1": slider_spin,
                "label_2": "FB IVB",
                "value_2": ivb,
                "label_3": "VAA",
                "value_3": None,
                "label_4": "Movement Edge",
                "value_4": None,
            },
            "results": {
                "label_1": "K %",
                "value_1": k_rate,
                "label_2": "Whiff %",
                "value_2": whiff_pct,
                "label_3": "K-BB %",
                "value_3": kbb,
                "label_4": "Signal",
                "value_4": signal_label,
            },
            "briefing": " ".join(parts) if parts else "Live pitcher profile loaded.",
        }

    return out


def write_scout_metrics(df: pd.DataFrame) -> dict:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "players": {
            **build_scout_hitter_metrics(df),
            **build_scout_pitcher_metrics(df),
        },
    }
    (DIST_DIR / "scout_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("Wrote dist/scout_metrics.json")
    return payload



def load_promotion_watch_dossier_players() -> list[dict]:
    """
    Local dossier supplement for current Promotion Watch candidates.

    Purpose:
    - Promotion Watch can surface fresh AAA/AA/A/D1 candidates before they exist
      in the main MLB Statcast-derived player_index or Supabase supplemental universe.
    - The Scout Dossier shell should still generate a valid player page for those
      candidates when Promotion Watch has a resolved MLBAM/player id.
    - This does not invent metrics. It creates identity + context scaffold records
      so the dossier can hydrate whatever support/scout metrics are available later.
    """
    path = DIST_DIR / "typical-call-up" / "promotion_watch.json"
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[DOSSIER_SUPPLEMENT] promotion watch unavailable: {exc}")
        return []

    top = payload.get("top_signals") or {}
    if not isinstance(top, dict):
        return []

    sections = [
        "pitchers_72hr",
        "hitters_72hr",
        "pitchers_14day",
        "hitters_14day",
        "recent_arrivals",
        "depth_radar",
    ]

    supplemental: list[dict] = []
    seen: set[str] = set()

    for section in sections:
        rows = top.get(section) or []
        if not isinstance(rows, list):
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue

            pid = safe_int(
                row.get("resolved_player_id")
                or row.get("player_id")
                or row.get("mlbam_id")
                or row.get("id")
            )
            if pid is None:
                continue

            pid_key = str(pid)
            if pid_key in seen:
                continue
            seen.add(pid_key)

            player_name = (
                str(row.get("player_name") or row.get("name") or "").strip()
                or f"Player {pid}"
            )

            team_value = (
                str(
                    row.get("team")
                    or row.get("team_name")
                    or row.get("display_team")
                    or row.get("display_org")
                    or row.get("org")
                    or row.get("affiliate")
                    or ""
                ).strip()
            )

            level_value = str(row.get("level") or "").strip()
            if not level_value:
                if section == "depth_radar":
                    level_value = "AA_A_D1_SURVEILLANCE"
                elif "72hr" in section or "14day" in section:
                    level_value = "AAA_PROMOTION_WATCH"
                else:
                    level_value = "PROMOTION_WATCH"

            position_value = (
                str(
                    row.get("position")
                    or row.get("position_group")
                    or row.get("player_type")
                    or row.get("role")
                    or ""
                ).strip()
            )

            supplemental.append(
                {
                    "player_id": pid,
                    "full_name": player_name,
                    "first_name": "",
                    "last_name": "",
                    "team": team_value,
                    "team_name": team_value,
                    "level": level_value,
                    "position": position_value,
                    "bats": "",
                    "throws": "",
                    "age": None,
                    "status": f"PROMOTION_WATCH_SUPPLEMENTAL_UNIVERSE:{section}",
                    "headshot_url": build_headshot_url(pid),
                    "promotion_watch_section": section,
                    "promotion_watch_score": row.get("edge_score") or row.get("live_score"),
                    "promotion_watch_why": row.get("why") or "",
                    "promotion_watch_source": row.get("source_badge") or "",
                }
            )

    print(f"[DOSSIER_SUPPLEMENT] promotion watch players loaded: {len(supplemental)}")
    return supplemental


def load_supplemental_milb_dossier_players() -> list[dict]:
    """
    Disabled for the isolated mobile live canary route.

    The canary route is a static visual/reference build and must not use
    private Supabase service-role environment variables or supplemental
    database reads. Normal production builders remain responsible for any
    private-data-backed supplemental identity loading.
    """
    print("[MOBILE_LIVE_CANARY_ROUTE_V1] supplemental MiLB dossier loading disabled")
    return []

def load_player_signal_context_lookup() -> dict[str, dict]:
    """Load season-context + signal metrics keyed by canonical player_id for Performance Audit support tiles."""
    path = DIST_DIR / "admin" / "player_signal_index.json"
    if not path.exists():
        print("[DOSSIER_CONTEXT] missing player_signal_index.json")
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[DOSSIER_CONTEXT] unable to read player_signal_index.json: {exc}")
        return {}

    lookup: dict[str, dict] = {}
    players = payload.get("players", []) if isinstance(payload, dict) else []

    for player in players:
        pid = str(player.get("player_id") or "").strip()
        if not pid or pid.lower() == "none":
            continue

        ctx = player.get("season_context") if isinstance(player.get("season_context"), dict) else None
        signals = player.get("signals") if isinstance(player.get("signals"), list) else []

        merged_metrics = {}
        for sig in signals:
            metrics = sig.get("metrics") if isinstance(sig, dict) else None
            if isinstance(metrics, dict):
                merged_metrics.update({k: v for k, v in metrics.items() if v not in (None, "")})

        lookup[pid] = {
            "season_context": ctx,
            "signals": signals,
            "support_metrics": build_support_metrics(ctx, merged_metrics, player),
        }

    print(f"[DOSSIER_CONTEXT] loaded support context for {len(lookup)} players")
    return lookup





def build_scout_support_metrics(scout: Optional[dict]) -> Optional[dict]:
    """Fallback support strip for any player with scout_metrics.

    Reads the normalized scout_metrics zone shape:
    ballistics/movement/results -> label_1/value_1 ... label_4/value_4.
    """
    if not scout:
        return None

    ballistics = scout.get("ballistics") or {}
    movement = scout.get("movement") or {}
    results = scout.get("results") or {}
    player_type = str(scout.get("player_type") or "").lower()

    def zone_lookup(zone: dict) -> dict:
        out = {}
        for i in range(1, 5):
            label = zone.get(f"label_{i}")
            value = zone.get(f"value_{i}")
            if label and value not in (None, "", "—", "--"):
                out[str(label).strip().upper()] = value
        return out

    merged = {}
    for zone in (ballistics, movement, results):
        merged.update(zone_lookup(zone))

    def get(label: str):
        return merged.get(label.upper())

    def fmt(value, suffix: str = ""):
        if value in (None, "", "—", "--"):
            return "—"
        if isinstance(value, (int, float)):
            if suffix == "%":
                return f"{value:.1f}%"
            if "VELO" in suffix:
                return f"{value:.1f}"
            if suffix == "FT":
                return f"{value:.1f} ft"
            if suffix == "IN":
                return f"{value:.1f} in"
            return f"{value:.1f}"
        return str(value)

    if player_type == "pitcher":
        tiles = [
            {"label": "FB AVG VELO", "value": fmt(get("FB Avg Velo"))},
            {"label": "FB MAX VELO", "value": fmt(get("FB Max Velo"))},
            {"label": "FB IVB", "value": fmt(get("FB IVB"), "IN")},
            {"label": "K%", "value": fmt(get("K %"), "%")},
            {"label": "WHIFF%", "value": fmt(get("Whiff %"), "%")},
        ]
    else:
        tiles = [
            {"label": "AVG EV", "value": fmt(get("Avg Exit Velo"))},
            {"label": "MAX EV", "value": fmt(get("Max Exit Velo"))},
            {"label": "HARD HIT%", "value": fmt(get("Hard Hit %"), "%")},
            {"label": "BARREL%", "value": fmt(get("Barrel %"), "%")},
            {"label": "XBA", "value": fmt(get("xBA"))},
        ]

    tiles = [t for t in tiles if t["value"] != "—"]
    return {"profile": "SCOUT_PROFILE", "tiles": tiles[:5]} if tiles else None



def build_promotion_watch_context(player: Optional[dict]) -> Optional[dict]:
    """
    Build a compact, evidence-preserving context block for Promotion Watch candidates.

    This does not fabricate scout metrics. It preserves the active Promotion Watch
    signal that caused the player to enter the dossier universe, so the scout page
    can explain why the page exists even before full Statcast/scout_metrics hydrate.
    """
    player = player if isinstance(player, dict) else {}

    section = str(player.get("promotion_watch_section") or "").strip()
    score = player.get("promotion_watch_score")
    why = str(player.get("promotion_watch_why") or "").strip()
    source = str(player.get("promotion_watch_source") or "").strip()
    level = str(player.get("level") or "").strip()
    status = str(player.get("status") or "").strip()

    if not any([section, score not in (None, ""), why, source, level, status.startswith("PROMOTION_WATCH")]):
        return None

    section_label = section.replace("_", " ").upper() if section else "PROMOTION WATCH"
    tiles = [
        {"label": "SOURCE", "value": section_label},
        {"label": "SCORE", "value": score if score not in (None, "") else "—"},
        {"label": "LEVEL", "value": level or "—"},
        {"label": "FEED", "value": source or "—"},
    ]

    return {
        "profile": "PROMOTION_WATCH_CONTEXT",
        "section": section,
        "section_label": section_label,
        "score": score,
        "why": why,
        "source": source,
        "level": level,
        "tiles": [tile for tile in tiles if tile["value"] != "—"],
    }


def build_support_metrics(ctx: Optional[dict], metrics: Optional[dict], player: Optional[dict] = None) -> dict:
    """Normalize support metrics for all Performance Audit cards."""
    ctx = ctx if isinstance(ctx, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    player = player if isinstance(player, dict) else {}

    def pick(*keys, default="—"):
        for source in (ctx, metrics, player):
            for key in keys:
                val = source.get(key)
                if val not in (None, ""):
                    return val
        return default

    context_type = str(
        pick("season_context", "context", "profile_type", default="")
    ).upper()

    position = str(player.get("position") or metrics.get("position") or "").upper()
    is_hitter = context_type == "PLATE_DISCIPLINE" or any(pos in position for pos in ["B", "OF", "DH", "SS", "C"])

    if is_hitter:
        return {
            "profile": "PLATE_DISCIPLINE",
            "tiles": [
                {"label": "SEAGER", "value": pick("seager_score")},
                {"label": "BABIP", "value": pick("babip")},
                {"label": "BB/K", "value": pick("bb_k_ratio")},
                {"label": "BB%", "value": pick("bb_pct")},
                {"label": "K%", "value": pick("k_pct")},
            ],
        }

    return {
        "profile": "COMMAND_PROFILE",
        "tiles": [
            {"label": "K/BB", "value": pick("k_bb_ratio")},
            {"label": "BABIP", "value": pick("babip")},
            {"label": "K%", "value": pick("k_pct")},
            {"label": "BB%", "value": pick("bb_pct")},
            {"label": "BF", "value": pick("batters_faced")},
        ],
    }


def write_dossier_canon(
    df: pd.DataFrame,
    hitter_signals: pd.DataFrame,
    pitcher_signals: pd.DataFrame,
) -> dict:
    player_index_payload = build_player_index(df)
    player_rows = (
        player_index_payload.get("players", [])
        if isinstance(player_index_payload, dict)
        else []
    )
    supplemental_players = load_supplemental_milb_dossier_players()
    supplemental_players.extend(load_promotion_watch_dossier_players())


    waiver_path = DIST_DIR / "waiver_wire.json"
    if waiver_path.exists():
        try:
            waiver_payload = json.loads(waiver_path.read_text(encoding="utf-8"))
            waiver_rows = waiver_payload.get("all_assets", []) or waiver_payload.get("assets", [])

            for row in waiver_rows:
                pid = safe_int(row.get("player_id"))
                if pid is None:
                    continue

                supplemental_players.append({
                    "player_id": pid,
                    "full_name": str(row.get("player_name") or "").strip() or f"Player {pid}",
                    "first_name": "",
                    "last_name": "",
                    "team": str(row.get("team") or "").strip(),
                    "team_name": str(row.get("team") or "").strip(),
                    "level": "MLB_OR_WAIVER",
                    "position": str(row.get("position") or "").strip(),
                    "bats": "",
                    "throws": "",
                    "status": "WAIVER_SUPPLEMENTAL_UNIVERSE",
                    "headshot_url": build_headshot_url(pid),
                })

            print(f"[DOSSIER_SUPPLEMENT] waiver players considered: {len(waiver_rows)}")
        except Exception as exc:
            print(f"[DOSSIER_SUPPLEMENT] waiver supplemental unavailable: {exc}")

    promotion_watch_context_lookup = {
        str(p.get("player_id") or "").strip(): build_promotion_watch_context(p)
        for p in supplemental_players
        if str(p.get("player_id") or "").strip()
        and str(p.get("status") or "").startswith("PROMOTION_WATCH")
        and build_promotion_watch_context(p)
    }

    existing_ids = {
        str(p.get("player_id") or "").strip()
        for p in player_rows
        if str(p.get("player_id") or "").strip()
    }

    for extra_player in supplemental_players:
        extra_id = str(extra_player.get("player_id") or "").strip()
        if extra_id and extra_id not in existing_ids:
            player_rows.append(extra_player)
            existing_ids.add(extra_id)

    scout_metrics = {
        **build_scout_hitter_metrics(df),
        **build_scout_pitcher_metrics(df),
    }

    player_signal_context_lookup = load_player_signal_context_lookup()

    signal_lookup: dict[str, dict] = {}

    if not hitter_signals.empty:
        for _, row in hitter_signals.iterrows():
            pid = safe_int(row.get("batter"))
            if pid is not None:
                signal_lookup[str(pid)] = {
                    "canonical_score": safe_float(row.get("edge_score")),
                    "canonical_score_label": "Edge Score",
                    "trend_points_7d": row.get("trend_points") or "",
                    "trend_note": "7-Day Trend",
                }

    if not pitcher_signals.empty:
        for _, row in pitcher_signals.iterrows():
            pid = safe_int(row.get("pitcher"))
            if pid is not None:
                signal_lookup[str(pid)] = {
                    "canonical_score": safe_float(row.get("edge_score")),
                    "canonical_score_label": "Edge Score",
                    "trend_points_7d": row.get("trend_points") or "",
                    "trend_note": "7-Day Trend",
                }

    canon_players: dict[str, dict] = {}

    for player in player_rows:
        player_id = str(player.get("player_id") or "").strip()
        if not player_id:
            continue

        scout = scout_metrics.get(player_id, {})
        signal = signal_lookup.get(player_id, {})
        signal_context = player_signal_context_lookup.get(player_id, {})

        canon_players[player_id] = {
            "player_id": player_id,
            "player_name": player.get("full_name") or "Unknown Player",
            "team": player.get("team") or "",
            "current_team": player.get("team") or "",
            "position": player.get("position") or "",
            "age": player.get("age"),
            "bats": player.get("bats") or "",
            "throws": player.get("throws") or "",
            "headshot_url": player.get("headshot_url") or "",
            "canonical_score": signal.get("canonical_score"),
            "canonical_score_label": signal.get("canonical_score_label"),
            "trend_points_7d": signal.get("trend_points_7d", ""),
            "trend_note": signal.get("trend_note"),
            "rostered_by_user": False,
            "season_context": signal_context.get("season_context"),
            "signals": signal_context.get("signals", []),
            "support_metrics": signal_context.get("support_metrics") or build_scout_support_metrics(scout),
            "promotion_watch_context": build_promotion_watch_context(player) or promotion_watch_context_lookup.get(player_id),
        }

        if scout:
            canon_players[player_id]["scout_metrics"] = scout

    payload = {
        "generated_at": datetime.now().isoformat(),
        "players": canon_players,
    }

    (DIST_DIR / "dossier_canon.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print("Wrote dist/dossier_canon.json")
    return payload


def copy_static_assets() -> None:
    js_src = Path("src/js/player-search.js")
    js_dest = DIST_DIR / "player-search.js"
    if js_src.exists():
        js_dest.write_text(js_src.read_text(encoding="utf-8"), encoding="utf-8")
        print("Wrote dist/player-search.js")

    actions_src = Path("src/js/player-card-actions.js")
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
  <title>DiamondSignals — Signal Wall</title>
  <style>
/* DIAMONDSIGNALS_INSTANT_METRIC_TOOLTIP_V1
   Fast desktop-only metric tooltips for Signal Wall.
   Mobile layout/menu behavior intentionally untouched.
*/
@media screen and (min-width: 981px) {
  [data-tooltip] {
    position: relative;
    cursor: help;
  }

  [data-tooltip]::after {
    content: attr(data-tooltip);
    position: absolute;
    left: 50%;
    bottom: calc(100% + 10px);
    transform: translateX(-50%) translateY(2px);
    z-index: 9999;
    width: max-content;
    max-width: 280px;
    padding: 9px 11px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(6,10,18,0.96);
    color: rgba(255,255,255,0.94);
    box-shadow:
      0 12px 28px rgba(0,0,0,0.42),
      0 0 0 1px rgba(255,255,255,0.04);
    font-family: var(--mono);
    font-size: 10px;
    line-height: 1.35;
    letter-spacing: 0;
    text-transform: none;
    font-weight: 800;
    white-space: normal;
    text-align: left;
    opacity: 0;
    pointer-events: none;
    transition: opacity 70ms ease, transform 70ms ease;
  }

  [data-tooltip]::before {
    content: "";
    position: absolute;
    left: 50%;
    bottom: calc(100% + 4px);
    transform: translateX(-50%);
    z-index: 10000;
    border: 6px solid transparent;
    border-top-color: rgba(6,10,18,0.96);
    opacity: 0;
    pointer-events: none;
    transition: opacity 70ms ease;
  }

  [data-tooltip]:hover::after,
  [data-tooltip]:focus-visible::after {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }

  [data-tooltip]:hover::before,
  [data-tooltip]:focus-visible::before {
    opacity: 1;
  }
}

    :root {
      --bg: #080808;
      --card-radial: radial-gradient(circle at top left, #1a1a1a 0%, #080808 100%);
      --border: #2d2d2d;
      --text: #f0f0f0;
      --muted: #71717a;
      --soft: #a1a1aa;
      --tiny: #8a8a93;
      --emerald: #4ade80;
      --lime-hot: #b6ff00;
      --cyan-hot: #00e5ff;
      --blue: #6aa6ff;
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
        radial-gradient(circle at top right, rgba(239,68,68,0.04), transparent 20%),
        linear-gradient(180deg, #101010 0%, #080808 34%, #050505 100%);
      line-height: 1.35;
    }
    .topbar { position: sticky; top: 0; z-index: 50; background: rgba(8,8,8,0.90); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(255,255,255,0.05); }
    .topbar-inner, .app { width: min(1180px, calc(100% - 24px)); margin: 0 auto; }
    .topbar-inner { min-height: 62px; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 0; }
    .brand { display: flex; align-items: center; gap: 10px; }
    .brand-mark { width: 11px; height: 11px; border-radius: 999px; background: var(--lime-hot); box-shadow: 0 0 10px rgba(182,255,0,0.35); }
    .brand-kicker { font-size: 10px; line-height: 1; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 800; margin-bottom: 4px; }
    .brand-white { color: var(--text); }
    .brand-blue { color: var(--blue); }
    .brand-title { font-size: 16px; line-height: 1.05; letter-spacing: -0.02em; font-weight: 800; }
    .info-trigger { height: 34px; border-radius: 999px; border: 1px solid rgba(182,255,0,0.22); background: rgba(255,255,255,0.05); color: var(--text); display: inline-flex; align-items: center; justify-content: center; padding: 0 12px; font-family: var(--mono); font-size: 12px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; cursor: pointer; box-shadow: 0 0 10px rgba(182,255,0,0.08); }
    .livebox { text-align: right; }
    .live-label { display: inline-flex; align-items: center; gap: 7px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.16em; color: var(--lime-hot); font-weight: 800; margin-bottom: 4px; }
    .live-dot { width: 7px; height: 7px; border-radius: 999px; background: var(--lime-hot); box-shadow: 0 0 10px rgba(182,255,0,0.35); }
    .live-time { font-family: var(--mono); font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
    .glossary-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.52); opacity: 0; pointer-events: none; transition: opacity 0.22s ease; z-index: 80; }
    .glossary-overlay.open { opacity: 1; pointer-events: auto; }
    .glossary-drawer { position: fixed; top: 0; right: 0; width: min(560px,100vw); height: 100vh; background: linear-gradient(180deg, #101010 0%, #080808 100%); border-left: 1px solid rgba(255,255,255,0.08); box-shadow: -12px 0 40px rgba(0,0,0,0.42); transform: translateX(100%); transition: transform 0.24s ease; z-index: 90; display: flex; flex-direction: column; }
    .glossary-drawer.open { transform: translateX(0); }
    .glossary-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; padding: 18px; border-bottom: 1px solid rgba(255,255,255,0.06); background: rgba(255,255,255,0.02); }
    .glossary-kicker { font-size: 10px; line-height: 1; letter-spacing: 0.16em; text-transform: uppercase; color: var(--blue); font-weight: 800; margin-bottom: 8px; }
    .glossary-title { margin: 0; font-size: 20px; line-height: 1.05; letter-spacing: -0.03em; text-transform: uppercase; font-weight: 900; color: var(--text); }
    .glossary-close { width: 34px; height: 34px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.10); background: rgba(255,255,255,0.03); color: var(--text); display: inline-flex; align-items: center; justify-content: center; font-size: 18px; cursor: pointer; }
    .glossary-body { overflow-y: auto; padding: 18px; display: grid; gap: 18px; }

    
    /* SIGNAL_WALL_MOBILE_FIELD_GUIDE_BOTTOM_SHEET_REPAIR_V12
       Mobile-only: Field Guide becomes a bottom sheet with an obvious Back to Report control.
       Desktop drawer behavior remains unchanged. */
    .mobile-back-label {
      display: none;
    }
    .desktop-close-symbol {
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    @media (max-width: 640px) {
      body.mobile-glossary-open,
      body.mobile-field-guide-active {
        overflow: hidden !important;
        touch-action: none;
      }

      body .glossary-overlay {
        z-index: 120 !important;
        background: rgba(0,0,0,0.62) !important;
        backdrop-filter: blur(4px);
      }

      body .glossary-drawer {
        top: auto !important;
        right: 8px !important;
        left: 8px !important;
        bottom: 0 !important;
        width: auto !important;
        height: min(78dvh, 680px) !important;
        max-height: calc(100dvh - 72px) !important;
        border-left: 1px solid rgba(255,255,255,0.10) !important;
        border-right: 1px solid rgba(255,255,255,0.10) !important;
        border-top: 1px solid rgba(182,255,0,0.18) !important;
        border-bottom: 0 !important;
        border-radius: 24px 24px 0 0 !important;
        transform: translateY(calc(100% + 22px)) !important;
        transition: transform 220ms ease, opacity 180ms ease !important;
        z-index: 130 !important;
        box-shadow:
          0 -18px 50px rgba(0,0,0,0.58),
          0 0 24px rgba(182,255,0,0.08) !important;
        overflow: hidden !important;
      }

      body .glossary-drawer.open {
        transform: translateY(0) !important;
      }

      body .glossary-head {
        position: sticky !important;
        top: 0 !important;
        z-index: 2 !important;
        align-items: center !important;
        padding: 14px 14px 12px !important;
        background:
          linear-gradient(180deg, rgba(16,16,16,0.98), rgba(10,10,10,0.94)) !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
      }

      body .glossary-head::before {
        content: "";
        position: absolute;
        left: 50%;
        top: 7px;
        width: 46px;
        height: 4px;
        border-radius: 999px;
        transform: translateX(-50%);
        background: rgba(255,255,255,0.22);
      }

      body .glossary-kicker {
        margin-top: 8px !important;
        font-size: 8px !important;
        letter-spacing: 0.16em !important;
      }

      body .glossary-title {
        font-size: 18px !important;
      }

      body .glossary-close {
        width: auto !important;
        min-width: 132px !important;
        height: 36px !important;
        padding: 0 13px !important;
        border-radius: 999px !important;
        border-color: rgba(182,255,0,0.24) !important;
        background: rgba(182,255,0,0.08) !important;
        color: rgba(248,250,252,0.96) !important;
        font-family: var(--mono) !important;
        font-size: 10px !important;
        font-weight: 900 !important;
        letter-spacing: 0.10em !important;
        text-transform: uppercase !important;
      }

      body .glossary-close .desktop-close-symbol {
        display: none !important;
      }

      body .glossary-close .mobile-back-label {
        display: inline-flex !important;
        align-items: center;
        justify-content: center;
        white-space: nowrap;
      }

      body .glossary-body {
        padding: 14px 14px calc(22px + env(safe-area-inset-bottom)) !important;
        overflow-y: auto !important;
        -webkit-overflow-scrolling: touch;
      }

      body.mobile-glossary-open .player-card.mobile-card-open,
      body.mobile-field-guide-active .player-card.mobile-card-open {
        opacity: 0.52;
        pointer-events: none;
      }

      body.mobile-glossary-open .info-trigger,
      body.mobile-field-guide-active .info-trigger {
        opacity: 0 !important;
        pointer-events: none !important;
      }
    }
    .glossary-section { border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; background: rgba(255,255,255,0.02); padding: 14px; }
    .glossary-section-title { margin: 0 0 12px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--lime-hot); font-weight: 800; font-family: var(--mono); }
    .glossary-item { margin-bottom: 12px; }
    .glossary-term { display: block; margin-bottom: 4px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text); font-weight: 800; font-family: var(--mono); }
    .glossary-definition { font-size: 13px; line-height: 1.5; color: var(--soft); }
    .app { padding: 18px 0 34px; }
    .hero, .board { display: grid; gap: 16px; }
    .hero-grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
    .hero-card, .meta-card, .section, .player-card { background: var(--card-radial); border: 0.5px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); position: relative; overflow: hidden; }
    .hero-card::before, .meta-card::before, .section::before, .player-card::before { content: ""; position: absolute; inset: 0; pointer-events: none; border-radius: inherit; padding: 0.5px; background: linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.01)); -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0); -webkit-mask-composite: xor; mask-composite: exclude; opacity: 0.55; }
    .hero-card { padding: 18px; }
    .eyebrow { font-size: 10px; line-height: 1; letter-spacing: 0.18em; text-transform: uppercase; color: var(--blue); font-weight: 800; margin-bottom: 10px; }
    .hero-title { margin: 0 0 10px; font-size: clamp(34px, 4.8vw, 56px); line-height: 0.96; letter-spacing: -0.04em; font-weight: 900; text-transform: uppercase; color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; text-shadow: 0 0 10px rgba(255,255,255,0.03); }
    .hero-copy { margin: 0; max-width: 760px; color: var(--soft); font-size: 14px; }
    .hero-audit-copy {
      margin-top: 12px;
      color: rgba(255,255,255,0.82);
      font-weight: 700;
    }
    .hero-audit-label {
      display: inline-block;
      margin-right: 8px;
      color: var(--lime-hot);
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 900;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    .meta-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .meta-card { padding: 14px; }
    .meta-label, .metric-label, .sparkline-label, .section-kicker, .score-label, .rankline, .status-badge { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 800; }
    .meta-label { margin-bottom: 6px; }
    .meta-value { font-family: var(--mono); font-size: 13px; color: var(--text); font-variant-numeric: tabular-nums; }
    .section-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 16px 16px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); background: linear-gradient(180deg, rgba(214,164,58,0.06), rgba(255,255,255,0.01)); }
    .section-title { margin: 0; font-size: 18px; font-weight: 800; letter-spacing: -0.02em; text-transform: uppercase; }
    .section-badge { font-family: var(--mono); font-size: 11px; color: #d7dbe6; border: 1px solid rgba(255,255,255,0.12); border-radius: 999px; padding: 7px 10px; background: rgba(255,255,255,0.04); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02); }
    .cards { display: grid; gap: 10px; padding: 10px; }
    .player-card { padding: 14px; transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease; }
    .player-card.js-player-card { cursor: pointer; }
    .player-card.js-player-card:hover { transform: translateY(-1px); box-shadow: var(--shadow), 0 0 12px rgba(106,166,255,0.08); border-color: rgba(106,166,255,0.18); }
    .player-card.high-edge { border-color: rgba(74,222,128,0.22); box-shadow: var(--shadow), 0 0 8px rgba(74,222,128,0.07); }
    .player-top { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px; align-items: start; margin-bottom: 12px; }
    .player-head { display: flex; align-items: start; gap: 18px; min-width: 0; }
    .avatar { width: 42px; height: 42px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.10); display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.03); color: var(--text); font-size: 13px; font-weight: 800; flex: 0 0 auto; }
    .player-ident { min-width: 0; max-width: 100%; }
    .rankline { margin-bottom: 4px; }
    .player-name { font-size: clamp(18px, 1.55vw, 26px); line-height: 1; letter-spacing: -0.03em; font-weight: 800; margin: 0 0 4px; text-transform: uppercase; color: var(--text); word-break: break-word; overflow-wrap: anywhere; }
    .signal-line { font-size: 10px; color: var(--soft); font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.10em; }
    .provision-row-desktop { display: flex; align-items: center; justify-content: flex-start; min-width: 240px; padding-top: 6px; }
    .provision-row-mobile { display: none; }
    .scorebox { display: flex; align-items: flex-start; justify-content: flex-end; gap: 8px; min-width: 120px; flex: 0 0 120px; padding-top: 2px; }
    .score-meta { display: flex; flex-direction: column; align-items: flex-end; justify-content: center; text-align: right; gap: 3px; min-width: 74px; flex: 0 0 auto; }
    .score-label { font-size: 9px; letter-spacing: 0.08em; }
    .score-value { font-family: var(--sans); font-size: 36px; line-height: 0.92; font-weight: 900; font-style: italic; letter-spacing: -0.05em; color: var(--hg-amber); text-shadow: 0 0 8px rgba(214,164,58,0.14); }
    .score-value.edge-up { color: var(--hg-lime); text-shadow: 0 0 10px rgba(183,240,0,0.22); }
    .sparkline-wrap { margin: 0 0 12px; padding: 8px 10px; border: 1px solid rgba(255,255,255,0.04); border-radius: 12px; background: rgba(255,255,255,0.015); }
    .sparkline-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
    .sparkline-note { font-family: var(--mono); font-size: 10px; color: var(--tiny); text-transform: uppercase; letter-spacing: 0.1em; }
    svg.sparkline { display: block; width: 100%; height: 34px; }
    .sparkline-path { stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; fill: none; filter: drop-shadow(0 0 2px rgba(173, 255, 47, 0.4)); }
    .sparkline-path.glow { filter: drop-shadow(0 0 3px rgba(183,240,0,0.40)); }
    .metric-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-bottom: 12px; }
    .metric { border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 10px 10px 9px; background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.018)); min-width: 0; }
    .metric-value { font-family: var(--mono); font-size: 16px; line-height: 1.1; color: var(--text); font-weight: 800; word-break: break-word; font-variant-numeric: tabular-nums; }
    .metric-value.value-edge { color: #D4FF00; text-shadow: 0 0 8px rgba(212,255,0,0.16); }
    .metric-value.value-ballistics { color: #4A90E2; text-shadow: 0 0 8px rgba(74,144,226,0.14); }
    .metric-value.value-pulse { color: var(--text); }
    .metric-value.value-apex { color: #BB86FC; text-shadow: 0 0 8px rgba(187,134,252,0.14); }
    .metric-label { font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 800; opacity: 1; color: #A0A0A0; }
    .metric-label.label-alpha { color: #A0A0A0; text-shadow: none; }
    .metric-label.label-ballistics { color: #A0A0A0; text-shadow: none; }
    .metric-label.label-pulse { color: #A0A0A0; }
    .metric-label.label-apex { color: #A0A0A0; text-shadow: none; }
    .badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 12px; }
    .status-badge { line-height: 1; border-radius: 999px; padding: 7px 9px; border: 1px solid rgba(255,255,255,0.08); color: var(--soft); background: rgba(255,255,255,0.02); font-family: var(--mono); font-variant-numeric: tabular-nums; }
    .status-badge.positive { color: var(--hg-lime); border-color: rgba(183,240,0,0.22); box-shadow: 0 0 8px rgba(183,240,0,0.10); background: rgba(183,240,0,0.05); }
    .status-badge.neutral { color: #cfd4df; border-color: rgba(255,255,255,0.12); background: rgba(255,255,255,0.04); }
    .status-badge.apex-status { color: #BB86FC; border-color: rgba(187,134,252,0.22); background: rgba(187,134,252,0.06); box-shadow: 0 0 8px rgba(187,134,252,0.10); }
    .status-badge.active-pulse { animation: badgePulse 2.2s infinite ease-in-out; }
    .why { font-size: 11px; line-height: 1.55; color: var(--soft); font-family: var(--mono); font-variant-numeric: tabular-nums; }
    .provision-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 0 18px;
      border-radius: 10px;
      border: 1px solid rgba(96,165,250,0.28);
      background: rgba(18,18,18,0.94);
      color: white;
      font-family: var(--mono);
      font-size: 8px;
      font-weight: 800;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      cursor: pointer;
      transition: 160ms ease;
      white-space: nowrap;
    }
    .provision-btn:hover {
      background: rgba(24,24,24,0.98);
      border-color: rgba(96,165,250,0.40);
      transform: translateY(-1px);
      box-shadow: 0 0 14px rgba(59,130,246,0.12);
    }
    .footer { padding: 16px 4px 0; color: var(--muted); font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; font-variant-numeric: tabular-nums; }
    @keyframes badgePulse { 0%,100% { opacity: 0.82; } 50% { opacity: 1; } }


    /* Signal Wall split-board layout: preserve existing board markup, render Pitchers left / Hitters right */
    @media (min-width: 1100px) {
      .app {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 18px;
        align-items: start;
      }

      .app > .hero {
        grid-column: 1 / -1;
      }

      .app > .board {
        grid-column: 1 / -1;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 18px;
        margin-top: 0;
        min-width: 0;
      }

      .app > .board .player-card {
        min-width: 0;
      }

      .app > .board .player-top,
      .app > .board .player-head {
        min-width: 0;
      }

      .app > .board .player-name {
        font-size: clamp(22px, 1.95vw, 34px);
        line-height: 0.95;
      }

      .app > .board .metric-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .app > .board .season-context-grid {
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }
    }

    @media (max-width: 1099px) {
      .app {
        display: block;
      }

      .app > .board + .board {
        margin-top: 18px;
      }
    }

    {{ shell_styles | safe }}

      /* SIGNAL_WALL_MOBILE_NAV_FIXED_LOCK_V2
         Signal Wall page-level override: keep mobile command nav persistent above all live-page layers. */
      @media screen and (max-width: 760px) {
        body {
          padding-top: 58px !important;
        }

        .topnav.ds-shell-nav {
          position: fixed !important;
          top: 0 !important;
          left: 0 !important;
          right: 0 !important;
          width: 100vw !important;
          z-index: 2147483000 !important;
          transform: none !important;
          pointer-events: auto !important;
        }

        .topnav.ds-shell-nav .topnav-inner {
          position: relative !important;
          z-index: 2147483001 !important;
        }

        .ds-mobile-menu-trigger {
          position: relative !important;
          z-index: 2147483002 !important;
          pointer-events: auto !important;
        }

        .ds-mobile-menu-backdrop {
          position: fixed !important;
          inset: 0 !important;
          z-index: 2147483003 !important;
        }

        .ds-mobile-menu-drawer {
          position: fixed !important;
          top: 0 !important;
          right: 0 !important;
          bottom: 0 !important;
          z-index: 2147483004 !important;
          max-height: 100dvh !important;
          overflow-y: auto !important;
          -webkit-overflow-scrolling: touch !important;
        }

        body.ds-mobile-menu-open {
          overflow: hidden !important;
        }
      }

    {{ ledger_styles | safe }}
        @media (max-width: 640px) {
      .topbar-inner, .app, .topnav-inner, .search-strip-inner { width: min(100%, calc(100% - 16px)); }
      .search-strip-inner { justify-content: stretch; }
      .player-search { width: 100%; }
      .player-search-input { height: 36px; font-size: 12px; }

      .brand-title {
        font-size: 15px;
        line-height: 1.02;
        letter-spacing: -0.03em;
        font-weight: 900;
      }

      .hero-title { font-size: 34px; letter-spacing: -0.035em; }
      .meta-grid { grid-template-columns: 1fr; }

      .player-card {
        padding: 14px;
      }

      .player-top {
        grid-template-columns: auto minmax(0, 1fr);
        gap: 10px;
        margin-bottom: 8px;
      }

      .player-head {
        display: contents;
      }

      .provision-row-desktop {
        display: none;
      }

      .provision-row-mobile {
        display: block;
      }

      .avatar {
        width: 42px;
        height: 42px;
        font-size: 13px;
      }

      .player-ident {
        min-width: 0;
        max-width: 100%;
      }

      .player-name {
        display: block;
        width: 100%;
        font-size: 26px;
        line-height: 0.96;
        letter-spacing: -0.04em;
        font-weight: 900;
        text-transform: uppercase;
        word-break: normal;
        overflow-wrap: break-word;
        margin: 0 0 6px;
      }

      .signal-line {
        font-size: 11px;
        line-height: 1.3;
        letter-spacing: 0.05em;
      }
    
      .scorebox {
        width: 100%;
        min-width: 0;
        flex: 0 0 auto;
        display: flex;
        align-items: flex-start;
        justify-content: flex-start;
        gap: 12px;
        margin: 0 0 8px;
      }

      .score-meta {
        min-width: 0;
        flex: 0 0 auto;
        align-items: flex-start;
        text-align: left;
      }

      .score-label {
        font-size: 9px;
        letter-spacing: 0.10em;
      }

      .score-value {
        font-family: var(--sans);
        font-size: 48px;
        line-height: 0.88;
        font-weight: 900;
        font-style: italic;
        letter-spacing: -0.06em;
      }

      .provision-row {
        width: 100%;
        margin: 0 0 12px;
      }

      .provision-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        min-width: 100%;
        min-height: 46px;
        padding: 0 16px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.04em;
        white-space: nowrap;
        border: 1px solid rgba(96,165,250,0.28);
        background: rgba(18,18,18,0.94);
        color: white;
        box-shadow: 0 0 14px rgba(59,130,246,0.12);
      }

      .badge-row {
        gap: 8px;
      }

      .metric-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
      }

      .metric {
        padding: 10px 10px 9px;
      }

      .metric-label {
        font-size: 9px;
        letter-spacing: 0.06em;
      }

      .metric-value {
        font-size: 16px;
        line-height: 1.05;
      }

      .player-search-result { grid-template-columns: 120px 1fr; }
      .player-search-avatar { width: 120px; height: 120px; }
    }

    /* FINAL SIGNAL WALL FIELD GUIDE FLOAT NORMALIZATION */
    .glossary-btn,
    .guide-btn,
    .field-guide-btn,
    .field-guide-pill,
    button[onclick*="Glossary"],
    button[onclick*="glossary"],
    button[onclick*="Guide"],
    button[onclick*="guide"] {
      position: fixed !important;
      right: max(22px, calc((100vw - 1180px) / 2 + 22px)) !important;
      bottom: 22px !important;
      top: auto !important;
      left: auto !important;
      z-index: 95 !important;
      border-radius: 999px !important;
      padding: 12px 18px !important;
      border: 1px solid rgba(204,255,0,.28) !important;
      background: rgba(5,7,10,.88) !important;
      color: #ffffff !important;
      box-shadow: 0 0 22px rgba(204,255,0,.10), inset 0 1px 0 rgba(255,255,255,.08) !important;
      backdrop-filter: blur(14px) !important;
      font-family: var(--mono, "JetBrains Mono", "Roboto Mono", monospace) !important;
      font-size: 11px !important;
      font-weight: 900 !important;
      letter-spacing: .16em !important;
      text-transform: uppercase !important;
    }

    .topbar-center,
    .header-center,
    .glossary-center {
      display: none !important;
    }

    @media (max-width: 1280px) {
      .glossary-btn,
      .guide-btn,
      .field-guide-btn,
      .field-guide-pill,
      button[onclick*="Glossary"],
      button[onclick*="glossary"],
      button[onclick*="Guide"],
      button[onclick*="guide"] {
        right: 22px !important;
        bottom: 22px !important;
      }
    }

    @media (max-width: 640px) {
      .glossary-btn,
      .guide-btn,
      .field-guide-btn,
      .field-guide-pill,
      button[onclick*="Glossary"],
      button[onclick*="glossary"],
      button[onclick*="Guide"],
      button[onclick*="guide"] {
        right: 14px !important;
        bottom: 14px !important;
      }
    }




  
      .signal-audit-access-strip {
        margin: 22px auto 24px;
        padding: 14px 18px;
        max-width: 1280px;
        border: 1px solid rgba(182, 255, 0, 0.34);
        border-radius: 18px;
        background:
          linear-gradient(135deg, rgba(182, 255, 0, 0.11), rgba(0, 229, 255, 0.05)),
          rgba(8, 14, 18, 0.82);
        box-shadow:
          0 0 28px rgba(182, 255, 0, 0.10),
          inset 0 0 0 1px rgba(255,255,255,0.035);
        display: flex;
        align-items: center;
        gap: 13px;
      }

      .signal-audit-access-strip .audit-dot {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        background: #b6ff00;
        box-shadow: 0 0 18px rgba(182, 255, 0, 0.95);
        flex: 0 0 auto;
      }

      .signal-audit-access-strip .audit-copy {
        display: grid;
        gap: 3px;
      }

      .signal-audit-access-strip .audit-title {
        color: #b6ff00;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }

      .signal-audit-access-strip .audit-subtitle {
        color: rgba(255,255,255,0.78);
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.02em;
      }

      @media (max-width: 760px) {
        .signal-audit-access-strip {
          margin: 14px 12px 18px;
          padding: 12px 13px;
          border-radius: 15px;
          align-items: flex-start;
        }

        .signal-audit-access-strip .audit-title {
          font-size: 10px;
          letter-spacing: 0.14em;
        }

        .signal-audit-access-strip .audit-subtitle {
          font-size: 12px;
          line-height: 1.35;
        }
      }

    
    


    


    
    /* SIGNAL_WALL_MOBILE_FIELD_GUIDE_MENU_TWIN_V16
       True fix:
       Field Guide is inserted into the same parent as the mobile Menu pill.
       It behaves like Menu because it lives with Menu.
       No independent top/right/bottom positioning. No scroll drift. */
    @media (max-width: 640px) {
      body .info-trigger.ds-original-field-guide-hidden {
        display: none !important;
      }

      body .ds-mobile-menu-twin-host {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        gap: 12px !important;
        overflow: visible !important;
      }

      body .ds-mobile-menu-twin-host .ds-field-guide-menu-twin {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;

        height: 42px !important;
        min-width: 156px !important;
        width: auto !important;
        padding: 0 18px !important;
        margin: 0 !important;

        border-radius: 999px !important;
        border: 1px solid rgba(182,255,0,0.30) !important;
        background:
          radial-gradient(circle at 18% 0%, rgba(182,255,0,0.16), transparent 52%),
          rgba(5,7,10,0.92) !important;
        color: rgba(248,250,252,0.98) !important;
        box-shadow:
          0 10px 28px rgba(0,0,0,0.38),
          0 0 18px rgba(182,255,0,0.11),
          inset 0 1px 0 rgba(255,255,255,0.09) !important;
        backdrop-filter: blur(14px) !important;

        font-family: var(--mono) !important;
        font-size: 10px !important;
        font-weight: 900 !important;
        letter-spacing: 0.13em !important;
        line-height: 1 !important;
        text-transform: uppercase !important;
        white-space: nowrap !important;

        -webkit-appearance: none !important;
        appearance: none !important;
        cursor: pointer !important;
        -webkit-tap-highlight-color: transparent !important;
      }

      body .ds-mobile-menu-twin-host .ds-field-guide-menu-twin .ds-field-guide-icon {
        font-size: 15px !important;
        line-height: 1 !important;
        color: var(--hg-lime, #b6ff00) !important;
        text-shadow: 0 0 12px rgba(182,255,0,0.28) !important;
      }

      body.mobile-glossary-open .ds-field-guide-menu-twin,
      body.mobile-field-guide-active .ds-field-guide-menu-twin {
        opacity: 0.38 !important;
        pointer-events: none !important;
      }

      body .livebox,
      body .live-label,
      body .live-time {
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
      }
    }

    /* SIGNAL_WALL_MOBILE_TWIN_PILL_STYLE_POLISH_V18
       Keep the approved twin layout, but make Field Guide match the Menu pill's
       stronger border/glow language. Signal Wall mobile-only. */
    @media (max-width: 640px) {
      body .ds-mobile-menu-twin-host .ds-field-guide-menu-twin {
        border-color: rgba(182,255,0,0.48) !important;
        box-shadow:
          0 10px 28px rgba(0,0,0,0.38),
          0 0 22px rgba(182,255,0,0.20),
          0 0 44px rgba(182,255,0,0.08),
          inset 0 1px 0 rgba(255,255,255,0.11) !important;
      }

      body .ds-mobile-menu-twin-host .ds-field-guide-menu-twin .ds-field-guide-icon {
        text-shadow:
          0 0 12px rgba(182,255,0,0.42),
          0 0 22px rgba(182,255,0,0.20) !important;
      }
    }


    
    /* SIGNAL_WALL_MOBILE_HIDE_ORIGINAL_FIELD_GUIDE_TRIGGER_V17
       Keep only the Field Guide twin beside Menu on mobile.
       Hide the original topbar/hero Field Guide trigger so it cannot duplicate near LIVE. */
    @media (max-width: 640px) {
      body button.info-trigger[onclick="openGlossary()"]:not(.ds-field-guide-menu-twin),
      body .topbar button.info-trigger[onclick="openGlossary()"]:not(.ds-field-guide-menu-twin),
      body:has(.hero-title) button.info-trigger[onclick="openGlossary()"]:not(.ds-field-guide-menu-twin),
      body .info-trigger.ds-original-field-guide-hidden {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        min-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
      }

      body .ds-field-guide-menu-twin {
        display: inline-flex !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
      }
    }

/* SIGNAL_WALL_DESKTOP_LIVE_RAIL_REPAIR_V1
       Desktop-only /live rail repair.
       Keeps the accepted pro nav, centers the hero/meta zone, and aligns the board rail.
       Scoped to Signal Wall's body lock only. */
    @media screen and (min-width: 761px) {
      body.signal-wall-v2-typography-lock .topbar-inner,
      body.signal-wall-v2-typography-lock .ds-pro-desktop-nav-inner,
      body.signal-wall-v2-typography-lock .app {
        width: min(1120px, calc(100vw - 96px)) !important;
        max-width: 1120px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        box-sizing: border-box !important;
      }

      body.signal-wall-v2-typography-lock .app {
        display: block !important;
        padding-top: 28px !important;
        padding-bottom: 40px !important;
        overflow: visible !important;
      }

      body.signal-wall-v2-typography-lock .app > section.hero {
        width: 100% !important;
        max-width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        box-sizing: border-box !important;
      }

      body.signal-wall-v2-typography-lock .hero-card {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
      }

      body.signal-wall-v2-typography-lock .meta-grid {
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
        gap: 12px !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
      }

      body.signal-wall-v2-typography-lock .meta-card {
        min-width: 0 !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board {
        width: 100% !important;
        max-width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
        gap: 18px !important;
        align-items: start !important;
        box-sizing: border-box !important;
        overflow: visible !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board > .section {
        min-width: 0 !important;
        width: auto !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .cards,
      body.signal-wall-v2-typography-lock .app > section.board .player-card {
        min-width: 0 !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
      }

      /* SIGNAL_WALL_DESKTOP_CARD_FIREWALL_V6
         Desktop-only firewall.
         Purpose: preserve approved mobile ledger/card CSS unchanged while preventing
         mobile compact-row card behavior from taking over desktop /live.
         Scope: Signal Wall body lock + desktop breakpoint only. */
      body.signal-wall-v2-typography-lock .app > section.board .mobile-scan-row,
      body.signal-wall-v2-typography-lock .app > section.board .mobile-drawer-close,
      body.signal-wall-v2-typography-lock .app > section.board .mobile-signal-toggle,
      body.signal-wall-v2-typography-lock .app > section.board .provision-row-mobile {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-card,
      body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open),
      body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open {
        padding: 14px !important;
        height: auto !important;
        min-height: 0 !important;
        overflow: hidden !important;
        background: var(--card-radial) !important;
        border: 0.5px solid var(--border) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow) !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-card .player-top,
      body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .player-top,
      body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .player-top {
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) auto !important;
        gap: 18px !important;
        align-items: start !important;
        margin-bottom: 12px !important;
        visibility: visible !important;
        opacity: 1 !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-head {
        display: flex !important;
        align-items: start !important;
        gap: 18px !important;
        min-width: 0 !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .scorebox {
        display: flex !important;
        align-items: flex-start !important;
        justify-content: flex-end !important;
        gap: 8px !important;
        min-width: 120px !important;
        flex: 0 0 120px !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .provision-row-desktop {
        display: flex !important;
        visibility: visible !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .mobile-signal-detail,
      body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .mobile-signal-detail,
      body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .mobile-signal-detail {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        margin-top: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .mobile-signal-detail > summary {
        display: none !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .mobile-signal-tray,
      body.signal-wall-v2-typography-lock .app > section.board .mobile-signal-detail:not([open]) > .mobile-signal-tray,
      body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .mobile-signal-tray {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        animation: none !important;
      }


      /* SIGNAL_WALL_DESKTOP_CARD_CONTENT_RESTORE_V7
         Desktop-only add-on to V6.
         V6 restored the desktop card shell; V7 restores the hidden audit content
         that the compact/mobile ledger rules still suppress on wide screens. */
      body.signal-wall-v2-typography-lock .app > section.board .sparkline-wrap {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        margin: 0 0 12px !important;
        padding: 8px 10px !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .badge-row {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 6px !important;
        visibility: visible !important;
        opacity: 1 !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        margin: 0 0 12px !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .metric-grid {
        display: grid !important;
        grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
        gap: 8px !important;
        visibility: visible !important;
        opacity: 1 !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        margin: 0 0 12px !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .metric {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        min-height: 0 !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .season-context-strip {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        margin: 0 0 12px !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .season-context-grid {
        display: grid !important;
        grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
        gap: 7px !important;
        visibility: visible !important;
        opacity: 1 !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .why {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        margin-top: 10px !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .provision-row-desktop {
        grid-column: 1 / -1 !important;
        width: 100% !important;
        min-width: 0 !important;
        margin: 10px 0 0 !important;
        padding-top: 0 !important;
        justify-content: flex-start !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .provision-row-desktop .provision-btn {
        width: min(260px, 100%) !important;
        max-width: 260px !important;
        min-width: 0 !important;
      }

      
      /* SIGNAL_WALL_DESKTOP_TRACKING_BUTTON_LAYOUT_V9
         Desktop-only: move Initiate Tracking out of the identity/score collision zone.
         Keeps approved mobile expandable-card layout untouched. */
      body.signal-wall-v2-typography-lock .app > section.board .player-top {
        grid-template-columns: minmax(0, 1fr) auto !important;
        grid-auto-flow: row !important;
        align-items: start !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-head {
        grid-column: 1 / 2 !important;
        grid-row: 1 !important;
        min-width: 0 !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .scorebox {
        grid-column: 2 / 3 !important;
        grid-row: 1 !important;
        align-self: start !important;
        justify-self: end !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-top > .provision-row-desktop {
        grid-column: 1 / -1 !important;
        grid-row: 2 !important;
        position: static !important;
        inset: auto !important;
        transform: none !important;
        z-index: auto !important;
        width: 100% !important;
        min-width: 0 !important;
        margin: 12px 0 0 !important;
        padding: 0 !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-head .provision-row-desktop {
        flex: 0 0 100% !important;
        width: 100% !important;
        min-width: 0 !important;
        order: 99 !important;
        margin: 10px 0 0 !important;
        position: static !important;
        transform: none !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .provision-row-desktop .provision-btn {
        width: min(220px, 100%) !important;
        max-width: 220px !important;
        min-width: 0 !important;
        height: 38px !important;
        min-height: 38px !important;
        padding: 0 18px !important;
        position: static !important;
        transform: none !important;
        white-space: nowrap !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .mobile-signal-detail {
        margin-top: 12px !important;
      }


      /* SIGNAL_WALL_DESKTOP_CARD_HEADER_GRID_V10
         Desktop-only: stop tracking button from colliding with player name.
         Creates a stable desktop card header grid while leaving mobile ledger CSS untouched. */
      body.signal-wall-v2-typography-lock .app > section.board .player-card {
        padding: 16px !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-top {
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) 112px !important;
        grid-template-rows: auto auto !important;
        column-gap: 18px !important;
        row-gap: 10px !important;
        align-items: start !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-head {
        grid-column: 1 / 2 !important;
        grid-row: 1 / 2 !important;
        display: grid !important;
        grid-template-columns: 46px minmax(0, 1fr) !important;
        grid-template-rows: auto auto !important;
        column-gap: 14px !important;
        row-gap: 8px !important;
        align-items: start !important;
        min-width: 0 !important;
        width: 100% !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .avatar {
        grid-column: 1 / 2 !important;
        grid-row: 1 / 3 !important;
        width: 42px !important;
        height: 42px !important;
        min-width: 42px !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-ident {
        grid-column: 2 / 3 !important;
        grid-row: 1 / 2 !important;
        min-width: 0 !important;
        max-width: 100% !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-name {
        max-width: 100% !important;
        overflow-wrap: normal !important;
        word-break: normal !important;
        white-space: normal !important;
        line-height: 0.94 !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .player-head .provision-row-desktop,
      body.signal-wall-v2-typography-lock .app > section.board .player-top > .provision-row-desktop {
        grid-column: 2 / 3 !important;
        grid-row: 2 / 3 !important;
        display: flex !important;
        position: static !important;
        inset: auto !important;
        transform: none !important;
        z-index: auto !important;
        width: 100% !important;
        min-width: 0 !important;
        margin: 2px 0 0 !important;
        padding: 0 !important;
        justify-content: flex-start !important;
        align-items: center !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .provision-row-desktop .provision-btn {
        width: min(210px, 100%) !important;
        max-width: 210px !important;
        min-width: 0 !important;
        height: 34px !important;
        min-height: 34px !important;
        padding: 0 16px !important;
        font-size: 9px !important;
        letter-spacing: 0.12em !important;
        position: static !important;
        inset: auto !important;
        transform: none !important;
        white-space: nowrap !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .scorebox {
        grid-column: 2 / 3 !important;
        grid-row: 1 / 2 !important;
        justify-self: end !important;
        align-self: start !important;
        min-width: 96px !important;
        flex: 0 0 96px !important;
      }

      body.signal-wall-v2-typography-lock .app > section.board .mobile-signal-detail {
        margin-top: 14px !important;
        clear: both !important;
      }


    }

    @media screen and (min-width: 761px) and (max-width: 1100px) {
      body.signal-wall-v2-typography-lock .app > section.board {
        grid-template-columns: minmax(0, 1fr) !important;
      }

      body.signal-wall-v2-typography-lock .meta-grid {
        grid-template-columns: minmax(0, 1fr) !important;

      }
    }

</style>
</head>
<body class="signal-wall-v2-typography-lock">
  <div class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="brand-mark"></div>
        <div class="brand-text">
          <div class="brand-kicker"><span class="brand-white">DIAMOND</span><span class="brand-blue">SIGNALS</span></div>
          <div class="brand-title">Signal Wall // Institutional Elite</div>
        </div>
      </div>
      <button class="info-trigger" type="button" onclick="openGlossary()" aria-label="Open field guide">ⓘ Field Guide</button>
      <div class="livebox">
        <div class="live-label"><span class="live-dot"></span>LIVE</div>
        <div class="live-time">{{ generated_at }}</div>
      </div>
    </div>
  </div>

  {{ desktop_nav_html | safe }}
  {{ nav_html | safe }}
  {{ search_html | safe }}

  <div id="glossaryOverlay" class="glossary-overlay" onclick="closeGlossary()"></div>
  <aside id="glossaryDrawer" class="glossary-drawer" aria-hidden="true">
    <div class="glossary-head">
      <div>
        <div class="glossary-kicker">DiamondSignals Intelligence</div>
        <h2 class="glossary-title">Field Guide</h2>
      </div>
      <button class="glossary-close" type="button" onclick="closeGlossary()" aria-label="Back to report"><span class="desktop-close-symbol">×</span><span class="mobile-back-label">← Back to Report</span></button>
    </div>
    <div class="glossary-body">
      <section class="glossary-section">
        <h3 class="glossary-section-title">I. Global System Metrics</h3>
        <div class="glossary-item"><span class="glossary-term">System Status</span><div class="glossary-definition">Confirms the live state of the Statcast-driven pipeline.</div></div>
        <div class="glossary-item"><span class="glossary-term" {{ metric_tooltip_attr("Edge Score") | safe }}>Edge Score</span><div class="glossary-definition">A 0 to 100 ranking summarizing signal strength versus baseline.</div></div>
      </section>
      <section class="glossary-section">
        <h3 class="glossary-section-title">II. Pitching Signal Terms</h3>
        <div class="glossary-item"><span class="glossary-term">Whiff %</span><div class="glossary-definition">Share of recent pitches that generated a swinging strike event.</div></div>
        <div class="glossary-item"><span class="glossary-term">FB Velo</span><div class="glossary-definition">Average recent fastball velocity.</div></div>
        <div class="glossary-item"><span class="glossary-term">Extension</span><div class="glossary-definition">Release extension in feet.</div></div>
      </section>
      <section class="glossary-section">
        <h3 class="glossary-section-title">III. Hitting Signal Terms</h3>
        <div class="glossary-item"><span class="glossary-term">Avg EV</span><div class="glossary-definition">Mean exit velocity on tracked batted-ball events.</div></div>
        <div class="glossary-item"><span class="glossary-term">Barrel-like %</span><div class="glossary-definition">Share of batted balls in the DiamondSignals barrel-like bucket.</div></div>
        <div class="glossary-item"><span class="glossary-term">EV Burst</span><div class="glossary-definition">A recent jump in average exit velocity versus baseline.</div></div>
      </section>
    </div>
  </aside>

  <div class="app">
    <section class="hero">
      <div class="hero-grid">
        <div class="hero-card">
          <div class="eyebrow">Executive Terminal</div>
          <h1 class="hero-title">Today’s Signal Wall</h1>
          <p class="hero-copy">Live pitcher and hitter movement board built from the MLB Extraction Ledger chassis. Edge Score, SEAGER, BABIP, BB%, K%, BB/K, K/BB, and season-context command data remain preserved.</p>
          <p class="hero-copy hero-audit-copy">
            <span class="hero-audit-label">AUDIT LAYER ACTIVE</span>
            Click any player card to inspect the full performance audit.
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

      </div>
    </section>


    <section class="board">
      <div class="section pitching-section">
        <div class="section-head">
          <div>
            <div class="section-kicker">Pitching Board</div>
            <h2 class="section-title">Top 5 Pitchers</h2>
          </div>
          <div class="section-badge">Live Ranked</div>
        </div>

        <div class="cards">
          {% for row in pitchers %}
          {{ home_signal_ledger_card.render(
            row=row,
            player_type="pitcher",
            rank=loop.index,
            trigger_label="Pitcher Trigger",
            signal_line="Pitcher // Live Edge Signal // " ~ row.sample_note,
            gradient_id="pitcherGradient" ~ loop.index,
            gradient_to=("#b6ff00" if row.edge_score >= 65 else "#00e5ff"),
            pulse_badges=[]
          ) | safe }}
          {% endfor %}
        </div>
      </div>

      <div class="section hitting-section">
        <div class="section-head">
          <div>
            <div class="section-kicker">Hitting Board</div>
            <h2 class="section-title">Top 5 Hitters</h2>
          </div>
          <div class="section-badge">Live Ranked</div>
        </div>

        <div class="cards">
          {% for row in hitters %}
          {{ home_signal_ledger_card.render(
            row=row,
            player_type="hitter",
            rank=loop.index,
            trigger_label="Hitter Trigger",
            signal_line="Hitter // Live Edge Signal // " ~ row.sample_note,
            gradient_id="hitterGradient" ~ loop.index,
            gradient_to=("#b6ff00" if row.edge_score >= 65 else "#00e5ff"),
            pulse_badges=['EV Burst', 'Barrel Jump']
          ) | safe }}
          {% endfor %}
        </div>
      </div>
    </section>

    {{ footer_html | safe }}
  </div>

  <script src="/player-search.js"></script>
  <script src="/player-card-actions.js"></script>
  <script>


    // SIGNAL_WALL_MOBILE_FIELD_GUIDE_MENU_TWIN_JS_V16
    // Make Field Guide a true sibling/twin of the mobile Menu pill.
    // SIGNAL_WALL_MOBILE_MENU_LABEL_MODULE_FEEDS_V18
    // Signal Wall mobile-only brand label: Menu -> MODULE / FEEDS.
    function dsSetSignalWallMobileMenuLabel(menu, label) {
      if (!menu) return;
      const spans = Array.from(menu.querySelectorAll('span'));
      const labelSpan = spans.find((span) => !span.classList.contains('ds-mobile-menu-icon')) || spans[spans.length - 1];
      if (labelSpan) labelSpan.textContent = label;
    }

    function dsInstallFieldGuideMenuTwin() {
      const isMobile = window.matchMedia("(max-width: 640px)").matches;
      const originalGuide = document.querySelector('button.info-trigger[onclick="openGlossary()"]');

      const menu =
        document.querySelector('.ds-mobile-menu-trigger:not(.ds-field-guide-menu-twin)') ||
        document.querySelector('[data-mobile-menu-trigger]:not(.ds-field-guide-menu-twin)') ||
        Array.from(document.querySelectorAll('button, a')).find(el =>
          !el.classList.contains('ds-field-guide-menu-twin') &&
          (
            (el.textContent || '').trim().toUpperCase() === 'MENU' ||
            (el.textContent || '').trim().toUpperCase().includes('MENU') ||
            (el.textContent || '').trim().toUpperCase().includes('MODULE / FEEDS')
          )
        );

      if (!originalGuide || !menu || !menu.parentElement) return;

      let twin = document.querySelector('.ds-field-guide-menu-twin');

      if (!isMobile) {
        dsSetSignalWallMobileMenuLabel(menu, 'Menu');
        originalGuide.classList.remove('ds-original-field-guide-hidden');
        if (twin) twin.remove();
        menu.parentElement.classList.remove('ds-mobile-menu-twin-host');
        return;
      }

      originalGuide.classList.add('ds-original-field-guide-hidden');

      const host = menu.parentElement;
      host.classList.add('ds-mobile-menu-twin-host');
      dsSetSignalWallMobileMenuLabel(menu, 'MODULE / FEEDS');

      if (!twin) {
        twin = document.createElement('button');
        twin.type = 'button';
        twin.className = `${menu.className || ''} ds-field-guide-menu-twin`.trim();
        twin.setAttribute('aria-label', 'Open Field Guide');
        twin.addEventListener('click', function(event) {
          event.preventDefault();
          event.stopPropagation();
          openGlossary();
        });
      }

      // SIGNAL_WALL_MOBILE_FIELD_GUIDE_LABEL_RESET_V19
      // Prevent MODULE / FEEDS from contaminating the Field Guide twin on reinstall.
      twin.innerHTML = '<span class="ds-field-guide-icon">ⓘ</span><span>Field Guide</span>';

      if (twin.parentElement !== host) {
        host.insertBefore(twin, menu);
      } else if (twin.nextElementSibling !== menu) {
        host.insertBefore(twin, menu);
      }
    }

    window.addEventListener("resize", dsInstallFieldGuideMenuTwin, { passive: true });
    window.addEventListener("orientationchange", dsInstallFieldGuideMenuTwin, { passive: true });
    window.addEventListener("load", dsInstallFieldGuideMenuTwin, { passive: true });
    dsInstallFieldGuideMenuTwin();
    setTimeout(dsInstallFieldGuideMenuTwin, 250);
    setTimeout(dsInstallFieldGuideMenuTwin, 750);


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
      if (event.key === "Escape") closeGlossary();
    });
  </script>
</body>
</html>
"""
)


def render_html(pitchers: pd.DataFrame, hitters: pd.DataFrame) -> str:
    return HTML_TEMPLATE.render(
        metric_tooltip_attr=metric_tooltip_attr,
        generated_at=datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        threshold=f"{ALERT_THRESHOLD:.0f}+",
        timezone_label=TIMEZONE_LABEL,
        desktop_nav_html=Template(NAV_V2_TEMPLATE).render(active_nav="signal_wall"),
        nav_html=Template(NAV_TEMPLATE).render(active_nav="signal_wall"),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
        ledger_styles=LEDGER_STYLES_TEMPLATE,
        home_signal_ledger_card=HOME_SIGNAL_LEDGER_CARD,
        pitchers=pitchers.to_dict(orient="records"),
        hitters=hitters.to_dict(orient="records"),
    )


def render_signals_front_door() -> str:
    return SIGNALS_FRONT_DOOR_TEMPLATE


def scout_shell_html() -> str:
    html = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals // Scout</title>
  <style>
    :root {
      --bg: #080808;
      --surface: #121212;
      --card-radial: radial-gradient(circle at top left, #1a1a1a 0%, #080808 100%);
      --border: #2d2d2d;
      --text: #f0f0f0;
      --muted: #71717a;
      --soft: #a1a1aa;
      --blue: #6aa6ff;
      --lime-hot: #b6ff00;
      --shadow: 0 14px 34px rgba(0, 0, 0, 0.34);
      --radius: 18px;
      --mono: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--sans); }
    body {
      background:
        radial-gradient(circle at top left, rgba(106,166,255,0.06), transparent 24%),
        radial-gradient(circle at top right, rgba(239,68,68,0.04), transparent 20%),
        linear-gradient(180deg, #101010 0%, #080808 34%, #050505 100%);
      line-height: 1.35;
    }
    .topbar { position: sticky; top: 0; z-index: 50; background: rgba(8, 8, 8, 0.90); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(255,255,255,0.05); }
    .topbar-inner, .app { width: min(1180px, calc(100% - 24px)); margin: 0 auto; }
    .topbar-inner { min-height: 62px; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 0; }
    .brand { display: flex; align-items: center; gap: 10px; min-width: 0; }
    .brand-mark { width: 11px; height: 11px; border-radius: 999px; background: var(--lime-hot); box-shadow: 0 0 10px rgba(182,255,0,0.35); flex: 0 0 auto; }
    .brand-kicker { font-size: 10px; line-height: 1; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 800; margin-bottom: 4px; }
    .brand-white { color: var(--text); }
    .brand-blue { color: var(--blue); }
    .brand-title { font-size: 16px; line-height: 1.05; letter-spacing: -0.02em; font-weight: 800; }
    .header-actions { display: flex; align-items: center; gap: 10px; }
    .info-trigger { height: 34px; border-radius: 999px; border: 1px solid rgba(182,255,0,0.22); background: rgba(255,255,255,0.05); color: var(--text); display: inline-flex; align-items: center; justify-content: center; padding: 0 12px; font-family: var(--mono); font-size: 12px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; text-decoration: none; box-shadow: 0 0 10px rgba(182,255,0,0.08); }

    .app { padding: 28px 0 40px; }
    .hero-card, .section-card, .metric-card, .briefing-card {
      background: var(--card-radial);
      border: 0.5px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }
    .hero-card::before, .section-card::before, .metric-card::before, .briefing-card::before {
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

    .hero-card { padding: 22px; margin-bottom: 16px; }
    .eyebrow { font-size: 10px; line-height: 1; letter-spacing: 0.18em; text-transform: uppercase; color: var(--blue); font-weight: 800; margin-bottom: 10px; }
    
    .hero-copy { margin: 0; max-width: 760px; color: var(--soft); font-size: 14px; }

    .player-id-grid { display: grid; grid-template-columns: 120px 1fr auto; gap: 18px; align-items: center; padding: 18px; margin-bottom: 16px; }
    .headshot-shell { width: 120px; height: 120px; border-radius: 24px; border: 1px solid rgba(255,255,255,0.10); background: rgba(255,255,255,0.03); display: flex; align-items: center; justify-content: center; color: var(--soft); font-family: var(--mono); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; overflow: hidden; }
    .player-kicker { font-size: 10px; line-height: 1; letter-spacing: 0.16em; text-transform: uppercase; color: var(--blue); font-weight: 800; margin-bottom: 8px; }
    .player-name { margin: 0 0 8px; font-size: clamp(34px, 4.8vw, 56px); line-height: 0.96; letter-spacing: -0.04em; font-weight: 900; text-transform: uppercase; }
    .player-meta { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .meta-pill, .signal-pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 8px 10px; border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); font-family: var(--mono); font-size: 11px; color: var(--soft); text-transform: uppercase; letter-spacing: 0.08em; }
    .signal-stack { display: grid; gap: 8px; justify-items: end; min-width: 120px; }
    .signal-pill.strong { color: var(--lime-hot); border-color: rgba(182,255,0,0.22); box-shadow: 0 0 8px rgba(182,255,0,0.08); }

    .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px; }
    .metric-card { padding: 16px; }
    .metric-kicker { font-size: 10px; line-height: 1; letter-spacing: 0.16em; text-transform: uppercase; color: var(--lime-hot); font-weight: 800; margin-bottom: 12px; font-family: var(--mono); }
    .metric-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .metric-row:last-child { border-bottom: 0; padding-bottom: 0; }
    .metric-label { font-size: 11px; color: var(--soft); text-transform: uppercase; letter-spacing: 0.08em; font-family: var(--mono); }
    .metric-value { font-size: 18px; color: var(--text); font-family: var(--mono); font-weight: 700; white-space: nowrap; }

    .briefing-card { padding: 18px; }
    .briefing-kicker { font-size: 10px; line-height: 1; letter-spacing: 0.16em; text-transform: uppercase; color: var(--blue); font-weight: 800; margin-bottom: 12px; }
    .briefing-copy { margin: 0; color: var(--soft); font-size: 15px; line-height: 1.6; max-width: 900px; }

    
      
      .support-card {
        display: none;
        padding: 16px;
        margin-bottom: 16px;
      }

      .support-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
      }

      .support-title {
        font-family: var(--mono);
        font-size: 10px;
        line-height: 1;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--blue);
        font-weight: 800;
      }

      .support-profile {
        font-family: var(--mono);
        font-size: 10px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--lime-hot);
        border: 1px solid rgba(182,255,0,0.18);
        background: rgba(182,255,0,0.04);
        border-radius: 999px;
        padding: 7px 10px;
        white-space: nowrap;
      }

      .support-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
      }

      .support-tile {
        min-width: 0;
        border: 1px solid rgba(255,255,255,0.07);
        background: rgba(255,255,255,0.025);
        border-radius: 14px;
        padding: 11px 12px;
      }

      .support-label {
        font-family: var(--mono);
        font-size: 10px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--soft);
        margin-bottom: 6px;
      }

      .support-value {
        font-family: var(--mono);
        font-size: 18px;
        line-height: 1.05;
        color: var(--text);
        font-weight: 800;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      .promotion-context-card {
        display: none;
        padding: 16px;
        margin-bottom: 16px;
        border-color: rgba(182,255,0,0.16);
        background:
          radial-gradient(circle at top left, rgba(182,255,0,0.06), transparent 34%),
          rgba(255,255,255,0.025);
      }

      .promotion-context-copy {
        margin: 0;
        color: var(--soft);
        font-size: 14px;
        line-height: 1.55;
      }

      .promotion-context-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
      }


      /* PERFORMANCE AUDIT SCOUT LAYOUT LOCK */
      /* DIAMONDSIGNALS_SCOUT_AUDIT_DESKTOP_CANVAS_POLISH_V1
         Desktop-only: widen the audit canvas and reduce excessive black dead space
         without touching mobile behavior. */
      @media (min-width: 1000px) {
        .scout-page {
          width: min(1360px, calc(100vw - 64px));
          max-width: none;
          margin-left: auto;
          margin-right: auto;
          padding-bottom: 28px;
        }

        .scout-page .hero-card,
        .scout-page .section-card,
        .scout-page .metric-card,
        .scout-page .briefing-card {
          box-shadow: 0 18px 54px rgba(0,0,0,0.28);
        }

        .scout-page .dashboard-grid,
        .scout-page .audit-grid,
        .scout-page .metric-grid,
        .scout-page .metrics-grid {
          width: 100%;
        }

        .scout-page .briefing-card {
          margin-bottom: 0;
        }
      }

      /* DIAMONDSIGNALS_SCOUT_AUDIT_POLISH_V1 */
      .scout-page .hero-card {
        overflow: hidden;
      }
      .scout-page .hero-title,
      .scout-page .hero-copy {
        max-width: 100%;
        overflow-wrap: anywhere;
      }
      .scout-page .hero-copy {
        line-height: 1.55;
      }

      /* SCOUT_AUDIT_ARTIFACT_CLEANUP_V1
         Disable decorative pseudo-border masks on Performance Audit cards.
         These masks can render as visible rectangular artifacts over audit panels. */
      .hero-card::before,
      .section-card::before,
      .metric-card::before,
      .briefing-card::before {
        content: none;
        display: none;
      }

      @media (min-width: 761px) {
        .metrics-grid {
          display: grid !important;
          grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
          gap: 16px !important;
          align-items: stretch !important;
        }

        .metrics-grid .metric-card {
          min-width: 0 !important;
          width: auto !important;
        }

        .support-grid {
          display: grid !important;
          grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
          gap: 10px !important;
        }
      }

      @media (max-width: 900px) {
        .player-id-grid {
          grid-template-columns: 1fr;
          justify-items: start;
        }

        .signal-stack {
          justify-items: start;
          min-width: 0;
        }
      }

      @media (max-width: 760px) {
        .support-head {
          align-items: flex-start;
          flex-direction: column;
        }

        .metrics-grid {
          display: grid !important;
          grid-template-columns: 1fr !important;
          gap: 14px !important;
        }

        .support-grid {
          display: grid !important;
          grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
          gap: 10px !important;
        }
      }


/* MOBILE_LIVE_CANARY_FORCE_COMPACT_ROWS_V1
   Final canary-only mobile override.
   Purpose: undo the desktop firewall on /mobile-live-canary/ mobile widths and
   restore the compact row-first expandable layout. Desktop /live/ is untouched. */
@media (max-width: 760px) {
  body.signal-wall-v2-typography-lock .app > section.board {
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board > .section {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 0 14px !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board .cards {
    gap: 0 !important;
    padding: 2px 8px 16px !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board .player-card,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) {
    padding: 0 !important;
    min-height: 0 !important;
    height: auto !important;
    border-radius: 0 !important;
    border: 0 !important;
    border-bottom: 1px solid rgba(148,163,184,0.10) !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board .mobile-scan-row,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .mobile-scan-row,
  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .mobile-scan-row {
    display: grid !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    grid-template-columns: auto minmax(0, 1fr) auto !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
    min-height: 48px !important;
    padding: 8px 10px !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    color: inherit !important;
    text-align: left !important;
    cursor: pointer !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .player-top,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .provision-row-desktop,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .provision-row-mobile,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .mobile-drawer-close,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .mobile-signal-detail,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .mobile-signal-tray,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .sparkline-wrap,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .badge-row,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .metric-grid,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .season-context-strip,
  body.signal-wall-v2-typography-lock .app > section.board .player-card:not(.mobile-card-open) .why {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open {
    padding: 10px !important;
    margin: 8px 0 12px !important;
    border: 1px solid rgba(148,163,184,0.20) !important;
    border-radius: 18px !important;
    background: var(--card-radial) !important;
    box-shadow: var(--shadow) !important;
    overflow: visible !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .player-top {
    display: grid !important;
    visibility: visible !important;
    opacity: 1 !important;
    grid-template-columns: minmax(0, 1fr) auto !important;
    gap: 12px !important;
    margin-bottom: 10px !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .mobile-drawer-close {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .provision-row-mobile {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .mobile-signal-detail,
  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .mobile-signal-tray,
  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .sparkline-wrap,
  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .badge-row,
  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .metric-grid,
  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .season-context-strip,
  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .why {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
  }

  body.signal-wall-v2-typography-lock .app > section.board .player-card.mobile-card-open .metric-grid {
    display: grid !important;
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
          <div class="brand-title">Scout Terminal // Performance Audit</div>
        </div>
      </div>
      <div class="header-actions">
        <a class="info-trigger" href="/">Back to Signal Wall</a>
      </div>
    </div>
  </div>

  {nav_html}
  {search_html}

  <div class="app">
        <section class="hero-card">
      <div class="eyebrow">Scout Terminal</div>
      <h1 class="hero-title">Performance Audit</h1>
      <p class="hero-copy">
        Canonical performance dossier built from DiamondSignals live extraction, season context, and player audit layers.
      </p>
    </section>

    <section class="section-card player-id-grid">
      <div class="headshot-shell" id="scoutHeadshotWrap">
        <img id="scoutHeadshot" alt="Player headshot" style="width:100%;height:100%;object-fit:cover;border-radius:24px;display:none;" />
        <span id="scoutHeadshotFallback">Headshot</span>
      </div>

      <div>
        <div class="player-kicker">Player ID Card</div>
        <h2 class="player-name" id="scoutPlayerName">Player Name</h2>

        <div class="player-meta">
          <span class="meta-pill" id="scoutTeam">Team</span>
          <span class="meta-pill" id="scoutPosition">Position</span>
          <span class="meta-pill" id="scoutBT">B/T</span>
          <span class="meta-pill" id="scoutStatus">Status</span>
        </div>
      </div>

      <div class="signal-stack">
        <span class="signal-pill strong" id="scoutSignalPill">Signal Score</span>
        <span class="signal-pill" id="scoutTrendPill">Trend</span>
        <span class="signal-pill" id="scoutConfidencePill">Confidence</span>
      </div>
    </section>

      <section class="section-card support-card" id="supportMetricsCard">
        <div class="support-head">
          <div class="support-title">Support Metrics // Season Context</div>
          <div class="support-profile" id="supportProfile">CONTEXT</div>
        </div>
        <div class="support-grid" id="supportMetricsGrid"></div>
      </section>

      <section class="section-card promotion-context-card" id="promotionWatchContextCard">
        <div class="support-head">
          <div class="support-title">Promotion Watch // Current Signal Context</div>
          <div class="support-profile" id="promotionWatchContextProfile">WATCHLIST</div>
        </div>
        <p class="promotion-context-copy" id="promotionWatchContextCopy"></p>
        <div class="promotion-context-grid" id="promotionWatchContextGrid"></div>
      </section>



    <section class="metrics-grid">
      <article class="metric-card">
        <div class="metric-kicker" id="zone1Title">Ballistics</div>
        <div class="metric-row"><div class="metric-label" id="zone1Label1">Metric 1</div><div class="metric-value" id="zone1Value1">--</div></div>
        <div class="metric-row"><div class="metric-label" id="zone1Label2">Metric 2</div><div class="metric-value" id="zone1Value2">--</div></div>
        <div class="metric-row"><div class="metric-label" id="zone1Label3">Metric 3</div><div class="metric-value" id="zone1Value3">--</div></div>
        <div class="metric-row"><div class="metric-label" id="zone1Label4">Metric 4</div><div class="metric-value" id="zone1Value4">--</div></div>
      </article>

      <article class="metric-card">
        <div class="metric-kicker" id="zone2Title">Movement</div>
        <div class="metric-row"><div class="metric-label" id="zone2Label1">Metric 1</div><div class="metric-value" id="zone2Value1">--</div></div>
        <div class="metric-row"><div class="metric-label" id="zone2Label2">Metric 2</div><div class="metric-value" id="zone2Value2">--</div></div>
        <div class="metric-row"><div class="metric-label" id="zone2Label3">Metric 3</div><div class="metric-value" id="zone2Value3">--</div></div>
        <div class="metric-row"><div class="metric-label" id="zone2Label4">Metric 4</div><div class="metric-value" id="zone2Value4">--</div></div>
      </article>

      <article class="metric-card">
        <div class="metric-kicker" id="zone3Title">Results</div>
        <div class="metric-row"><div class="metric-label" id="zone3Label1">Metric 1</div><div class="metric-value" id="zone3Value1">--</div></div>
        <div class="metric-row"><div class="metric-label" id="zone3Label2">Metric 2</div><div class="metric-value" id="zone3Value2">--</div></div>
        <div class="metric-row"><div class="metric-label" id="zone3Label3">Metric 3</div><div class="metric-value" id="zone3Value3">--</div></div>
        <div class="metric-row"><div class="metric-label" id="zone3Label4">Metric 4</div><div class="metric-value" id="zone3Value4">--</div></div>
      </article>
    </section>

    <section class="briefing-card">
      <div class="briefing-kicker">Analyst Briefing</div>
      <p class="briefing-copy" id="scoutBriefingCopy">
        This area will hold the short DiamondSignals analyst summary generated from structured player data.
      </p>
    </section>
  </div>

  <script src="/player-search.js"></script>
  <script src="/player-card-actions.js"></script>
  <script>
    function formatPct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return `${Number(value).toFixed(1)}%`;
    }

    function format1(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return Number(value).toFixed(1);
    }

    function formatAvg(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return Number(value).toFixed(3).replace(/^0/, "");
    }

    function formatSigned3(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      const n = Number(value);
      const s = n.toFixed(3).replace(/^0/, "");
      return n > 0 ? `+${s}` : s;
    }

    function formatValue(label, value) {
      const upper = String(label || "").toUpperCase();
      if (value === null || value === undefined || value === "") return "--";
      if (typeof value === "string") return value;
      if (upper.includes("%")) return formatPct(value);
      if (upper.includes("AVG") || upper.includes("XBA")) return formatAvg(value);
      if (upper.includes("SPIN")) return Number(value).toFixed(0);
      if (upper.includes("EXTENSION")) return `${Number(value).toFixed(1)} ft`;
      if (upper.includes("IVB")) return `${Number(value).toFixed(1)} in`;
      if (upper.includes("DELTA") || upper.includes("EDGE")) return formatSigned3(value);
      return format1(value);
    }

    function setZone(prefix, zoneData) {
      for (let i = 1; i <= 4; i++) {
        const labelEl = document.getElementById(`${prefix}Label${i}`);
        const valueEl = document.getElementById(`${prefix}Value${i}`);
        const label = zoneData[`label_${i}`] || `Metric ${i}`;
        const value = zoneData[`value_${i}`];
        if (labelEl) labelEl.textContent = label;
        if (valueEl) valueEl.textContent = formatValue(label, value);
      }
    }

    async function fetchJsonWithFallback(paths) {
      for (const path of paths) {
        try {
          const bust = path.includes("?") ? "&" : "?";
          const res = await fetch(path + bust + "v=" + Date.now(), { cache: "no-store" });
          if (res.ok) {
            console.log("[SCOUT] loaded JSON from", path);
            return await res.json();
          }
        } catch (err) {
          console.log("[SCOUT] fetch failed for", path, err);
        }
      }
      throw new Error("All JSON fetch paths failed");
    }

      function escapeHtml(value) {
        return String(value ?? "—")
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#039;");
      }

      function renderSupportMetrics(player) {
        const card = document.getElementById("supportMetricsCard");
        const grid = document.getElementById("supportMetricsGrid");
        const profile = document.getElementById("supportProfile");

        if (!card || !grid || !profile) return;

        const support = player && player.support_metrics ? player.support_metrics : null;
        const tiles = support && Array.isArray(support.tiles) ? support.tiles : [];

        if (!support || !tiles.length) {
          card.style.display = "none";
          grid.innerHTML = "";
          return;
        }

        profile.textContent = support.profile || "SUPPORT";
        grid.innerHTML = tiles.map(tile => `
          <div class="support-tile">
            <div class="support-label">${escapeHtml(tile.label || "Metric")}</div>
            <div class="support-value">${escapeHtml(tile.value ?? "—")}</div>
          </div>
        `).join("");

        card.style.display = "block";
      }


      function renderPromotionWatchContext(player) {
        const card = document.getElementById("promotionWatchContextCard");
        const profile = document.getElementById("promotionWatchContextProfile");
        const copy = document.getElementById("promotionWatchContextCopy");
        const grid = document.getElementById("promotionWatchContextGrid");

        if (!card || !profile || !copy || !grid) return;

        const ctx = player && player.promotion_watch_context ? player.promotion_watch_context : null;
        const tiles = ctx && Array.isArray(ctx.tiles) ? ctx.tiles : [];

        if (!ctx) {
          card.style.display = "none";
          copy.textContent = "";
          grid.innerHTML = "";
          return;
        }

        profile.textContent = ctx.section_label || ctx.profile || "PROMOTION WATCH";
        copy.textContent = ctx.why || "This player is currently present in the Promotion Watch surveillance feed.";
        grid.innerHTML = tiles.map(tile => `
          <div class="support-tile">
            <div class="support-label">${escapeHtml(tile.label || "Context")}</div>
            <div class="support-value">${escapeHtml(tile.value ?? "—")}</div>
          </div>
        `).join("");

        card.style.display = "block";
      }

    async function loadScoutPlayer() {
      const pathParts = window.location.pathname.split("/").filter(Boolean);
      const scoutIndex = pathParts.indexOf("scout");
      const playerId = scoutIndex >= 0 && pathParts[scoutIndex + 1] ? pathParts[scoutIndex + 1] : null;

      console.log("[SCOUT] pathname", window.location.pathname);
      console.log("[SCOUT] parsed playerId", playerId);

      if (!playerId) {
        document.getElementById("scoutPlayerName").textContent = "SELECT A PLAYER";
        document.getElementById("scoutBriefingCopy").textContent =
          "This scout shell is live. Open a real dossier URL such as /scout/671096/ to hydrate a player profile.";
        return;
      }

      try {
        window.__DS_SCOUT_PLAYER__ = __SCOUT_PLAYER_JSON__;

        const embeddedPlayer = window.__DS_SCOUT_PLAYER__ || null;
        const dossierPayload = embeddedPlayer && String(embeddedPlayer.player_id) === String(playerId)
          ? { players: { [String(playerId)]: embeddedPlayer } }
          : await fetchJsonWithFallback([
              "/dossier_canon.json",
              "../../dossier_canon.json",
              "../dossier_canon.json",
              "/dist/dossier_canon.json"
            ]);

        const players = dossierPayload && dossierPayload.players ? dossierPayload.players : {};
        const player = players[String(playerId)] || null;
        const scoutMetrics = player && player.scout_metrics ? player.scout_metrics : null;

        console.log("[SCOUT] hasPlayer", !!player);
        console.log("[SCOUT] hasScoutMetrics", !!scoutMetrics);

        if (!player) {
          document.getElementById("scoutPlayerName").textContent = "PLAYER NOT FOUND";
          document.getElementById("scoutBriefingCopy").textContent =
            "This dossier path resolved, but no player record was found in dossier_canon.json.";
          return;
        }

        document.title = `DiamondSignals // ${player.player_name}`;
        document.getElementById("scoutPlayerName").textContent = player.player_name || "Unknown Player";
        document.getElementById("scoutTeam").textContent = player.current_team || player.team || "Team";
        document.getElementById("scoutPosition").textContent = player.position || "Position";
        document.getElementById("scoutBT").textContent = `${player.bats || "-"} / ${player.throws || "-"}`;
        document.getElementById("scoutStatus").textContent = player.current_team || player.team || "Status";

        const img = document.getElementById("scoutHeadshot");
        const fallback = document.getElementById("scoutHeadshotFallback");
        if (player.headshot_url) {
          img.src = player.headshot_url;
          img.onload = () => {
            img.style.display = "block";
            fallback.style.display = "none";
          };
        }

        document.getElementById("scoutSignalPill").textContent =
          `${player.canonical_score_label || "Signal Score"} ${player.canonical_score ?? "--"}`;
        document.getElementById("scoutTrendPill").textContent = player.trend_note || "Trend";
        document.getElementById("scoutConfidencePill").textContent = "CANONICAL";
          renderSupportMetrics(player);
          renderPromotionWatchContext(player);

        if (scoutMetrics) {
          setZone("zone1", scoutMetrics.ballistics || {});
          setZone("zone2", scoutMetrics.movement || {});
          setZone("zone3", scoutMetrics.results || {});
          document.getElementById("scoutBriefingCopy").textContent =
            scoutMetrics.briefing || "Live player profile loaded.";
        } else {
          document.getElementById("scoutBriefingCopy").textContent =
            "This player loaded from dossier_canon.json, but no scout_metrics block was found yet.";
        }
      } catch (error) {
        console.error("[SCOUT] load error", error);
        document.getElementById("scoutPlayerName").textContent = "SCOUT LOAD ERROR";
        document.getElementById("scoutBriefingCopy").textContent =
          "The dossier shell loaded, but canonical dossier data could not be fetched.";
      }
    }

    loadScoutPlayer();
  </script>
</body>
</html>
"""
    # Scout Performance Audit uses its own clean topbar.
    # Do not inject global nav/search shells here; they create duplicate command-nav artifacts.
    nav_html = ""
    search_html = ""
    shell_styles = SHELL_STYLES_TEMPLATE

    return (
        html.replace("{nav_html}", nav_html)
        .replace("{search_html}", search_html)
        .replace("{shell_styles}", shell_styles)
    )


def write_scout_pages(dossier_payload: dict) -> None:
    scout_dir = DIST_DIR / "scout"
    scout_dir.mkdir(parents=True, exist_ok=True)

    shell_html = scout_shell_html()
    index_shell_html = shell_html.replace("__SCOUT_PLAYER_JSON__", "null")

    (scout_dir / "index.html").write_text(index_shell_html, encoding="utf-8")
    print("Wrote dist/scout/index.html")

    players = dossier_payload.get("players", {}) if isinstance(dossier_payload, dict) else {}
    player_ids = sorted(players.keys())

    for player_id in player_ids:
        player_dir = scout_dir / str(player_id)
        player_dir.mkdir(parents=True, exist_ok=True)
        player_html = shell_html.replace(
            "__SCOUT_PLAYER_JSON__",
            json.dumps(players[player_id], separators=(",", ":"), ensure_ascii=False),
        )
        (player_dir / "index.html").write_text(player_html, encoding="utf-8")

    print(f"Wrote {len(player_ids)} player dossier pages under dist/scout/<player_id>/index.html")


def write_status_file(status_payload: dict) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    if status_payload.get("report_id") == "signal_wall":
        status_payload["mode"] = SIGNAL_WALL_MODE
        status_payload["pipeline_layers"] = SIGNAL_WALL_PIPELINE_LAYERS
        status_payload["hardening_notes"] = SIGNAL_WALL_HARDENING_NOTES
    SIGNAL_WALL_STATUS_PATH.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {SIGNAL_WALL_STATUS_PATH}")



def _season_context_key(value):
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _season_context_player_id(value):
    try:
        if value in (None, ""):
            return None
        return str(int(float(str(value).strip())))
    except Exception:
        return None


def load_signal_wall_season_context_lookup():
    path = DIST_DIR / "admin" / "player_signal_index.json"
    if not path.exists():
        print("SIGNAL_WALL_SEASON_CONTEXT_MISSING:", path)
        return {}, {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("SIGNAL_WALL_SEASON_CONTEXT_ERROR:", exc)
        return {}, {}

    by_id = {}
    by_name = {}

    for player in data.get("players", []):
        ctx = player.get("season_context")
        if not ctx:
            continue

        pid = _season_context_player_id(player.get("player_id"))
        name = _season_context_key(player.get("player_name") or player.get("name"))

        if pid:
            by_id[pid] = ctx
        if name:
            by_name[name] = ctx

    print("SIGNAL_WALL_SEASON_CONTEXT_LOOKUP:", len(by_id), "ids //", len(by_name), "names")
    return by_id, by_name




def _format_pct(value):
    try:
        if value is None or pd.isna(value):
            return None
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return None


def _format_ratio(value):
    try:
        if value is None or pd.isna(value):
            return None
        return f"{float(value):.2f}"
    except Exception:
        return None


def _format_babip(value):
    try:
        if value is None or pd.isna(value):
            return None
        return f"{float(value):.3f}".replace("0.", ".")
    except Exception:
        return None


def build_signal_wall_hitter_plate_context(raw: pd.DataFrame) -> dict:
    """Build direct BABIP / BB% / K% / BB-K context for Signal Wall top hitters.

    This prevents final cards from depending only on player_signal_index.json coverage.
    The context is derived from the same fresh Statcast window used to produce Signal Wall.
    """
    if raw is None or raw.empty or "batter" not in raw.columns:
        return {}

    df = raw.copy()
    df = df[df["batter"].notna()].copy()
    if df.empty:
        return {}

    # Plate appearance proxy: use unique game/pitch-at-bat grouping when available,
    # otherwise fall back to event rows.
    if {"game_pk", "at_bat_number", "batter"}.issubset(df.columns):
        pa_df = df.drop_duplicates(["game_pk", "at_bat_number", "batter"]).copy()
    else:
        pa_df = df[df.get("events").notna()].copy() if "events" in df.columns else df.copy()

    if "events" not in pa_df.columns:
        return {}

    events = pa_df["events"].fillna("").astype(str).str.lower()

    walks = events.isin(["walk", "intent_walk"])
    strikeouts = events.isin(["strikeout", "strikeout_double_play"])

    # BABIP approximation from Statcast event taxonomy:
    # (H - HR) / (AB - K - HR + SF)
    hits = events.isin(["single", "double", "triple", "home_run"])
    homers = events.eq("home_run")
    sac_flies = events.eq("sac_fly")
    non_ab = events.isin([
        "walk",
        "intent_walk",
        "hit_by_pitch",
        "sac_bunt",
        "sac_fly",
        "catcher_interf",
    ])
    at_bats = (~non_ab).astype(int)

    work = pa_df[["batter"]].copy()
    work["pa"] = 1
    work["walks"] = walks.astype(int)
    work["strikeouts"] = strikeouts.astype(int)
    work["hits"] = hits.astype(int)
    work["homers"] = homers.astype(int)
    work["sac_flies"] = sac_flies.astype(int)
    work["at_bats"] = at_bats

    grouped = work.groupby("batter", dropna=True).sum(numeric_only=True).reset_index()

    context = {}
    for _, row in grouped.iterrows():
        pa = int(row.get("pa") or 0)
        if pa <= 0:
            continue

        bb = int(row.get("walks") or 0)
        k = int(row.get("strikeouts") or 0)
        h = int(row.get("hits") or 0)
        hr = int(row.get("homers") or 0)
        ab = int(row.get("at_bats") or 0)
        sf = int(row.get("sac_flies") or 0)

        bb_pct = bb / pa if pa else None
        k_pct = k / pa if pa else None
        bb_k_ratio = bb / k if k else None

        babip_den = ab - k - hr + sf
        babip = ((h - hr) / babip_den) if babip_den > 0 else None

        pid = _season_context_player_id(row.get("batter"))
        if not pid:
            continue

        context[pid] = {
            "season_context": "PLATE_DISCIPLINE",
            "source": "signal_wall_statcast_window",
            "bb_pct": _format_pct(bb_pct),
            "k_pct": _format_pct(k_pct),
            "bb_k_ratio": _format_ratio(bb_k_ratio),
            "babip": _format_babip(babip),
            "plate_appearances": pa,
            "strikeouts": k,
            "walks": bb,
        }

    return context


def enrich_signal_wall_hitter_plate_context(raw: pd.DataFrame, hitters: pd.DataFrame) -> pd.DataFrame:
    """Fill missing hitter season_context directly from fresh Signal Wall raw data."""
    if hitters is None or hitters.empty:
        return hitters

    context_by_id = build_signal_wall_hitter_plate_context(raw)
    if not context_by_id:
        return hitters

    out = hitters.copy()

    def pick_direct_context(row):
        existing = row.get("season_context")
        if isinstance(existing, dict) and existing:
            return existing

        for field in ["resolved_player_id", "player_id", "batter", "mlbam_id", "id"]:
            pid = _season_context_player_id(row.get(field))
            if pid and pid in context_by_id:
                return context_by_id[pid]

        return existing if existing is not None else None

    out["season_context"] = out.apply(pick_direct_context, axis=1)
    return out




def build_signal_wall_pitcher_command_context(raw: pd.DataFrame) -> dict:
    """Build direct K% / BB% / K-BB / BABIP command context for Signal Wall top pitchers.

    This prevents pitcher cards from depending only on player_signal_index.json coverage.
    The context is derived from the same fresh Statcast window used to produce Signal Wall.
    """
    if raw is None or raw.empty or "pitcher" not in raw.columns:
        return {}

    df = raw.copy()
    df = df[df["pitcher"].notna()].copy()
    if df.empty:
        return {}

    # Batter-faced proxy: use unique game/pitch-at-bat/pitcher grouping when available,
    # otherwise fall back to event rows.
    if {"game_pk", "at_bat_number", "pitcher"}.issubset(df.columns):
        bf_df = df.drop_duplicates(["game_pk", "at_bat_number", "pitcher"]).copy()
    else:
        bf_df = df[df.get("events").notna()].copy() if "events" in df.columns else df.copy()

    if "events" not in bf_df.columns:
        return {}

    events = bf_df["events"].fillna("").astype(str).str.lower()

    walks = events.isin(["walk", "intent_walk"])
    strikeouts = events.isin(["strikeout", "strikeout_double_play"])

    # BABIP approximation from Statcast event taxonomy:
    # (H - HR) / (AB - K - HR + SF)
    hits = events.isin(["single", "double", "triple", "home_run"])
    homers = events.eq("home_run")
    sac_flies = events.eq("sac_fly")
    non_ab = events.isin([
        "walk",
        "intent_walk",
        "hit_by_pitch",
        "sac_bunt",
        "sac_fly",
        "catcher_interf",
    ])
    at_bats = (~non_ab).astype(int)

    work = bf_df[["pitcher"]].copy()
    work["batters_faced"] = 1
    work["walks"] = walks.astype(int)
    work["strikeouts"] = strikeouts.astype(int)
    work["hits"] = hits.astype(int)
    work["homers"] = homers.astype(int)
    work["sac_flies"] = sac_flies.astype(int)
    work["at_bats"] = at_bats

    grouped = work.groupby("pitcher", dropna=True).sum(numeric_only=True).reset_index()

    context = {}
    for _, row in grouped.iterrows():
        bf = int(row.get("batters_faced") or 0)
        if bf <= 0:
            continue

        bb = int(row.get("walks") or 0)
        k = int(row.get("strikeouts") or 0)
        h = int(row.get("hits") or 0)
        hr = int(row.get("homers") or 0)
        ab = int(row.get("at_bats") or 0)
        sf = int(row.get("sac_flies") or 0)

        k_pct = k / bf if bf else None
        bb_pct = bb / bf if bf else None
        k_bb_ratio = k / bb if bb else None

        babip_den = ab - k - hr + sf
        babip = ((h - hr) / babip_den) if babip_den > 0 else None

        pid = _season_context_player_id(row.get("pitcher"))
        if not pid:
            continue

        context[pid] = {
            "season_context": "COMMAND_PROFILE",
            "source": "signal_wall_statcast_window",
            "k_pct": _format_pct(k_pct),
            "bb_pct": _format_pct(bb_pct),
            "k_bb_ratio": _format_ratio(k_bb_ratio),
            "babip": _format_babip(babip),
            "batters_faced": bf,
            "strikeouts": k,
            "walks": bb,
        }

    return context


def enrich_signal_wall_pitcher_command_context(raw: pd.DataFrame, pitchers: pd.DataFrame) -> pd.DataFrame:
    """Fill missing pitcher season_context directly from fresh Signal Wall raw data."""
    if pitchers is None or pitchers.empty:
        return pitchers

    context_by_id = build_signal_wall_pitcher_command_context(raw)
    if not context_by_id:
        return pitchers

    out = pitchers.copy()

    def pick_direct_context(row):
        existing = row.get("season_context")
        if isinstance(existing, dict) and existing:
            # Normalize legacy pitcher BABIP display to hitter-style .290 format when possible.
            if existing.get("babip"):
                existing = dict(existing)
                existing["babip"] = _format_babip(str(existing.get("babip")).replace("N/A", "")) or existing.get("babip")
            return existing

        for field in ["resolved_player_id", "player_id", "pitcher", "mlbam_id", "id"]:
            pid = _season_context_player_id(row.get(field))
            if pid and pid in context_by_id:
                return context_by_id[pid]

        return existing if existing is not None else None

    out["season_context"] = out.apply(pick_direct_context, axis=1)
    return out


def apply_signal_wall_season_context(df):
    if df is None or df.empty:
        return df

    by_id, by_name = load_signal_wall_season_context_lookup()
    if not by_id and not by_name:
        return df

    out = df.copy()

    def pick_ctx(row):
        for field in ["resolved_player_id", "player_id", "batter", "pitcher", "mlbam_id", "id"]:
            pid = _season_context_player_id(row.get(field))
            if pid and pid in by_id:
                return by_id[pid]

        for field in ["player_name", "name", "displayName"]:
            name = _season_context_key(row.get(field))
            if name and name in by_name:
                return by_name[name]

        return None

    out["season_context"] = out.apply(pick_ctx, axis=1)
    return out


def main() -> None:
    build_started_at = utc_now_iso()
    raw = fetch_statcast_window(START_DATE, END_DATE)
    if raw.empty:
        status_payload = build_report_status(
            "signal_wall",
            build_success=False,
            threshold_minutes=180,
            build_started_at=build_started_at,
            build_finished_at=utc_now_iso(),
            errors=["Statcast fallback returned no fresh data."],
            notes=["Keeping existing dist assets."],
        )
        write_status_file(status_payload)
        print(
            "Skipping dashboard rebuild because Statcast fallback returned no fresh data. Keeping existing dist assets."
        )
        return

    hitter_signals = build_hitter_signals(raw)
    hitter_signals = add_seager_score(raw, hitter_signals)
    pitcher_signals = build_pitcher_signals(raw)

    if hitter_signals.empty and pitcher_signals.empty:
        status_payload = build_report_status(
            "signal_wall",
            build_success=False,
            threshold_minutes=180,
            build_started_at=build_started_at,
            build_finished_at=utc_now_iso(),
            errors=["No hitter or pitcher signals were produced."],
        )
        write_status_file(status_payload)
        raise RuntimeError("No hitter or pitcher signals were produced.")

    top_hitters = hitter_signals.head(10).copy()
    top_pitchers = pitcher_signals.head(10).copy()

    top_hitters["player_id"] = top_hitters["batter"].fillna("").astype(str).str.strip()
    top_pitchers["player_id"] = top_pitchers["pitcher"].fillna("").astype(str).str.strip()

    top_hitters = backfill_resolved_player_ids(top_hitters)
    top_pitchers = backfill_resolved_player_ids(top_pitchers)

    top_hitters = apply_signal_wall_season_context(top_hitters)
    top_pitchers = apply_signal_wall_season_context(top_pitchers)

    top_hitters = enrich_signal_wall_hitter_plate_context(raw, top_hitters)
    top_pitchers = enrich_signal_wall_pitcher_command_context(raw, top_pitchers)

    print("SIGNAL_WALL_RENDER_CONTEXT_HITTERS:", int(top_hitters["season_context"].notna().sum()) if "season_context" in top_hitters.columns else 0)
    print("SIGNAL_WALL_RENDER_CONTEXT_PITCHERS:", int(top_pitchers["season_context"].notna().sum()) if "season_context" in top_pitchers.columns else 0)

    combined_alerts = pd.concat([top_pitchers, top_hitters], ignore_index=True)
    combined_alerts = combined_alerts.sort_values(
        "edge_score", ascending=False
    ).reset_index(drop=True)

    sections = {
        "top_signals": len(combined_alerts),
    }
    validation = build_validation_report(
        "signal_wall",
        [
            validate_min_rows("top_signals", sections["top_signals"], 1),
        ],
    )

    html = render_html(top_pitchers, top_hitters)

    live_path = DIST_DIR / "mobile-live-canary" / "index.html"
    temp_live_path = write_temp_output(str(live_path), html)
    promoted_live = promote_output_if_valid(temp_live_path, str(live_path), validation["ok"])
    if promoted_live:
        save_snapshot(str(live_path), str(SNAPSHOT_DIR / "mobile-live-canary-index.html"))
        print(f"Wrote {live_path}")
    else:
        print("Skipped publishing live signal wall due to failed validation.")

    front_door_html = render_signals_front_door()
    output_path = DIST_DIR / "mobile-live-canary" / "front-door.html"
    temp_front_door_path = write_temp_output(str(output_path), front_door_html)
    promoted_front_door = promote_output_if_valid(temp_front_door_path, str(output_path), validation["ok"])
    if promoted_front_door:
        print(f"Wrote {output_path}")
    else:
        print("Skipped publishing signal wall front door due to failed validation.")

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
                  "season_context",
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
                  "season_context",
                  "seager_score",
                "why",
                "badges",
            ]
        ].to_dict(orient="records"),
    }

    (DIST_DIR / "signals.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Wrote dist/signals.json")

    build_finished_at = utc_now_iso()
    status_payload = build_report_status(
        "signal_wall",
        build_success=bool(promoted_live and promoted_front_door),
        threshold_minutes=180,
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        source_updated_at=build_finished_at,
        section_counts=sections,
        degraded=not validation["ok"],
        errors=validation["messages"],
        notes=[] if validation["ok"] else ["Validation blocked one or more signal wall publishes."],
    )
    write_status_file(status_payload)

    player_index = build_player_index(raw)
    (DIST_DIR / "player_index.json").write_text(
        json.dumps(player_index, indent=2), encoding="utf-8"
    )
    print("Wrote dist/player_index.json")

    write_scout_metrics(raw)
    dossier_payload = write_dossier_canon(raw, hitter_signals, pitcher_signals)
    copy_static_assets()
    write_scout_pages(dossier_payload)
    send_telegram_alerts(combined_alerts)


if __name__ == "__main__":
    main()
