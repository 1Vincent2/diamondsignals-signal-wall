#!/usr/bin/env python3
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from jinja2 import Template
from pybaseball import playerid_reverse_lookup, statcast

DIST_DIR = Path("dist")
DIST_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = Path(__file__).parent / "templates"
NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")
LEDGER_STYLES_TEMPLATE = (TEMPLATES_DIR / "ledger_styles.css").read_text(encoding="utf-8")
HOME_SIGNAL_LEDGER_CARD_TEMPLATE = (TEMPLATES_DIR / "components" / "home_signal_ledger_card.html").read_text(encoding="utf-8")
HOME_SIGNAL_LEDGER_CARD = Template(HOME_SIGNAL_LEDGER_CARD_TEMPLATE)
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

    missing_mask = out["resolved_player_id"].eq("")
    if missing_mask.any() and "player_name" in out.columns:
        out.loc[missing_mask, "resolved_player_id"] = (
            out.loc[missing_mask, "player_name"]
            .fillna("")
            .astype(str)
            .map(resolve_player_id_by_name)
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
    merged["metric_1_label"] = "Whiff %"
    merged["metric_2"] = merged["recent_fb_velo"].round(1)
    merged["metric_2_label"] = "FB Velo"
    merged["metric_3"] = merged["recent_extension"].round(1).map(
        lambda x: f"{x:.1f} ft" if pd.notna(x) else "—"
    )
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

def load_supplemental_aaa_dossier_players() -> list[dict]:
    try:
        from supabase import create_client
        import os

        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        sb = create_client(url, key)

        resp = (
            sb.table("milb_aaa_weekly_signal_base")
            .select("player_id,player_name,org,week_start")
            .order("week_start", desc=True)
            .limit(1200)
            .execute()
)

        data = resp.data or []
        if not data:
            return []

        df_extra = pd.DataFrame(data)
        if df_extra.empty or "player_id" not in df_extra.columns:
            return []

        df_extra["player_id"] = pd.to_numeric(df_extra["player_id"], errors="coerce")
        df_extra = df_extra[df_extra["player_id"].notna()].copy()
        if df_extra.empty:
            return []

        df_extra["player_id"] = df_extra["player_id"].astype(int)
        if "week_start" in df_extra.columns:
            df_extra["week_start"] = pd.to_datetime(df_extra["week_start"], errors="coerce")

        df_extra = df_extra.sort_values(
            ["player_id", "week_start"] if "week_start" in df_extra.columns else ["player_id"],
            ascending=[True, False] if "week_start" in df_extra.columns else [True],
        ).drop_duplicates(subset=["player_id"], keep="first")

        supplemental_players: list[dict] = []

        for _, row in df_extra.iterrows():
            pid = safe_int(row.get("player_id"))
            if pid is None:
                continue

            full_name = str(row.get("player_name") or "").strip() or f"Player {pid}"

            team_value = str(row.get("org") or "").strip()  

            supplemental_players.append(
                {
                    "player_id": pid,
                    "full_name": full_name,
                    "first_name": "",
                    "last_name": "",
                    "team": team_value,
                    "team_name": team_value,
                    "position": "",
                    "bats": "",
                    "throws": "",
                    "status": "",
                    "headshot_url": build_headshot_url(pid),
                }
            )

        return supplemental_players

    except Exception:
        return []
    
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
    supplemental_players = load_supplemental_aaa_dossier_players()

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
    .meta-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .meta-card { padding: 14px; }
    .meta-label, .metric-label, .sparkline-label, .section-kicker, .score-label, .rankline, .status-badge { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 800; }
    .meta-label { margin-bottom: 6px; }
    .meta-value { font-family: var(--mono); font-size: 13px; color: var(--text); font-variant-numeric: tabular-nums; }
    .slate-heat-card { padding: 14px; }
    .slate-heat-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; }
    .slate-heat-bar { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.04); overflow: hidden; }
    .slate-heat-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, rgba(214,164,58,0.55) 0%, #b7f000 100%); box-shadow: 0 0 10px rgba(183,240,0,0.18); }
    .slate-heat-value { font-family: var(--mono); font-size: 13px; color: var(--text); font-variant-numeric: tabular-nums; }
    .section-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 16px 16px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); background: linear-gradient(180deg, rgba(214,164,58,0.06), rgba(255,255,255,0.01)); }
    .section-title { margin: 0; font-size: 18px; font-weight: 800; letter-spacing: -0.02em; text-transform: uppercase; }
    .section-badge { font-family: var(--mono); font-size: 11px; color: #d7dbe6; border: 1px solid rgba(255,255,255,0.12); border-radius: 999px; padding: 7px 10px; background: rgba(255,255,255,0.04); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02); }
    .cards { display: grid; gap: 10px; padding: 10px; }
    .player-card { padding: 14px; transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease; }
    .player-card.js-player-card { cursor: pointer; }
    .player-card.js-player-card:hover { transform: translateY(-1px); box-shadow: var(--shadow), 0 0 12px rgba(106,166,255,0.08); border-color: rgba(106,166,255,0.18); }
    .player-card.high-edge { border-color: rgba(74,222,128,0.22); box-shadow: var(--shadow), 0 0 8px rgba(74,222,128,0.07); }
    .player-top { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 10px; align-items: start; margin-bottom: 12px; }
    .avatar { width: 42px; height: 42px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.10); display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.03); color: var(--text); font-size: 13px; font-weight: 800; flex: 0 0 auto; }
    .player-ident { min-width: 0; max-width: 100%; }
    .rankline { margin-bottom: 4px; }
    .player-name { font-size: clamp(18px, 1.55vw, 26px); line-height: 0.98; letter-spacing: -0.035em; font-weight: 900; margin: 0 0 4px; text-transform: uppercase; color: var(--text); word-break: break-word; overflow-wrap: anywhere; }
    .signal-line { font-size: 10px; color: var(--soft); font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.05em; }
    .scorebox { display: flex; align-items: center; justify-content: flex-end; gap: 8px; min-width: 238px; flex: 0 0 238px; }
    .score-meta { display: flex; flex-direction: column; align-items: flex-end; justify-content: center; text-align: right; gap: 3px; min-width: 74px; flex: 0 0 74px; }
    .score-value { font-family: var(--mono); font-size: 28px; line-height: 1; font-weight: 800; color: var(--hg-amber); text-shadow: 0 0 8px rgba(214,164,58,0.14); }
    .score-value.edge-up { color: var(--hg-lime); text-shadow: 0 0 10px rgba(183,240,0,0.22); }
    .sparkline-wrap { margin: 0 0 12px; padding: 8px 10px; border: 1px solid rgba(255,255,255,0.04); border-radius: 12px; background: rgba(255,255,255,0.015); }
    .sparkline-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
    .sparkline-note { font-family: var(--mono); font-size: 10px; color: var(--tiny); text-transform: uppercase; letter-spacing: 0.1em; }
    svg.sparkline { display: block; width: 100%; height: 34px; }
    .sparkline-path { stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; fill: none; }
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
      min-height: 34px;
      padding: 0 12px;
      border-radius: 10px;
      border: 1px solid rgba(96,165,250,0.26);
      background: linear-gradient(180deg, rgba(18,18,18,0.96) 0%, rgba(8,8,8,0.96) 100%);
      color: #dbeafe;
      font-family: var(--mono);
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      cursor: pointer;
      transition: 160ms ease;
      white-space: nowrap;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }
    .provision-btn:hover {
      transform: translateY(-1px);
      border-color: rgba(96,165,250,0.42);
      color: #ffffff;
      box-shadow: 0 0 14px rgba(59,130,246,0.14);
    }
    .footer { padding: 16px 4px 0; color: var(--muted); font-family: var(--mono); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; font-variant-numeric: tabular-nums; }
    @keyframes badgePulse { 0%,100% { opacity: 0.82; } 50% { opacity: 1; } }
    @media (min-width: 900px) {
      .hero-grid { grid-template-columns: 1.35fr 0.9fr; align-items: stretch; }
      .board { grid-template-columns: 1fr 1fr; }
    }
    {{ shell_styles | safe }}
    {{ ledger_styles | safe }}
    @media (max-width: 640px) {
      .topbar-inner, .app, .topnav-inner, .search-strip-inner { width: min(100%, calc(100% - 16px)); }
      .search-strip-inner { justify-content: stretch; }
      .player-search { width: 100%; }
      .player-search-input { height: 36px; font-size: 12px; }
      .brand-title { font-size: 14px; }
      .hero-title { font-size: 34px; letter-spacing: -0.035em; }
      .meta-grid { grid-template-columns: 1fr; }
      .player-name { font-size: 17px; }
      .score-value { font-size: 23px; }
      .scorebox { min-width: 0; flex: 0 0 auto; gap: 8px; }
      .score-meta { min-width: 64px; flex: 0 0 64px; }
      .player-search-result { grid-template-columns: 120px 1fr; }
      .player-search-avatar { width: 120px; height: 120px; }
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
      <button class="info-trigger" type="button" onclick="openGlossary()" aria-label="Open glossary">ⓘ Glossary</button>
      <div class="livebox">
        <div class="live-label"><span class="live-dot"></span>LIVE</div>
        <div class="live-time">{{ generated_at }}</div>
      </div>
    </div>
  </div>

  {{ nav_html | safe }}
  {{ search_html | safe }}

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
        <div class="glossary-item"><span class="glossary-term">Slate Heat</span><div class="glossary-definition">A model-driven index of total opportunity across the day's schedule.</div></div>
        <div class="glossary-item"><span class="glossary-term">System Status</span><div class="glossary-definition">Confirms the live state of the Statcast-driven pipeline.</div></div>
        <div class="glossary-item"><span class="glossary-term">Edge Score</span><div class="glossary-definition">A 0 to 100 ranking summarizing signal strength versus baseline.</div></div>
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
          <p class="hero-copy">A live, mobile-first DiamondSignals board built for fast scan readability.</p>
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
            <div class="slate-heat-bar"><div class="slate-heat-fill" style="width: {{ slate_heat }}%;"></div></div>
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

    @media (max-width: 900px) {
      .player-id-grid { grid-template-columns: 1fr; justify-items: start; }
      .signal-stack { justify-items: start; min-width: 0; }
      .metrics-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 640px) {
      .topbar-inner, .app, .topnav-inner, .search-strip-inner { width: min(100%, calc(100% - 16px)); }
      .search-strip-inner { justify-content: stretch; }
      .player-search { width: 100%; }
      .player-search-input { height: 36px; font-size: 12px; }
      .headshot-shell { width: 92px; height: 92px; border-radius: 18px; }
      .player-name { font-size: 28px; }
    }

    {shell_styles}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="brand-mark"></div>
        <div class="brand-text">
          <div class="brand-kicker"><span class="brand-white">DIAMOND</span><span class="brand-blue">SIGNALS</span></div>
          <div class="brand-title">Scout Terminal // Player Dossier</div>
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
      <h1 class="hero-title">Player Dossier Reference</h1>
      <p class="hero-copy">
        DiamondSignals scout shell v2. This page hydrates dossiers from the canonical dossier JSON payload.
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
          const res = await fetch(path);
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
        const dossierPayload = await fetchJsonWithFallback([
          "../../dossier_canon.json",
          "../dossier_canon.json",
          "/dossier_canon.json",
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
    nav_html = Template(NAV_TEMPLATE).render(active_nav="scout_dossier")
    search_html = SEARCH_TEMPLATE
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

    (scout_dir / "index.html").write_text(shell_html, encoding="utf-8")
    print("Wrote dist/scout/index.html")

    players = dossier_payload.get("players", {}) if isinstance(dossier_payload, dict) else {}
    player_ids = sorted(players.keys())

    for player_id in player_ids:
        player_dir = scout_dir / str(player_id)
        player_dir.mkdir(parents=True, exist_ok=True)
        (player_dir / "index.html").write_text(shell_html, encoding="utf-8")

    print(f"Wrote {len(player_ids)} player dossier pages under dist/scout/<player_id>/index.html")


def main() -> None:
    raw = fetch_statcast_window(START_DATE, END_DATE)
    if raw.empty:
        print(
            "Skipping dashboard rebuild because Statcast fallback returned no fresh data. Keeping existing dist assets."
        )
        return

    hitter_signals = build_hitter_signals(raw)
    pitcher_signals = build_pitcher_signals(raw)

    if hitter_signals.empty and pitcher_signals.empty:
        raise RuntimeError("No hitter or pitcher signals were produced.")

    top_hitters = hitter_signals.head(5).copy()
    top_pitchers = pitcher_signals.head(5).copy()

    top_hitters["player_id"] = top_hitters["batter"].fillna("").astype(str).str.strip()
    top_pitchers["player_id"] = top_pitchers["pitcher"].fillna("").astype(str).str.strip()

    top_hitters = backfill_resolved_player_ids(top_hitters)
    top_pitchers = backfill_resolved_player_ids(top_pitchers)

    combined_alerts = pd.concat([top_pitchers, top_hitters], ignore_index=True)
    combined_alerts = combined_alerts.sort_values(
        "edge_score", ascending=False
    ).reset_index(drop=True)

    html = render_html(top_pitchers, top_hitters)

    live_path = DIST_DIR / "live" / "index.html"
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text(html, encoding="utf-8")
    print(f"Wrote {live_path}")

    front_door_html = render_signals_front_door()
    output_path = DIST_DIR / "index.html"
    output_path.write_text(front_door_html, encoding="utf-8")
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

    (DIST_DIR / "signals.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Wrote dist/signals.json")

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