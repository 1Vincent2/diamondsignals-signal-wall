from __future__ import annotations

from datetime import datetime
import os

import pandas as pd
from supabase import create_client

SOURCE_BADGE = "SRC: AAA_PIPELINE_v1"
SCORE_VERSION = "EDGE_v2.0"
TABLE_NAME = "prospect_intelligence_daily"


def get_supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)


def as_int_or_none(value):
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def as_float_or_none(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def fetch_source_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from build_typical_call_up import (
        fetch_latest_aaa_weekly_signal_base,
        build_aaa_hitter_promotion_watch,
        build_aaa_pitcher_promotion_watch,
    )

    base_df = fetch_latest_aaa_weekly_signal_base()
    if base_df is None or base_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    base_df = base_df.copy()
    hitters = build_aaa_hitter_promotion_watch(base_df)
    pitchers = build_aaa_pitcher_promotion_watch(base_df)

    return base_df, hitters, pitchers

def build_prospect_intelligence_rows(
    base_df: pd.DataFrame,
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    snapshot_date,
) -> list[dict]:
    if base_df.empty:
        return []

    hitter_map: dict[int, dict] = {}
    pitcher_map: dict[int, dict] = {}

    if not hitters.empty:
        for _, row in hitters.iterrows():
            player_id = row.get("player_id") or row.get("mlbam_id") or row.get("mlb_id")
            try:
                player_id_int = int(player_id)
            except Exception:
                continue
            hitter_map[player_id_int] = row.to_dict()

    if not pitchers.empty:
        for _, row in pitchers.iterrows():
            player_id = row.get("player_id") or row.get("mlbam_id") or row.get("mlb_id")
            try:
                player_id_int = int(player_id)
            except Exception:
                continue
            pitcher_map[player_id_int] = row.to_dict()

    rows: list[dict] = []

    snapshot_ts = pd.Timestamp(snapshot_date)

    for _, row in base_df.iterrows():
        player_id = row.get("player_id") or row.get("mlbam_id") or row.get("mlb_id")
        try:
            player_id = int(player_id)
        except Exception:
            continue

        player_name = row.get("player_name") or "Unknown Player"
        org = row.get("org") or row.get("team") or None
        level = row.get("level") or "AAA"

        hitter_row = hitter_map.get(player_id)
        pitcher_row = pitcher_map.get(player_id)

        signal_type = None
        position_group = None
        edge_score = None
        signal_archetype = None
        scout_narrative = None

        last_7d_iso = None
        k_bb_ratio = None
        exit_velo_90th = None
        bb_rate = None
        k_rate = None
        k_rate_proxy = None
        bb_rate_proxy = None
        bf = None
        pa = None
        trend_points = None
        trend_glow = False

        source_date = row.get("week_start")
        source_ts = pd.to_datetime(source_date, errors="coerce")
        data_freshness_hours = None
        latency_warning = False

        if pd.notna(source_ts):
            freshness_hours = (snapshot_ts - source_ts).total_seconds() / 3600.0
            if freshness_hours >= 0:
                data_freshness_hours = round(freshness_hours, 1)
                latency_warning = data_freshness_hours > 48

        if hitter_row is not None:
            signal_type = "Hitter"
            position_group = "Hitter"
            edge_score = hitter_row.get("edge_score")

            last_7d_iso = row.get("iso")

            if pd.notna(row.get("kbb_h")):
                try:
                    k_bb_ratio = float(row.get("kbb_h"))
                except Exception:
                    k_bb_ratio = None

            ev90_candidates = [
                row.get("exit_velo_90th"),
                row.get("ev_90"),
                row.get("ev90"),
                row.get("p90_ev"),
                row.get("exit_velocity_90th"),
            ]
            for candidate in ev90_candidates:
                if pd.notna(candidate):
                    try:
                        exit_velo_90th = float(candidate)
                        break
                    except Exception:
                        pass

            bb_rate = hitter_row.get("bb_rate")
            k_rate = hitter_row.get("k_rate")
            pa = row.get("pa")
            trend_points = hitter_row.get("trend_points")
            trend_glow = bool(hitter_row.get("trend_glow", False))

            iso_val = as_float_or_none(last_7d_iso)
            bb_rate_val = as_float_or_none(bb_rate)

            if iso_val is not None and iso_val >= 0.250:
                signal_archetype = "Impact Bat"
            elif k_bb_ratio is not None and k_bb_ratio <= 1.50:
                signal_archetype = "Zone Control"
            elif pd.notna(row.get("hr")) and float(row.get("hr")) >= 2:
                signal_archetype = "HR Surge"
            elif bb_rate_val is not None and bb_rate_val >= 0.12:
                signal_archetype = "On-Base Pressure"
            else:
                signal_archetype = "Promotion Watch"

            score_txt = f"{float(edge_score):.1f}" if pd.notna(edge_score) else "--"
            iso_txt = f"{iso_val:.3f}" if iso_val is not None else "--"
            bb_txt = f"{bb_rate_val * 100:.1f}%" if bb_rate_val is not None else "--"
            kbb_txt = f"{float(k_bb_ratio):.2f}" if k_bb_ratio is not None else "--"

            scout_narrative = (
                f"{player_name}'s {score_txt} Edge Score is driven by ISO of {iso_txt}, "
                f"K/BB of {kbb_txt}, and BB rate of {bb_txt}."
            )

        elif pitcher_row is not None:
            signal_type = "Pitcher"
            position_group = "Pitcher"
            edge_score = pitcher_row.get("edge_score")

            if pd.notna(row.get("kbb_p")):
                try:
                    k_bb_ratio = float(row.get("kbb_p"))
                except Exception:
                    k_bb_ratio = None

            k_rate_proxy = pitcher_row.get("k_rate_proxy")
            bb_rate_proxy = pitcher_row.get("bb_rate_proxy")
            bf = row.get("bf")
            trend_points = pitcher_row.get("trend_points")
            trend_glow = bool(pitcher_row.get("trend_glow", False))

            kbb_val = as_float_or_none(k_bb_ratio)
            k_rate_proxy_val = as_float_or_none(k_rate_proxy)
            bb_rate_proxy_val = as_float_or_none(bb_rate_proxy)

            if kbb_val is not None and kbb_val >= 4.0:
                signal_archetype = "Bat-Miss Ready"
            elif bb_rate_proxy_val is not None and bb_rate_proxy_val <= 0.08:
                signal_archetype = "Command Hold"
            elif pd.notna(row.get("so_p")) and float(row.get("so_p")) >= 10:
                signal_archetype = "Whiff Volume"
            else:
                signal_archetype = "Promotion Watch"

            score_txt = f"{float(edge_score):.1f}" if pd.notna(edge_score) else "--"
            kbb_txt = f"{kbb_val:.2f}" if kbb_val is not None else "--"
            k_txt = f"{k_rate_proxy_val * 100:.1f}%" if k_rate_proxy_val is not None else "--"
            bb_txt = f"{bb_rate_proxy_val * 100:.1f}%" if bb_rate_proxy_val is not None else "--"

            scout_narrative = (
                f"{player_name}'s {score_txt} Edge Score is driven by K/BB of {kbb_txt}, "
                f"K rate proxy of {k_txt}, and BB rate proxy of {bb_txt}."
            )

        else:
            continue

        rows.append(
            {
                "snapshot_date": str(snapshot_date),
                "player_id": as_int_or_none(player_id),
                "player_name": player_name,
                "org": org,
                "level": level,
                "signal_type": signal_type,
                "position_group": position_group,
                "edge_score": as_float_or_none(edge_score),
                "score_version": SCORE_VERSION,
                "source_badge": SOURCE_BADGE,
                "data_freshness_hours": as_float_or_none(data_freshness_hours),
                "latency_warning": bool(latency_warning),
                "signal_archetype": signal_archetype,
                "is_recent_arrival": False,
                "arrival_type": None,
                "arrival_date": None,
                "last_7d_iso": as_float_or_none(last_7d_iso),
                "k_bb_ratio": as_float_or_none(k_bb_ratio),
                "exit_velo_90th": as_float_or_none(exit_velo_90th),
                "bb_rate": as_float_or_none(bb_rate),
                "k_rate": as_float_or_none(k_rate),
                "k_rate_proxy": as_float_or_none(k_rate_proxy),
                "bb_rate_proxy": as_float_or_none(bb_rate_proxy),
                "bf": as_int_or_none(bf),
                "pa": as_int_or_none(pa),
                "trend_points": trend_points,
                "trend_glow": bool(trend_glow),
                "scout_narrative": scout_narrative,
            }
        )

    return rows

def upsert_rows(rows: list[dict]) -> None:
    if not rows:
        return

    client = get_supabase_client()
    client.table(TABLE_NAME).upsert(
        rows,
        on_conflict="snapshot_date,player_id",
    ).execute()


def main() -> None:
    snapshot_date = datetime.now().date()
    base_df, hitters, pitchers = fetch_source_frame()
    rows = build_prospect_intelligence_rows(base_df, hitters, pitchers, snapshot_date)
    upsert_rows(rows)
    print(f"Upserted {len(rows)} rows into {TABLE_NAME}")


if __name__ == "__main__":
    main()