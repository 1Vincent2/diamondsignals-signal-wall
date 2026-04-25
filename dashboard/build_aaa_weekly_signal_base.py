from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import pandas as pd
from supabase import create_client

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DIST_DIR = REPO_ROOT / "dist"


def safe_name(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return "Unknown"
    return " ".join(part.capitalize() for part in text.split())


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / std


def classify_score(score: float) -> str:
    try:
        score = float(score)
    except Exception:
        return "neutral"
    if score >= 80:
        return "elite"
    if score >= 65:
        return "positive"
    if score >= 50:
        return "watch"
    return "neutral"


def fetch_raw_weekly() -> pd.DataFrame:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    sb = create_client(url, key)

    resp = (
        sb.table("milb_raw_weekly")
        .select("*")
        .eq("level", "AAA")
        .order("week_start", desc=True)
        .limit(5000)
        .execute()
    )

    data = resp.data or []
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].apply(safe_name)
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    return df


def build_signal_base(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()

    numeric_cols = ["pa", "bb", "so", "hr", "iso", "ev90", "wrc_plus", "bf", "so_p", "bb_allowed"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out["pa"].notna() | out["bf"].notna()].copy()
    if out.empty:
        return pd.DataFrame()

    if {"so", "bb"}.issubset(out.columns):
        out["kbb_h"] = out.apply(
            lambda r: (float(r["so"]) / float(r["bb"])) if pd.notna(r["bb"]) and float(r["bb"]) > 0 else (float(r["so"]) if pd.notna(r["so"]) else 0.0),
            axis=1,
        )
    else:
        out["kbb_h"] = None

    if {"so_p", "bb_allowed"}.issubset(out.columns):
        out["kbb_p"] = out.apply(
            lambda r: (float(r["so_p"]) / float(r["bb_allowed"])) if pd.notna(r["bb_allowed"]) and float(r["bb_allowed"]) > 0 else (float(r["so_p"]) if pd.notna(r["so_p"]) else 0.0),
            axis=1,
        )
    else:
        out["kbb_p"] = None

    hitter_mask = out["pa"].notna()
    pitcher_mask = out["bf"].notna()

    out["edge_score"] = None
    out["score_class"] = "neutral"

    if hitter_mask.any():
        hitters = out.loc[hitter_mask].copy()
        hitters["bb_rate"] = (hitters["bb"] / hitters["pa"]).fillna(0)
        hitters["k_rate"] = (hitters["so"] / hitters["pa"]).fillna(0)
        kbb_series = hitters["kbb_h"].replace(0, pd.NA)
        kbb_fill = kbb_series.dropna().median()
        if pd.isna(kbb_fill):
            kbb_fill = 1.0

        hitters["edge_score_raw"] = (
            50
            + 12 * zscore(hitters["iso"].fillna(0))
            - 10 * zscore(kbb_series.fillna(kbb_fill))
            + 8 * zscore(hitters["bb_rate"])
            - 6 * zscore(hitters["k_rate"])
            + 4 * zscore(hitters["hr"].fillna(0))
            + 2 * zscore(hitters["pa"].fillna(0))
        )
        hitters["edge_score"] = hitters["edge_score_raw"].clip(5, 95).round(1)
        hitters["score_class"] = hitters["edge_score"].apply(classify_score)
        out.loc[hitters.index, "edge_score"] = hitters["edge_score"]
        out.loc[hitters.index, "score_class"] = hitters["score_class"]

    if pitcher_mask.any():
        pitchers = out.loc[pitcher_mask].copy()
        kbb_series = pitchers["kbb_p"].replace(0, pd.NA)
        kbb_fill = kbb_series.dropna().median()
        if pd.isna(kbb_fill):
            kbb_fill = 1.0

        pitchers["k_rate_proxy"] = (pitchers["so_p"] / pitchers["bf"]).fillna(0)
        pitchers["bb_rate_proxy"] = (pitchers["bb_allowed"] / pitchers["bf"]).fillna(0)
        pitchers["edge_score_raw"] = (
            50
            + 14 * zscore(kbb_series.fillna(kbb_fill))
            + 8 * zscore(pitchers["k_rate_proxy"])
            - 7 * zscore(pitchers["bb_rate_proxy"])
            + 3 * zscore(pitchers["bf"].fillna(0))
        )
        pitchers["edge_score"] = pitchers["edge_score_raw"].clip(5, 95).round(1)
        pitchers["score_class"] = pitchers["edge_score"].apply(classify_score)
        out.loc[pitchers.index, "edge_score"] = pitchers["edge_score"]
        out.loc[pitchers.index, "score_class"] = pitchers["score_class"]

    out["updated_at"] = datetime.utcnow().isoformat()
    return out


def main() -> None:
    df = fetch_raw_weekly()
    if df.empty:
        print("No AAA raw weekly rows found.")
        return

    out = build_signal_base(df)
    path = DIST_DIR / "aaa_weekly_signal_base_preview.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
