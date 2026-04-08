from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import re

import pandas as pd
from jinja2 import Template

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DIST_DIR = REPO_ROOT / "dist"
HIDDEN_GEMS_DIR = DIST_DIR / "hidden-gems"
TEMPLATES_DIR = BASE_DIR / "templates"

NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")

TIMEZONE_LABEL = "America/New_York"
SOURCE_BADGE = "SRC: EDGE_PIPELINE_v1"
MODEL_BADGE = "HIDDEN_GEMS_v1"

MLB_CODES = {
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
}


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


def safe_text(value, fallback="—") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def initials(name: str) -> str:
    parts = [p for p in str(name).split() if p]
    if not parts:
        return "—"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][:1] + parts[-1][:1]).upper()


def build_polyline(values: list[float], width: int = 120, height: int = 34, pad: int = 3) -> str:
    cleaned = []
    for v in values:
        try:
            f = float(v)
            if math.isfinite(f):
                cleaned.append(f)
        except Exception:
            continue

    if not cleaned:
        cleaned = [0.0, 0.0]
    if len(cleaned) == 1:
        cleaned = [cleaned[0], cleaned[0]]

    vmin = min(cleaned)
    vmax = max(cleaned)
    if vmax == vmin:
        vmax = vmin + 1.0

    if len(cleaned) == 2:
        xs = [0, width]
    else:
        xs = [i * width / (len(cleaned) - 1) for i in range(len(cleaned))]

    points = []
    for x, v in zip(xs, cleaned):
        pct = (v - vmin) / (vmax - vmin)
        y = (height - pad) - (pct * (height - pad * 2))
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


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


def clean_candidate_code(value: str) -> str:
    text = re.sub(r"[^A-Za-z]", "", str(value or "")).upper()
    return text[:4] if text else ""


def map_team_to_code(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "—"
    code = clean_candidate_code(text)
    if code in MLB_CODES:
        return code
    return "—"


def first_present(row: pd.Series, candidates: list[str], fallback="—") -> str:
    for col in candidates:
        if col in row.index:
            val = row.get(col)
            if val is None or pd.isna(val):
                continue
            text = str(val).strip()
            if text and text.lower() not in {"nan", "none", "null"}:
                return text
    return fallback


def derive_display_team(row: pd.Series) -> str:
    for col in ["team_abbrev", "org_code", "mlb_org_code", "team_code", "team"]:
        if col in row.index:
            code = map_team_to_code(row.get(col))
            if code != "—":
                return code
    return "—"


def derive_display_org(row: pd.Series) -> str:
    return first_present(
        row,
        ["org", "parent_org", "mlb_org", "team", "team_name", "affiliate_name"],
        fallback="—",
    )


def load_hidden_gems_source_frame() -> pd.DataFrame:
    from supabase import create_client
    import os

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    sb = create_client(url, key)

    resp = (
        sb.table("milb_aaa_weekly_signal_base")
        .select("*")
        .order("week_start", desc=True)
        .limit(300)
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


def build_pitcher_trend_lookup(df: pd.DataFrame) -> dict[str, dict]:
    if df.empty:
        return {}
    pitchers = df[df["bf"].notna()].copy()
    if pitchers.empty:
        return {}

    metric_col = "kbb_p" if "kbb_p" in pitchers.columns else None
    if metric_col is None:
        return {}

    out = {}
    pitchers["player_name"] = pitchers["player_name"].apply(safe_name)

    for player, group in pitchers.groupby("player_name"):
        group = group.sort_values("week_start")
        vals = pd.to_numeric(group[metric_col], errors="coerce").dropna().tolist()
        if not vals:
            continue
        out[player] = {
            "trend_points": build_polyline(vals),
            "trend_glow": len(vals) >= 2 and vals[-1] > vals[-2],
            "trend_note": f"{len(vals)}W K/BB",
        }
    return out


def build_hitter_trend_lookup(df: pd.DataFrame) -> dict[str, dict]:
    if df.empty:
        return {}
    hitters = df[df["pa"].notna()].copy()
    if hitters.empty:
        return {}

    metric_col = None
    if "iso" in hitters.columns:
        metric_col = "iso"
    elif "hr" in hitters.columns:
        metric_col = "hr"

    if metric_col is None:
        return {}

    out = {}
    hitters["player_name"] = hitters["player_name"].apply(safe_name)

    for player, group in hitters.groupby("player_name"):
        group = group.sort_values("week_start")
        vals = pd.to_numeric(group[metric_col], errors="coerce").dropna().tolist()
        if not vals:
            continue
        out[player] = {
            "trend_points": build_polyline(vals),
            "trend_glow": len(vals) >= 2 and vals[-1] > vals[-2],
            "trend_note": f"{len(vals)}W {metric_col.upper()}",
        }
    return out


def build_hidden_gems_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    pitchers = df[df["bf"].notna()].copy()
    if pitchers.empty:
        return pd.DataFrame()

    numeric_cols = [
        "bf", "so_p", "bb_allowed", "kbb_p", "era", "whiff_pct",
        "ivb", "velo", "rostered_pct"
    ]
    for col in numeric_cols:
        if col in pitchers.columns:
            pitchers[col] = pd.to_numeric(pitchers[col], errors="coerce")

    trend_lookup = build_pitcher_trend_lookup(pitchers)
    latest = pitchers.sort_values("week_start").groupby("player_name", as_index=False).tail(1).copy()

    latest["trait_score"] = 0.0
    if "ivb" in latest.columns:
        latest["trait_score"] = latest["trait_score"] + 0.30 * zscore(latest["ivb"].fillna(0))
    if "whiff_pct" in latest.columns:
        latest["trait_score"] = latest["trait_score"] + 0.28 * zscore(latest["whiff_pct"].fillna(0))
    if "velo" in latest.columns:
        latest["trait_score"] = latest["trait_score"] + 0.22 * zscore(latest["velo"].fillna(0))
    if "kbb_p" in latest.columns:
        latest["trait_score"] = latest["trait_score"] + 0.20 * zscore(latest["kbb_p"].fillna(0))

    latest["divergence_score"] = 0.0
    if "era" in latest.columns:
        latest["divergence_score"] = latest["divergence_score"] + 0.58 * zscore(latest["era"].fillna(latest["era"].median()))
    latest["divergence_score"] = latest["divergence_score"] + 0.42 * zscore(latest["trait_score"].fillna(0))

    latest["market_score"] = 0.0
    if "rostered_pct" in latest.columns:
        latest["market_score"] = -1.0 * zscore(latest["rostered_pct"].fillna(latest["rostered_pct"].median()))

    latest["trigger_score"] = 0.0

    latest["hidden_gems_score_raw"] = (
        0.38 * latest["trait_score"].fillna(0)
        + 0.30 * latest["divergence_score"].fillna(0)
        + 0.20 * latest["market_score"].fillna(0)
        + 0.12 * latest["trigger_score"].fillna(0)
    )

    latest["hidden_gems_score"] = (50 + 15 * zscore(latest["hidden_gems_score_raw"].fillna(0))).clip(5, 95).round(1)
    latest["score_class"] = latest["hidden_gems_score"].apply(classify_score)
    latest["avatar"] = latest["player_name"].map(initials)
    latest["player_type"] = "Pitcher"
    latest["display_team"] = latest.apply(derive_display_team, axis=1)
    latest["display_org"] = latest.apply(derive_display_org, axis=1)

    latest["metric_1_label"] = "Trait"
    latest["metric_1"] = latest["trait_score"].fillna(0).map(lambda x: f"{x:.2f}")
    latest["metric_2_label"] = "Divergence"
    latest["metric_2"] = latest["divergence_score"].fillna(0).map(lambda x: f"{x:.2f}")
    latest["metric_3_label"] = "Market"
    latest["metric_3"] = latest["market_score"].fillna(0).map(lambda x: f"{x:.2f}")

    latest["why_hidden"] = latest.apply(
        lambda r: "Strong shape and bat-miss traits are running ahead of visible results; public line may still be lagging.",
        axis=1,
    )

    latest["pill_1"] = latest.apply(
        lambda r: "Ballistic Breakout" if float(r["trait_score"]) >= 0.40 else "Under-the-Hood Elite",
        axis=1,
    )
    latest["pill_2"] = latest.apply(
        lambda r: "Latent Alpha" if float(r["divergence_score"]) >= 0.50 else "Divergence Watch",
        axis=1,
    )
    latest["pill_3"] = latest.apply(
        lambda r: "Low Market Price" if float(r["market_score"]) >= 0.15 else "Market Early",
        axis=1,
    )

    latest["trend_points"] = "0,24 24,22 48,20 72,19 96,16 120,14"
    latest["trend_glow"] = latest["hidden_gems_score"] >= 65
    latest["trend_note"] = "Recent Form"

    for idx, row in latest.iterrows():
        info = trend_lookup.get(row["player_name"])
        if info:
            latest.at[idx, "trend_points"] = info["trend_points"]
            latest.at[idx, "trend_glow"] = bool(info["trend_glow"])
            latest.at[idx, "trend_note"] = info["trend_note"]

    latest["source_badge"] = SOURCE_BADGE
    latest["model_badge"] = MODEL_BADGE

    return latest.sort_values("hidden_gems_score", ascending=False).head(10).reset_index(drop=True)


def build_hidden_gems_hitters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    hitters = df[df["pa"].notna()].copy()
    if hitters.empty:
        return pd.DataFrame()

    numeric_cols = [
        "pa", "iso", "hr", "avg", "ev90", "ev_90", "rostered_pct"
    ]
    for col in numeric_cols:
        if col in hitters.columns:
            hitters[col] = pd.to_numeric(hitters[col], errors="coerce")

    trend_lookup = build_hitter_trend_lookup(hitters)
    latest = hitters.sort_values("week_start").groupby("player_name", as_index=False).tail(1).copy()

    ev90_col = "ev90" if "ev90" in latest.columns else "ev_90" if "ev_90" in latest.columns else None

    latest["trait_score"] = 0.0
    if ev90_col:
        latest["trait_score"] = latest["trait_score"] + 0.42 * zscore(latest[ev90_col].fillna(0))
    if "iso" in latest.columns:
        latest["trait_score"] = latest["trait_score"] + 0.34 * zscore(latest["iso"].fillna(0))
    if "hr" in latest.columns:
        latest["trait_score"] = latest["trait_score"] + 0.24 * zscore(latest["hr"].fillna(0))

    latest["divergence_score"] = 0.0
    if "avg" in latest.columns:
        latest["divergence_score"] = latest["divergence_score"] + 0.58 * zscore((-1 * latest["avg"]).fillna(0))
    latest["divergence_score"] = latest["divergence_score"] + 0.42 * zscore(latest["trait_score"].fillna(0))

    latest["market_score"] = 0.0
    if "rostered_pct" in latest.columns:
        latest["market_score"] = -1.0 * zscore(latest["rostered_pct"].fillna(latest["rostered_pct"].median()))

    latest["trigger_score"] = 0.0

    latest["hidden_gems_score_raw"] = (
        0.40 * latest["trait_score"].fillna(0)
        + 0.28 * latest["divergence_score"].fillna(0)
        + 0.22 * latest["market_score"].fillna(0)
        + 0.10 * latest["trigger_score"].fillna(0)
    )

    latest["hidden_gems_score"] = (50 + 15 * zscore(latest["hidden_gems_score_raw"].fillna(0))).clip(5, 95).round(1)
    latest["score_class"] = latest["hidden_gems_score"].apply(classify_score)
    latest["avatar"] = latest["player_name"].map(initials)
    latest["player_type"] = "Hitter"
    latest["display_team"] = latest.apply(derive_display_team, axis=1)
    latest["display_org"] = latest.apply(derive_display_org, axis=1)

    latest["metric_1_label"] = "Trait"
    latest["metric_1"] = latest["trait_score"].fillna(0).map(lambda x: f"{x:.2f}")
    latest["metric_2_label"] = "Divergence"
    latest["metric_2"] = latest["divergence_score"].fillna(0).map(lambda x: f"{x:.2f}")
    latest["metric_3_label"] = "Market"
    latest["metric_3"] = latest["market_score"].fillna(0).map(lambda x: f"{x:.2f}")

    latest["why_hidden"] = latest.apply(
        lambda r: "Ballistic quality is stronger than visible production; public-facing results may still be understating the bat.",
        axis=1,
    )

    latest["pill_1"] = latest.apply(
        lambda r: "Ballistic Breakout" if float(r["trait_score"]) >= 0.40 else "Under-the-Hood Elite",
        axis=1,
    )
    latest["pill_2"] = latest.apply(
        lambda r: "Latent Alpha" if float(r["divergence_score"]) >= 0.50 else "Divergence Watch",
        axis=1,
    )
    latest["pill_3"] = latest.apply(
        lambda r: "Low Market Price" if float(r["market_score"]) >= 0.15 else "Market Early",
        axis=1,
    )

    latest["trend_points"] = "0,25 24,23 48,22 72,18 96,16 120,13"
    latest["trend_glow"] = latest["hidden_gems_score"] >= 65
    latest["trend_note"] = "Recent Form"

    for idx, row in latest.iterrows():
        info = trend_lookup.get(row["player_name"])
        if info:
            latest.at[idx, "trend_points"] = info["trend_points"]
            latest.at[idx, "trend_glow"] = bool(info["trend_glow"])
            latest.at[idx, "trend_note"] = info["trend_note"]

    latest["source_badge"] = SOURCE_BADGE
    latest["model_badge"] = MODEL_BADGE

    return latest.sort_values("hidden_gems_score", ascending=False).head(10).reset_index(drop=True)


HTML_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals // Hidden Gems</title>
  <style>
    :root {
      --bg: #080808;
      --surface: #121212;
      --card-radial: radial-gradient(circle at top left, #171717 0%, #080808 100%);
      --border: #2d2d2d;
      --text: #f0f0f0;
      --muted: #71717a;
      --soft: #a1a1aa;
      --tiny: #7c7c84;
      --blue: #6aa6ff;
      --blue-soft: #8ab4ff;
      --lime-hot: #b6ff00;
      --red: #ef4444;
      --gold: #fbbf24;
      --shadow: 0 14px 34px rgba(0, 0, 0, 0.34);
      --radius: 18px;
      --mono: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--sans); }
    body {
      background:
        radial-gradient(circle at top left, rgba(106,166,255,0.05), transparent 24%),
        radial-gradient(circle at top right, rgba(239,68,68,0.03), transparent 20%),
        linear-gradient(180deg, #101010 0%, #080808 34%, #050505 100%);
      line-height: 1.35;
    }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 50;
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
      gap: 10px;
      min-width: 0;
    }

    .brand-mark {
      width: 11px;
      height: 11px;
      border-radius: 999px;
      background: var(--lime-hot);
      box-shadow: 0 0 10px rgba(182,255,0,0.35);
      flex: 0 0 auto;
    }

    .brand-kicker {
      font-size: 10px;
      line-height: 1;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-weight: 800;
      margin-bottom: 4px;
    }

    .brand-white { color: var(--text); }
    .brand-blue { color: var(--blue); }

    .brand-title {
      font-size: 16px;
      line-height: 1.05;
      letter-spacing: -0.02em;
      font-weight: 800;
    }

    .livebox { text-align: right; }
    .live-label {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--soft);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .live-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--red);
      box-shadow: 0 0 10px rgba(239,68,68,0.35);
    }

    .live-time {
      margin-top: 6px;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--tiny);
    }

    .app {
      padding: 28px 0 56px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
      margin-bottom: 20px;
    }

    .hero-card,
    .summary-card,
    .section,
    .drawer-panel {
      background: var(--card-radial);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }

    .hero-card { padding: 22px 22px 20px; }
    .summary-card { padding: 18px; display: grid; gap: 12px; align-content: start; }
    .section { padding: 18px; margin-bottom: 16px; }

    .eyebrow {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--blue);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      margin-bottom: 12px;
    }

    .hero-title {
      margin: 0;
      font-size: clamp(32px, 6vw, 56px);
      line-height: 0.95;
      letter-spacing: -0.05em;
      text-transform: uppercase;
      font-weight: 900;
    }

    .hero-sub {
      margin: 14px 0 0;
      max-width: 60ch;
      color: var(--soft);
      font-size: 14px;
      line-height: 1.6;
    }

    .summary-label {
      font-family: var(--mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--tiny);
    }

    .summary-value {
      font-size: 28px;
      line-height: 1;
      font-weight: 800;
      letter-spacing: -0.03em;
    }

    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 16px;
    }

    .section-head-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .section-kicker {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--blue);
      margin-bottom: 6px;
    }

    .section-title {
      margin: 0;
      font-size: 18px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .section-badge,
    .field-guide-btn {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      color: var(--soft);
      border-radius: 999px;
      padding: 7px 11px;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .field-guide-btn {
      cursor: pointer;
      transition: 160ms ease;
    }

    .field-guide-btn:hover {
      color: var(--text);
      border-color: rgba(106,166,255,0.20);
      background: rgba(106,166,255,0.06);
    }

    .cards {
      display: grid;
      gap: 12px;
    }

    .player-card {
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255,255,255,0.02);
    }

    .player-top {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 14px;
      align-items: start;
    }

    .avatar {
      width: 42px;
      height: 42px;
      border-radius: 999px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 800;
      color: var(--soft);
      flex: 0 0 auto;
    }

    .rankline {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--tiny);
      margin-bottom: 5px;
    }

    .player-name {
      margin: 0;
      font-size: 20px;
      line-height: 1;
      letter-spacing: -0.03em;
      font-weight: 800;
    }

    .signal-line {
      margin-top: 6px;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--soft);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .card-meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }

    .card-meta-badge {
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      border-radius: 999px;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      color: var(--soft);
    }

    .card-meta-badge.team {
      color: var(--blue-soft);
      border-color: rgba(106,166,255,0.18);
      background: rgba(106,166,255,0.06);
    }

    .scorebox { text-align: right; min-width: 72px; }

    .score-label {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--tiny);
      margin-bottom: 6px;
    }

    .score-value {
      font-size: 30px;
      line-height: 1;
      font-weight: 800;
      letter-spacing: -0.04em;
    }

    .score-value.positive,
    .score-value.elite {
      color: var(--lime-hot);
    }

    .score-value.watch {
      color: var(--gold);
    }

    .score-value.neutral {
      color: var(--text);
    }

    .sparkline-wrap {
      margin-top: 14px;
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px;
      background: rgba(255,255,255,0.025);
      padding: 10px 12px;
    }

    .sparkline-head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }

    .sparkline-label,
    .sparkline-note {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--tiny);
    }

    .sparkline {
      width: 100%;
      height: 34px;
      display: block;
    }

    .sparkline-path {
      fill: none;
      stroke-width: 2.5;
      stroke-linecap: round;
      stroke-linejoin: round;
    }

    .sparkline-path.glow {
      filter: drop-shadow(0 0 4px rgba(182,255,0,0.22));
    }

    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 5px 9px;
      border-radius: 999px;
      font-family: var(--mono);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      border: 1px solid rgba(255,255,255,0.10);
      white-space: nowrap;
    }

    .pill.primary {
      color: var(--lime-hot);
      border-color: rgba(182,255,0,0.20);
      background: rgba(182,255,0,0.05);
    }

    .pill.secondary {
      color: var(--blue-soft);
      border-color: rgba(106,166,255,0.20);
      background: rgba(106,166,255,0.06);
    }

    .pill.tertiary {
      color: var(--gold);
      border-color: rgba(251,191,36,0.20);
      background: rgba(251,191,36,0.06);
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }

    .metric {
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px;
      padding: 10px;
      background: rgba(255,255,255,0.02);
      min-width: 0;
    }

    .metric-label {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--tiny);
    }

    .metric-value {
      margin-top: 6px;
      font-size: 18px;
      font-weight: 800;
      line-height: 1.15;
      word-break: break-word;
    }

    .why {
      margin-top: 14px;
      font-size: 10px;
      line-height: 1.5;
      color: #8b8b94;
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
    }

    .drawer-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.48);
      opacity: 0;
      pointer-events: none;
      transition: opacity 180ms ease;
      z-index: 120;
    }

    .drawer-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }

    .drawer-panel {
      position: fixed;
      top: 18px;
      right: 18px;
      width: min(460px, calc(100vw - 24px));
      max-height: calc(100vh - 36px);
      overflow: auto;
      padding: 18px;
      transform: translateX(108%);
      transition: transform 220ms ease;
      z-index: 130;
    }

    .drawer-panel.open {
      transform: translateX(0);
    }

    .drawer-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }

    .drawer-title-wrap {
      min-width: 0;
    }

    .drawer-kicker {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--blue);
      margin-bottom: 6px;
    }

    .drawer-title {
      margin: 0;
      font-size: 22px;
      line-height: 1;
      letter-spacing: -0.03em;
      font-weight: 800;
      text-transform: uppercase;
    }

    .drawer-close {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      color: var(--soft);
      border-radius: 999px;
      padding: 8px 11px;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      cursor: pointer;
    }

    .drawer-close:hover {
      color: var(--text);
      border-color: rgba(255,255,255,0.14);
    }

    .guide-list {
      display: grid;
      gap: 10px;
    }

    .guide-item {
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
      padding: 12px;
    }

    .guide-term {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text);
      margin-bottom: 6px;
    }

    .guide-def {
      color: var(--soft);
      font-size: 13px;
      line-height: 1.5;
    }

    {{ shell_styles | safe }}

    @media (max-width: 900px) {
      .hero {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 640px) {
      .topbar-inner,
      .app,
      .topnav-inner,
      .search-strip-inner {
        width: min(100%, calc(100% - 16px));
      }

      .player-top {
        grid-template-columns: auto 1fr;
      }

      .scorebox {
        grid-column: 2;
        text-align: left;
        margin-top: 8px;
      }

      .metric-grid {
        grid-template-columns: 1fr;
      }

      .player-name {
        font-size: 17px;
      }

      .score-value {
        font-size: 24px;
      }

      .section-head {
        flex-direction: column;
        align-items: stretch;
      }

      .section-head-actions {
        justify-content: flex-start;
      }

      .drawer-panel {
        top: 8px;
        right: 8px;
        width: calc(100vw - 16px);
        max-height: calc(100vh - 16px);
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
          <div class="brand-title">Hidden Gems // Institutional Edge</div>
        </div>
      </div>
      <div class="livebox">
        <div class="live-label"><span class="live-dot"></span>LIVE</div>
        <div class="live-time">{{ generated_at }}</div>
      </div>
    </div>
  </div>

  {{ nav_html | safe }}
  {{ search_html | safe }}

  <div class="app">
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">Signal Wall // Edge</div>
        <h1 class="hero-title">Hidden Gems</h1>
        <p class="hero-sub">
          Under-the-hood elite profiles whose visible surface results may still be masking breakout value.
        </p>
      </div>

      <div class="summary-card">
        <div>
          <div class="summary-label">Mode</div>
          <div class="summary-value">EDGE</div>
        </div>
        <div>
          <div class="summary-label">Pitchers</div>
          <div class="summary-value">{{ pitchers|length }}</div>
        </div>
        <div>
          <div class="summary-label">Hitters</div>
          <div class="summary-value">{{ hitters|length }}</div>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <div class="section-kicker">Latent Alpha</div>
          <h2 class="section-title">Pitcher Hidden Gems</h2>
        </div>
        <div class="section-head-actions">
          <button type="button" class="field-guide-btn" onclick="openGlossary()">Field Guide</button>
          <div class="section-badge">Top {{ pitchers|length }}</div>
        </div>
      </div>

      {% if pitchers %}
      <div class="cards">
        {% for row in pitchers %}
        <article class="player-card">
          <div class="player-top">
            <div class="avatar">{{ row.avatar }}</div>
            <div class="player-ident">
              <div class="rankline">#{{ loop.index }} Divergence Pitcher</div>
              <h3 class="player-name">{{ row.player_name }}</h3>
              <div class="signal-line">{{ row.display_org }} // Pitcher // Hidden Gems</div>
              <div class="card-meta-row">
                <span class="card-meta-badge">{{ row.source_badge }}</span>
                <span class="card-meta-badge">{{ row.model_badge }}</span>
                {% if row.display_team != "—" %}<span class="card-meta-badge team">{{ row.display_team }}</span>{% endif %}
              </div>
            </div>
            <div class="scorebox">
              <div class="score-label">Edge Score</div>
              <div class="score-value {{ row.score_class }}">{{ row.hidden_gems_score }}</div>
            </div>
          </div>

          <div class="sparkline-wrap">
            <div class="sparkline-head">
              <div class="sparkline-label">Ballistics vs Surface</div>
              <div class="sparkline-note">{{ row.trend_note }}</div>
            </div>
            <svg class="sparkline" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <linearGradient id="pitcherGemGradient{{ loop.index }}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#444444" stop-opacity="0.65"></stop>
                  <stop offset="100%" stop-color="{% if row.hidden_gems_score >= 65 %}#b6ff00{% else %}#6aa6ff{% endif %}" stop-opacity="1"></stop>
                </linearGradient>
              </defs>
              <polyline class="sparkline-path {% if row.trend_glow %}glow{% endif %}" stroke="url(#pitcherGemGradient{{ loop.index }})" points="{{ row.trend_points }}" />
            </svg>
          </div>

          <div class="pill-row">
            <span class="pill primary">{{ row.pill_1 }}</span>
            <span class="pill secondary">{{ row.pill_2 }}</span>
            <span class="pill tertiary">{{ row.pill_3 }}</span>
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

          <div class="why">{{ row.why_hidden }}</div>
        </article>
        {% endfor %}
      </div>
      {% else %}
      <div class="why">No pitcher hidden gems available yet.</div>
      {% endif %}
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <div class="section-kicker">Latent Alpha</div>
          <h2 class="section-title">Hitter Hidden Gems</h2>
        </div>
        <div class="section-head-actions">
          <button type="button" class="field-guide-btn" onclick="openGlossary()">Field Guide</button>
          <div class="section-badge">Top {{ hitters|length }}</div>
        </div>
      </div>

      {% if hitters %}
      <div class="cards">
        {% for row in hitters %}
        <article class="player-card">
          <div class="player-top">
            <div class="avatar">{{ row.avatar }}</div>
            <div class="player-ident">
              <div class="rankline">#{{ loop.index }} Divergence Hitter</div>
              <h3 class="player-name">{{ row.player_name }}</h3>
              <div class="signal-line">{{ row.display_org }} // Hitter // Hidden Gems</div>
              <div class="card-meta-row">
                <span class="card-meta-badge">{{ row.source_badge }}</span>
                <span class="card-meta-badge">{{ row.model_badge }}</span>
                {% if row.display_team != "—" %}<span class="card-meta-badge team">{{ row.display_team }}</span>{% endif %}
              </div>
            </div>
            <div class="scorebox">
              <div class="score-label">Edge Score</div>
              <div class="score-value {{ row.score_class }}">{{ row.hidden_gems_score }}</div>
            </div>
          </div>

          <div class="sparkline-wrap">
            <div class="sparkline-head">
              <div class="sparkline-label">Ballistics vs Surface</div>
              <div class="sparkline-note">{{ row.trend_note }}</div>
            </div>
            <svg class="sparkline" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden="true">
              <defs>
                <linearGradient id="hitterGemGradient{{ loop.index }}" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stop-color="#444444" stop-opacity="0.65"></stop>
                  <stop offset="100%" stop-color="{% if row.hidden_gems_score >= 65 %}#b6ff00{% else %}#6aa6ff{% endif %}" stop-opacity="1"></stop>
                </linearGradient>
              </defs>
              <polyline class="sparkline-path {% if row.trend_glow %}glow{% endif %}" stroke="url(#hitterGemGradient{{ loop.index }})" points="{{ row.trend_points }}" />
            </svg>
          </div>

          <div class="pill-row">
            <span class="pill primary">{{ row.pill_1 }}</span>
            <span class="pill secondary">{{ row.pill_2 }}</span>
            <span class="pill tertiary">{{ row.pill_3 }}</span>
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

          <div class="why">{{ row.why_hidden }}</div>
        </article>
        {% endfor %}
      </div>
      {% else %}
      <div class="why">No hitter hidden gems available yet.</div>
      {% endif %}
    </section>

    {{ footer_html | safe }}
  </div>

  <div id="drawerOverlay" class="drawer-overlay" onclick="closeGlossary()"></div>

  <aside id="glossaryDrawer" class="drawer-panel" aria-hidden="true">
    <div class="drawer-head">
      <div class="drawer-title-wrap">
        <div class="drawer-kicker">Hidden Gems // Field Guide</div>
        <h2 class="drawer-title">Field Guide</h2>
      </div>
      <button type="button" class="drawer-close" onclick="closeGlossary()">Close</button>
    </div>

    <div class="guide-list">
      <div class="guide-item">
        <div class="guide-term">Hidden Gem</div>
        <div class="guide-def">A player whose underlying traits look stronger than the public-facing results currently suggest.</div>
      </div>

      <div class="guide-item">
        <div class="guide-term">Latent Alpha</div>
        <div class="guide-def">Unpriced upside that may not yet be reflected in roster rate, ownership, or visible surface stats.</div>
      </div>

      <div class="guide-item">
        <div class="guide-term">Ballistic Breakout</div>
        <div class="guide-def">A profile where movement, shape, power, or impact quality is beginning to point toward a stronger future result set.</div>
      </div>

      <div class="guide-item">
        <div class="guide-term">Divergence</div>
        <div class="guide-def">The gap between strong underlying traits and weaker box-score outcomes. This is the core Hidden Gems lens.</div>
      </div>

      <div class="guide-item">
        <div class="guide-term">Under-the-Hood Elite</div>
        <div class="guide-def">A player whose deeper traits are materially stronger than their recent public perception.</div>
      </div>

      <div class="guide-item">
        <div class="guide-term">Trait Score</div>
        <div class="guide-def">A composite measure of underlying quality, such as shape, whiff, velocity, or impact traits.</div>
      </div>

      <div class="guide-item">
        <div class="guide-term">Divergence Score</div>
        <div class="guide-def">A measure of how much the player’s visible results lag their deeper trait profile.</div>
      </div>

      <div class="guide-item">
        <div class="guide-term">Market Score</div>
        <div class="guide-def">A rough estimate of how overlooked the player still appears to be relative to the strength of the profile.</div>
      </div>

      <div class="guide-item">
        <div class="guide-term">Ballistics vs Surface</div>
        <div class="guide-def">The sparkline area is meant to frame the relationship between rising underlying quality and slower-moving public results.</div>
      </div>

      <div class="guide-item">
        <div class="guide-term">Low Market Price</div>
        <div class="guide-def">A player who may still be available cheaply because the market has not fully recognized the profile yet.</div>
      </div>
    </div>
  </aside>

  <script>
    function openGlossary() {
      const overlay = document.getElementById("drawerOverlay");
      const drawer = document.getElementById("glossaryDrawer");
      if (overlay) overlay.classList.add("open");
      if (drawer) {
        drawer.classList.add("open");
        drawer.setAttribute("aria-hidden", "false");
      }
    }

    function closeGlossary() {
      const overlay = document.getElementById("drawerOverlay");
      const drawer = document.getElementById("glossaryDrawer");
      if (overlay) overlay.classList.remove("open");
      if (drawer) {
        drawer.classList.remove("open");
        drawer.setAttribute("aria-hidden", "true");
      }
    }

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") closeGlossary();
    });
  </script>
</body>
</html>
"""
)


def render_html() -> str:
    df = load_hidden_gems_source_frame()
    pitchers = build_hidden_gems_pitchers(df)
    hitters = build_hidden_gems_hitters(df)

    return HTML_TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        timezone_label=TIMEZONE_LABEL,
        nav_html=Template(NAV_TEMPLATE).render(active_nav="hidden_gems"),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
        pitchers=pitchers.to_dict(orient="records"),
        hitters=hitters.to_dict(orient="records"),
    )


def main() -> None:
    HIDDEN_GEMS_DIR.mkdir(parents=True, exist_ok=True)
    html = render_html()
    output_path = HIDDEN_GEMS_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()