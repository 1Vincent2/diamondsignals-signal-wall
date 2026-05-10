from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import json
import re

import pandas as pd
from jinja2 import Template

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DIST_DIR = REPO_ROOT / "dist"
HIDDEN_GEMS_DIR = DIST_DIR / "hidden-gems"
HIDDEN_GEMS_JSON = HIDDEN_GEMS_DIR / "mlb_extraction_ledger.json"
TEMPLATES_DIR = BASE_DIR / "templates"

NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")

TIMEZONE_LABEL = "America/New_York"
SOURCE_BADGE = "SRC: EDGE_PIPELINE_v1"
MODEL_BADGE = "MLB_EXTRACTION_v1"

MLB_CODES = {
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
}

MLB_NAME_TO_CODE = {
    "diamondbacks": "ARI",
    "braves": "ATL",
    "orioles": "BAL",
    "red sox": "BOS",
    "cubs": "CHC",
    "white sox": "CWS",
    "reds": "CIN",
    "guardians": "CLE",
    "rockies": "COL",
    "tigers": "DET",
    "astros": "HOU",
    "royals": "KC",
    "angels": "LAA",
    "dodgers": "LAD",
    "marlins": "MIA",
    "brewers": "MIL",
    "twins": "MIN",
    "mets": "NYM",
    "yankees": "NYY",
    "athletics": "ATH",
    "phillies": "PHI",
    "pirates": "PIT",
    "padres": "SD",
    "mariners": "SEA",
    "giants": "SF",
    "cardinals": "STL",
    "rays": "TB",
    "rangers": "TEX",
    "blue jays": "TOR",
    "nationals": "WSH",
}
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

def fetch_player_team_identity(player_id: int) -> dict:
    try:
        import requests

        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=currentTeam"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
        people = payload.get("people", []) or []
        if not people:
            return {}

        person = people[0]
        current_team = person.get("currentTeam") or {}
        team_name = str(current_team.get("name") or "").strip()

        return {
            "display_org": team_name or "Active MLB",
            "display_team": map_team_to_code(team_name) if team_name else "MLB",
        }
    except Exception:
        return {}


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

    lowered = text.lower()
    for name, abbr in MLB_NAME_TO_CODE.items():
        if name in lowered:
            return abbr

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
    for col in ["team_abbrev", "org_code", "mlb_org_code", "team_code", "team", "org", "parent_org", "mlb_org"]:
        if col in row.index:
            code = map_team_to_code(row.get(col))
            if code != "—":
                return code
    return "—"


def derive_display_org(row: pd.Series) -> str:
    text = first_present(
        row,
        ["org", "parent_org", "mlb_org", "team", "team_name", "affiliate_name"],
        fallback="—",
    )
    if text != "—":
        return text
    return derive_display_team(row)


def load_hidden_gems_source_frame() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    from datetime import date, timedelta
    try:
        from build_dashboard import fetch_statcast_window, build_hitter_signals, build_pitcher_signals
    except ModuleNotFoundError:
        from dashboard.build_dashboard import fetch_statcast_window, build_hitter_signals, build_pitcher_signals

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=14)

    raw = fetch_statcast_window(start_dt, end_dt)
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame(), f"MLB Statcast // {start_dt.isoformat()} to {end_dt.isoformat()}"

    hitters = build_hitter_signals(raw)
    pitchers = build_pitcher_signals(raw)

    if not hitters.empty and "player_name" in hitters.columns:
        hitters["player_name"] = hitters["player_name"].apply(safe_name)

    if not pitchers.empty and "player_name" in pitchers.columns:
        pitchers["player_name"] = pitchers["player_name"].apply(safe_name)

    return hitters, pitchers, f"MLB Statcast // {start_dt.isoformat()} to {end_dt.isoformat()}"


def build_hidden_gems_hitters_from_mlb(signals: pd.DataFrame, team_lookup: dict[int, dict] | None = None) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame()

    latest = signals.copy()
    latest["hidden_gems_score"] = pd.to_numeric(latest.get("edge_score"), errors="coerce").fillna(0).round(1)
    latest["score_class"] = latest["hidden_gems_score"].apply(classify_score)

    latest.loc[latest["hidden_gems_score"] < 70, "score_class"] = "watch"
    latest.loc[latest["hidden_gems_score"] < 65, "score_class"] = "neutral"
    latest["avatar"] = latest["player_name"].map(initials)
    latest["player_id"] = latest.get("batter", "").fillna("").astype(str).str.strip() if "batter" in latest.columns else ""
    latest = backfill_resolved_player_ids(latest)

    team_lookup = team_lookup or {}

    latest["display_org"] = "Active MLB"
    latest["display_team"] = "MLB"

    if "batter" in latest.columns:
        for idx, val in latest["batter"].items():
            try:
                pid = int(val)
            except Exception:
                continue
            info = team_lookup.get(pid) or {}
            org = str(info.get("display_org") or "").strip()
            team = str(info.get("display_team") or "").strip()
            if org:
                latest.at[idx, "display_org"] = org
            if team and team != "—":
                latest.at[idx, "display_team"] = team

    latest["source_badge"] = SOURCE_BADGE
    latest["model_badge"] = MODEL_BADGE

    latest["metric_1_label"] = "Physics Core"
    latest["metric_1"] = latest.get("recent_ev", pd.Series([None] * len(latest))).map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "—")

    latest["metric_2_label"] = "Market Gap"
    if "ev_delta" in latest.columns:
        latest["metric_2"] = latest["ev_delta"].map(lambda x: f"{float(x):+.1f}" if pd.notna(x) else "—")
    else:
        latest["metric_2"] = "—"

    latest["metric_3_label"] = "Market Attention Feed"
    if "rostered_pct" in latest.columns:
        latest["metric_3"] = latest["rostered_pct"].map(
            lambda x: f"{float(x):.1f}%" if pd.notna(x) else "FEED OFFLINE"
        )
    else:
        latest["metric_3"] = "FEED OFFLINE"

    def pick_pills(row):
        score = float(pd.to_numeric(row.get("hidden_gems_score"), errors="coerce") or 0)
        badges = list(row.get("badges") or [])

        if score >= 75:
            pills = badges[:3]
            while len(pills) < 3:
                pills.append("Trend Confirming")
        elif score >= 70:
            pills = badges[:2]
            while len(pills) < 2:
                pills.append("Trend Confirming")
            pills.append("Watchlist")
        else:
            pills = ["Trend Confirming", "Watchlist", "Monitor Only"]

        return pd.Series(pills[:3])

    latest[["pill_1", "pill_2", "pill_3"]] = latest.apply(pick_pills, axis=1)

    if "display_team" in latest.columns:
        latest = latest[latest["display_team"].isin(MLB_CODES)].copy()

    if "ev_delta" in latest.columns:
        latest = latest[pd.to_numeric(latest["ev_delta"], errors="coerce").abs() <= 12].copy()

    if "recent_ev" in latest.columns:
        latest = latest[pd.to_numeric(latest["recent_ev"], errors="coerce").between(85, 110, inclusive="both")].copy()

    if "recent_max_ev" in latest.columns:
        latest = latest[pd.to_numeric(latest["recent_max_ev"], errors="coerce").between(95, 122, inclusive="both")].copy()

    def build_hitter_thesis(row):
        ev = pd.to_numeric(row.get("recent_ev"), errors="coerce")
        ev_delta = pd.to_numeric(row.get("ev_delta"), errors="coerce")
        barrel = pd.to_numeric(row.get("recent_barrel_rate"), errors="coerce")
        bbe = pd.to_numeric(row.get("recent_bbe"), errors="coerce")
        max_ev = pd.to_numeric(row.get("recent_max_ev"), errors="coerce")
        score = pd.to_numeric(row.get("hidden_gems_score"), errors="coerce")

        if pd.notna(ev_delta) and ev_delta >= 4.0 and pd.notna(barrel) and barrel >= 0.08:
            return "Impact quality is already ahead of the production line. The bat has moved before the market has reacted."
        if pd.notna(ev) and ev >= 92 and pd.notna(max_ev) and max_ev >= 112:
            return "The damage profile is live even if the visible output still looks incomplete. This is a hard-contact mismatch."
        if pd.notna(barrel) and barrel >= 0.10 and pd.notna(bbe) and bbe >= 15:
            return "Barrel-like contact is showing up often enough to matter. The surface stats have not fully caught up yet."
        if pd.notna(ev_delta) and ev_delta >= 2.0:
            return "Average impact has improved faster than the box-score line. The market is still staring at the wrong layer."
        if pd.notna(score) and score >= 75:
            return "Underlying offensive quality is stronger than current market recognition. This is a clean extraction candidate."
        return "The ballistic profile is firmer than the visible production line. Market recognition still looks late."

    latest["why_hidden"] = latest.apply(build_hitter_thesis, axis=1)

    latest["trend_note"] = latest.get("sample_note", "MLB Statcast")
    latest["trend_glow"] = latest.get("trend_glow", False)

    return latest.sort_values("hidden_gems_score", ascending=False).head(12).reset_index(drop=True)


def build_hidden_gems_pitchers_from_mlb(signals: pd.DataFrame, team_lookup: dict[int, dict] | None = None) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame()

    latest = signals.copy()
    latest["hidden_gems_score"] = pd.to_numeric(latest.get("edge_score"), errors="coerce").fillna(0).round(1)
    latest["score_class"] = latest["hidden_gems_score"].apply(classify_score)

    latest.loc[latest["hidden_gems_score"] < 70, "score_class"] = "watch"
    latest.loc[latest["hidden_gems_score"] < 65, "score_class"] = "neutral"
    latest["avatar"] = latest["player_name"].map(initials)
    latest["player_id"] = latest.get("pitcher", "").fillna("").astype(str).str.strip() if "pitcher" in latest.columns else ""
    latest = backfill_resolved_player_ids(latest)

    team_lookup = team_lookup or {}

    latest["display_org"] = "Active MLB"
    latest["display_team"] = "MLB"

    if "pitcher" in latest.columns:
        for idx, val in latest["pitcher"].items():
            try:
                pid = int(val)
            except Exception:
                continue
            info = team_lookup.get(pid) or {}
            org = str(info.get("display_org") or "").strip()
            team = str(info.get("display_team") or "").strip()
            if org:
                latest.at[idx, "display_org"] = org
            if team and team != "—":
                latest.at[idx, "display_team"] = team

    latest["source_badge"] = SOURCE_BADGE
    latest["model_badge"] = MODEL_BADGE

    latest["metric_1_label"] = "Physics Core"
    if "recent_whiff_rate" in latest.columns:
        latest["metric_1"] = latest["recent_whiff_rate"].map(lambda x: f"{100*float(x):.1f}%" if pd.notna(x) else "—")
    else:
        latest["metric_1"] = "—"

    latest["metric_2_label"] = "Market Gap"
    if "velo_delta" in latest.columns:
        latest["metric_2"] = latest["velo_delta"].map(lambda x: f"{float(x):+.1f}" if pd.notna(x) else "—")
    else:
        latest["metric_2"] = "—"

    latest["metric_3_label"] = "Market Attention Feed"
    if "rostered_pct" in latest.columns:
        latest["metric_3"] = latest["rostered_pct"].map(
            lambda x: f"{float(x):.1f}%" if pd.notna(x) else "FEED OFFLINE"
        )
    else:
        latest["metric_3"] = "FEED OFFLINE"

    def pick_pills(row):
        score = float(pd.to_numeric(row.get("hidden_gems_score"), errors="coerce") or 0)
        badges = list(row.get("badges") or [])

        if score >= 75:
            pills = badges[:3]
            while len(pills) < 3:
                pills.append("Trend Confirming")
        elif score >= 70:
            pills = badges[:2]
            while len(pills) < 2:
                pills.append("Trend Confirming")
            pills.append("Watchlist")
        else:
            pills = ["Trend Confirming", "Watchlist", "Monitor Only"]

        return pd.Series(pills[:3])

    latest[["pill_1", "pill_2", "pill_3"]] = latest.apply(pick_pills, axis=1)

    if "display_team" in latest.columns:
        latest = latest[latest["display_team"].isin(MLB_CODES)].copy()

    def build_pitcher_thesis(row):
        whiff = pd.to_numeric(row.get("recent_whiff_rate"), errors="coerce")
        whiff_delta = pd.to_numeric(row.get("whiff_delta"), errors="coerce")
        velo = pd.to_numeric(row.get("recent_fb_velo"), errors="coerce")
        velo_delta = pd.to_numeric(row.get("velo_delta"), errors="coerce")
        ext_delta = pd.to_numeric(row.get("extension_delta"), errors="coerce")
        score = pd.to_numeric(row.get("hidden_gems_score"), errors="coerce")

        if pd.notna(velo_delta) and velo_delta >= 1.0 and pd.notna(whiff) and whiff >= 0.18:
            return "Velocity has moved first and the bat-miss support is already there. The visible line usually catches up later."
        if pd.notna(whiff_delta) and whiff_delta >= 0.015 and pd.notna(velo_delta) and velo_delta >= 0.5:
            return "Whiff support is improving while the fastball is still firm. The market has not fully priced in the shape change yet."
        if pd.notna(ext_delta) and ext_delta >= 0.15 and pd.notna(velo) and velo >= 96:
            return "Extension and carry support are stronger than the public read. This is a live stuff-over-surface mismatch."
        if pd.notna(whiff) and whiff >= 0.20:
            return "The miss profile is stronger than the surface line suggests. Underlying pitch quality is beating public perception."
        if pd.notna(velo_delta) and velo_delta >= 1.5:
            return "The fastball has materially improved before the surface results have corrected. That is a classic extraction setup."
        if pd.notna(score) and score >= 75:
            return "The underlying arsenal is grading stronger than the visible results. This is the type of mismatch the market is late to recognize."
        return "Underlying pitch quality is holding firmer than the surface line. The market read still looks behind the actual signal."

    latest["why_hidden"] = latest.apply(build_pitcher_thesis, axis=1)

    latest["trend_note"] = latest.get("sample_note", "MLB Statcast")
    latest["trend_glow"] = latest.get("trend_glow", False)

    return latest.sort_values("hidden_gems_score", ascending=False).head(12).reset_index(drop=True)


def build_pitcher_trend_lookup(df: pd.DataFrame) -> dict[str, dict]:
    if df.empty:
        return {}
    pitchers = df[df["bf"].notna()].copy()
    if pitchers.empty or "kbb_p" not in pitchers.columns:
        return {}

    out: dict[str, dict] = {}
    pitchers["player_name"] = pitchers["player_name"].apply(safe_name)

    for player, group in pitchers.groupby("player_name"):
        group = group.sort_values("week_start")
        vals = pd.to_numeric(group["kbb_p"], errors="coerce").dropna().tolist()
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

    out: dict[str, dict] = {}
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

    latest["trait_score_raw"] = 0.0
    if "ivb" in latest.columns:
        latest["trait_score_raw"] += 0.30 * zscore(latest["ivb"].fillna(0))
    if "whiff_pct" in latest.columns:
        latest["trait_score_raw"] += 0.28 * zscore(latest["whiff_pct"].fillna(0))
    if "velo" in latest.columns:
        latest["trait_score_raw"] += 0.22 * zscore(latest["velo"].fillna(0))
    if "kbb_p" in latest.columns:
        latest["trait_score_raw"] += 0.20 * zscore(latest["kbb_p"].fillna(0))

    latest["surface_pressure_raw"] = 0.0
    if "era" in latest.columns:
        latest["surface_pressure_raw"] += 0.58 * zscore(latest["era"].fillna(latest["era"].median()))
    latest["surface_pressure_raw"] += 0.42 * zscore(latest["trait_score_raw"].fillna(0))

    latest["market_score_raw"] = 0.0
    if "rostered_pct" in latest.columns:
        latest["market_score_raw"] = -1.0 * zscore(latest["rostered_pct"].fillna(latest["rostered_pct"].median()))

    latest["trigger_score_raw"] = 0.0

    latest["hidden_gems_score_raw"] = (
        0.38 * latest["trait_score_raw"].fillna(0)
        + 0.30 * latest["surface_pressure_raw"].fillna(0)
        + 0.20 * latest["market_score_raw"].fillna(0)
        + 0.12 * latest["trigger_score_raw"].fillna(0)
    )

    latest["hidden_gems_score"] = (50 + 15 * zscore(latest["hidden_gems_score_raw"].fillna(0))).clip(5, 95).round(1)
    latest["score_class"] = latest["hidden_gems_score"].apply(classify_score)
    latest["avatar"] = latest["player_name"].map(initials)
    latest["player_type"] = "Pitcher"
    latest = backfill_resolved_player_ids(latest)
    latest["display_team"] = latest.apply(derive_display_team, axis=1)
    latest["display_org"] = latest.apply(derive_display_org, axis=1)

    latest["metric_1_label"] = "Physics Core"
    latest["metric_1"] = latest["trait_score_raw"].fillna(0).map(lambda x: f"{x:.2f}")
    latest["metric_2_label"] = "Market Gap"
    latest["metric_2"] = latest["surface_pressure_raw"].fillna(0).map(lambda x: f"{x:.2f}")
    latest["metric_3_label"] = "Market Attention Feed"
    latest["metric_3"] = latest["market_score_raw"].fillna(0).map(lambda x: f"{x:.2f}")

    def pitcher_summary(r: pd.Series) -> str:
        trait_score = float(r.get("trait_score_raw", 0) or 0)
        divergence_score = float(r.get("surface_pressure_raw", 0) or 0)
        market_score = float(r.get("market_score_raw", 0) or 0)

        ivb = r.get("ivb")
        whiff = r.get("whiff_pct")
        velo = r.get("velo")
        kbb = r.get("kbb_p")

        lead_trait = "underlying pitch quality"
        if pd.notna(whiff) and pd.notna(kbb):
            lead_trait = "bat-miss and strike-efficiency support"
        if pd.notna(ivb) and pd.notna(velo):
            lead_trait = "shape and velocity support"
        elif pd.notna(ivb):
            lead_trait = "shape support"
        elif pd.notna(velo):
            lead_trait = "velocity support"
        elif pd.notna(kbb):
            lead_trait = "command and strikeout support"

        if trait_score >= 0.55 and divergence_score >= 0.65:
            return f"{lead_trait.capitalize()} is running materially stronger than the current visible line."
        if divergence_score >= 0.70:
            return f"Surface results still look weaker than the underlying {lead_trait}."
        if market_score >= 0.20 and trait_score >= 0.30:
            return f"{lead_trait.capitalize()} looks stronger than current market attention implies."
        if trait_score >= 0.45:
            return f"{lead_trait.capitalize()} gives this arm a stronger hidden profile than the public line suggests."
        return "Underlying support remains better than the current surface read."

    latest["why_hidden"] = latest.apply(pitcher_summary, axis=1)

    latest["pill_1"] = latest.apply(
        lambda r: "Ballistic Breakout" if float(r["trait_score_raw"]) >= 0.40 else "Under-the-Hood Elite",
        axis=1,
    )
    latest["pill_2"] = latest.apply(
        lambda r: "Latent Alpha" if float(r["surface_pressure_raw"]) >= 0.50 else "Divergence Watch",
        axis=1,
    )
    latest["pill_3"] = latest.apply(
        lambda r: "Low Market Price" if float(r["market_score_raw"]) >= 0.15 else "Market Early",
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

    numeric_cols = ["pa", "iso", "hr", "avg", "ev90", "ev_90", "rostered_pct"]
    for col in numeric_cols:
        if col in hitters.columns:
            hitters[col] = pd.to_numeric(hitters[col], errors="coerce")

    trend_lookup = build_hitter_trend_lookup(hitters)
    latest = hitters.sort_values("week_start").groupby("player_name", as_index=False).tail(1).copy()

    ev90_col = "ev90" if "ev90" in latest.columns else "ev_90" if "ev_90" in latest.columns else None

    latest["trait_score_raw"] = 0.0
    if ev90_col:
        latest["trait_score_raw"] += 0.42 * zscore(latest[ev90_col].fillna(0))
    if "iso" in latest.columns:
        latest["trait_score_raw"] += 0.34 * zscore(latest["iso"].fillna(0))
    if "hr" in latest.columns:
        latest["trait_score_raw"] += 0.24 * zscore(latest["hr"].fillna(0))

    latest["surface_pressure_raw"] = 0.0
    if "avg" in latest.columns:
        latest["surface_pressure_raw"] += 0.58 * zscore((-1 * latest["avg"]).fillna(0))
    latest["surface_pressure_raw"] += 0.42 * zscore(latest["trait_score_raw"].fillna(0))

    latest["market_score_raw"] = 0.0
    if "rostered_pct" in latest.columns:
        latest["market_score_raw"] = -1.0 * zscore(latest["rostered_pct"].fillna(latest["rostered_pct"].median()))

    latest["trigger_score_raw"] = 0.0

    latest["hidden_gems_score_raw"] = (
        0.40 * latest["trait_score_raw"].fillna(0)
        + 0.28 * latest["surface_pressure_raw"].fillna(0)
        + 0.22 * latest["market_score_raw"].fillna(0)
        + 0.10 * latest["trigger_score_raw"].fillna(0)
    )

    latest["hidden_gems_score"] = (50 + 15 * zscore(latest["hidden_gems_score_raw"].fillna(0))).clip(5, 95).round(1)
    latest["score_class"] = latest["hidden_gems_score"].apply(classify_score)
    latest["avatar"] = latest["player_name"].map(initials)
    latest["player_type"] = "Hitter"
    latest = backfill_resolved_player_ids(latest)
    latest["display_team"] = latest.apply(derive_display_team, axis=1)
    latest["display_org"] = latest.apply(derive_display_org, axis=1)

    latest["metric_1_label"] = "Physics Core"
    latest["metric_1"] = latest["trait_score_raw"].fillna(0).map(lambda x: f"{x:.2f}")
    latest["metric_2_label"] = "Market Gap"
    latest["metric_2"] = latest["surface_pressure_raw"].fillna(0).map(lambda x: f"{x:.2f}")
    latest["metric_3_label"] = "Market Attention Feed"
    latest["metric_3"] = latest["market_score_raw"].fillna(0).map(lambda x: f"{x:.2f}")

    def hitter_summary(r: pd.Series) -> str:
        trait_score = float(r.get("trait_score_raw", 0) or 0)
        divergence_score = float(r.get("surface_pressure_raw", 0) or 0)
        market_score = float(r.get("market_score_raw", 0) or 0)

        iso = r.get("iso")
        hr = r.get("hr")
        ev = r.get("ev90") if "ev90" in r.index else r.get("ev_90") if "ev_90" in r.index else None

        lead_trait = "underlying offensive quality"
        if pd.notna(ev) and pd.notna(iso):
            lead_trait = "impact quality and power shape"
        elif pd.notna(ev):
            lead_trait = "impact quality"
        elif pd.notna(iso) and pd.notna(hr):
            lead_trait = "power shape and damage support"
        elif pd.notna(iso):
            lead_trait = "power shape"
        elif pd.notna(hr):
            lead_trait = "damage support"

        if trait_score >= 0.55 and divergence_score >= 0.65:
            return f"{lead_trait.capitalize()} is materially stronger than the current production line."
        if divergence_score >= 0.70:
            return f"Visible production still looks behind the deeper {lead_trait}."
        if market_score >= 0.20 and trait_score >= 0.30:
            return f"{lead_trait.capitalize()} looks better than current market pricing suggests."
        if trait_score >= 0.45:
            return f"{lead_trait.capitalize()} gives this bat more hidden value than the public line implies."
        return "Underlying offensive support still looks better than the current surface read."

    latest["why_hidden"] = latest.apply(hitter_summary, axis=1)

    latest["pill_1"] = latest.apply(
        lambda r: "Ballistic Breakout" if float(r["trait_score_raw"]) >= 0.40 else "Under-the-Hood Elite",
        axis=1,
    )
    latest["pill_2"] = latest.apply(
        lambda r: "Latent Alpha" if float(r["surface_pressure_raw"]) >= 0.50 else "Divergence Watch",
        axis=1,
    )
    latest["pill_3"] = latest.apply(
        lambda r: "Low Market Price" if float(r["market_score_raw"]) >= 0.15 else "Market Early",
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


MLB_EXTRACTION_TEMPLATE = (TEMPLATES_DIR / "hidden_gems" / "mlb_extraction.html").read_text(encoding="utf-8")
HTML_TEMPLATE = Template(MLB_EXTRACTION_TEMPLATE)



def _json_clean_value(value):
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _json_records(df, kind: str) -> list[dict]:
    if df is None or df.empty:
        return []

    records = []
    for idx, row in df.head(12).reset_index(drop=True).iterrows():
        raw = {k: _json_clean_value(v) for k, v in row.to_dict().items()}

        player_id = str(raw.get("player_id") or raw.get("batter") or raw.get("pitcher") or "").strip()
        name = str(raw.get("player_name") or raw.get("name") or "Player X").strip()
        team = str(raw.get("display_team") or raw.get("team") or raw.get("display_org") or "MLB").strip()
        score = raw.get("hidden_gems_score") or raw.get("edge_score") or raw.get("signal_score") or 0

        try:
            score_display = f"{float(score):.1f}".rstrip("0").rstrip(".")
        except Exception:
            score_display = str(score or "0")

        metric_value = None
        metric_label = "MARKET ATTENTION CONTEXT"

        if kind == "pitcher":
            if raw.get("recent_velo_delta") is not None:
                metric_label = "PITCHING PHYSICS CONTEXT"
                metric_value = f"{float(raw.get('recent_velo_delta')):+.1f} mph velocity delta"
            elif raw.get("recent_whiff_rate") is not None:
                metric_label = "PITCHING PHYSICS CONTEXT"
                metric_value = f"{float(raw.get('recent_whiff_rate')):.1%} recent whiff rate"
            elif raw.get("recent_fastball_velo") is not None:
                metric_label = "PITCHING PHYSICS CONTEXT"
                metric_value = f"{float(raw.get('recent_fastball_velo')):.1f} mph fastball"
        else:
            if raw.get("recent_ev_delta") is not None:
                metric_label = "CONTACT QUALITY CONTEXT"
                metric_value = f"{float(raw.get('recent_ev_delta')):+.1f} mph EV delta"
            elif raw.get("recent_avg_ev") is not None:
                metric_label = "CONTACT QUALITY CONTEXT"
                metric_value = f"{float(raw.get('recent_avg_ev')):.1f} mph recent EV"
            elif raw.get("recent_barrel_rate") is not None:
                metric_label = "CONTACT QUALITY CONTEXT"
                metric_value = f"{float(raw.get('recent_barrel_rate')):.1%} barrel rate"

        if not metric_value:
            metric_value = str(raw.get("trait_label") or raw.get("supporting_metric") or "Roster % lagging signal score")

        records.append({
            "rank": int(idx + 1),
            "kind": kind,
            "player_id": player_id,
            "name": name,
            "displayName": name.upper(),
            "team": team,
            "role": "SP" if kind == "pitcher" else "BAT",
            "score": score_display,
            "signal_id": f"DS-MLB-EXTRACTION-{player_id or idx + 1}",
            "diagnosis": str(raw.get("verdict") or raw.get("pill_1") or "UNDERPRICED MLB ASSET").upper(),
            "metric_label": metric_label,
            "metric_value": metric_value,
            "body_copy": str(raw.get("why_hidden") or "Standard fantasy market behavior is lagging. This profile shows stronger underlying support than the visible market has fully absorbed."),
            "raw": raw,
        })

    return records


def write_mlb_extraction_json(pitchers, hitters, generated_at: str, latest_week_start: str) -> None:
    pitcher_records = _json_records(pitchers, "pitcher")
    hitter_records = _json_records(hitters, "hitter")

    top_signals = sorted(
        pitcher_records + hitter_records,
        key=lambda r: float(r.get("score") or 0),
        reverse=True,
    )[:12]

    payload = {
        "report": "MLB Extraction Ledger",
        "subtitle": "Market Latency / Hidden Gems Surface",
        "version": "mlb_extraction_ledger_v0.1",
        "generated_at": generated_at,
        "latest_week_start": latest_week_start,
        "status": "real_data_v0.1",
        "top_signals": top_signals,
        "top_pitchers": pitcher_records,
        "top_hitters": hitter_records,
    }

    HIDDEN_GEMS_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def render_html() -> str:
    hitter_source, pitcher_source, source_window = load_hidden_gems_source_frame()

    hitter_team_lookup = {}
    if hitter_source is not None and not hitter_source.empty and "batter" in hitter_source.columns:
        for _, row in hitter_source.iterrows():
            try:
                pid = int(row.get("batter"))
            except Exception:
                continue
            hitter_team_lookup[pid] = fetch_player_team_identity(pid) or {}

    pitcher_team_lookup = {}
    if pitcher_source is not None and not pitcher_source.empty and "pitcher" in pitcher_source.columns:
        for _, row in pitcher_source.iterrows():
            try:
                pid = int(row.get("pitcher"))
            except Exception:
                continue
            pitcher_team_lookup[pid] = fetch_player_team_identity(pid) or {}

    pitchers = build_hidden_gems_pitchers_from_mlb(pitcher_source, pitcher_team_lookup)
    hitters = build_hidden_gems_hitters_from_mlb(hitter_source, hitter_team_lookup)

    generated_at = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    write_mlb_extraction_json(pitchers, hitters, generated_at, source_window)

    return HTML_TEMPLATE.render(
        generated_at=generated_at,
        latest_week_start=source_window,
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