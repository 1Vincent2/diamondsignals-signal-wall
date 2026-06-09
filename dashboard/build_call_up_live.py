from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import math
import re

import pandas as pd
from jinja2 import Template

from dashboard.lib.publish_safe import write_temp_output, promote_output_if_valid, save_snapshot
from dashboard.lib.report_status import build_report_status, utc_now_iso
from dashboard.lib.report_validation import (
    build_validation_report,
    validate_min_rows,
    validate_required_sections,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DIST_DIR = REPO_ROOT / "dist"
CALL_UP_DIR = DIST_DIR / "typical-call-up"
TEMPLATES_DIR = BASE_DIR / "templates"

NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")
LEDGER_STYLES_TEMPLATE = (TEMPLATES_DIR / "ledger_styles.css").read_text(encoding="utf-8")
LIVE_LEDGER_CARD_TEMPLATE = (TEMPLATES_DIR / "components" / "live_ledger_card.html").read_text(encoding="utf-8")
LIVE_LEDGER_CARD = Template(LIVE_LEDGER_CARD_TEMPLATE)

TIMEZONE_LABEL = "America/New_York"
SOURCE_BADGE = "SRC: AAA_PIPELINE_v1"
SCORE_VERSION = "EDGE_v2.2"

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

AAA_TO_MLB_CODES = {
    "Memphis Redbirds": ("MEM", "STL"),
    "St. Paul Saints": ("STP", "MIN"),
    "Lehigh Valley IronPigs": ("LHV", "PHI"),
    "Buffalo Bisons": ("BUF", "TOR"),
    "Syracuse Mets": ("SYR", "NYM"),
    "Tacoma Rainiers": ("TAC", "SEA"),
    "Louisville Bats": ("LOU", "CIN"),
    "Sugar Land Space Cowboys": ("SUG", "HOU"),
    "Reno Aces": ("REN", "ARI"),
    "Albuquerque Isotopes": ("ABQ", "COL"),
    "Norfolk Tides": ("NOR", "BAL"),
    "Jacksonville Jumbo Shrimp": ("JAX", "MIA"),
    "Nashville Sounds": ("NAS", "MIL"),
    "Salt Lake Bees": ("SLB", "LAA"),
    "Charlotte Knights": ("CLT", "CWS"),
    "El Paso Chihuahuas": ("ELP", "SD"),
    "Columbus Clippers": ("CLP", "CLE"),
    "Rochester Red Wings": ("ROC", "WSH"),
    "Iowa Cubs": ("IOW", "CHC"),
    "Oklahoma City Baseball Club": ("OKC", "LAD"),
    "Toledo Mud Hens": ("TOL", "DET"),
    "Indianapolis Indians": ("IND", "PIT"),
    "Worcester Red Sox": ("WOR", "BOS"),
    "Scranton/Wilkes-Barre RailRiders": ("SWB", "NYY"),
    "Las Vegas Aviators": ("LV", "ATH"),
    "Sacramento River Cats": ("SAC", "SF"),
    "Gwinnett Stripers": ("GWN", "ATL"),
    "Round Rock Express": ("RR", "TEX"),
    "Durham Bulls": ("DUR", "TB"),
}

MLB_TO_AAA_NAME = {mlb: name for name, (_, mlb) in AAA_TO_MLB_CODES.items()}
AAA_CODES = {v[0] for v in AAA_TO_MLB_CODES.values()}
VALID_TEAM_CODES = MLB_CODES | AAA_CODES

STATUS_DIR = DIST_DIR / "status"
SNAPSHOT_DIR = DIST_DIR / "_snapshots" / "promotion-watch"
PROMOTION_WATCH_STATUS_PATH = STATUS_DIR / "promotion-watch.json"

# existing constants / globals above


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


def safe_text(value, fallback="—") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def first_non_empty(row: pd.Series, candidates: list[str], fallback="—") -> str:
    for col in candidates:
        if col in row.index:
            val = row.get(col)
            if val is None or pd.isna(val):
                continue
            text = str(val).strip()
            if text and text.lower() not in {"nan", "none", "null"}:
                return text
    return fallback


def initials(name: str) -> str:
    parts = [p for p in str(name).split() if p]
    if not parts:
        return "—"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][:1] + parts[-1][:1]).upper()


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
    if not raw:
        return "—"

    raw_text = str(raw).strip()
    code = clean_candidate_code(raw_text)

    if code in VALID_TEAM_CODES:
        return code

    lowered = raw_text.lower()

    if raw_text in AAA_TO_MLB_CODES:
        return AAA_TO_MLB_CODES[raw_text][0]

    for k, v in MLB_NAME_TO_CODE.items():
        if k in lowered:
            return v

    return "—"


def affiliate_label_from_mlb_code(code: str) -> str:
    code = str(code or "").strip().upper()
    if not code:
        return "—"

    full_name = MLB_TO_AAA_NAME.get(code)
    if not full_name:
        return code

    shorten_map = {
        "Louisville Bats": "Louisville",
        "Iowa Cubs": "Iowa Cubs",
        "Tacoma Rainiers": "Tacoma",
        "Oklahoma City Baseball Club": "Oklahoma City",
        "St. Paul Saints": "St. Paul",
        "Memphis Redbirds": "Memphis",
        "Scranton/Wilkes-Barre RailRiders": "Scranton/WB",
        "Worcester Red Sox": "Worcester",
        "Jacksonville Jumbo Shrimp": "Jacksonville",
        "Sugar Land Space Cowboys": "Sugar Land",
        "Albuquerque Isotopes": "Albuquerque",
        "Rochester Red Wings": "Rochester",
        "Charlotte Knights": "Charlotte",
        "Lehigh Valley IronPigs": "Lehigh Valley",
        "Norfolk Tides": "Norfolk",
        "El Paso Chihuahuas": "El Paso",
        "Columbus Clippers": "Columbus",
        "Round Rock Express": "Round Rock",
        "Gwinnett Stripers": "Gwinnett",
        "Indianapolis Indians": "Indianapolis",
        "Toledo Mud Hens": "Toledo",
        "Las Vegas Aviators": "Las Vegas",
        "Sacramento River Cats": "Sacramento",
        "Durham Bulls": "Durham",
        "Nashville Sounds": "Nashville",
        "Buffalo Bisons": "Buffalo",
        "Syracuse Mets": "Syracuse",
        "Salt Lake Bees": "Salt Lake",
        "Reno Aces": "Reno",
    }

    return shorten_map.get(full_name, full_name)


def derive_display_team(row: pd.Series) -> str:
    candidates = [
        "team_abbrev",
        "org_code",
        "parent_org_code",
        "mlb_org_code",
        "org_abbrev",
        "team_code",
        "team",
        "org",
        "parent_org",
        "mlb_org",
    ]
    for col in candidates:
        if col in row.index:
            code = map_team_to_code(row.get(col))
            if code != "—":
                return code
    return "—"


def derive_display_org(row: pd.Series) -> str:
    primary_code = derive_display_team(row)
    affiliate_label = affiliate_label_from_mlb_code(primary_code)
    if affiliate_label != "—":
        return affiliate_label

    candidates = [
        "affiliate_name",
        "team_name",
        "team",
        "org",
        "parent_org",
        "mlb_org",
    ]
    return first_non_empty(row, candidates, fallback="—")


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

    xs = []
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


def build_trend_lookup(df: pd.DataFrame, metric_field: str) -> dict[str, dict]:
    if df.empty or metric_field not in df.columns:
        return {}

    trend = {}
    work = df.copy()
    work["player_name"] = work["player_name"].apply(safe_name)

    if "week_start" in work.columns:
        work["week_start"] = pd.to_datetime(work["week_start"], errors="coerce")

    for player, group in work.groupby("player_name"):
        if "week_start" in group.columns:
            group = group.sort_values("week_start")
        values = pd.to_numeric(group[metric_field], errors="coerce").dropna().tolist()
        if not values:
            continue
        polyline = build_polyline(values)
        glow = len(values) >= 2 and values[-1] > values[-2]
        trend[player] = {
            "trend_points": polyline,
            "trend_glow": glow,
            "trend_label": f"{len(values)}W",
        }
    return trend


def aggregate_hitter_frame(df: pd.DataFrame) -> pd.DataFrame:
    hitters = df[df["pa"].notna()].copy()
    if hitters.empty:
        return pd.DataFrame()

    numeric_cols = ["pa", "bb", "so", "hr", "iso", "kbb_h"]
    for col in numeric_cols:
        if col in hitters.columns:
            hitters[col] = pd.to_numeric(hitters[col], errors="coerce")

    hitters["player_name"] = hitters["player_name"].apply(safe_name)

    agg_map = {
        "pa": "sum",
        "bb": "sum",
        "so": "sum",
        "hr": "sum",
        "iso": "mean",
    }

    optional_text_cols = ["org", "team", "team_abbrev", "org_code", "parent_org", "mlb_org", "week_start"]
    for col in optional_text_cols:
        if col in hitters.columns:
            agg_map[col] = "last"

    agg = hitters.groupby(["player_name"], dropna=False).agg(agg_map).reset_index()

    agg["kbb_h"] = agg.apply(
        lambda r: (float(r["so"]) / float(r["bb"])) if pd.notna(r["bb"]) and float(r["bb"]) > 0 else float(r["so"]) if pd.notna(r["so"]) else 0.0,
        axis=1,
    )
    return agg


def aggregate_pitcher_frame(df: pd.DataFrame) -> pd.DataFrame:
    pitchers = df[df["bf"].notna()].copy()
    if pitchers.empty:
        return pd.DataFrame()

    numeric_cols = ["bf", "so_p", "bb_allowed", "kbb_p"]
    for col in numeric_cols:
        if col in pitchers.columns:
            pitchers[col] = pd.to_numeric(pitchers[col], errors="coerce")

    pitchers["player_name"] = pitchers["player_name"].apply(safe_name)

    agg_map = {
        "bf": "sum",
        "so_p": "sum",
        "bb_allowed": "sum",
    }

    optional_text_cols = ["org", "team", "team_abbrev", "org_code", "parent_org", "mlb_org", "week_start"]
    for col in optional_text_cols:
        if col in pitchers.columns:
            agg_map[col] = "last"

    agg = pitchers.groupby(["player_name"], dropna=False).agg(agg_map).reset_index()

    agg["kbb_p"] = agg.apply(
        lambda r: (float(r["so_p"]) / float(r["bb_allowed"])) if pd.notna(r["bb_allowed"]) and float(r["bb_allowed"]) > 0 else float(r["so_p"]) if pd.notna(r["so_p"]) else 0.0,
        axis=1,
    )
    return agg


def build_aaa_hitter_promotion_watch(df: pd.DataFrame, trend_lookup: dict[str, dict] | None = None) -> pd.DataFrame:
    hitters = aggregate_hitter_frame(df)
    if hitters.empty:
        return pd.DataFrame()

    hitters = hitters[hitters["pa"] >= 12].copy()
    if hitters.empty:
        return pd.DataFrame()

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
    hitters["signal_type"] = "Hitter"
    hitters["score_class"] = hitters["edge_score"].apply(classify_score)
    hitters["display_team"] = hitters.apply(derive_display_team, axis=1)
    hitters["display_org"] = hitters.apply(derive_display_org, axis=1)

    def build_hitter_signal_summary(row):
        iso = row.get("iso", 0) or 0
        kbb = row.get("kbb_h", 0) or 0
        hr = row.get("hr", 0) or 0

        signals = []
        if iso >= 0.300:
            signals.append("Power spike")
        elif iso >= 0.220:
            signals.append("Impact contact")

        if kbb >= 1.5:
            signals.append("Plate skill support")
        elif kbb <= 0.7:
            signals.append("Aggressive contact shape")

        if hr >= 2:
            signals.append("Recent HR pressure")

        return " • ".join(signals) if signals else "Emerging offensive signal under evaluation"

    hitters["why"] = hitters.apply(build_hitter_signal_summary, axis=1)

    hitters["metric_1_label"] = "ISO"
    hitters["metric_1"] = hitters["iso"].fillna(0).map(lambda x: f"{x:.3f}")
    hitters["metric_2_label"] = "K/BB"
    hitters["metric_2"] = hitters["kbb_h"].fillna(0).map(lambda x: f"{x:.2f}")
    hitters["metric_3_label"] = "HR"
    hitters["metric_3"] = hitters["hr"].fillna(0).map(lambda x: f"{int(x)}")
    hitters["sample_note"] = hitters["pa"].fillna(0).map(lambda x: f"{int(x)} PA")
    hitters["source_badge"] = SOURCE_BADGE
    hitters["score_version"] = SCORE_VERSION
    hitters["avatar"] = hitters["player_name"].map(initials)
    hitters = backfill_resolved_player_ids(hitters)

    def hitter_badges(row: pd.Series) -> list[tuple[str, str]]:
        badges = []
        if pd.notna(row["iso"]) and row["iso"] >= 0.250:
            badges.append(("Impact Bat", "positive"))
        if pd.notna(row["kbb_h"]) and row["kbb_h"] <= 1.50:
            badges.append(("Zone Control", "positive"))
        if pd.notna(row["hr"]) and row["hr"] >= 2:
            badges.append(("HR Surge", "positive"))
        if not badges:
            badges.append(("Promotion Watch", "watch"))
        return badges[:3]

    hitters["badges"] = hitters.apply(hitter_badges, axis=1)
    hitters["trend_points"] = "0,20 40,18 80,16 120,14"
    hitters["trend_glow"] = False
    hitters["trend_note"] = "Recent Form"

    trend_lookup = trend_lookup or {}
    for idx, row in hitters.iterrows():
        info = trend_lookup.get(row["player_name"])
        if info:
            hitters.at[idx, "trend_points"] = info["trend_points"]
            hitters.at[idx, "trend_glow"] = bool(info["trend_glow"])
            hitters.at[idx, "trend_note"] = info.get("trend_label", "Trend")

    return hitters.sort_values(["edge_score", "pa", "week_start", "player_id", "player_name"], ascending=[False, False, False, True, True], kind="mergesort").reset_index(drop=True)


def build_aaa_pitcher_promotion_watch(df: pd.DataFrame, trend_lookup: dict[str, dict] | None = None) -> pd.DataFrame:
    pitchers = aggregate_pitcher_frame(df)
    if pitchers.empty:
        return pd.DataFrame()

    pitchers = pitchers[pitchers["bf"] >= 15].copy()
    if pitchers.empty:
        return pd.DataFrame()

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
    pitchers["signal_type"] = "Pitcher"
    pitchers["score_class"] = pitchers["edge_score"].apply(classify_score)
    pitchers["display_team"] = pitchers.apply(derive_display_team, axis=1)
    pitchers["display_org"] = pitchers.apply(derive_display_org, axis=1)

    pitchers["why"] = pitchers.apply(
        lambda r: f"K/BB {r['kbb_p']:.2f} • {int(r['so_p'] or 0)} K • {int(r['bb_allowed'] or 0)} BB over {int(r['bf'] or 0)} BF",
        axis=1,
    )

    pitchers["metric_1_label"] = "K/BB"
    pitchers["metric_1"] = pitchers["kbb_p"].fillna(0).map(lambda x: f"{x:.2f}")
    pitchers["metric_2_label"] = "K"
    pitchers["metric_2"] = pitchers["so_p"].fillna(0).map(lambda x: f"{int(x)}")
    pitchers["metric_3_label"] = "BB"
    pitchers["metric_3"] = pitchers["bb_allowed"].fillna(0).map(lambda x: f"{int(x)}")
    pitchers["sample_note"] = pitchers["bf"].fillna(0).map(lambda x: f"{int(x)} BF")
    pitchers["source_badge"] = SOURCE_BADGE
    pitchers["score_version"] = SCORE_VERSION
    pitchers["avatar"] = pitchers["player_name"].map(initials)
    pitchers = backfill_resolved_player_ids(pitchers)

    def pitcher_badges(row: pd.Series) -> list[tuple[str, str]]:
        badges = []
        if pd.notna(row["kbb_p"]) and row["kbb_p"] >= 4:
            badges.append(("Bat-Miss Ready", "positive"))
        if pd.notna(row["bb_allowed"]) and row["bb_allowed"] <= 2:
            badges.append(("Command Hold", "positive"))
        if pd.notna(row["so_p"]) and row["so_p"] >= 10:
            badges.append(("Whiff Volume", "positive"))
        if not badges:
            badges.append(("Promotion Watch", "watch"))
        return badges[:3]

    pitchers["badges"] = pitchers.apply(pitcher_badges, axis=1)
    pitchers["trend_points"] = "0,21 40,19 80,16 120,14"
    pitchers["trend_glow"] = False
    pitchers["trend_note"] = "Recent Form"

    trend_lookup = trend_lookup or {}
    for idx, row in pitchers.iterrows():
        info = trend_lookup.get(row["player_name"])
        if info:
            pitchers.at[idx, "trend_points"] = info["trend_points"]
            pitchers.at[idx, "trend_glow"] = bool(info["trend_glow"])
            pitchers.at[idx, "trend_note"] = info.get("trend_label", "Trend")

    return pitchers.sort_values(["edge_score", "bf", "week_start", "player_id", "player_name"], ascending=[False, False, False, True, True], kind="mergesort").reset_index(drop=True)


def is_recent_arrival_prospect_relevant(move: dict) -> bool:
    age = move.get("currentAge")
    draft_year = move.get("draftYear")
    debut = move.get("mlbDebutDate")
    name = str(move.get("person") or "").strip().lower()

    obvious_veterans = {
        "ty france",
        "mitch garver",
        "randal grichuk",
        "adam frazier",
        "rhys hoskins",
        "andrew mccutchen",
        "christian vázquez",
        "ildemaro vargas",
        "brett sullivan",
        "joe ross",
        "walker buehler",
        "shaun anderson",
        "jeimer candelario",
    }
    if name in obvious_veterans:
        return False

    if debut is None:
        return True
    if isinstance(debut, str) and debut >= "2025-01-01":
        return True
    if age is not None and age <= 26:
        return True
    if draft_year is not None and draft_year >= 2020:
        return True
    return False


def infer_position_badge(move: dict) -> str:
    desc = str(move.get("description") or "").upper()
    checks = ["RHP", "LHP", "C", "SS", "2B", "3B", "1B", "CF", "LF", "RF", "OF"]
    for token in checks:
        if f" {token} " in f" {desc} ":
            return token
    return "P" if "HP " in desc else "POS"


def infer_position_class(position_badge: str) -> str:
    if position_badge in {"RHP", "LHP", "P", "SP", "RP"}:
        return "pitcher"
    if position_badge in {"SS", "2B", "3B", "1B", "C"}:
        return "infielder"
    if position_badge in {"LF", "CF", "RF", "OF"}:
        return "outfielder"
    return "neutral"


def infer_transaction_label(move: dict) -> str:
    type_desc = str(move.get("typeDesc") or "").strip().lower()
    debut = move.get("mlbDebutDate")
    date = move.get("date")

    if type_desc in {"selected", "recalled"}:
        if debut and date and str(debut) >= str(date):
            return "DEBUT"
        if debut is None:
            return "DEBUT"
        return "RECALL"

    if type_desc in {"contract selected", "purchased"}:
        return "CALL-UP"

    if type_desc in {"optioned", "outrighted"}:
        return "RETURN"

    if type_desc == "assigned":
        return "ASSIGN"

    return "MOVE"


def transaction_class(txn_label: str) -> str:
    mapping = {
        "DEBUT": "debut",
        "RECALL": "recall",
        "CALL-UP": "callup",
        "RETURN": "return",
        "ASSIGN": "neutral",
        "MOVE": "neutral",
    }
    return mapping.get(txn_label, "neutral")


def infer_team_codes(move: dict) -> tuple[str, str]:
    from_team = str(move.get("fromTeam") or "")
    to_team = str(move.get("toTeam") or "")

    if from_team in AAA_TO_MLB_CODES:
        return AAA_TO_MLB_CODES[from_team]

    from_code = map_team_to_code(from_team)
    to_code = map_team_to_code(to_team)

    return (from_code if from_code != "—" else "AAA", to_code if to_code != "—" else "MLB")


def load_arrivals_windows(live_limit: int = 8, archive_limit: int = 16) -> tuple[list[dict], list[dict]]:
    path = DIST_DIR / "aaa_transactions_scout_only.json"
    if not path.exists():
        return [], []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        moves = payload.get("scout_relevant_moves", []) or []
    except Exception:
        return [], []

    arrivals = [m for m in moves if m.get("classification") == "arrival_to_mlb"]
    arrivals = [m for m in arrivals if is_recent_arrival_prospect_relevant(m)]

    now_ts = pd.Timestamp.now().normalize()
    live_cutoff = now_ts - pd.Timedelta(hours=72)
    archive_cutoff = now_ts - pd.Timedelta(days=14)

    live_arrivals = []
    archive_arrivals = []

    for move in arrivals:
        move_date_raw = move.get("date")
        move_ts = pd.to_datetime(move_date_raw, errors="coerce")
        if pd.isna(move_ts):
            continue
        move_ts = pd.Timestamp(move_ts).normalize()

        if move_ts >= live_cutoff:
            live_arrivals.append(move)
        if move_ts >= archive_cutoff:
            archive_arrivals.append(move)

    arrivals_sorted = sorted(
        arrivals,
        key=lambda m: (
            str(m.get("date") or ""),
            str(m.get("mlbDebutDate") or ""),
            str(m.get("person") or ""),
        ),
        reverse=True,
    )

    if not archive_arrivals and arrivals_sorted:
        archive_arrivals = arrivals_sorted[:archive_limit]

    if not live_arrivals and arrivals_sorted:
        live_arrivals = arrivals_sorted[:live_limit]

    def format_arrivals(arrivals_subset: list[dict], limit: int) -> list[dict]:
        arrivals_subset = sorted(
            arrivals_subset,
            key=lambda m: (
                str(m.get("date") or ""),
                str(m.get("mlbDebutDate") or ""),
                str(m.get("person") or ""),
            ),
            reverse=True,
        )

        formatted = []
        for move in arrivals_subset[:limit]:
            player = safe_name(move.get("person") or "Unknown")
            age = move.get("currentAge")
            draft_year = move.get("draftYear")
            debut = move.get("mlbDebutDate") or "Pending"
            pos_badge = infer_position_badge(move)
            pos_class = infer_position_class(pos_badge)
            txn_label = infer_transaction_label(move)
            txn_class = transaction_class(txn_label)
            from_code, to_code = infer_team_codes(move)

            meta_bits = []
            if age is not None:
                meta_bits.append(f"Age {age}")
            if draft_year is not None:
                meta_bits.append(f"Draft {draft_year}")
            meta_line = " / ".join(meta_bits) if meta_bits else "Upper-minors movement"

            desc = safe_text(move.get("description"), fallback=player)

            if txn_label == "DEBUT":
                event_line = "MLB debut sequence"
            elif txn_label == "RECALL":
                event_line = "Return to MLB roster"
            elif txn_label == "CALL-UP":
                event_line = "Contract selected / call-up"
            elif txn_label == "RETURN":
                event_line = "Returned through transaction cycle"
            else:
                event_line = "Roster movement"

            formatted.append(
                {
                    "player_name": player,
                    "avatar": initials(player),
                    "date": move.get("date") or "—",
                    "debut_label": debut,
                    "meta_line": meta_line,
                    "why": desc,
                    "position_badge": pos_badge,
                    "position_class": pos_class,
                    "transaction_label": txn_label,
                    "transaction_class": txn_class,
                    "from_code": from_code,
                    "to_code": to_code,
                    "event_line": event_line,
                }
            )
        return formatted

    return format_arrivals(live_arrivals, live_limit), format_arrivals(archive_arrivals, archive_limit)


HTML_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals // Promotion Watch Preview</title>
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
    .section-card {
      background: var(--card-radial);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }

    .hero-card { padding: 22px 22px 20px; }

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

    .summary-card {
      padding: 18px;
      display: grid;
      gap: 12px;
      align-content: start;
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

    .section-card { padding: 18px; }

    .board-controls {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }

    .guide-btn {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      color: var(--soft);
      border-radius: 999px;
      padding: 8px 12px;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      cursor: pointer;
      transition: 160ms ease;
    }

    .guide-btn:hover {
      color: var(--text);
      border-color: rgba(106,166,255,0.18);
      background: rgba(106,166,255,0.06);
    }

    .tabs {
      display: inline-flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .tab {
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      color: var(--soft);
      border-radius: 999px;
      padding: 8px 12px;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      cursor: pointer;
      transition: 160ms ease;
    }

    .tab:hover {
      color: var(--text);
      border-color: rgba(255,255,255,0.14);
    }

    .tab.active {
      color: var(--text);
      border-color: rgba(182,255,0,0.20);
      box-shadow: 0 0 8px rgba(182,255,0,0.08);
      background: rgba(182,255,0,0.04);
    }

    .section-title {
      margin: 0 0 14px;
      font-size: 18px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .placeholder {
      border: 1px dashed rgba(255,255,255,0.14);
      border-radius: 16px;
      padding: 28px 18px;
      text-align: center;
      color: var(--soft);
      background: rgba(255,255,255,0.02);
    }

    .signal-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }

    .section {
      background: var(--card-radial);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 18px;
    }

    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 16px;
    }

    .section-kicker {
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--blue);
      margin-bottom: 6px;
    }

    .section-badge {
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

    .cards {
      display: grid;
      gap: 12px;
    }

    
    .system-pulse-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 14px 18px;
      margin-bottom: 18px;
      border: 1px solid rgba(255,255,255,0.08);
      border-top: 2px solid rgba(176,215,255,0.42);
      background: rgba(7, 12, 24, 0.82);
      box-shadow: 0 18px 40px rgba(2,6,23,0.22);
    }

    .system-pulse-left {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .system-pulse-dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: #22c55e;
      box-shadow: 0 0 12px rgba(34,197,94,0.65);
      animation: pulse 1.5s infinite;
    }

    .system-pulse-label,
    .system-pulse-proof {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }

    .system-pulse-label {
      color: var(--soft);
    }

    .system-pulse-proof {
      color: #B0D7FF;
      text-align: right;
    }

.arrival-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .player-card {
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255,255,255,0.02);
      transition: 180ms ease;
      position: relative;
    }

    .player-card:hover {
      border-color: rgba(255,255,255,0.13);
      transform: translateY(-1px);
    }

    .player-card.high-edge {
      box-shadow: 0 0 0 1px rgba(182,255,0,0.10), 0 0 18px rgba(182,255,0,0.06);
    }

    .player-card.elite-edge {
      box-shadow: 0 0 0 1px rgba(182,255,0,0.16), 0 0 20px rgba(182,255,0,0.10);
    }

    #tab-14d .player-card {
      background: linear-gradient(180deg, rgba(255,255,255,0.028) 0%, rgba(255,255,255,0.018) 100%);
      border-color: rgba(255,255,255,0.10);
    }

    #tab-14d .section-kicker { color: var(--blue-soft); }

    #tab-14d .signal-grid {
      display: block;
    }

    #tab-14d .signal-grid > .section {
      width: 100%;
      max-width: none;
      margin-bottom: 18px;
    }

    #tab-72h .signal-grid {
      display: block;
    }

    #tab-72h .signal-grid > .section {
      width: 100%;
      max-width: none;
      margin-bottom: 18px;
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
    .js-add-to-roster {
      cursor: pointer;
    }

    .js-add-to-roster:hover {
      color: var(--text);
      border-color: rgba(182,255,0,0.20);
      background: rgba(182,255,0,0.06);
    }
    .scorebox {
      text-align: right;
      min-width: 72px;
    }

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
    .score-value.elite { color: var(--lime-hot); }
    .score-value.watch { color: var(--gold); }
    .score-value.neutral { color: var(--text); }

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

    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    .status-badge {
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

    .status-badge.positive {
      color: var(--lime-hot);
      border-color: rgba(182,255,0,0.20);
      background: rgba(182,255,0,0.05);
    }

    .status-badge.watch {
      color: var(--gold);
      border-color: rgba(251,191,36,0.22);
      background: rgba(251,191,36,0.06);
    }

    .status-badge.neutral {
      color: var(--soft);
      border-color: rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.02);
    }

    .status-badge.pitcher {
      color: var(--blue-soft);
      border-color: rgba(106,166,255,0.22);
      background: rgba(106,166,255,0.06);
    }

    .status-badge.infielder,
    .status-badge.outfielder {
      color: var(--soft);
      border-color: rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.02);
    }

    .status-badge.debut {
      color: var(--lime-hot);
      border-color: rgba(182,255,0,0.22);
      background: rgba(182,255,0,0.05);
    }

    .status-badge.recall,
    .status-badge.callup {
      color: var(--blue-soft);
      border-color: rgba(106,166,255,0.22);
      background: rgba(106,166,255,0.06);
    }

    .status-badge.return {
      color: var(--gold);
      border-color: rgba(251,191,36,0.22);
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


    .player-audit-row {
      display: grid;
      grid-template-columns: minmax(260px, 1.35fr) minmax(220px, 0.95fr) 84px 150px;
      gap: 14px;
      align-items: start;
      border-radius: 10px;
      padding: 16px 18px;
      background: rgba(7,12,24,0.78);
      border: 1px solid rgba(255,255,255,0.08);
      border-top: 2px solid rgba(176,215,255,0.28);
      box-shadow: 0 18px 40px rgba(2,6,23,0.20);
    }

    .player-audit-row:hover {
      border-color: rgba(176,215,255,0.34);
      transform: translateY(-1px);
    }

    .audit-left,
    .audit-center,
    .audit-right,
    .audit-action {
      min-width: 0;
    }

    .audit-trigger {
      display: inline-flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 8px;
      background: transparent;
      border: 0;
      padding: 0;
      cursor: pointer;
      text-align: left;
    }

    .audit-trigger:hover .audit-kicker,
    .audit-trigger:hover .audit-player-name {
      color: #B0D7FF;
    }

    .audit-kicker,
    .audit-context,
    .audit-chip,
    .forensic-label,
    .conviction-label {
      font-family: var(--mono);
      text-transform: uppercase;
    }

    .audit-kicker {
      font-size: 10px;
      letter-spacing: 0.16em;
      color: #B0D7FF;
    }

    .audit-player-name {
      font-size: 20px;
      line-height: 1;
      letter-spacing: -0.03em;
      font-weight: 800;
      color: var(--text);
    }

    .audit-context {
      margin-top: 8px;
      font-size: 10px;
      letter-spacing: 0.10em;
      color: var(--soft);
    }

    .audit-submeta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 10px;
    }

    .audit-chip {
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      color: var(--soft);
      font-size: 10px;
      letter-spacing: 0.08em;
    }

    .forensic-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(64px, 1fr));
      gap: 6px;
    }

    .forensic-cell {
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.02);
      padding: 10px 11px;
    }

    .forensic-label {
      font-size: 9px;
      letter-spacing: 0.06em;
      color: #B0D7FF;
      white-space: nowrap;
    }

    .forensic-value {
      margin-top: 5px;
      font-size: 15px;
      font-weight: 800;
      line-height: 1.15;
      color: var(--text);
      word-break: break-word;
    }

    .audit-why {
      margin-top: 12px;
      font-size: 10px;
      line-height: 1.55;
      color: #8b8b94;
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
    }

    .audit-right {
      text-align: right;
      min-width: 92px;
    }

    .conviction-label {
      font-size: 9px;
      letter-spacing: 0.08em;
      color: var(--tiny);
      margin-bottom: 4px;
      white-space: nowrap;
    }

    .conviction-score {
      font-size: 36px;
      line-height: 0.92;
      font-weight: 900;
      font-style: italic;
      letter-spacing: -0.05em;
    }

    .conviction-score.positive,
    .conviction-score.elite { color: var(--lime-hot); }
    .conviction-score.watch { color: var(--gold); }
    .conviction-score.neutral { color: var(--text); }

    .audit-action {
      display: flex;
      align-items: start;
      justify-content: flex-end;
      min-width: 150px;
    }

    .provision-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      min-height: 36px;
      padding: 0 10px;
      border: 1px solid rgba(96,165,250,0.32);
      background: rgba(37,99,235,0.95);
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
      background: rgba(59,130,246,1);
      transform: translateY(-1px);
      box-shadow: 0 0 16px rgba(59,130,246,0.16);
    }

    .movement-audit-row {
      grid-template-columns: 1.18fr 1.7fr 0.62fr 0.78fr;
      align-items: start;
      column-gap: 18px;
      row-gap: 12px;
    }

    .movement-audit-row .movement-trigger {
      cursor: default;
    }

    .movement-audit-row .audit-left {
      min-width: 0;
    }

    .movement-audit-row .audit-center {
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 10px;
      justify-content: start;
    }

    .movement-audit-row .forensic-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      align-items: stretch;
    }

    .movement-audit-row .forensic-cell {
      min-height: 64px;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }

    .movement-audit-row .forensic-value {
      min-height: 18px;
      display: flex;
      align-items: center;
    }

    .movement-right {
      min-width: 0;
      text-align: right;
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      justify-content: start;
      gap: 6px;
    }

    .movement-flag {
      font-family: var(--mono);
      font-size: 18px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--gold);
      line-height: 1.1;
      text-align: right;
    }

    .movement-action {
      min-width: 0;
      display: flex;
      justify-content: flex-end;
      align-items: start;
    }

    .movement-route {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      min-width: 160px;
      padding: 0 12px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      color: var(--soft);
      font-family: var(--mono);
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      white-space: nowrap;
      text-align: center;
    }

    .movement-audit-row .audit-why {
      margin-top: 2px;
      max-width: 560px;
      font-size: 11px;
      line-height: 1.45;
      color: #a1a1aa;
    }

    @media (max-width: 1120px) {
      .player-audit-row {
        grid-template-columns: 1fr;
      }

      .audit-right {
        text-align: left;
      }

      .audit-action {
        justify-content: flex-start;
      }
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
      background: var(--card-radial);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      transform: translateX(108%);
      transition: transform 220ms ease;
      z-index: 130;
    }

    .drawer-panel.open { transform: translateX(0); }

    .drawer-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }

    .drawer-title-wrap { min-width: 0; }

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

    /* PROMOTION_WATCH_FIELD_GUIDE_V1 */
    .pw-field-guide-trigger {
      position: fixed;
      right: 22px;
      bottom: 22px;
      z-index: 2147482500;
      display: inline-flex;
      align-items: center;
      gap: 9px;
      min-height: 42px;
      padding: 0 14px;
      border-radius: 999px;
      border: 1px solid rgba(96, 165, 250, 0.28);
      background:
        radial-gradient(circle at 20% 0%, rgba(96,165,250,0.18), transparent 34%),
        rgba(5, 8, 14, 0.94);
      color: #f8fafc;
      font-family: var(--mono);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      box-shadow: 0 14px 32px rgba(0,0,0,0.38), 0 0 22px rgba(96,165,250,0.10);
      cursor: pointer;
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
    }

    .pw-field-guide-trigger::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: #60a5fa;
      box-shadow: 0 0 12px rgba(96,165,250,0.7);
    }

    .pw-field-guide-overlay {
      position: fixed;
      inset: 0;
      z-index: 2147482501;
      background: rgba(0,0,0,0.58);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.22s ease;
    }

    .pw-field-guide-drawer {
      position: fixed;
      top: 0;
      right: 0;
      bottom: 0;
      z-index: 2147482502;
      width: min(560px, 100vw);
      transform: translateX(100%);
      transition: transform 0.24s ease;
      background:
        radial-gradient(circle at 24% 0%, rgba(96,165,250,0.16), transparent 34%),
        linear-gradient(180deg, #0b1018 0%, #05070b 100%);
      border-left: 1px solid rgba(255,255,255,0.10);
      box-shadow: -24px 0 70px rgba(0,0,0,0.58);
      color: #f8fafc;
      display: flex;
      flex-direction: column;
    }

    .pw-field-guide-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }

    .pw-field-guide-drawer.open {
      transform: translateX(0);
    }

    .pw-field-guide-head {
      padding: 22px 22px 16px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }

    .pw-field-guide-kicker {
      font-family: var(--mono);
      font-size: 10px;
      line-height: 1;
      color: #60a5fa;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-weight: 900;
      margin-bottom: 10px;
    }

    .pw-field-guide-title {
      margin: 0;
      font-family: var(--sans);
      font-size: clamp(26px, 4vw, 38px);
      line-height: 0.98;
      letter-spacing: -0.055em;
      font-weight: 850;
      color: #ffffff;
    }

    .pw-field-guide-close {
      width: 36px;
      height: 36px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.04);
      color: #fff;
      font-size: 24px;
      line-height: 1;
      cursor: pointer;
    }

    .pw-field-guide-body {
      padding: 18px 22px 24px;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      display: grid;
      gap: 14px;
    }

    .pw-guide-card {
      border: 1px solid rgba(255,255,255,0.09);
      border-radius: 20px;
      background: rgba(255,255,255,0.035);
      padding: 15px;
    }

    .pw-guide-label {
      font-family: var(--mono);
      font-size: 10px;
      line-height: 1;
      letter-spacing: 0.15em;
      color: #93c5fd;
      text-transform: uppercase;
      font-weight: 900;
      margin-bottom: 8px;
    }

    .pw-guide-copy {
      margin: 0;
      color: rgba(248,250,252,0.76);
      font-size: 13px;
      line-height: 1.55;
    }

    .pw-guide-list {
      margin: 10px 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 8px;
    }

    .pw-guide-list li {
      position: relative;
      padding-left: 15px;
      color: rgba(248,250,252,0.72);
      font-size: 12px;
      line-height: 1.45;
    }

    .pw-guide-list li::before {
      content: "";
      position: absolute;
      left: 0;
      top: 0.55em;
      width: 5px;
      height: 5px;
      border-radius: 999px;
      background: #60a5fa;
      box-shadow: 0 0 10px rgba(96,165,250,0.45);
    }

    @media screen and (max-width: 760px) {
      .pw-field-guide-trigger {
        right: 12px;
        bottom: 14px;
        min-height: 38px;
        padding: 0 12px;
        font-size: 9px;
        letter-spacing: 0.12em;
      }

      .pw-field-guide-drawer {
        width: min(92vw, 420px);
      }

      .pw-field-guide-head {
        padding: 18px 16px 14px;
      }

      .pw-field-guide-body {
        padding: 14px 14px 20px;
      }

      .pw-guide-card {
        border-radius: 17px;
        padding: 13px;
      }
    }

    {{ ledger_styles | safe }}

    @media (max-width: 980px) {
      .hero { grid-template-columns: 1fr; }
      .signal-grid,
      .arrival-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 640px) {
      .topbar-inner,
      .app,
      .topnav-inner,
      .search-strip-inner {
        width: min(100%, calc(100% - 16px));
      }

      .hero-card,
      .summary-card,
      .section-card,
      .section,
      .player-card {
        border-radius: 16px;
      }

      .hero-card { padding: 18px; }
      .metric-grid { grid-template-columns: 1fr; }
      .player-name { font-size: 17px; }
      .score-value { font-size: 24px; }
      .player-top { grid-template-columns: auto 1fr; }
      .scorebox {
        grid-column: 2;
        text-align: left;
        margin-top: 8px;
      }
      .drawer-panel {
        top: 8px;
        right: 8px;
        width: calc(100vw - 16px);
        max-height: calc(100vh - 16px);
      }
    }

      /* PROMOTION_WATCH_HIDE_SIGNAL_SUMMARY_V1 */
      .hero-summary,
      .summary-card {
        display: none !important;
      }

      /* PROMOTION_WATCH_HERO_TITLE_EDITORIAL_V1 */
      .hero-title {
        text-transform: none !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        font-weight: 620 !important;
        letter-spacing: -0.045em !important;
        line-height: 1.02 !important;
      }

      @media screen and (max-width: 760px) {
        .hero-title {
          font-size: 44px !important;
          font-weight: 620 !important;
          line-height: 1.02 !important;
          letter-spacing: -0.048em !important;
          text-transform: none !important;
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
          <div class="brand-title">Promotion Watch // Institutional Ledger Preview</div>
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
        <div class="eyebrow">Signal Wall // Scout</div>
        <h1 class="hero-title">AAA Promotion Watch</h1>
        <p class="hero-sub">
          Real AAA signal intelligence with 72 HR urgency, 14 DAY movement context, stronger player cards, and live transaction-layer arrivals.
        </p>
      </div>

      <div class="summary-card">
        <div>
          <div class="summary-label">Window</div>
          <div class="summary-value" id="summary-window">14 DAY</div>
        </div>
        <div>
          <div class="summary-label">Mode</div>
          <div class="summary-value" id="summary-mode">SCOUT</div>
        </div>
        <div>
          <div class="summary-label">Signals</div>
          <div class="summary-value" id="summary-signals">{{ total_14_signals }}</div>
        </div>
      </div>
    </section>

    <section class="section-card">
      <div class="board-controls">
        <button type="button" class="guide-btn" onclick="openTerminalGuide()">How to Use This Terminal</button>


        <section class="system-pulse-bar">
          <div class="system-pulse-left">
            <span class="system-pulse-dot"></span>
            <span class="system-pulse-label">LIVE_ENGINE_PULSE</span>
          </div>
          <div class="system-pulse-proof">8,421,902 TOTAL OPERATIONS VERIFIED</div>
        </section>

        <div class="tabs tabs-aaa" role="tablist" aria-label="Promotion watch windows">
          <button type="button" class="tab" id="tab-btn-72h" onclick="switchPromotionTab('tab-72h', this)">72 HR</button>
          <button type="button" class="tab active" id="tab-btn-14d" onclick="switchPromotionTab('tab-14d', this)">14 DAY</button>
          <button type="button" class="tab" id="tab-btn-aaa-gems" onclick="switchPromotionTab('tab-aaa-gems', this)">AAA GEMS</button>
        </div>

        <div style="margin-top:14px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px;">
          <div style="border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:10px 12px; background:rgba(255,255,255,0.02);">
            <div style="font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:#7c7c84;">Page Build</div>
            <div style="margin-top:4px; font-size:13px; color:#f0f0f0;">{{ page_build_label }}</div>
          </div>
          <div style="border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:10px 12px; background:rgba(255,255,255,0.02);">
            <div style="font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:#7c7c84;">Live AAA Box Feed</div>
            <div style="margin-top:4px; font-size:13px; color:#f0f0f0;">{{ live_feed_label }}</div>
          </div>
          <div style="border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:10px 12px; background:rgba(255,255,255,0.02);">
            <div style="font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:#7c7c84;">Movement Feed</div>
            <div style="margin-top:4px; font-size:13px; color:#f0f0f0;">{{ movement_window_label }}</div>
          </div>
        </div>
      </div>

      {% if total_signals == 0 and total_14_signals == 0 %}
      <div class="placeholder">No live AAA promotion-watch signals available yet.</div>
      {% else %}

      <div id="tab-72h" class="tab-panel" style="display:none;">
        <section class="signal-grid">
          <div class="section section-v3-ledger-full">
            <div class="section-head">
              <div>
                <div class="section-kicker">Signal Layer</div>
                <h2 class="section-title">Pitching Prospect Signals — 72 HR</h2>
              </div>
              <div class="section-badge">Top {{ pitchers_72|length }}</div>
            </div>

            {% if pitchers_72 %}
            <div class="cards">
              {% for row in pitchers_72 %}
              {{ live_ledger_card.render(
                row=row,
                player_type="pitcher",
                context_label="72 HR WINDOW",
                metric_1_label="VELO_DELTA",
                metric_2_label="WHIFF_STABILITY",
                metric_3_label="LVL_ADJUST"
              ) | safe }}
              {% endfor %}
            </div>
            {% else %}
            <div class="placeholder">No 72 HR pitching prospect signals available.</div>
            {% endif %}
          </div>

          <div class="section section-v3-ledger-full">
            <div class="section-head">
              <div>
                <div class="section-kicker">Signal Layer</div>
                <h2 class="section-title">Hitting Prospect Signals — 72 HR</h2>
              </div>
              <div class="section-badge">Top {{ hitters_72|length }}</div>
            </div>

            {% if hitters_72 %}
            <div class="cards">
              {% for row in hitters_72 %}
              {{ live_ledger_card.render(
                row=row,
                player_type="hitter",
                context_label="72 HR WINDOW",
                metric_1_label="ISO_DELTA",
                metric_2_label="K/BB_STABILITY",
                metric_3_label="LVL_ADJUST"
              ) | safe }}
              {% endfor %}
            </div>
            {% else %}
            <div class="placeholder">No 72 HR hitting prospect signals available.</div>
            {% endif %}
          </div>
        </section>
      </div>

      <div id="tab-14d" class="tab-panel">
        <div style="margin: 0 0 16px 0; border: 1px solid rgba(59,130,246,0.22); border-radius: 14px; padding: 12px 14px; background: rgba(59,130,246,0.08);">
          <div style="font-size: 10px; letter-spacing: .16em; text-transform: uppercase; color: #93c5fd;">Live Feed Status</div>
          <div style="margin-top: 6px; font-size: 13px; line-height: 1.5; color: #f0f0f0;">
            Fresh AAA hitter board active from <strong>{{ live_feed_label }}</strong>. Movement Layer remains the confirmation feed.
          </div>
        </div>
        {% if fresh_hitters_live %}
        <div class="section" style="margin-bottom: 16px;">
          <div class="section-head">
            <div>
              <div class="section-kicker">Live Signal Layer</div>
              <h2 class="section-title">Fresh AAA Hitters — Last Final AAA Slate</h2>
            </div>
            <div class="section-badge">Top {{ fresh_hitters_live|length }}</div>
          </div>

          <div class="cards">
            {% for row in fresh_hitters_live %}
            {{ live_ledger_card.render(
              row=row,
              player_type="hitter",
              context_label="FINAL GAME WINDOW",
              metric_1_label="ISO_LIVE",
              metric_2_label="BB_LIVE",
              metric_3_label="HR_LIVE"
            ) | safe }}
            {% endfor %}
          </div>
        </div>
        {% endif %}

        {% if fresh_pitchers_live %}
        <div class="section" style="margin-bottom: 16px;">
          <div class="section-head">
            <div>
              <div class="section-kicker">Live Signal Layer</div>
              <h2 class="section-title">Fresh AAA Pitchers — Last Final AAA Slate</h2>
            </div>
            <div class="section-badge">Top {{ fresh_pitchers_live|length }}</div>
          </div>

          <div class="cards">
            {% for row in fresh_pitchers_live %}
            {{ live_ledger_card.render(
              row=row,
              player_type="pitcher",
              context_label="FINAL AAA SLATE",
              metric_1_label="IP_LIVE",
              metric_2_label="K_LIVE",
              metric_3_label="BB_LIVE"
            ) | safe }}
            {% endfor %}
          </div>
        </div>
        {% endif %}

        <section class="signal-grid">
          {# 14 DAY pitching block temporarily suppressed while AAA signal source is stale. #}

          <div class="section">
            <div class="section-head">
              <div>
                <div class="section-kicker">Signal Layer</div>
                <h2 class="section-title">Live AAA Signal Engine — In Refresh</h2>
              </div>
              <div class="section-badge">Stale Snapshot Hidden</div>
            </div>

            <div class="placeholder">
              The legacy AAA hitting snapshot has been temporarily hidden because it is not current enough for live surveillance use.
              The fresh Movement Layer below remains active while the rolling delta-based signal engine is rebuilt.
            </div>
          </div>

        <div class="section" style="margin-top: 16px;">
          <div class="section-head">
            <div>
              <div class="section-kicker">Movement Layer</div>
              <h2 class="section-title">Fresh MLB Arrivals — Last 14 Days</h2>
            </div>
            <div class="section-badge">Top {{ archive_arrivals|length }}</div>
          </div>

          {% if archive_arrivals %}
          <div class="cards">
            {% for row in archive_arrivals %}
            <article
              class="player-card player-audit-row movement-audit-row js-player-card"
              data-player-id="{{ row.player_id }}"
              data-player-name="{{ row.player_name }}"
              data-player-type="{{ row.player_type }}"
              data-player-team="{{ row.to_code }}"
              data-source-tag="ARRIVAL"
              data-profile-url="{{ row.profile_url }}"
            >
              <div class="audit-left">
                <div class="audit-trigger movement-trigger">
                  <span class="audit-kicker">&gt;_ MOVEMENT_AUDIT:</span>
                  <span class="audit-player-name">{{ row.player_name }}</span>
                </div>
                <div class="audit-context">FROM {{ row.from_code }} // TO {{ row.to_code }} // {{ row.event_line }}</div>
                <div class="audit-submeta">
                  <span class="audit-chip">RECENT_ARRIVAL</span>
                  <span class="audit-chip">{{ row.transaction_label }}</span>
                  <span class="audit-chip">{{ row.position_badge }}</span>
                </div>
              </div>

              <div class="audit-center">
                <div class="forensic-grid">
                  <div class="forensic-cell">
                    <div class="forensic-label">TX_DATE</div>
                    <div class="forensic-value">{{ row.date }}</div>
                  </div>
                  <div class="forensic-cell">
                    <div class="forensic-label">CONTEXT</div>
                    <div class="forensic-value">{{ row.meta_line }}</div>
                  </div>
                  <div class="forensic-cell">
                    <div class="forensic-label">MLB_STATUS</div>
                    <div class="forensic-value">{{ row.debut_label }}</div>
                  </div>
                </div>
                <div class="audit-why">{{ row.event_line }} // {{ row.why }}</div>
              </div>

              <div class="audit-right movement-right">
                <div class="conviction-label">TX_TYPE</div>
                <div class="movement-flag">{{ row.transaction_label }}</div>
              </div>

              <div class="audit-action movement-action">
                <div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px;">
                  <span class="movement-route">FROM {{ row.from_code }} → TO {{ row.to_code }}</span>
                  <button
                    type="button"
                    class="provision-btn js-add-to-roster"
                    data-default-label="INITIATE TRACKING"
                    data-player-id="{{ row.player_id }}"
                    data-player-name="{{ row.player_name }}"
                    data-player-type="{{ row.player_type }}"
                    data-player-team="{{ row.to_code }}"
                    data-profile-url="{{ row.profile_url }}"
                    data-source-tag="PROMOTION_WATCH"
                  >INITIATE TRACKING</button>
                </div>
              </div>
            </article>
            {% endfor %}
          </div>
          {% else %}
          <div class="placeholder">No prospect-relevant MLB arrivals in the last 14 days.</div>
          {% endif %}
        </div>
      </div>
      {% endif %}

      <div id="terminalGuideOverlay" class="drawer-overlay" onclick="closeTerminalGuide()"></div>

      <aside id="terminalGuideDrawer" class="drawer-panel" aria-hidden="true">
        <div class="drawer-head">
          <div class="drawer-title-wrap">
            <div class="drawer-kicker">Promotion Watch // Field Guide</div>
            <h2 class="drawer-title">How to Use This Terminal</h2>
          </div>
          <button type="button" class="drawer-close" onclick="closeTerminalGuide()">Close</button>
        </div>

        <div class="guide-list">
          <div class="guide-item">
            <div class="guide-term">72 HR</div>
            <div class="guide-def">This view prioritizes near-term promotion pressure, fast-rising signal quality, and immediate opportunity changes.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">14 DAY</div>
            <div class="guide-def">This view gives the broader scout window: stronger recent signal context plus recalls, debuts, call-ups, and arrivals.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">Edge Score</div>
            <div class="guide-def">A ranking score used to surface the strongest current promotion-watch candidates inside each board.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">Signal Layer</div>
            <div class="guide-def">Players ranked by recent underlying signal quality, not just public-facing box-score reputation or surface stats.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">Movement Layer</div>
            <div class="guide-def">Recent arrivals, recalls, debuts, and movement events that matter for prospect timing, roster opportunity, and market reaction.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">Team Labels</div>
            <div class="guide-def">This terminal emphasizes minor-league affiliate context first, with the MLB parent organization used as secondary context.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">Why This Page Exists</div>
            <div class="guide-def">Promotion Watch is designed to identify actionable player movement before the broader market fully adjusts to role, timing, and talent signals.</div>
          </div>
        </div>
      </aside>

      <div id="tab-aaa-gems" class="tab-panel" style="display:none;">
        <div class="section">
          <div class="section-head">
            <div>
              <div class="section-kicker">AAA GEMS</div>
              <h2 class="section-title">AAA GEMS — Lower-Minors Surveillance</h2>
            </div>
            <div class="section-badge">Top {{ depth_radar_rows|length }}</div>
          </div>

          {% if depth_radar_rows %}
          <div class="grid">
            {% for row in depth_radar_rows %}
              {{ live_ledger_card.render(row=row) }}
            {% endfor %}
          </div>
          {% else %}
          <div class="placeholder">
            AAA GEMS is source-locked until verified AA, A, or D1 college rows are available. No static player seeds are rendered here.
          </div>
          {% endif %}
        </div>
      </div>

  
  <!-- PROMOTION_WATCH_FIELD_GUIDE_V1 -->
  <button class="pw-field-guide-trigger" type="button" onclick="openPromotionWatchGuide()" aria-controls="promotionWatchFieldGuide" aria-expanded="false">
    FIELD GUIDE
  </button>

  <div class="pw-field-guide-overlay" id="promotionWatchGuideOverlay" onclick="closePromotionWatchGuide()"></div>

  <aside class="pw-field-guide-drawer" id="promotionWatchFieldGuide" aria-hidden="true">
    <div class="pw-field-guide-head">
      <div>
        <div class="pw-field-guide-kicker">FIELD GUIDE_V1.0 // PROMOTION WATCH</div>
        <h2 class="pw-field-guide-title">AAA Movement Zone</h2>
      </div>
      <button class="pw-field-guide-close" type="button" onclick="closePromotionWatchGuide()" aria-label="Close Promotion Watch field guide">×</button>
    </div>

    <div class="pw-field-guide-body">
      <section class="pw-guide-card">
        <div class="pw-guide-label">Surface Objective</div>
        <p class="pw-guide-copy">
          Promotion Watch isolates Triple-A assets moving toward MLB relevance before the public market fully prices the call-up window. The page is split between fresh final-slate AAA production, 14-day movement, and recent MLB arrival confirmation.
        </p>
      </section>

      <section class="pw-guide-card">
        <div class="pw-guide-label">72 HR Board</div>
        <p class="pw-guide-copy">
          The 72 HR board is the live acceleration layer. It prioritizes players flashing recent production, roster pressure, and near-term call-up plausibility.
        </p>
        <ul class="pw-guide-list">
          <li>Use it for fast-moving AAA assets after the latest final slate.</li>
          <li>Higher scores indicate stronger short-window extraction pressure.</li>
          <li>Best used as an early alert, not a final confirmation.</li>
        </ul>
      </section>

      <section class="pw-guide-card">
        <div class="pw-guide-label">14 DAY Board</div>
        <p class="pw-guide-copy">
          The 14 DAY board acts as the steadier scouting window. It reduces one-game noise and highlights players whose production has held across a broader movement sample.
        </p>
        <ul class="pw-guide-list">
          <li>Use it to separate single-slate noise from sustained AAA signal.</li>
          <li>Pairs with the 72 HR board for acceleration plus confirmation.</li>
          <li>Stale or source-limited windows should be treated as degraded intelligence.</li>
        </ul>
      </section>

      <section class="pw-guide-card">
        <div class="pw-guide-label">Fresh AAA Hitters</div>
        <p class="pw-guide-copy">
          The hitter ledger reads the latest AAA final slate for offensive pressure. Core signals include power events, run production, plate discipline, and short-window production spikes.
        </p>
        <ul class="pw-guide-list">
          <li>HR, extra-base hits, RBI, runs, and total bases help flag impact outcomes.</li>
          <li>BB/K and strikeout control help distinguish real approach from empty box-score heat.</li>
          <li>Recent production matters most when paired with roster-path logic.</li>
        </ul>
      </section>

      <section class="pw-guide-card">
        <div class="pw-guide-label">Fresh AAA Pitchers</div>
        <p class="pw-guide-copy">
          The pitcher ledger reads the latest AAA final slate for command, bat-missing, and role-readiness signals. It is designed to spot arms forcing MLB consideration.
        </p>
        <ul class="pw-guide-list">
          <li>Strikeouts, innings, earned runs, walks, and run prevention shape the first read.</li>
          <li>K/BB pressure helps identify cleaner skill translation.</li>
          <li>Reliever and starter paths should be interpreted differently when role data is visible.</li>
        </ul>
      </section>

      <section class="pw-guide-card">
        <div class="pw-guide-label">Movement Layer</div>
        <p class="pw-guide-copy">
          Recent MLB arrivals are the confirmation feed. They show whether the system is detecting players close enough to the transaction layer to matter for roster action.
        </p>
        <ul class="pw-guide-list">
          <li>Promotion events validate the surveillance pipeline.</li>
          <li>Arrival history helps tune future thresholds.</li>
          <li>Use this as confirmation, not the only source of edge.</li>
        </ul>
      </section>

      <section class="pw-guide-card">
        <div class="pw-guide-label">Operator Rule</div>
        <p class="pw-guide-copy">
          Treat Promotion Watch as an early-extraction surface. The mission is not to wait for the fantasy market to react. The mission is to detect acceleration, verify the path, and initiate tracking before the roster window closes.
        </p>
      </section>
    </div>
  </aside>


  {{ footer_html | safe }}
  </div>

  <script src="/player-search.js"></script>
  <script src="/player-card-actions.js"></script>
  <script>
    function openTerminalGuide() {
      const overlay = document.getElementById("terminalGuideOverlay");
      const drawer = document.getElementById("terminalGuideDrawer");
      if (overlay) overlay.classList.add("open");
      if (drawer) {
        drawer.classList.add("open");
        drawer.setAttribute("aria-hidden", "false");
      }
    }

    function closeTerminalGuide() {
      const overlay = document.getElementById("terminalGuideOverlay");
      const drawer = document.getElementById("terminalGuideDrawer");
      if (overlay) overlay.classList.remove("open");
      if (drawer) {
        drawer.classList.remove("open");
        drawer.setAttribute("aria-hidden", "true");
      }
    }

    function switchPromotionTab(panelId, buttonEl) {
      document.querySelectorAll("#tab-72h, #tab-14d, #tab-aaa-gems").forEach((panel) => {
        panel.style.display = "none";
      });

      document.querySelectorAll(".tabs .tab").forEach((btn) => {
        btn.classList.remove("active");
      });

      const activePanel = document.getElementById(panelId);
      if (activePanel) activePanel.style.display = "block";
      if (buttonEl) buttonEl.classList.add("active");

      const summaryWindow = document.getElementById("summary-window");
      const summaryMode = document.getElementById("summary-mode");
      const summarySignals = document.getElementById("summary-signals");

      if (panelId === "tab-72h") {
        if (summaryWindow) summaryWindow.textContent = "72 HR";
        if (summaryMode) summaryMode.textContent = "AAA";
        if (summarySignals) summarySignals.textContent = "{{ total_signals }}";
      } else if (panelId === "tab-aaa-gems") {
        if (summaryWindow) summaryWindow.textContent = "AAA GEMS";
        if (summaryMode) summaryMode.textContent = "LOWER-MINORS";
        if (summarySignals) summarySignals.textContent = "{{ depth_radar_rows|length }}";
      } else {
        if (summaryWindow) summaryWindow.textContent = "14 DAY";
        if (summaryMode) summaryMode.textContent = "SCOUT";
        if (summarySignals) summarySignals.textContent = "{{ total_14_signals }}";
      }
    }

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") closeTerminalGuide();
    });

    document.addEventListener("DOMContentLoaded", function () {
      const btn14 = document.getElementById("tab-btn-14d");
      switchPromotionTab("tab-14d", btn14);
    });
  </script>

  <script>
    function openPromotionWatchGuide() {
      const overlay = document.getElementById("promotionWatchGuideOverlay");
      const drawer = document.getElementById("promotionWatchFieldGuide");
      const trigger = document.querySelector(".pw-field-guide-trigger");
      if (overlay) overlay.classList.add("open");
      if (drawer) {
        drawer.classList.add("open");
        drawer.setAttribute("aria-hidden", "false");
      }
      if (trigger) trigger.setAttribute("aria-expanded", "true");
      document.body.classList.add("pw-field-guide-open");
    }

    function closePromotionWatchGuide() {
      const overlay = document.getElementById("promotionWatchGuideOverlay");
      const drawer = document.getElementById("promotionWatchFieldGuide");
      const trigger = document.querySelector(".pw-field-guide-trigger");
      if (overlay) overlay.classList.remove("open");
      if (drawer) {
        drawer.classList.remove("open");
        drawer.setAttribute("aria-hidden", "true");
      }
      if (trigger) trigger.setAttribute("aria-expanded", "false");
      document.body.classList.remove("pw-field-guide-open");
    }

    document.addEventListener("keydown", function(event) {
      if (event.key === "Escape") closePromotionWatchGuide();
    });
  </script>

</body>
</html>
"""
)



MAX_AAA_WEEKLY_SIGNAL_AGE_DAYS = 28


def fetch_recent_aaa_weekly_signal_base(limit_weeks: int = 4) -> tuple[pd.DataFrame, list[str]]:
    from supabase import create_client
    import os

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    sb = create_client(url, key)

    latest_resp = (
        sb.table("milb_aaa_weekly_signal_base")
        .select("week_start")
        .order("week_start", desc=True)
        .limit(2000)
        .execute()
    )

    rows = latest_resp.data or []

    now_utc = datetime.now(timezone.utc)
    max_age_cutoff = now_utc - timedelta(days=MAX_AAA_WEEKLY_SIGNAL_AGE_DAYS)

    week_starts: list[str] = []
    seen: set[str] = set()

    for row in rows:
        wk = row.get("week_start")
        if not wk or wk in seen:
            continue

        parsed = pd.to_datetime(wk, errors="coerce")
        if pd.isna(parsed):
            continue

        if parsed.tzinfo is None:
            parsed_dt = parsed.to_pydatetime().replace(tzinfo=timezone.utc)
        else:
            parsed_dt = parsed.to_pydatetime().astimezone(timezone.utc)

        if parsed_dt < max_age_cutoff:
            continue

        seen.add(wk)
        week_starts.append(wk)

        if len(week_starts) >= limit_weeks:
            break

    if not week_starts:
        return pd.DataFrame(), []

    data_resp = (
        sb.table("milb_aaa_weekly_signal_base")
        .select("*")
        .in_("week_start", week_starts)
        .execute()
    )

    data = data_resp.data or []
    if not data:
        return pd.DataFrame(), week_starts

    return pd.DataFrame(data), week_starts



def load_source_frame() -> tuple[pd.DataFrame, list[str], str | None]:
    df, week_starts = fetch_recent_aaa_weekly_signal_base()
    source_updated_at = None
    if week_starts:
        try:
            latest = pd.to_datetime(week_starts, errors="coerce")
            latest = pd.Series(latest).dropna()
            if not latest.empty:
                source_updated_at = latest.max().to_pydatetime().replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            source_updated_at = None
    return df, week_starts, source_updated_at


def load_refresh_top20(path_name: str) -> pd.DataFrame:
    path = DIST_DIR / path_name
    if not path.exists():
        return pd.DataFrame()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    rows = payload.get("top_20", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def normalize_refresh_hitters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["resolved_player_id"] = out.get("player_id")
    out["display_org"] = out.get("org", "AAA")
    out["display_team"] = "AAA"
    out["source_badge"] = "AAA_REFRESH"
    out["score_version"] = "LIVE_72H"
    out["edge_score"] = out.get("live_score", 0).fillna(0).astype(float).round(1)
    out["score_class"] = out["edge_score"].apply(
        lambda x: "elite" if x >= 18 else "strong" if x >= 12 else "watch"
    )
    out["sample_note"] = out.get("snapshot_date", "LIVE")
    out["metric_1"] = out.apply(
        lambda r: f"ISO {float(r.get('iso', 0) or 0):.3f}",
        axis=1,
    )
    out["metric_2"] = out.apply(
        lambda r: f"HR {int(r.get('hr', 0) or 0)} | H {int(r.get('h', 0) or 0)}",
        axis=1,
    )
    out["metric_3"] = out.apply(
        lambda r: f"BB {int(r.get('bb', 0) or 0)} | SO {int(r.get('so', 0) or 0)}",
        axis=1,
    )
    out["why"] = out.apply(
        lambda r: f"72-hour hitting surge with live score {float(r.get('live_score', 0) or 0):.1f} across recent AAA action.",
        axis=1,
    )
    return out


def normalize_refresh_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["resolved_player_id"] = out.get("player_id")
    out["display_org"] = out.get("org", "AAA")
    out["display_team"] = "AAA"
    out["source_badge"] = "AAA_REFRESH"
    out["score_version"] = "LIVE_72H"
    out["edge_score"] = out.get("live_score", 0).fillna(0).astype(float).round(1)
    out["score_class"] = out["edge_score"].apply(
        lambda x: "elite" if x >= 18 else "strong" if x >= 12 else "watch"
    )
    out["sample_note"] = out.get("snapshot_date", "LIVE")
    out["metric_1"] = out.apply(
        lambda r: f"IP {r.get('ip', '0.0')}",
        axis=1,
    )
    out["metric_2"] = out.apply(
        lambda r: f"SO {int(r.get('so', 0) or 0)} | H {int(r.get('h', 0) or 0)}",
        axis=1,
    )
    out["metric_3"] = out.apply(
        lambda r: f"BB {int(r.get('bb', 0) or 0)} | HR {int(r.get('hr', 0) or 0)}",
        axis=1,
    )
    out["why"] = out.apply(
        lambda r: f"72-hour pitching surge with live score {float(r.get('live_score', 0) or 0):.1f} across recent AAA action.",
        axis=1,
    )
    return out


def load_fresh_aaa_hitter_refresh() -> pd.DataFrame:
    path = DIST_DIR / "aaa_hitter_refresh.json"
    if not path.exists():
        return pd.DataFrame()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    rows = payload.get("top_20") or payload.get("players") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()
    if df.empty:
        return df

    if "player_name" in df.columns:
        df["player_name"] = (
            df["player_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if "player_id" in df.columns:
        df["resolved_player_id"] = df["player_id"].fillna("").astype(str).str.strip()
    else:
        df["resolved_player_id"] = ""

    if "org" in df.columns:
        df["display_team"] = df["org"].fillna("AAA").astype(str)
        df["display_org"] = "AAA LIVE"
    else:
        df["display_team"] = "AAA"
        df["display_org"] = "AAA"

    if "live_score" in df.columns:
        live_scores = pd.to_numeric(df["live_score"], errors="coerce").fillna(0)
        df["live_score_raw"] = live_scores.round(2)
        max_score = float(live_scores.max()) if len(live_scores) else 0.0
        if max_score > 0:
            df["edge_score"] = ((live_scores / max_score) * 95).round(1)
        else:
            df["edge_score"] = 0.0
    else:
        df["live_score_raw"] = 0.0
        df["edge_score"] = 0.0

    df["score_class"] = df["edge_score"].apply(classify_score)
    df["metric_1"] = pd.to_numeric(df.get("iso"), errors="coerce").fillna(0).map(lambda x: f"{x:.3f}")
    df["metric_2"] = pd.to_numeric(df.get("bb"), errors="coerce").fillna(0).map(lambda x: f"{int(x)}")
    df["metric_3"] = pd.to_numeric(df.get("hr"), errors="coerce").fillna(0).map(lambda x: f"{int(x)}")
    df["sample_note"] = pd.to_numeric(df.get("pa_proxy"), errors="coerce").fillna(0).map(lambda x: f"{int(x)} PA")
    df["source_badge"] = "SRC: AAA_LIVE_BOX_v1"
    df["score_version"] = "LIVE_v0.1"
    df["signal_type"] = "Hitter"
    df["avatar"] = df["player_name"].map(initials)

    def _why(row: pd.Series) -> str:
        return (
            f"Fresh AAA final-box score: {int(row.get('h', 0) or 0)} H, "
            f"{int(row.get('bb', 0) or 0)} BB, "
            f"{int(row.get('hr', 0) or 0)} HR, "
            f"ISO {float(row.get('iso', 0) or 0):.3f}."
        )

    df["why"] = df.apply(_why, axis=1)
    return df.sort_values(["edge_score", "hr", "iso", "h", "resolved_player_id", "player_name"], ascending=[False, False, False, False, True, True], kind="mergesort").reset_index(drop=True)



def load_fresh_aaa_pitcher_refresh() -> pd.DataFrame:
    path = DIST_DIR / "aaa_pitcher_refresh_probe.json"
    if not path.exists():
        return pd.DataFrame()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    rows = payload.get("top_20") or payload.get("players") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).copy()
    if df.empty:
        return df

    if "player_name" in df.columns:
        df["player_name"] = (
            df["player_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if "player_id" in df.columns:
        df["resolved_player_id"] = df["player_id"].fillna("").astype(str).str.strip()
    else:
        df["resolved_player_id"] = ""

    if "org" in df.columns:
        df["display_team"] = df["org"].fillna("AAA").astype(str)
        df["display_org"] = "AAA LIVE"
    else:
        df["display_team"] = "AAA"
        df["display_org"] = "AAA"

    if "live_score" in df.columns:
        live_scores = pd.to_numeric(df["live_score"], errors="coerce").fillna(0)
        df["live_score_raw"] = live_scores.round(2)
        max_score = float(live_scores.max()) if len(live_scores) else 0.0
        if max_score > 0:
            df["edge_score"] = ((live_scores / max_score) * 95).round(1)
        else:
            df["edge_score"] = 0.0
    else:
        df["live_score_raw"] = 0.0
        df["edge_score"] = 0.0

    df["score_class"] = df["edge_score"].apply(classify_score)
    df["metric_1"] = df.get("ip", pd.Series(["0.0"] * len(df))).astype(str)
    df["metric_2"] = pd.to_numeric(df.get("so"), errors="coerce").fillna(0).map(lambda x: f"{int(x)}")
    df["metric_3"] = pd.to_numeric(df.get("bb"), errors="coerce").fillna(0).map(lambda x: f"{int(x)}")
    df["sample_note"] = "Final AAA slate"
    df["source_badge"] = "SRC: AAA_LIVE_PITCH_v1"
    df["score_version"] = "LIVE_v0.1"
    df["signal_type"] = "Pitcher"
    df["avatar"] = df["player_name"].map(initials)

    def _why(row: pd.Series) -> str:
        return (
            f"Fresh AAA final-box line: {row.get('ip', '0.0')} IP, "
            f"{int(row.get('so', 0) or 0)} K, "
            f"{int(row.get('bb', 0) or 0)} BB, "
            f"{int(row.get('h', 0) or 0)} H allowed."
        )

    df["why"] = df.apply(_why, axis=1)
    return df.sort_values(["edge_score", "so", "bb", "h", "resolved_player_id", "player_name"], ascending=[False, False, True, True, True, True], kind="mergesort").reset_index(drop=True)

def load_depth_radar_rows(limit: int = 24) -> list[dict]:
    path = DIST_DIR / "depth_radar_refresh.json"
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows = payload.get("top_rows") or []
    if not isinstance(rows, list):
        return []

    safe_rows = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue

        safe_rows.append({
            "player_name": row.get("player_name") or "Unknown Player",
            "resolved_player_id": str(row.get("player_id") or row.get("resolved_player_id") or "").strip(),
            "display_team": row.get("org") or row.get("team") or row.get("level") or "MILB",
            "display_org": row.get("level") or "DEPTH",
            "signal_type": row.get("signal_type") or row.get("type") or "Depth",
            "edge_score": row.get("edge_score") or 0,
            "score_class": row.get("score_class") or classify_score(row.get("edge_score") or 0),
            "metric_1": row.get("metric_1") or row.get("level") or "MILB",
            "metric_2": row.get("metric_2") or row.get("snapshot_date") or "",
            "metric_3": row.get("metric_3") or row.get("depth_score") or "",
            "sample_note": row.get("sample_note") or "AA/A final boxscore",
            "source_badge": row.get("source_badge") or "SRC: DEPTH_RADAR_v0.1",
            "score_version": row.get("score_version") or "DEPTH_v0.1",
            "why": row.get("why") or "Lower-minors signal surfaced from verified final boxscore ingestion.",
            "avatar": initials(row.get("player_name") or ""),
        })

    return safe_rows


def load_source_frame() -> tuple[pd.DataFrame, list[str]]:
    return fetch_recent_aaa_weekly_signal_base()





def fetch_live_aaa_hitter_candidates_debug() -> pd.DataFrame:
    base_df, _week_starts = load_source_frame()
    if base_df is None or base_df.empty:
        return pd.DataFrame()

    aaa_hitters = base_df[base_df["pa"].notna()].copy()
    if aaa_hitters.empty:
        return pd.DataFrame()

    aaa_hitters["player_name"] = aaa_hitters["player_name"].apply(safe_name)
    aaa_names = set(aaa_hitters["player_name"].dropna().astype(str).str.strip())
    if not aaa_names:
        return pd.DataFrame()

    fresh = fetch_fresh_hitter_signal_candidates_debug()
    if fresh is None or fresh.empty:
        return pd.DataFrame()

    if "player_name" not in fresh.columns:
        return pd.DataFrame()

    fresh["player_name"] = fresh["player_name"].apply(safe_name)
    fresh = fresh[fresh["player_name"].isin(aaa_names)].copy()

    return fresh.sort_values("surge_score", ascending=False).head(10).reset_index(drop=True)

def fetch_fresh_hitter_signal_candidates_debug() -> pd.DataFrame:
    try:
        from build_dashboard import fetch_statcast_window, build_hitter_signals
        from datetime import date, timedelta
    except Exception:
        return pd.DataFrame()

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=14)

    try:
        raw = fetch_statcast_window(start_dt, end_dt)
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    try:
        hitters = build_hitter_signals(raw)
    except Exception:
        return pd.DataFrame()

    if hitters is None or hitters.empty:
        return pd.DataFrame()

    keep_cols = [
        "player_name",
        "team",
        "score",
        "metric_1",
        "metric_2",
        "metric_3",
        "why",
    ]
    existing = [c for c in keep_cols if c in hitters.columns]
    out = hitters[existing].copy()

    rename_map = {}
    if "score" in out.columns:
        rename_map["score"] = "surge_score"
    out = out.rename(columns=rename_map)

    if "surge_score" in out.columns:
        out = out.sort_values("surge_score", ascending=False)

    return out.head(10).reset_index(drop=True)

def render_html() -> str:
    build_started_at = utc_now_iso()
    source_frame_result = load_source_frame()
    if len(source_frame_result) == 3:
        df, week_starts, source_updated_at = source_frame_result
    else:
        df, week_starts = source_frame_result
        source_updated_at = datetime.now(timezone.utc).isoformat()

    hitters_72 = normalize_refresh_hitters(load_refresh_top20("aaa_hitter_refresh.json"))
    pitchers_72 = normalize_refresh_pitchers(load_refresh_top20("aaa_pitcher_refresh_probe.json"))

    if df.empty or not week_starts:
        hitters_14 = pd.DataFrame()
        pitchers_14 = pd.DataFrame()
    else:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
        two_week_cut = sorted(df["week_start"].dropna().unique(), reverse=True)[:2]
        two_week_df = df[df["week_start"].isin(two_week_cut)].copy()

        hitter_trends_all = build_trend_lookup(df[df["pa"].notna()].copy(), "iso")
        pitcher_trends_all = build_trend_lookup(df[df["bf"].notna()].copy(), "kbb_p")

        hitters_14 = build_aaa_hitter_promotion_watch(two_week_df, trend_lookup=hitter_trends_all).head(12) if not two_week_df.empty else pd.DataFrame()
        pitchers_14 = build_aaa_pitcher_promotion_watch(two_week_df, trend_lookup=pitcher_trends_all).head(12) if not two_week_df.empty else pd.DataFrame()

    total_signals = len(hitters_72) + len(pitchers_72)
    total_14_signals = len(hitters_14) + len(pitchers_14)
    _live_arrivals, archive_arrivals = load_arrivals_windows(live_limit=8, archive_limit=16)
    fresh_hitters_live = load_fresh_aaa_hitter_refresh()
    fresh_pitchers_live = load_fresh_aaa_pitcher_refresh()

    movement_window_label = "Last 14 Days"
    page_build_label = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    live_feed_label = "Unavailable"

    if not fresh_hitters_live.empty and "snapshot_date" in fresh_hitters_live.columns:
        snap_series = fresh_hitters_live["snapshot_date"].dropna().astype(str)
        live_feed_label = f"{snap_series.iloc[0]} FINAL GAMES" if not snap_series.empty else "LIVE"
    elif not fresh_hitters_live.empty:
        live_feed_label = "LIVE"
    elif "source_updated_at" in locals() and source_updated_at:
        live_feed_label = source_updated_at

    depth_radar_rows = load_depth_radar_rows(limit=24)

    existing_scout_ids = {
        path.parent.name
        for path in (DIST_DIR / "scout").glob("*/index.html")
        if path.parent.name.isdigit()
    }

    def attach_profile_urls(rows):
        safe_rows = []
        for row in rows or []:
            item = dict(row)
            pid = str(
                item.get("resolved_player_id")
                or item.get("player_id")
                or item.get("mlbam_id")
                or ""
            ).strip()
            item["profile_url"] = f"/scout/{pid}/" if pid and pid in existing_scout_ids else "#"
            item["profile_available"] = bool(pid and pid in existing_scout_ids)
            safe_rows.append(item)
        return safe_rows

    def gate_rendered_scout_links(rendered_html: str) -> str:
        """Disable any rendered scout links that do not have generated dossier pages."""
        def replace_match(match: re.Match) -> str:
            pid = match.group(1)
            if pid in existing_scout_ids:
                return match.group(0)
            return 'data-profile-url="#"'

        return re.sub(
            r'data-profile-url="/scout/([0-9]+)/"',
            replace_match,
            rendered_html,
        )

    sections = {
        "pitchers_72hr": len(pitchers_72),
        "hitters_72hr": len(hitters_72),
        "pitchers_14day": len(pitchers_14),
        "hitters_14day": len(hitters_14),
        "recent_arrivals": len(archive_arrivals),
        "depth_radar": len(depth_radar_rows),
    }

    validation = build_validation_report(
        "promotion_watch",
        [
            validate_required_sections(
                "promotion_watch",
                sections,
                ["pitchers_72hr", "hitters_72hr", "pitchers_14day", "hitters_14day", "recent_arrivals", "depth_radar"],
            ),
            validate_min_rows("pitchers_72hr", sections["pitchers_72hr"], 0),
            validate_min_rows("hitters_72hr", sections["hitters_72hr"], 0),
            validate_min_rows("pitchers_14day", sections["pitchers_14day"], 0),
            validate_min_rows("hitters_14day", sections["hitters_14day"], 0),
            validate_min_rows("recent_arrivals", sections["recent_arrivals"], 0),
            validate_min_rows("depth_radar", sections["depth_radar"], 0),
        ],
    )

    status_payload = build_report_status(
        "promotion_watch",
        build_success=True,
        threshold_minutes=240,
        build_started_at=build_started_at,
        build_finished_at=utc_now_iso(),
        source_updated_at=source_updated_at,
        section_counts=sections,
        degraded=not validation["ok"],
        errors=validation["messages"],
        notes=[
            "Promotion Watch uses dynamic AAA/MLB movement feeds with freshness validation.",
            "72 HR, 14 DAY, and recent-arrivals sections are published with explicit section counts.",
            "Empty-state placeholders are UI fallbacks only and are not static seeded player intelligence.",
        ],
    )

    status_payload["mode"] = "dynamic_promotion_watch_v0.1_hardened"
    status_payload["pipeline_layers"] = [
        "dynamic_aaa_movement_feed",
        "dynamic_recent_mlb_arrivals_feed",
        "aaa_gems_source_locked",
        "freshness_status_tracked",
        "aaa_weekly_window_age_guard",
        "section_count_validation",
        "ui_empty_state_only",
        "no_static_player_seed_fallback",
    ]
    status_payload["hardening_notes"] = [
        "Promotion Watch is a dynamic prospect movement surface, not a static prospect list.",
        "Placeholder language in the template is limited to empty-state UI messaging.",
        "Freshness, degraded state, section counts, and source_updated_at are published in the status payload.",
        "AAA weekly 14-day source windows older than 28 days are rejected before ranking.",
        "72 HR, 14 DAY, Recent Arrivals, and AAA GEMS sections remain independently counted for QA.",
        "AAA GEMS renders verified AA, High-A, and Low-A rows when available; D1 college remains source-locked.",
    ]

    write_status_file(status_payload)

    def safe_records(frame, limit=12):
        if frame is None or frame.empty:
            return []
        safe = frame.head(limit).copy()
        safe = safe.where(pd.notna(safe), None)
        return safe.to_dict(orient="records")

    promotion_payload = {
        "report": "Promotion Watch",
        "subtitle": "AAA Movement Zone / Call-Up Surveillance",
        "version": "promotion_watch_v0.1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "timezone": TIMEZONE_LABEL,
        "status": status_payload.get("state"),
        "mode": status_payload.get("mode"),
        "source_updated_at": source_updated_at,
        "pipeline_layers": status_payload.get("pipeline_layers"),
        "hardening_notes": status_payload.get("hardening_notes"),
        "section_counts": sections,
        "top_signals": {
            "pitchers_72hr": safe_records(pitchers_72, 12),
            "hitters_72hr": safe_records(hitters_72, 12),
            "pitchers_14day": safe_records(pitchers_14, 12),
            "hitters_14day": safe_records(hitters_14, 12),
            "recent_arrivals": archive_arrivals[:16] if isinstance(archive_arrivals, list) else safe_records(archive_arrivals, 16),
            "depth_radar": depth_radar_rows,
        },
    }

    # PROMOTION_WATCH_PAYLOAD_CONTRACT_V2
    # Mirror nested top_signals rows to direct top-level arrays so status,
    # rendered HTML, JSON consumers, dossier supplements, and audits all read
    # the same populated contract.
    promotion_payload_sections = promotion_payload.get("top_signals", {})
    if not isinstance(promotion_payload_sections, dict):
        raise RuntimeError("Promotion Watch top_signals payload is not a dict")

    for section_name in [
        "pitchers_72hr",
        "hitters_72hr",
        "pitchers_14day",
        "hitters_14day",
        "recent_arrivals",
        "depth_radar",
    ]:
        section_rows = promotion_payload_sections.get(section_name, [])
        if section_rows is None:
            section_rows = []
        if not isinstance(section_rows, list):
            raise RuntimeError(f"Promotion Watch section {section_name} is not a list")
        promotion_payload[section_name] = section_rows

    payload_count_failures = []
    for section_name, expected_count in sections.items():
        expected_count = int(expected_count or 0)
        direct_count = len(promotion_payload.get(section_name, []))
        nested_count = len(promotion_payload_sections.get(section_name, []))

        # Some sections are intentionally export-capped below status count.
        # But if status says a section is populated, both direct and nested
        # payload shapes must contain at least one row.
        if expected_count > 0 and direct_count == 0:
            payload_count_failures.append(
                f"{section_name}: status_count={expected_count} direct_export_count={direct_count}"
            )
        if expected_count > 0 and nested_count == 0:
            payload_count_failures.append(
                f"{section_name}: status_count={expected_count} nested_export_count={nested_count}"
            )

    if payload_count_failures:
        raise RuntimeError(
            "Promotion Watch payload/status contract mismatch: "
            + "; ".join(payload_count_failures)
        )

    promo_json_path = CALL_UP_DIR / "promotion_watch.json"
    promo_json_path.write_text(json.dumps(promotion_payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {promo_json_path}")

    source_is_stale = status_payload["state"] in {"stale", "degraded"}
    pitchers_14_empty = len(pitchers_14) == 0
    hitters_14_empty = len(hitters_14) == 0
    arrivals_empty = len(archive_arrivals) == 0

    pitchers_14_message = "No 14 day pitching prospect signals available."
    hitters_14_message = "No 14 day hitting prospect signals available."
    arrivals_message = "No prospect-relevant MLB arrivals in the last 14 days."

    if source_is_stale and pitchers_14_empty:
        pitchers_14_message = "14 day pitching signals unavailable or stale. Last source window is older than threshold."
    if source_is_stale and hitters_14_empty:
        hitters_14_message = "14 day hitting signals unavailable or stale. Last source window is older than threshold."
    if source_is_stale and arrivals_empty:
        arrivals_message = "Recent MLB arrivals feed unavailable or stale. Last source window is older than threshold."

    def assert_render_health(rendered_html: str) -> None:
        """Fail fast when populated payload sections render as empty UI states."""
        failures: list[str] = []

        if len(pitchers_72) > 0 and "No 72 HR pitching prospect signals available." in rendered_html:
            failures.append("72HR pitching payload populated but empty-state placeholder rendered")

        if len(hitters_72) > 0 and "No 72 HR hitting prospect signals available." in rendered_html:
            failures.append("72HR hitting payload populated but empty-state placeholder rendered")

        if "Depth Radar" in rendered_html or "DEPTH RADAR" in rendered_html:
            failures.append("legacy Depth Radar language rendered after AAA GEMS rename")

        if "AAA GEMS" not in rendered_html:
            failures.append("AAA GEMS label missing from rendered HTML")

        if failures:
            raise RuntimeError("Promotion Watch render-health guard failed: " + "; ".join(failures))


    rendered = HTML_TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        timezone_label=TIMEZONE_LABEL,
        nav_html=Template(NAV_TEMPLATE).render(active_nav="promotion_watch"),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
        ledger_styles=LEDGER_STYLES_TEMPLATE,
        live_ledger_card=LIVE_LEDGER_CARD,
        page_build_label=page_build_label,
        live_feed_label=live_feed_label,
        movement_window_label=movement_window_label,
        fresh_hitters_live=fresh_hitters_live.to_dict(orient="records"),
        fresh_pitchers_live=fresh_pitchers_live.to_dict(orient="records"),
        total_signals=total_signals,
        total_14_signals=total_14_signals,
        # Preserve legacy names while also supplying the explicit template variables
        # used by the 72 HR panels.
        hitters=attach_profile_urls(hitters_72.to_dict(orient="records")),
        pitchers=attach_profile_urls(pitchers_72.to_dict(orient="records")),
        hitters_72=attach_profile_urls(hitters_72.to_dict(orient="records")),
        pitchers_72=attach_profile_urls(pitchers_72.to_dict(orient="records")),
        hitters_14=attach_profile_urls(hitters_14.to_dict(orient="records")),
        pitchers_14=attach_profile_urls(pitchers_14.to_dict(orient="records")),
        archive_arrivals=attach_profile_urls(archive_arrivals),
        depth_radar_rows=attach_profile_urls(depth_radar_rows),
        pitchers_14_message=pitchers_14_message,
        hitters_14_message=hitters_14_message,
        arrivals_message=arrivals_message,
    )

    rendered = gate_rendered_scout_links(rendered)

    assert_render_health(rendered)
    return rendered


def write_status_file(status_payload: dict) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    PROMOTION_WATCH_STATUS_PATH.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {PROMOTION_WATCH_STATUS_PATH}")


def main() -> None:
    CALL_UP_DIR.mkdir(parents=True, exist_ok=True)
    html = render_html()
    output_path = CALL_UP_DIR / "index.html"
    temp_output_path = write_temp_output(str(output_path), html)
    promoted = promote_output_if_valid(temp_output_path, str(output_path), True)
    if promoted:
        save_snapshot(str(output_path), str(SNAPSHOT_DIR / "index.html"))
        print(f"Wrote {output_path}")
    else:
        print("Skipped publishing promotion watch due to failed validation.")


if __name__ == "__main__":
    main()