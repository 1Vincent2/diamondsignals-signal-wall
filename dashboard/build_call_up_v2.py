from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import math
import re

import pandas as pd
from jinja2 import Template

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DIST_DIR = REPO_ROOT / "dist"
CALL_UP_DIR = DIST_DIR / "typical-call-up"
TEMPLATES_DIR = BASE_DIR / "templates"

NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")

TIMEZONE_LABEL = "America/New_York"
SOURCE_BADGE = "SRC: AAA_PIPELINE_v1"
SCORE_VERSION = "EDGE_v2.1"

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

AAA_CODES = {v[0] for v in AAA_TO_MLB_CODES.values()}
VALID_TEAM_CODES = MLB_CODES | AAA_CODES


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
    candidates = [
        "org",
        "parent_org",
        "mlb_org",
        "team",
        "team_name",
        "affiliate_name",
    ]
    text = first_non_empty(row, candidates, fallback="—")
    code = map_team_to_code(text)
    return code if code != "—" else text


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

    group_cols = ["player_name"]
    agg_map = {
        "pa": "sum",
        "bb": "sum",
        "so": "sum",
        "hr": "sum",
        "iso": "mean",
    }

    optional_text_cols = [
        "org", "team", "team_abbrev", "org_code", "parent_org", "mlb_org", "week_start"
    ]
    for col in optional_text_cols:
        if col in hitters.columns:
            agg_map[col] = "last"

    agg = hitters.groupby(group_cols, dropna=False).agg(agg_map).reset_index()

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

    group_cols = ["player_name"]
    agg_map = {
        "bf": "sum",
        "so_p": "sum",
        "bb_allowed": "sum",
    }

    optional_text_cols = [
        "org", "team", "team_abbrev", "org_code", "parent_org", "mlb_org", "week_start"
    ]
    for col in optional_text_cols:
        if col in pitchers.columns:
            agg_map[col] = "last"

    agg = pitchers.groupby(group_cols, dropna=False).agg(agg_map).reset_index()

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
            signals.append("Strong plate control")
        elif kbb <= 0.7:
            signals.append("Aggressive contact profile")

        if hr >= 2:
            signals.append("Early HR surge")

        return " • ".join(signals) if signals else "Emerging performance signal under evaluation"

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

    def hitter_badges(row: pd.Series) -> list[tuple[str, str]]:
        badges = []
        if pd.notna(row["iso"]) and row["iso"] >= 0.250:
            badges.append(("Impact Bat", "positive"))
        if pd.notna(row["kbb_h"]) and row["kbb_h"] <= 1.50:
            badges.append(("Zone Control", "positive"))
        if pd.notna(row["hr"]) and row["hr"] >= 2:
            badges.append(("HR Surge", "positive"))
        if row["display_team"] != "—":
            badges.append((row["display_team"], "team"))
        if not badges:
            badges.append(("Promotion Watch", "watch"))
        return badges[:4]

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

    return hitters.sort_values(["edge_score", "pa"], ascending=[False, False]).reset_index(drop=True)


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
        lambda r: (
            f"K/BB {r['kbb_p']:.2f} • {int(r['so_p'] or 0)} K • "
            f"{int(r['bb_allowed'] or 0)} BB over {int(r['bf'] or 0)} BF"
        ),
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

    def pitcher_badges(row: pd.Series) -> list[tuple[str, str]]:
        badges = []
        if pd.notna(row["kbb_p"]) and row["kbb_p"] >= 4:
            badges.append(("Bat-Miss Ready", "positive"))
        if pd.notna(row["bb_allowed"]) and row["bb_allowed"] <= 2:
            badges.append(("Command Hold", "positive"))
        if pd.notna(row["so_p"]) and row["so_p"] >= 10:
            badges.append(("Whiff Volume", "positive"))
        if row["display_team"] != "—":
            badges.append((row["display_team"], "team"))
        if not badges:
            badges.append(("Promotion Watch", "watch"))
        return badges[:4]

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

    return pitchers.sort_values(["edge_score", "bf"], ascending=[False, False]).reset_index(drop=True)


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
  <title>DiamondSignals // Promotion Watch</title>
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

    .hero-card {
      padding: 22px 22px 20px;
    }

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

    .section-card {
      padding: 18px;
    }

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

    #tab-14d .section-kicker {
      color: var(--blue-soft);
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

    .status-badge.team {
      color: var(--blue-soft);
      border-color: rgba(106,166,255,0.20);
      background: rgba(106,166,255,0.06);
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

    @media (max-width: 980px) {
      .hero {
        grid-template-columns: 1fr;
      }

      .signal-grid,
      .arrival-grid {
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

      .hero-card,
      .summary-card,
      .section-card,
      .section,
      .player-card {
        border-radius: 16px;
      }

      .hero-card {
        padding: 18px;
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

      .player-top {
        grid-template-columns: auto 1fr;
      }

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
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="brand-mark"></div>
        <div class="brand-text">
          <div class="brand-kicker"><span class="brand-white">DIAMOND</span><span class="brand-blue">SIGNALS</span></div>
          <div class="brand-title">Promotion Watch // Institutional Elite</div>
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

        <div class="tabs" role="tablist" aria-label="Promotion watch windows">
          <button type="button" class="tab" id="tab-btn-72h" onclick="switchPromotionTab('tab-72h', this)">72 HR</button>
          <button type="button" class="tab active" id="tab-btn-14d" onclick="switchPromotionTab('tab-14d', this)">14 DAY</button>
        </div>
      </div>

      {% if total_signals == 0 and total_14_signals == 0 %}
      <div class="placeholder">
        No live AAA promotion-watch signals available yet.
      </div>
      {% else %}

      <div id="tab-72h" style="display:none;">
        <section class="signal-grid">
          <div class="section">
            <div class="section-head">
              <div>
                <div class="section-kicker">Signal Layer</div>
                <h2 class="section-title">Pitching Prospect Signals — 72 HR</h2>
              </div>
              <div class="section-badge">Top {{ pitchers|length }}</div>
            </div>

            <div class="cards">
              {% for row in pitchers %}
              <article class="player-card {% if row.edge_score >= 80 %}elite-edge{% elif row.edge_score >= 65 %}high-edge{% endif %}">
                <div class="player-top">
                  <div class="avatar">{{ row.avatar }}</div>
                  <div class="player-ident">
                    <div class="rankline">#{{ loop.index }} Pitcher Trigger</div>
                    <h3 class="player-name">{{ row.player_name }}</h3>
                    <div class="signal-line">{{ row.display_org }} // Pitcher // {{ row.sample_note }}</div>
                    <div class="card-meta-row">
                      <span class="card-meta-badge">{{ row.source_badge }}</span>
                      <span class="card-meta-badge">{{ row.score_version }}</span>
                      {% if row.display_team != "—" %}<span class="card-meta-badge team">{{ row.display_team }}</span>{% endif %}
                    </div>
                  </div>
                  <div class="scorebox">
                    <div class="score-label">Edge Score</div>
                    <div class="score-value {{ row.score_class }}">{{ row.edge_score }}</div>
                  </div>
                </div>

                <div class="sparkline-wrap">
                  <div class="sparkline-head">
                    <div class="sparkline-label">Recent Trend</div>
                    <div class="sparkline-note">{{ row.trend_note }}</div>
                  </div>
                  <svg class="sparkline" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden="true">
                    <defs>
                      <linearGradient id="pitcherGradient{{ loop.index }}" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#444444" stop-opacity="0.65"></stop>
                        <stop offset="100%" stop-color="{% if row.edge_score >= 65 %}#b6ff00{% else %}#6aa6ff{% endif %}" stop-opacity="1"></stop>
                      </linearGradient>
                    </defs>
                    <polyline class="sparkline-path {% if row.trend_glow %}glow{% endif %}" stroke="url(#pitcherGradient{{ loop.index }})" points="{{ row.trend_points }}" />
                  </svg>
                </div>

                <div class="badge-row">
                  {% for badge, badge_class in row.badges %}
                  <span class="status-badge {{ badge_class }}">{{ badge }}</span>
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
                <div class="section-kicker">Signal Layer</div>
                <h2 class="section-title">Hitting Prospect Signals — 72 HR</h2>
              </div>
              <div class="section-badge">Top {{ hitters|length }}</div>
            </div>

            <div class="cards">
              {% for row in hitters %}
              <article class="player-card {% if row.edge_score >= 80 %}elite-edge{% elif row.edge_score >= 65 %}high-edge{% endif %}">
                <div class="player-top">
                  <div class="avatar">{{ row.avatar }}</div>
                  <div class="player-ident">
                    <div class="rankline">#{{ loop.index }} Hitter Trigger</div>
                    <h3 class="player-name">{{ row.player_name }}</h3>
                    <div class="signal-line">{{ row.display_org }} // Hitter // {{ row.sample_note }}</div>
                    <div class="card-meta-row">
                      <span class="card-meta-badge">{{ row.source_badge }}</span>
                      <span class="card-meta-badge">{{ row.score_version }}</span>
                      {% if row.display_team != "—" %}<span class="card-meta-badge team">{{ row.display_team }}</span>{% endif %}
                    </div>
                  </div>
                  <div class="scorebox">
                    <div class="score-label">Edge Score</div>
                    <div class="score-value {{ row.score_class }}">{{ row.edge_score }}</div>
                  </div>
                </div>

                <div class="sparkline-wrap">
                  <div class="sparkline-head">
                    <div class="sparkline-label">Recent Trend</div>
                    <div class="sparkline-note">{{ row.trend_note }}</div>
                  </div>
                  <svg class="sparkline" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden="true">
                    <defs>
                      <linearGradient id="hitterGradient{{ loop.index }}" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#444444" stop-opacity="0.65"></stop>
                        <stop offset="100%" stop-color="{% if row.edge_score >= 65 %}#b6ff00{% else %}#6aa6ff{% endif %}" stop-opacity="1"></stop>
                      </linearGradient>
                    </defs>
                    <polyline class="sparkline-path {% if row.trend_glow %}glow{% endif %}" stroke="url(#hitterGradient{{ loop.index }})" points="{{ row.trend_points }}" />
                  </svg>
                </div>

                <div class="badge-row">
                  {% for badge, badge_class in row.badges %}
                  <span class="status-badge {{ badge_class }}">{{ badge }}</span>
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
      </div>

      <div id="tab-14d">
        <section class="signal-grid">
          <div class="section">
            <div class="section-head">
              <div>
                <div class="section-kicker">Signal Layer</div>
                <h2 class="section-title">Pitching Prospect Signals — Last 14 Days</h2>
              </div>
              <div class="section-badge">Top {{ pitchers_14|length }}</div>
            </div>

            {% if pitchers_14 %}
            <div class="cards">
              {% for row in pitchers_14 %}
              <article class="player-card {% if row.edge_score >= 80 %}elite-edge{% elif row.edge_score >= 65 %}high-edge{% endif %}">
                <div class="player-top">
                  <div class="avatar">{{ row.avatar }}</div>
                  <div class="player-ident">
                    <div class="rankline">#{{ loop.index }} Pitcher Trigger</div>
                    <h3 class="player-name">{{ row.player_name }}</h3>
                    <div class="signal-line">{{ row.display_org }} // 14 Day Window // {{ row.sample_note }}</div>
                    <div class="card-meta-row">
                      <span class="card-meta-badge">{{ row.source_badge }}</span>
                      <span class="card-meta-badge">{{ row.score_version }}</span>
                      {% if row.display_team != "—" %}<span class="card-meta-badge team">{{ row.display_team }}</span>{% endif %}
                    </div>
                  </div>
                  <div class="scorebox">
                    <div class="score-label">Edge Score</div>
                    <div class="score-value {{ row.score_class }}">{{ row.edge_score }}</div>
                  </div>
                </div>

                <div class="sparkline-wrap">
                  <div class="sparkline-head">
                    <div class="sparkline-label">Real Trend</div>
                    <div class="sparkline-note">{{ row.trend_note }}</div>
                  </div>
                  <svg class="sparkline" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden="true">
                    <defs>
                      <linearGradient id="pitcher14Gradient{{ loop.index }}" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#444444" stop-opacity="0.65"></stop>
                        <stop offset="100%" stop-color="{% if row.edge_score >= 65 %}#b6ff00{% else %}#6aa6ff{% endif %}" stop-opacity="1"></stop>
                      </linearGradient>
                    </defs>
                    <polyline class="sparkline-path {% if row.trend_glow %}glow{% endif %}" stroke="url(#pitcher14Gradient{{ loop.index }})" points="{{ row.trend_points }}" />
                  </svg>
                </div>

                <div class="badge-row">
                  {% for badge, badge_class in row.badges %}
                  <span class="status-badge {{ badge_class }}">{{ badge }}</span>
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
            {% else %}
            <div class="placeholder">No 14 day pitching prospect signals available.</div>
            {% endif %}
          </div>

          <div class="section">
            <div class="section-head">
              <div>
                <div class="section-kicker">Signal Layer</div>
                <h2 class="section-title">Hitting Prospect Signals — Last 14 Days</h2>
              </div>
              <div class="section-badge">Top {{ hitters_14|length }}</div>
            </div>

            {% if hitters_14 %}
            <div class="cards">
              {% for row in hitters_14 %}
              <article class="player-card {% if row.edge_score >= 80 %}elite-edge{% elif row.edge_score >= 65 %}high-edge{% endif %}">
                <div class="player-top">
                  <div class="avatar">{{ row.avatar }}</div>
                  <div class="player-ident">
                    <div class="rankline">#{{ loop.index }} Hitter Trigger</div>
                    <h3 class="player-name">{{ row.player_name }}</h3>
                    <div class="signal-line">{{ row.display_org }} // 14 Day Window // {{ row.sample_note }}</div>
                    <div class="card-meta-row">
                      <span class="card-meta-badge">{{ row.source_badge }}</span>
                      <span class="card-meta-badge">{{ row.score_version }}</span>
                      {% if row.display_team != "—" %}<span class="card-meta-badge team">{{ row.display_team }}</span>{% endif %}
                    </div>
                  </div>
                  <div class="scorebox">
                    <div class="score-label">Edge Score</div>
                    <div class="score-value {{ row.score_class }}">{{ row.edge_score }}</div>
                  </div>
                </div>

                <div class="sparkline-wrap">
                  <div class="sparkline-head">
                    <div class="sparkline-label">Real Trend</div>
                    <div class="sparkline-note">{{ row.trend_note }}</div>
                  </div>
                  <svg class="sparkline" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden="true">
                    <defs>
                      <linearGradient id="hitter14Gradient{{ loop.index }}" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#444444" stop-opacity="0.65"></stop>
                        <stop offset="100%" stop-color="{% if row.edge_score >= 65 %}#b6ff00{% else %}#6aa6ff{% endif %}" stop-opacity="1"></stop>
                      </linearGradient>
                    </defs>
                    <polyline class="sparkline-path {% if row.trend_glow %}glow{% endif %}" stroke="url(#hitter14Gradient{{ loop.index }})" points="{{ row.trend_points }}" />
                  </svg>
                </div>

                <div class="badge-row">
                  {% for badge, badge_class in row.badges %}
                  <span class="status-badge {{ badge_class }}">{{ badge }}</span>
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
            {% else %}
            <div class="placeholder">No 14 day hitting prospect signals available.</div>
            {% endif %}
          </div>
        </section>

        <div class="section" style="margin-top: 16px;">
          <div class="section-head">
            <div>
              <div class="section-kicker">Movement Layer</div>
              <h2 class="section-title">Recent MLB Arrivals — Last 14 Days</h2>
            </div>
            <div class="section-badge">{{ archive_arrivals|length }} Players</div>
          </div>

          {% if archive_arrivals %}
          <div class="arrival-grid">
            {% for row in archive_arrivals %}
            <article class="player-card">
              <div class="player-top">
                <div class="avatar">{{ row.avatar }}</div>
                <div class="player-ident">
                  <div class="rankline">Movement Trigger // {{ row.transaction_label }}</div>
                  <h3 class="player-name">{{ row.player_name }}</h3>
                  <div class="signal-line">{{ row.from_code }} → {{ row.to_code }} // {{ row.event_line }}</div>
                  <div class="card-meta-row">
                    <span class="card-meta-badge">Arrival</span>
                    <span class="card-meta-badge team">{{ row.from_code }}</span>
                    <span class="card-meta-badge team">{{ row.to_code }}</span>
                  </div>
                </div>
              </div>

              <div class="badge-row">
                <span class="status-badge {{ row.position_class }}">{{ row.position_badge }}</span>
                <span class="status-badge {{ row.transaction_class }}">{{ row.transaction_label }}</span>
              </div>

              <div class="metric-grid">
                <div class="metric">
                  <div class="metric-label">Move Date</div>
                  <div class="metric-value">{{ row.date }}</div>
                </div>
                <div class="metric">
                  <div class="metric-label">Profile</div>
                  <div class="metric-value">{{ row.meta_line }}</div>
                </div>
                <div class="metric">
                  <div class="metric-label">Debut Status</div>
                  <div class="metric-value">{{ row.debut_label }}</div>
                </div>
              </div>

              <div class="why">{{ row.why }}</div>
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
            <div class="guide-def">This window prioritizes near-term promotion pressure and immediate movement signals.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">14 DAY</div>
            <div class="guide-def">This window gives the broader scout view: recent signal strength plus transaction-layer movement and arrivals.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">Edge Score</div>
            <div class="guide-def">A ranking score used to surface the strongest current signal candidates inside each board.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">Signal Layer</div>
            <div class="guide-def">Players ranked by recent underlying signal quality rather than simple public-facing box-score reputation.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">Movement Layer</div>
            <div class="guide-def">Recent arrivals, recalls, debuts, and related transitions that matter for prospect timing and opportunity.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">How to Read Team Labels</div>
            <div class="guide-def">This board should emphasize the minor-league affiliate context first, with the MLB parent organization used as secondary context.</div>
          </div>

          <div class="guide-item">
            <div class="guide-term">Why This Page Exists</div>
            <div class="guide-def">Promotion Watch is designed to identify actionable player movement before the broader market fully adjusts.</div>
          </div>
        </div>
      </aside>

    {{ footer_html | safe }}
  </div>

  <script src="/player-search.js"></script>
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
      document.querySelectorAll("#tab-72h, #tab-14d").forEach((panel) => {
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

      if (panelId === "tab-14d") {
        if (summaryWindow) summaryWindow.textContent = "14 DAY";
        if (summaryMode) summaryMode.textContent = "SCOUT";
        if (summarySignals) summarySignals.textContent = "{{ total_14_signals }}";
      } else {
        if (summaryWindow) summaryWindow.textContent = "72 HR";
        if (summaryMode) summaryMode.textContent = "AAA";
        if (summarySignals) summarySignals.textContent = "{{ total_signals }}";
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
</body>
</html>
"""
)


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
        .limit(limit_weeks)
        .execute()
    )

    rows = latest_resp.data or []
    week_starts = [row["week_start"] for row in rows if row.get("week_start")]
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


def load_source_frame() -> tuple[pd.DataFrame, list[str]]:
    return fetch_recent_aaa_weekly_signal_base()


def render_html() -> str:
    df, week_starts = load_source_frame()

    if df.empty or not week_starts:
        hitters_72 = pd.DataFrame()
        pitchers_72 = pd.DataFrame()
        hitters_14 = pd.DataFrame()
        pitchers_14 = pd.DataFrame()
    else:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
        latest_week = pd.to_datetime(week_starts[0], errors="coerce")
        latest_df = df[df["week_start"] == latest_week].copy()

        two_week_cut = sorted(df["week_start"].dropna().unique(), reverse=True)[:2]
        two_week_df = df[df["week_start"].isin(two_week_cut)].copy()

        hitter_trends_all = build_trend_lookup(df[df["pa"].notna()].copy(), "iso")
        pitcher_trends_all = build_trend_lookup(df[df["bf"].notna()].copy(), "kbb_p")

        hitters_72 = build_aaa_hitter_promotion_watch(latest_df, trend_lookup=hitter_trends_all).head(6) if not latest_df.empty else pd.DataFrame()
        pitchers_72 = build_aaa_pitcher_promotion_watch(latest_df, trend_lookup=pitcher_trends_all).head(6) if not latest_df.empty else pd.DataFrame()

        hitters_14 = build_aaa_hitter_promotion_watch(two_week_df, trend_lookup=hitter_trends_all).head(6) if not two_week_df.empty else pd.DataFrame()
        pitchers_14 = build_aaa_pitcher_promotion_watch(two_week_df, trend_lookup=pitcher_trends_all).head(6) if not two_week_df.empty else pd.DataFrame()

    total_signals = len(hitters_72) + len(pitchers_72)
    total_14_signals = len(hitters_14) + len(pitchers_14)
    live_arrivals, archive_arrivals = load_arrivals_windows(live_limit=8, archive_limit=16)

    return HTML_TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        timezone_label=TIMEZONE_LABEL,
        nav_html=Template(NAV_TEMPLATE).render(active_nav="promotion_watch"),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
        total_signals=total_signals,
        total_14_signals=total_14_signals,
        hitters=hitters_72.to_dict(orient="records"),
        pitchers=pitchers_72.to_dict(orient="records"),
        hitters_14=hitters_14.to_dict(orient="records"),
        pitchers_14=pitchers_14.to_dict(orient="records"),
        live_arrivals=live_arrivals,
        archive_arrivals=archive_arrivals,
    )


def main() -> None:
    CALL_UP_DIR.mkdir(parents=True, exist_ok=True)
    html = render_html()
    output_path = CALL_UP_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()