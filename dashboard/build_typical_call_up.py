#!/usr/bin/env python3
import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from jinja2 import Template

DIST_DIR = Path("dist")
DIST_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = Path(__file__).parent / "templates"
NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "65"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
SITE_URL = os.getenv("SITE_URL", "").strip()
TIMEZONE_LABEL = os.getenv("TIMEZONE_LABEL", "America/New_York")
SOURCE_BADGE = "SRC: AAA_PIPELINE_v1"
SCORE_VERSION = "EDGE_v2.0"


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
        body += f"\n[Open Page]({markdown_escape(SITE_URL)})"
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


def fetch_latest_aaa_weekly_signal_base() -> pd.DataFrame:
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    sb = create_client(url, key)

    latest_resp = (
        sb.table("milb_aaa_weekly_signal_base")
        .select("week_start")
        .order("week_start", desc=True)
        .limit(1)
        .execute()
    )

    rows = latest_resp.data or []
    if not rows:
        raise RuntimeError("milb_aaa_weekly_signal_base returned no rows.")

    latest_week = rows[0]["week_start"]

    data_resp = (
        sb.table("milb_aaa_weekly_signal_base")
        .select("*")
        .eq("week_start", latest_week)
        .execute()
    )

    data = data_resp.data or []
    if not data:
        raise RuntimeError(f"No AAA weekly signal base rows found for {latest_week}.")

    df = pd.DataFrame(data)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def fetch_latest_prospect_intelligence() -> tuple[pd.DataFrame, pd.DataFrame]:
    from supabase import create_client

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

    sb = create_client(url, key)

    latest_resp = (
        sb.table("prospect_intelligence_daily")
        .select("snapshot_date")
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )

    rows = latest_resp.data or []
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    latest_snapshot = rows[0]["snapshot_date"]

    data_resp = (
        sb.table("prospect_intelligence_daily")
        .select("*")
        .eq("snapshot_date", latest_snapshot)
        .execute()
    )

    data = data_resp.data or []
    if not data:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(data)
    hitters = df[df["signal_type"] == "Hitter"].copy()
    pitchers = df[df["signal_type"] == "Pitcher"].copy()
    return hitters, pitchers

def normalize_prospect_intelligence_for_cards(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "player_name" in out.columns:
        out["player_name"] = out["player_name"].apply(safe_name)

    out["edge_score"] = pd.to_numeric(out.get("edge_score"), errors="coerce").fillna(0).round(1)

    if "trend_glow" not in out.columns:
        out["trend_glow"] = False
    else:
        out["trend_glow"] = out["trend_glow"].fillna(False).astype(bool)

    if "signal_type" not in out.columns:
        out["signal_type"] = ""
    else:
        out["signal_type"] = out["signal_type"].fillna("")

    if "sample_note" not in out.columns:
        out["sample_note"] = ""
    else:
        out["sample_note"] = out["sample_note"].fillna("")

    if "source_badge" not in out.columns:
        out["source_badge"] = SOURCE_BADGE
    else:
        out["source_badge"] = out["source_badge"].fillna(SOURCE_BADGE)

    if "score_version" not in out.columns:
        out["score_version"] = SCORE_VERSION
    else:
        out["score_version"] = out["score_version"].fillna(SCORE_VERSION)

    if "signal_archetype" not in out.columns:
        out["signal_archetype"] = "Promotion Watch"
    else:
        out["signal_archetype"] = out["signal_archetype"].fillna("Promotion Watch")

    if "data_freshness_hours" not in out.columns:
        out["data_freshness_hours"] = None
    if "latency_warning" not in out.columns:
        out["latency_warning"] = False
    else:
        out["latency_warning"] = out["latency_warning"].fillna(False).astype(bool)

    if "scout_narrative" in out.columns:
        out["why"] = out["scout_narrative"].fillna(out.get("why", ""))
    elif "why" not in out.columns:
        out["why"] = ""

    if "pa" in out.columns:
        out.loc[out["signal_type"] == "Hitter", "sample_note"] = out["pa"].map(
            lambda x: f"{int(x)} PA" if pd.notna(x) else "Sample N/A"
        )

    if "bf" in out.columns:
        out.loc[out["signal_type"] == "Pitcher", "sample_note"] = out["bf"].map(
            lambda x: f"{int(x)} BF" if pd.notna(x) else "Sample N/A"
        )

    hitter_mask = out["signal_type"].eq("Hitter")
    pitcher_mask = out["signal_type"].eq("Pitcher")

    out["metric_1_label"] = ""
    out["metric_1"] = "--"
    out["metric_2_label"] = ""
    out["metric_2"] = "--"
    out["metric_3_label"] = ""
    out["metric_3"] = "--"

    if hitter_mask.any():
        out.loc[hitter_mask, "metric_1_label"] = "ISO"
        if "last_7d_iso" in out.columns:
            out.loc[hitter_mask, "metric_1"] = out.loc[hitter_mask, "last_7d_iso"].map(
                lambda x: f"{float(x):.3f}" if pd.notna(x) else "--"
            )

        out.loc[hitter_mask, "metric_2_label"] = "K/BB"
        if "k_bb_ratio" in out.columns:
            out.loc[hitter_mask, "metric_2"] = out.loc[hitter_mask, "k_bb_ratio"].map(
                lambda x: f"{float(x):.2f}" if pd.notna(x) else "--"
            )

        out.loc[hitter_mask, "metric_3_label"] = "BB%"
        if "bb_rate" in out.columns:
            out.loc[hitter_mask, "metric_3"] = out.loc[hitter_mask, "bb_rate"].map(
                lambda x: f"{float(x) * 100:.1f}%" if pd.notna(x) else "--"
            )

    if pitcher_mask.any():
        out.loc[pitcher_mask, "metric_1_label"] = "K/BB"
        if "k_bb_ratio" in out.columns:
            out.loc[pitcher_mask, "metric_1"] = out.loc[pitcher_mask, "k_bb_ratio"].map(
                lambda x: f"{float(x):.2f}" if pd.notna(x) else "--"
            )

        out.loc[pitcher_mask, "metric_2_label"] = "K%"
        if "k_rate_proxy" in out.columns:
            out.loc[pitcher_mask, "metric_2"] = out.loc[pitcher_mask, "k_rate_proxy"].map(
                lambda x: f"{float(x) * 100:.1f}%" if pd.notna(x) else "--"
            )

        out.loc[pitcher_mask, "metric_3_label"] = "BB%"
        if "bb_rate_proxy" in out.columns:
            out.loc[pitcher_mask, "metric_3"] = out.loc[pitcher_mask, "bb_rate_proxy"].map(
                lambda x: f"{float(x) * 100:.1f}%" if pd.notna(x) else "--"
            )

    if "badges" not in out.columns:
        out["badges"] = out["signal_archetype"].map(lambda x: [x] if pd.notna(x) else ["Promotion Watch"])

    if "badge_classes" not in out.columns:
        out["badge_classes"] = out["badges"].map(
            lambda badges: ["positive" if b != "Promotion Watch" else "neutral" for b in badges]
        )

    default_hitter_trend = "0,24 20,22 40,20 60,18 80,16 100,14 120,12"
    default_pitcher_trend = "0,24 20,22 40,19 60,17 80,15 100,13 120,11"

    out.loc[hitter_mask & out["trend_points"].eq(""), "trend_points"] = default_hitter_trend
    out.loc[pitcher_mask & out["trend_points"].eq(""), "trend_points"] = default_pitcher_trend

    return out.sort_values("edge_score", ascending=False).reset_index(drop=True)

def build_aaa_hitter_promotion_watch(df: pd.DataFrame) -> pd.DataFrame:
    hitters = df[df["pa"].notna()].copy()
    if hitters.empty:
        return pd.DataFrame()

    for col in ["pa", "bb", "so", "hr", "iso", "kbb_h"]:
        hitters[col] = pd.to_numeric(hitters[col], errors="coerce")

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
    hitters["player_name"] = hitters["player_name"].apply(safe_name)
    hitters["signal_type"] = "Hitter"

    hitters["why"] = hitters.apply(
        lambda r: (
            f"ISO {r['iso']:.3f}, HR {int(r['hr'] or 0)}, "
            f"BB {int(r['bb'] or 0)} vs K {int(r['so'] or 0)} over {int(r['pa'] or 0)} PA."
        ),
        axis=1,
    )

    hitters["metric_1_label"] = "ISO"
    hitters["metric_1"] = hitters["iso"].fillna(0).map(lambda x: f"{x:.3f}")
    hitters["metric_2_label"] = "K/BB"
    hitters["metric_2"] = hitters["kbb_h"].fillna(0).map(lambda x: f"{x:.2f}")
    hitters["metric_3_label"] = "HR"
    hitters["metric_3"] = hitters["hr"].fillna(0).map(lambda x: f"{int(x)}")
    hitters["sample_note"] = hitters["pa"].fillna(0).map(lambda x: f"{int(x)} PA")
    hitters["trend_points"] = "0,24 20,22 40,20 60,18 80,16 100,14 120,12"
    hitters["trend_glow"] = hitters["edge_score"] >= 65
    hitters["source_badge"] = SOURCE_BADGE
    hitters["score_version"] = SCORE_VERSION

    def hitter_badges(row: pd.Series) -> list[str]:
        badges = []
        if pd.notna(row["iso"]) and row["iso"] >= 0.250:
            badges.append("Impact Bat")
        if pd.notna(row["kbb_h"]) and row["kbb_h"] <= 1.50:
            badges.append("Zone Control")
        if pd.notna(row["hr"]) and row["hr"] >= 2:
            badges.append("HR Surge")
        if not badges:
            badges.append("Promotion Watch")
        if "Promotion Watch" not in badges:
            badges.append("Promotion Watch")
        return badges[:3]

    def hitter_badge_classes(row: pd.Series) -> list[str]:
        classes = []
        for badge in row["badges"]:
            if badge in ["Impact Bat", "Zone Control", "HR Surge"]:
                classes.append("positive")
            else:
                classes.append("neutral")
        return classes

    hitters["badges"] = hitters.apply(hitter_badges, axis=1)
    hitters["badge_classes"] = hitters.apply(hitter_badge_classes, axis=1)

    return hitters.sort_values(["edge_score", "pa"], ascending=[False, False]).reset_index(drop=True)


def build_aaa_pitcher_promotion_watch(df: pd.DataFrame) -> pd.DataFrame:
    pitchers = df[df["bf"].notna()].copy()
    if pitchers.empty:
        return pd.DataFrame()

    for col in ["bf", "so_p", "bb_allowed", "kbb_p"]:
        pitchers[col] = pd.to_numeric(pitchers[col], errors="coerce")

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
    pitchers["player_name"] = pitchers["player_name"].apply(safe_name)
    pitchers["signal_type"] = "Pitcher"

    pitchers["why"] = pitchers.apply(
        lambda r: (
            f"K/BB {r['kbb_p']:.2f}, {int(r['so_p'] or 0)} K, "
            f"{int(r['bb_allowed'] or 0)} BB over {int(r['bf'] or 0)} BF."
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
    pitchers["trend_points"] = "0,24 20,22 40,19 60,17 80,15 100,13 120,11"
    pitchers["trend_glow"] = pitchers["edge_score"] >= 65
    pitchers["source_badge"] = SOURCE_BADGE
    pitchers["score_version"] = SCORE_VERSION

    def pitcher_badges(row: pd.Series) -> list[str]:
        badges = []
        if pd.notna(row["kbb_p"]) and row["kbb_p"] >= 4:
            badges.append("Bat-Miss Ready")
        if pd.notna(row["bb_allowed"]) and row["bb_allowed"] <= 2:
            badges.append("Command Hold")
        if pd.notna(row["so_p"]) and row["so_p"] >= 10:
            badges.append("Whiff Volume")
        if not badges:
            badges.append("Promotion Watch")
        if "Promotion Watch" not in badges:
            badges.append("Promotion Watch")
        return badges[:3]

    def pitcher_badge_classes(row: pd.Series) -> list[str]:
        classes = []
        for badge in row["badges"]:
            if badge in ["Bat-Miss Ready", "Command Hold", "Whiff Volume"]:
                classes.append("positive")
            else:
                classes.append("neutral")
        return classes

    pitchers["badges"] = pitchers.apply(pitcher_badges, axis=1)
    pitchers["badge_classes"] = pitchers.apply(pitcher_badge_classes, axis=1)

    return pitchers.sort_values(["edge_score", "bf"], ascending=[False, False]).reset_index(drop=True)


def fetch_mlb_debut_map(names: list[str]) -> dict[str, dict]:
    debut_map: dict[str, dict] = {}

    for name in names:
        try:
            resp = requests.get(
                "https://statsapi.mlb.com/api/v1/people/search",
                params={"names": name},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            people = data.get("people", []) or []

            if not people:
                debut_map[name] = {
                    "is_mlb": False,
                    "mlb_debut_date": None,
                    "mlb_id": None,
                }
                continue

            person = people[0]
            debut_date = person.get("mlbDebutDate")

            debut_map[name] = {
                "is_mlb": bool(debut_date),
                "mlb_debut_date": debut_date,
                "mlb_id": person.get("id"),
            }
        except Exception:
            debut_map[name] = {
                "is_mlb": False,
                "mlb_debut_date": None,
                "mlb_id": None,
            }

    return debut_map


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

    if " RHP " in f" {desc} ":
        return "RHP"
    if " LHP " in f" {desc} ":
        return "LHP"
    if " C " in f" {desc} ":
        return "C"
    if " SS " in f" {desc} ":
        return "SS"
    if " 2B " in f" {desc} ":
        return "2B"
    if " 3B " in f" {desc} ":
        return "3B"
    if " 1B " in f" {desc} ":
        return "1B"
    if " CF " in f" {desc} ":
        return "CF"
    if " LF " in f" {desc} ":
        return "LF"
    if " RF " in f" {desc} ":
        return "RF"
    if " OF " in f" {desc} ":
        return "OF"
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

    if type_desc in {"optioned", "outrighted"}:
        return "RETURN"

    if type_desc == "assigned":
        return "ASSIGN"

    return "MOVE"


AAA_TO_MLB_CODES = {
    "Memphis Redbirds": ("MEM", "STL"),
    "St. Paul Saints": ("STP", "MIN"),
    "Lehigh Valley IronPigs": ("LHV", "PHI"),
    "Buffalo Bisons": ("BUF", "TOR"),
    "Syracuse Mets": ("SYR", "NYM"),
    "Tacoma Rainiers": ("TAC", "SEA"),
    "Louisville Bats": ("LOU", "CIN"),
    "Sugar Land Space Cowboys": ("SL", "HOU"),
    "Reno Aces": ("REN", "ARI"),
    "Albuquerque Isotopes": ("ABQ", "COL"),
    "Norfolk Tides": ("NOR", "BAL"),
    "Jacksonville Jumbo Shrimp": ("JAX", "MIA"),
    "Nashville Sounds": ("NAS", "MIL"),
    "Salt Lake Bees": ("SLB", "LAA"),
    "Charlotte Knights": ("CHA", "CWS"),
    "El Paso Chihuahuas": ("ELP", "SD"),
    "Columbus Clippers": ("CLP", "CLE"),
    "Rochester Red Wings": ("ROC", "WSH"),
}


def infer_team_codes(move: dict) -> tuple[str, str]:
    from_team = str(move.get("fromTeam") or "")
    to_team = str(move.get("toTeam") or "")

    if from_team in AAA_TO_MLB_CODES:
        return AAA_TO_MLB_CODES[from_team]

    from_code = from_team[:3].upper() if from_team else "AAA"
    to_code = to_team[:3].upper() if to_team else "MLB"
    return from_code, to_code

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
            player = move.get("person") or "Unknown"
            age = move.get("currentAge")
            draft_year = move.get("draftYear")
            debut = move.get("mlbDebutDate") or "MLB debut pending"

            meta_bits = []
            if age is not None:
                meta_bits.append(f"Age {age}")
            if draft_year is not None:
                meta_bits.append(f"Draft {draft_year}")

            meta_line = " // ".join(meta_bits) if meta_bits else "Upper-minors movement"
            why = move.get("description") or player
            pos_badge = infer_position_badge(move)
            pos_class = infer_position_class(pos_badge)
            txn_label = infer_transaction_label(move)
            from_code, to_code = infer_team_codes(move)

            formatted.append(
                {
                    "player_name": player,
                    "date": move.get("date") or "—",
                    "debut_label": debut,
                    "meta_line": meta_line,
                    "why": why,
                    "position_badge": pos_badge,
                    "position_class": pos_class,
                    "transaction_label": txn_label,
                    "from_code": from_code,
                    "to_code": to_code,
                }
            )
        return formatted

    return format_arrivals(live_arrivals, live_limit), format_arrivals(archive_arrivals, archive_limit)


HTML_TEMPLATE = Template("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals — Prospect Tracker</title>
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
      height: 34px;
      border-radius: 999px;
      border: 1px solid rgba(182,255,0,0.22);
      background: rgba(255,255,255,0.05);
      color: var(--text);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0 12px;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      cursor: pointer;
      white-space: nowrap;
    }

    .livebox {
      text-align: right;
      flex: 0 0 auto;
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

    .glossary-drawer.open { transform: translateX(0); }

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

    .glossary-item { margin-bottom: 12px; }
    .glossary-item:last-child { margin-bottom: 0; }

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

    .app { padding: 18px 0 34px; }

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

    .hero-card { padding: 18px; }

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

    .meta-card { padding: 14px; }

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

    .meta-label { margin-bottom: 6px; }

    .meta-value {
      font-family: var(--mono);
      font-size: 13px;
      color: var(--text);
      word-break: break-word;
      font-variant-numeric: tabular-nums;
    }

    .slate-heat-card { padding: 14px; }

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

    .arrivals-section {
      margin-bottom: 16px;
    }
    .arrivals-tabs {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .arrivals-tab {
      appearance: none;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.02);
      color: var(--soft);
      border-radius: 999px;
      padding: 7px 11px;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      cursor: pointer;
      transition: all 0.18s ease;
    }

    .arrivals-tab.active {
      color: var(--text);
      border-color: rgba(106,166,255,0.22);
      background: rgba(106,166,255,0.08);
      box-shadow: 0 0 8px rgba(106,166,255,0.10);
    }

    .arrivals-panel {
      display: none;
    }

    .arrivals-panel.active {
      display: block;
    }

    .arrivals-empty {
      padding: 18px 16px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--tiny);
      border-top: 1px solid rgba(255,255,255,0.05);
    }
    .signal-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }

    .arrivals-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 10px;
    }

    .arrival-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }

    .arrival-name-row {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 4px;
    }

    .pos-badge,
    .txn-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 3px 7px;
      border-radius: 999px;
      font-family: var(--mono);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.03);
      color: var(--soft);
    }

    .pos-badge.pitcher {
      color: #67e8f9;
      border-color: rgba(103,232,249,0.28);
      background: rgba(103,232,249,0.06);
    }

    .pos-badge.infielder {
      color: #fbbf24;
      border-color: rgba(251,191,36,0.28);
      background: rgba(251,191,36,0.06);
    }

    .pos-badge.outfielder {
      color: #34d399;
      border-color: rgba(52,211,153,0.28);
      background: rgba(52,211,153,0.06);
    }

    .txn-badge {
      color: var(--text);
      border-color: rgba(255,255,255,0.12);
    }

    .arrival-dest {
      color: #39FF14;
      text-shadow: 0 0 6px rgba(57,255,20,0.18);
      font-weight: 800;
    }

    .arrival-card {
      border: 1px solid rgba(255,255,255,0.05);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.02);
    }

    .arrival-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }

    .arrival-name {
      margin: 0 0 4px;
      font-size: 17px;
      line-height: 1.05;
      letter-spacing: -0.02em;
      font-weight: 800;
    }

    .arrival-meta,
    .arrival-route,
    .arrival-note {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--soft);
      font-variant-numeric: tabular-nums;
    }

    .arrival-route {
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .arrival-note {
      color: var(--tiny);
      line-height: 1.45;
    }

    .board {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }

    .section { overflow: hidden; }

    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 16px 16px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      background: linear-gradient(180deg, rgba(255,255,255,0.022), rgba(255,255,255,0.008));
    }

    .section-kicker { margin-bottom: 5px; }

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
      font-variant-numeric: tabular-nums;
    }

    .cards {
      display: grid;
      gap: 10px;
      padding: 10px;
    }

    .player-card { padding: 14px; }

    .player-card.high-edge {
      border-color: rgba(74,222,128,0.22);
      box-shadow: var(--shadow), 0 0 8px rgba(74,222,128,0.07);
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

    .player-ident { min-width: 0; }
    .rankline { margin-bottom: 4px; }

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

    .card-meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0 0;
    }

    .card-meta-badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 9px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
      font-family: var(--mono);
      font-size: 10px;
      color: var(--soft);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .card-meta-badge.source {
      border-color: rgba(106,166,255,0.22);
      background: rgba(106,166,255,0.08);
      color: var(--blue);
    }

        .card-meta-badge.model {
      border-color: rgba(182,255,0,0.20);
      background: rgba(182,255,0,0.08);
      color: var(--lime-hot);
    }

    .card-meta-badge.warning {
      border-color: rgba(239,68,68,0.24);
      background: rgba(239,68,68,0.10);
      color: #ef4444;
    }

    .scorebox {

    .scorebox {
      text-align: right;
      min-width: 88px;
    }

    .score-label { margin-bottom: 4px; }

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

    .metric-label { margin-bottom: 6px; }

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

    .status-badge.neutral {
      color: var(--soft);
      border-color: rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.02);
    }

    .why {
      font-size: 10px;
      line-height: 1.45;
      color: var(--tiny);
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
    }

    @media (min-width: 900px) {
      .hero-grid {
        grid-template-columns: 1.35fr 0.9fr;
        align-items: stretch;
      }
    }

{{ shell_styles | safe }}
       

       @media (max-width: 640px) {
      .topbar-inner,
      .app,
      .topnav-inner {
        width: min(100%, calc(100% - 16px));
      }

      .hero-layout {
        grid-template-columns: 1fr;
      }

      .hero-side {
        grid-template-columns: 1fr;
      }

      .signal-grid {
        grid-template-columns: 1fr;
      }
      .arrivals-tabs {
        width: 100%;
      }
      .arrivals-grid {
        grid-template-columns: 1fr;
      }

      .player-name {
        font-size: 17px;
      }

      .score-value {
        font-size: 24px;
      }

      .arrival-top {
        flex-direction: column;
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
        <button class="info-trigger" type="button" onclick="openGlossary()" aria-label="Open glossary">ⓘ Glossary</button>
        <div class="livebox">
          <div class="live-label"><span class="live-dot"></span>LIVE</div>
          <div class="live-time">{{ generated_at }}</div>
        </div>
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
        <div class="glossary-item">
          <span class="glossary-term">Slate Heat</span>
          <div class="glossary-definition">
            A quick index built from the visible board scores. Higher values mean a stronger concentration of promotion-worthy AAA signals.
          </div>
        </div>
        <div class="glossary-item">
          <span class="glossary-term">Edge Score</span>
          <div class="glossary-definition">
            A simple 0–100 ranking built from the available AAA weekly signal fields in Supabase and prospect intelligence rows.
          </div>
        </div>
      </section>

      <section class="glossary-section">
        <h3 class="glossary-section-title">II. Hitting Metrics</h3>
        <div class="glossary-item">
          <span class="glossary-term">ISO</span>
          <div class="glossary-definition">
            Isolated power. Higher values point to stronger extra-base impact.
          </div>
        </div>
        <div class="glossary-item">
          <span class="glossary-term">K/BB</span>
          <div class="glossary-definition">
            Strikeouts divided by walks. Lower is generally cleaner for hitters.
          </div>
        </div>
      </section>

      <section class="glossary-section">
        <h3 class="glossary-section-title">III. Pitching Metrics</h3>
        <div class="glossary-item">
          <span class="glossary-term">K/BB</span>
          <div class="glossary-definition">
            Strikeouts divided by walks allowed. Higher is stronger for pitchers.
          </div>
        </div>
        <div class="glossary-item">
          <span class="glossary-term">BF</span>
          <div class="glossary-definition">
            Batters faced in the weekly sample.
          </div>
        </div>
      </section>
    </div>
  </aside>

  <div class="app">
        <section class="hero">
      <div class="hero-layout">
        <div class="hero-main">
          <div class="eyebrow">Executive Terminal</div>
          <h1 class="hero-title">Prospect Tracker</h1>
          <p class="hero-copy">
            A live AAA-first DiamondSignals board built from Supabase weekly signal data, with a new movement layer for fresh AAA-to-MLB prospect tracking.
          </p>
        </div>

        <div class="hero-side">
          <div class="meta-card">
            <div class="meta-label">Last Updated</div>
            <div class="meta-value">{{ generated_at }}</div>
          </div>

          <div class="meta-card">
            <div class="meta-label">Source</div>
            <div class="meta-value">AAA Weekly Signal Base + Scout Transactions</div>
          </div>

          <div class="meta-card">
            <div class="meta-label">Alert Threshold</div>
            <div class="meta-value">{{ threshold }}</div>
          </div>
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
    </section>

       {% if live_arrivals or archive_arrivals %}
    <section class="section arrivals-section">
      <div class="section-head">
        <div>
          <div class="section-kicker">Movement Layer</div>
          <h2 class="section-title">Recent Arrivals</h2>
        </div>
        <div class="arrivals-tabs" role="tablist" aria-label="Recent arrival windows">
          <button class="arrivals-tab active" type="button" data-tab="live-arrivals" onclick="switchArrivalsTab('live-arrivals', this)">LIVE 72H</button>
          <button class="arrivals-tab" type="button" data-tab="archive-arrivals" onclick="switchArrivalsTab('archive-arrivals', this)">ARCHIVE 14D</button>
        </div>
      </div>

      <div id="live-arrivals" class="arrivals-panel active">
        {% if live_arrivals %}
        <div class="arrivals-grid">
          {% for row in live_arrivals %}
          <article class="arrival-card">
            <div class="arrival-head">
              <div>
                <div class="arrival-name-row">
                  <h3 class="arrival-name">{{ row.player_name }}</h3>
                  <span class="pos-badge {{ row.position_class }}">[{{ row.position_badge }}]</span>
                  <span class="txn-badge">[{{ row.transaction_label }}]</span>
                </div>
                <div class="arrival-meta">{{ row.meta_line }}</div>
              </div>
              <div class="arrival-date">{{ row.date }}</div>
            </div>

            <div class="arrival-route">
              <span>[{{ row.from_code }}]</span>
              <span>➔</span>
              <span class="arrival-dest">[{{ row.to_code }}]</span>
            </div>

            <div class="arrival-meta">{{ row.debut_label }}</div>
            <div class="arrival-note">{{ row.why }}</div>
          </article>
          {% endfor %}
        </div>
        {% else %}
        <div class="arrivals-empty">No prospect-relevant MLB arrivals in the last 72 hours.</div>
        {% endif %}
      </div>

      <div id="archive-arrivals" class="arrivals-panel">
        {% if archive_arrivals %}
        <div class="arrivals-grid">
          {% for row in archive_arrivals %}
          <article class="arrival-card">
            <div class="arrival-head">
              <div>
                <div class="arrival-name-row">
                  <h3 class="arrival-name">{{ row.player_name }}</h3>
                  <span class="pos-badge {{ row.position_class }}">[{{ row.position_badge }}]</span>
                  <span class="txn-badge">[{{ row.transaction_label }}]</span>
                </div>
                <div class="arrival-meta">{{ row.meta_line }}</div>
              </div>
              <div class="arrival-date">{{ row.date }}</div>
            </div>

            <div class="arrival-route">
              <span>[{{ row.from_code }}]</span>
              <span>➔</span>
              <span class="arrival-dest">[{{ row.to_code }}]</span>
            </div>

            <div class="arrival-meta">{{ row.debut_label }}</div>
            <div class="arrival-note">{{ row.why }}</div>
          </article>
          {% endfor %}
        </div>
        {% else %}
        <div class="arrivals-empty">No prospect-relevant MLB arrivals in the last 14 days.</div>
        {% endif %}
      </div>
    </section>
    {% endif %}

    <section class="signal-grid">
      <div class="section">
        <div class="section-head">
          <div>
            <div class="section-kicker">Signal Layer</div>
            <h2 class="section-title">Pitching Prospect Signals</h2>
          </div>
          <div class="section-badge">Top 5</div>
        </div>

        <div class="cards">
          {% for row in pitchers %}
          <article class="player-card {% if row.edge_score >= 65 %} high-edge{% endif %}">
            <div class="player-top">
              <div class="avatar">{{ row.player_name[:2]|upper }}</div>
              <div class="player-ident">
                <div class="rankline">#{{ loop.index }} Pitcher Trigger</div>
                <h3 class="player-name">{{ row.player_name }}</h3>
                <div class="signal-line">Pitcher // Live Edge Signal // {{ row.sample_note }}</div>
                <div class="card-meta-row">
                  <span class="card-meta-badge source">{{ row.source_badge if row.source_badge else 'SRC: AAA_PIPELINE_v1' }}</span>
                  <span class="card-meta-badge model">{{ row.score_version if row.score_version else 'EDGE_v2.0' }}</span>
                  {% if row.latency_warning %}
                  <span class="card-meta-badge warning">LATENCY WARNING</span>
                  {% endif %}
                </div>
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
                <polyline class="sparkline-path {% if row.trend_glow %}glow{% endif %}" stroke="url(#pitcherGradient{{ loop.index }})" points="{{ row.trend_points }}" />
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
            <div class="section-kicker">Signal Layer</div>
            <h2 class="section-title">Hitting Prospect Signals</h2>
          </div>
          <div class="section-badge">Top 5</div>
        </div>

        <div class="cards">
          {% for row in hitters %}
          <article class="player-card {% if row.edge_score >= 65 %} high-edge{% endif %}">
            <div class="player-top">
              <div class="avatar">{{ row.player_name[:2]|upper }}</div>
              <div class="player-ident">
                <div class="rankline">#{{ loop.index }} Hitter Trigger</div>
                <h3 class="player-name">{{ row.player_name }}</h3>
                <div class="signal-line">Hitter // Live Edge Signal // {{ row.sample_note }}</div>
                <div class="card-meta-row">
                  <span class="card-meta-badge source">{{ row.source_badge if row.source_badge else 'SRC: AAA_PIPELINE_v1' }}</span>
                  <span class="card-meta-badge model">{{ row.score_version if row.score_version else 'EDGE_v2.0' }}</span>
                  {% if row.latency_warning %}
                  <span class="card-meta-badge warning">LATENCY WARNING</span>
                  {% endif %}
                </div>
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
                <polyline class="sparkline-path {% if row.trend_glow %}glow{% endif %}" stroke="url(#hitterGradient{{ loop.index }})" points="{{ row.trend_points }}" />
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
    </section>

    {{ footer_html | safe }}
  </div>

  <script src="/player-search.js"></script>
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
    function switchArrivalsTab(panelId, buttonEl) {
      document.querySelectorAll(".arrivals-panel").forEach((panel) => {
        panel.classList.remove("active");
      });

      document.querySelectorAll(".arrivals-tab").forEach((btn) => {
        btn.classList.remove("active");
      });

      const activePanel = document.getElementById(panelId);
      if (activePanel) activePanel.classList.add("active");
      if (buttonEl) buttonEl.classList.add("active");
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

    live_arrivals, archive_arrivals = load_arrivals_windows(live_limit=8, archive_limit=16)

    return HTML_TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        threshold=f"{ALERT_THRESHOLD:.0f}+",
        timezone_label=TIMEZONE_LABEL,
        slate_heat=slate_heat,
        nav_html=Template(NAV_TEMPLATE).render(active_nav="promotion_watch"),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
        live_arrivals=live_arrivals,
        archive_arrivals=archive_arrivals,
        pitchers=pitchers.to_dict(orient="records"),
        hitters=hitters.to_dict(orient="records"),
    )


def main() -> None:
    raw = fetch_latest_aaa_weekly_signal_base()
    hitter_signals = build_aaa_hitter_promotion_watch(raw)
    pitcher_signals = build_aaa_pitcher_promotion_watch(raw)

    pi_hitters, pi_pitchers = fetch_latest_prospect_intelligence()
    if not pi_hitters.empty:
        hitter_signals = normalize_prospect_intelligence_for_cards(pi_hitters)
    if not pi_pitchers.empty:
        pitcher_signals = normalize_prospect_intelligence_for_cards(pi_pitchers)

    if hitter_signals.empty and pitcher_signals.empty:
        raise RuntimeError("No AAA hitter or pitcher promotion-watch signals were produced.")

    top_hitters = hitter_signals.head(20).copy()
    top_pitchers = pitcher_signals.head(20).copy()

    combined_names = pd.concat(
        [top_pitchers["player_name"], top_hitters["player_name"]],
        ignore_index=True
    ).dropna().astype(str).unique().tolist()

    debut_map = fetch_mlb_debut_map(combined_names)

    top_pitchers["is_mlb"] = top_pitchers["player_name"].map(
        lambda n: debut_map.get(n, {}).get("is_mlb", False)
    )
    top_hitters["is_mlb"] = top_hitters["player_name"].map(
        lambda n: debut_map.get(n, {}).get("is_mlb", False)
    )

    top_pitchers = top_pitchers[top_pitchers["is_mlb"] != True].copy()
    top_hitters = top_hitters[top_hitters["is_mlb"] != True].copy()

    top_pitchers = top_pitchers.head(5).copy()
    top_hitters = top_hitters.head(5).copy()

    combined_alerts = pd.concat([top_pitchers, top_hitters], ignore_index=True)
    combined_alerts = combined_alerts.sort_values("edge_score", ascending=False).reset_index(drop=True)

    html = render_html(top_pitchers, top_hitters)
    output_path = DIST_DIR / "typical-call-up" / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "top_pitchers": top_pitchers[
            [
                "player_name",
                "org",
                "edge_score",
                "metric_1_label",
                "metric_1",
                "metric_2_label",
                "metric_2",
                "metric_3_label",
                "metric_3",
                "why",
            ]
        ].to_dict(orient="records"),
        "top_hitters": top_hitters[
            [
                "player_name",
                "org",
                "edge_score",
                "metric_1_label",
                "metric_1",
                "metric_2_label",
                "metric_2",
                "metric_3_label",
                "metric_3",
                "why",
            ]
        ].to_dict(orient="records"),
    }

    (DIST_DIR / "signals.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {DIST_DIR / 'signals.json'}")

    send_telegram_alerts(combined_alerts)


if __name__ == "__main__":
    main()