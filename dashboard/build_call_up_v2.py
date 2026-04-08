from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Template
import json

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

        if not signals:
            return "Emerging performance signal under evaluation"

        return " + ".join(signals)

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
    hitters["trend_points"] = "0,24 20,22 40,20 60,18 80,16 100,14 120,12"
    hitters["trend_glow"] = hitters["edge_score"] >= 65

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
    pitchers["source_badge"] = SOURCE_BADGE
    pitchers["score_version"] = SCORE_VERSION
    pitchers["trend_points"] = "0,24 20,22 40,19 60,17 80,15 100,13 120,11"
    pitchers["trend_glow"] = pitchers["edge_score"] >= 65

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
      --card-radial: radial-gradient(circle at top left, #1a1a1a 0%, #080808 100%);
      --border: #2d2d2d;
      --text: #f0f0f0;
      --muted: #71717a;
      --soft: #a1a1aa;
      --tiny: #7c7c84;
      --blue: #6aa6ff;
      --lime-hot: #b6ff00;
      --red: #ef4444;
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

    .tabs {
      display: inline-flex;
      gap: 8px;
      margin-bottom: 18px;
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
    }

    .tab.active {
      color: var(--text);
      border-color: rgba(182,255,0,0.20);
      box-shadow: 0 0 8px rgba(182,255,0,0.08);
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

    .player-card.high-edge {
      box-shadow: 0 0 0 1px rgba(182,255,0,0.10), 0 0 18px rgba(182,255,0,0.06);
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
      background: rgba(255,255,255,0.03);
      color: var(--soft);
    }

    .scorebox {
      text-align: right;
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

    .edge-up {
      color: var(--lime-hot);
    }

    .sparkline-wrap {
      margin-top: 14px;
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
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
      filter: drop-shadow(0 0 4px rgba(182,255,0,0.25));
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
    }

    .status-badge.positive {
      color: var(--lime-hot);
      border-color: rgba(182,255,0,0.20);
      background: rgba(182,255,0,0.05);
    }

    .status-badge.neutral {
      color: var(--soft);
      border-color: rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.02);
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
      line-height: 1;
    }

    .why {
      margin-top: 14px;
      font-size: 10px;
      line-height: 1.45;
      color: var(--tiny);
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
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

      .hero-card,
      .summary-card,
      .section-card {
        border-radius: 16px;
      }

      .hero-card {
        padding: 18px;
      }

      .signal-grid {
        grid-template-columns: 1fr;
      }

      .metric-grid {
        grid-template-columns: 1fr 1fr 1fr;
      }

      .player-name {
        font-size: 17px;
      }

      .score-value {
        font-size: 24px;
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
          Clean rebuild with real AAA signal data, shared shell architecture, and locked 72 HR / 14 DAY control windows.
        </p>
      </div>

         <div class="summary-card">
        <div>
          <div class="summary-label">Window</div>
          <div class="summary-value" id="summary-window">72 HR</div>
        </div>
        <div>
          <div class="summary-label">Mode</div>
          <div class="summary-value" id="summary-mode">AAA</div>
        </div>
        <div>
          <div class="summary-label">Signals</div>
          <div class="summary-value" id="summary-signals">{{ total_signals }}</div>
        </div>
      </div>   
    </section>

   <section class="section-card">
      <div class="tabs" role="tablist" aria-label="Promotion watch windows">
        <button type="button" class="tab active" onclick="switchPromotionTab('tab-72h', this)">72 HR</button>
        <button type="button" class="tab" onclick="switchPromotionTab('tab-14d', this)">14 DAY</button>
      </div>

            {% if total_signals == 0 %}
      <div class="placeholder">
        No live AAA promotion-watch signals available yet.
      </div>
      {% else %}
      <div id="tab-72h">
      <section class="signal-grid">
        <div class="section">
          <div class="section-head">
            <div>
              <div class="section-kicker">Signal Layer</div>
              <h2 class="section-title">Pitching Prospect Signals</h2>
            </div>
            <div class="section-badge">Top {{ pitchers|length }}</div>
          </div>

          <div class="cards">
            {% for row in pitchers %}
            <article class="player-card {% if row.edge_score >= 65 %}high-edge{% endif %}">
              <div class="player-top">
                <div class="avatar">{{ row.player_name[:2]|upper }}</div>
                <div class="player-ident">
                  <div class="rankline">#{{ loop.index }} Pitcher Trigger</div>
                  <h3 class="player-name">{{ row.player_name }}</h3>
                  <div class="signal-line">Pitcher // Live Edge Signal // {{ row.sample_note }}</div>
                  <div class="card-meta-row">
                    <span class="card-meta-badge source">{{ row.source_badge }}</span>
                    <span class="card-meta-badge model">{{ row.score_version }}</span>
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
              <div class="section-badge">Top {{ hitters|length }}</div>
            </div>

            <div class="cards">
              {% for row in hitters %}
              <article class="player-card {% if row.edge_score >= 65 %}high-edge{% endif %}">
                <div class="player-top">
                  <div class="avatar">{{ row.player_name[:2]|upper }}</div>
                  <div class="player-ident">
                    <div class="rankline">#{{ loop.index }} Hitter Trigger</div>
                    <h3 class="player-name">{{ row.player_name }}</h3>
                    <div class="signal-line">{{ row.org }} // Hitter // {{ row.sample_note }}</div>
                    <div class="card-meta-row">
                      <span class="card-meta-badge source">{{ row.source_badge }}</span>
                      <span class="card-meta-badge model">{{ row.score_version }}</span>
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
        </div>

           <div id="tab-14d" style="display:none;">
          <div class="section">
            <div class="section-head">
              <div>
                <div class="section-kicker">Movement Layer</div>
                <h2 class="section-title">Archive Arrivals — Last 14 Days</h2>
              </div>
              <div class="section-badge">{{ archive_arrivals|length }} Players</div>
            </div>

            {% if archive_arrivals %}
            <div class="cards">
              {% for row in archive_arrivals %}
              <article class="player-card">
                <div class="player-top">
                  <div class="avatar">{{ row.player_name[:2]|upper }}</div>
                  <div class="player-ident">
                    <div class="rankline">{{ row.transaction_label }}</div>
                    <h3 class="player-name">{{ row.player_name }}</h3>
                    <div class="signal-line">{{ row.from_code }} → {{ row.to_code }} // {{ row.date }}</div>
                    <div class="card-meta-row">
                      <span class="card-meta-badge source">{{ row.position_badge }}</span>
                      <span class="card-meta-badge model">{{ row.transaction_label }}</span>
                    </div>
                  </div>
                </div>

                <div class="badge-row">
                  <span class="status-badge {{ row.position_class }}">{{ row.position_badge }}</span>
                  <span class="status-badge neutral">{{ row.transaction_label }}</span>
                </div>

                <div class="metric-grid">
                  <div class="metric">
                    <div class="metric-label">Date</div>
                    <div class="metric-value">{{ row.date }}</div>
                  </div>
                  <div class="metric">
                    <div class="metric-label">Age / Draft</div>
                    <div class="metric-value">{{ row.meta_line }}</div>
                  </div>
                  <div class="metric">
                    <div class="metric-label">Debut</div>
                    <div class="metric-value">{{ row.debut_label }}</div>
                  </div>
                </div>

                <div class="why">{{ row.why }}</div>
              </article>
              {% endfor %}
            </div>
            {% else %}
            <div class="placeholder">
              No prospect-relevant MLB arrivals in the last 14 days.
            </div>
            {% endif %}
          </div>
        </div>     <
        {% endif %}
      </section>   

    {{ footer_html | safe }}
  </div>

         <script src="/player-search.js"></script>
    <script>
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
          if (summaryMode) summaryMode.textContent = "ARCHIVE";
          if (summarySignals) summarySignals.textContent = "{{ archive_arrivals|length }}";
        } else {
          if (summaryWindow) summaryWindow.textContent = "72 HR";
          if (summaryMode) summaryMode.textContent = "AAA";
          if (summarySignals) summarySignals.textContent = "{{ total_signals }}";
        }
      }  

      document.addEventListener("DOMContentLoaded", function () {
        const defaultTab = document.getElementById("tab-72h");
        if (defaultTab) defaultTab.style.display = "block";
      });
    </script>
</body>
</html>
"""
)


def fetch_recent_aaa_weekly_signal_base(limit_weeks: int = 2) -> tuple[pd.DataFrame, str, str | None]:
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
    if not rows:
        return pd.DataFrame(), "", None

    week_starts = [row["week_start"] for row in rows if row.get("week_start")]
    if not week_starts:
        return pd.DataFrame(), "", None

    latest_week = week_starts[0]
    prior_week = week_starts[1] if len(week_starts) > 1 else None

    data_resp = (
        sb.table("milb_aaa_weekly_signal_base")
        .select("*")
        .in_("week_start", week_starts)
        .execute()
    )

    data = data_resp.data or []
    if not data:
        return pd.DataFrame(), latest_week, prior_week

    return pd.DataFrame(data), latest_week, prior_week


def load_source_frame() -> tuple[pd.DataFrame, str, str | None]:
    return fetch_recent_aaa_weekly_signal_base()

def render_html() -> str:
    df, latest_week, prior_week = load_source_frame()

    if df.empty or not latest_week:
        hitters_72 = pd.DataFrame()
        pitchers_72 = pd.DataFrame()
        hitters_14 = pd.DataFrame()
        pitchers_14 = pd.DataFrame()
    else:
        latest_df = df[df["week_start"] == latest_week].copy()
        prior_df = df[df["week_start"] == prior_week].copy() if prior_week else pd.DataFrame()

        hitters_72 = build_aaa_hitter_promotion_watch(latest_df).head(6) if not latest_df.empty else pd.DataFrame()
        pitchers_72 = build_aaa_pitcher_promotion_watch(latest_df).head(6) if not latest_df.empty else pd.DataFrame()

        if not prior_df.empty:
            two_week_df = pd.concat([latest_df, prior_df], ignore_index=True)
        else:
            two_week_df = latest_df.copy()

        hitters_14 = build_aaa_hitter_promotion_watch(two_week_df).head(6) if not two_week_df.empty else pd.DataFrame()
        pitchers_14 = build_aaa_pitcher_promotion_watch(two_week_df).head(6) if not two_week_df.empty else pd.DataFrame()
      
    total_signals = len(hitters_72) + len(pitchers_72)
    live_arrivals, archive_arrivals = load_arrivals_windows(live_limit=8, archive_limit=16)

    return HTML_TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        timezone_label=TIMEZONE_LABEL,
        nav_html=Template(NAV_TEMPLATE).render(active_nav="promotion_watch"),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
        total_signals=total_signals,
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