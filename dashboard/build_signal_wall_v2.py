#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path
from jinja2 import Template

DIST_DIR = Path("dist")
SIGNALS_JSON = DIST_DIR / "signals.json"
OUT_DIR = DIST_DIR / "live-v2"
OUT_PATH = OUT_DIR / "index.html"

NAV_TEMPLATE_PATH = Path("dashboard/templates/components/nav.html")
SEARCH_TEMPLATE_PATH = Path("dashboard/templates/components/player_search.html")
SHELL_STYLES_PATH = Path("dashboard/templates/shell_styles.css")

def load_text(path: Path, fallback: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else fallback

def avatar(name: str) -> str:
    parts = str(name or "").replace(",", "").split()
    if not parts:
        return "DS"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def safe(v, fallback="—"):
    return fallback if v is None or v == "" else v

def score_class(score):
    try:
        s = float(score)
    except Exception:
        return "score-cool"
    if s >= 70:
        return "score-hot"
    if s >= 60:
        return "score-warm"
    return "score-cool"

def normalize_rows(rows, player_type):
    out = []
    for idx, r in enumerate(rows or [], start=1):
        ctx = r.get("season_context") or {}
        badges = r.get("badges") or []
        name = r.get("player_name") or r.get("name") or "Unknown Player"
        edge_score = r.get("edge_score")

        if player_type == "pitcher":
            diagnosis = [
                badges[0] if len(badges) > 0 else "COMMAND TREND",
                badges[1] if len(badges) > 1 else "LIVE ARM SIGNAL",
                "WATCHLIST",
            ]
            context_tiles = [
                ("K%", ctx.get("k_pct") or "—"),
                ("BB%", ctx.get("bb_pct") or "—"),
                ("K/BB", ctx.get("k_bb_ratio") or "—"),
                ("BABIP", ctx.get("babip") or "N/A"),
            ]
            board_label = "PITCHER"
            spark_label = "Ballistics vs Surface"
            gradient = "#6aa6ff"
        else:
            diagnosis = [
                badges[0] if len(badges) > 0 else "BLAST TREND",
                badges[1] if len(badges) > 1 else "CONTACT SIGNAL",
                "SEAGER CHECK",
            ]
            context_tiles = [
                ("BB%", ctx.get("bb_pct") or "—"),
                ("K%", ctx.get("k_pct") or "—"),
                ("BB/K", ctx.get("bb_k_ratio") or "—"),
                ("SEAGER", r.get("seager_score") if r.get("seager_score") is not None else "—"),
                ("BABIP", ctx.get("babip") or "N/A"),
            ]
            board_label = "HITTER"
            spark_label = "Contact Path vs Surface"
            gradient = "#b6ff00"

        out.append({
            **r,
            "rank": idx,
            "player_name": name,
            "avatar": avatar(name),
            "edge_score": edge_score,
            "score_class": score_class(edge_score),
            "player_type": player_type,
            "board_label": board_label,
            "spark_label": spark_label,
            "gradient": gradient,
            "diagnosis": diagnosis,
            "context_label": ctx.get("season_context") or ("COMMAND_PROFILE" if player_type == "pitcher" else "PLATE_DISCIPLINE"),
            "context_season": ctx.get("season") or "SEASON",
            "context_tiles": context_tiles,
            "metric_1_label": safe(r.get("metric_1_label")),
            "metric_1": safe(r.get("metric_1")),
            "metric_2_label": safe(r.get("metric_2_label")),
            "metric_2": safe(r.get("metric_2")),
            "metric_3_label": safe(r.get("metric_3_label")),
            "metric_3": safe(r.get("metric_3")),
            "why": safe(r.get("why"), "Signal thesis pending."),
            "sample_note": safe(r.get("sample_note"), "LIVE WINDOW"),
            "trend_points": safe(r.get("trend_points"), "0,26 20,22 40,24 60,15 80,18 100,10 120,8"),
            "resolved_player_id": r.get("resolved_player_id") or r.get("player_id") or "",
        })
    return out

HTML = r'''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals // Signal Wall</title>
  <style>
    {{ shell_styles | safe }}

    :root {
      --bg: #05070b;
      --panel: rgba(8, 13, 20, 0.94);
      --panel-2: rgba(15, 23, 42, 0.72);
      --line: rgba(255,255,255,0.10);
      --muted: rgba(148,163,184,0.80);
      --text: rgba(248,250,252,0.98);
      --lime: #b6ff00;
      --cyan: #22d3ee;
      --blue: #6aa6ff;
      --red: #fb7185;
      --mono: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    body {
      margin: 0;
      background:
        radial-gradient(circle at 18% -10%, rgba(34,211,238,0.12), transparent 32%),
        radial-gradient(circle at 82% 8%, rgba(182,255,0,0.08), transparent 34%),
        var(--bg);
      color: var(--text);
      font-family: var(--sans);
    }

    .topbar {
      border-bottom: 1px solid rgba(255,255,255,0.08);
      background: rgba(2,6,23,0.78);
      backdrop-filter: blur(18px);
      position: sticky;
      top: 0;
      z-index: 20;
    }

    .topbar-inner,
    .app {
      width: min(1840px, calc(100% - 48px));
      margin: 0 auto;
    }

    .topbar-inner {
      min-height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
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
      background: var(--lime);
      box-shadow: 0 0 14px rgba(182,255,0,0.55);
    }

    .brand-kicker {
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.16em;
    }

    .brand-blue { color: var(--cyan); }
    .brand-white { color: var(--text); }

    .brand-title {
      margin-top: 2px;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.12em;
      color: var(--muted);
      text-transform: uppercase;
    }

    .livebox {
      text-align: right;
      font-family: var(--mono);
    }

    .live-label {
      font-size: 10px;
      font-weight: 900;
      letter-spacing: 0.14em;
      color: var(--lime);
    }

    .live-time {
      margin-top: 3px;
      font-size: 10px;
      color: var(--muted);
    }

    .app {
      padding: 28px 0 46px;
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 420px;
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
    }

    .hero-card,
    .summary-card,
    .section {
      border: 1px solid var(--line);
      border-radius: 24px;
      background:
        radial-gradient(circle at 14% 0%, rgba(182,255,0,0.07), transparent 34%),
        linear-gradient(180deg, rgba(15,23,42,0.88), rgba(2,6,23,0.94));
      box-shadow: 0 20px 60px rgba(0,0,0,0.28);
    }

    .hero-card {
      padding: 30px;
    }

    .eyebrow,
    .summary-label,
    .section-kicker,
    .rankline,
    .signal-line,
    .score-label,
    .metric-label,
    .sparkline-label,
    .diagnosis-label,
    .season-context-head {
      font-family: var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }

    .eyebrow {
      color: var(--lime);
      font-size: 10px;
      font-weight: 900;
      margin-bottom: 12px;
    }

    .hero-title {
      margin: 0;
      font-family: var(--sans);
      font-size: clamp(46px, 4.85vw, 68px);
      line-height: 1.02;
      font-weight: 760;
      letter-spacing: -0.055em;
      color: #f7f8fa;
    }

    .hero-copy {
      margin: 14px 0 0;
      max-width: 880px;
      color: rgba(203,213,225,0.80);
      font-size: 15px;
      line-height: 1.65;
    }

    .summary-card {
      padding: 20px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .summary-tile {
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 14px;
      background: rgba(2,6,23,0.40);
    }

    .summary-label {
      font-size: 9px;
      font-weight: 900;
      color: var(--muted);
    }

    .summary-value {
      margin-top: 6px;
      font-family: var(--sans);
      font-size: 24px;
      font-weight: 900;
      letter-spacing: -0.045em;
    }

    .boards {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }

    .section {
      overflow: hidden;
    }

    .section-head {
      padding: 18px 18px 16px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
    }

    .section-kicker {
      font-size: 9px;
      font-weight: 900;
      color: var(--lime);
    }

    .section-title {
      margin: 5px 0 0;
      font-family: var(--sans);
      font-size: clamp(26px, 2.1vw, 34px);
      line-height: 1;
      font-weight: 900;
      letter-spacing: -0.05em;
    }

    .section-badge {
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 999px;
      padding: 7px 10px;
      font-family: var(--mono);
      font-size: 9px;
      font-weight: 900;
      letter-spacing: 0.12em;
      color: rgba(226,232,240,0.88);
      background: rgba(2,6,23,0.42);
      text-transform: uppercase;
      white-space: nowrap;
    }

    .cards-grid {
      display: grid;
      gap: 14px;
      padding: 14px;
    }

    .player-card {
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 22px;
      padding: 16px;
      background:
        radial-gradient(circle at 20% 0%, rgba(182,255,0,0.08), transparent 34%),
        linear-gradient(180deg, rgba(15,23,42,0.92), rgba(2,6,23,0.96));
      box-shadow: 0 18px 46px rgba(0,0,0,0.36);
    }

    .player-card:hover {
      border-color: rgba(182,255,0,0.26);
      transform: translateY(-1px);
      transition: 0.18s ease;
    }

    .player-top {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 132px;
      gap: 16px;
      align-items: start;
    }

    .player-head {
      display: grid;
      grid-template-columns: 42px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
      min-width: 0;
    }

    .avatar {
      width: 42px;
      height: 42px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.10);
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(255,255,255,0.035);
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 900;
      color: var(--text);
    }

    .player-ident {
      min-width: 0;
    }

    .rankline {
      font-size: 9px;
      font-weight: 800;
      color: var(--muted);
    }

    .player-name {
      margin: 4px 0 0;
      font-family: var(--sans);
      font-size: clamp(21px, 1.45vw, 28px);
      font-weight: 820;
      line-height: 0.98;
      letter-spacing: -0.045em;
      text-transform: uppercase;
      color: rgba(248,250,252,0.98);
      max-width: 100%;
    }

    .signal-line {
      margin-top: 7px;
      font-size: 9px;
      line-height: 1.45;
      color: rgba(148,163,184,0.86);
      letter-spacing: 0.10em;
    }

    .score-head {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 10px;
    }

    .score-meta {
      text-align: right;
    }

    .score-label {
      font-size: 8px;
      font-weight: 900;
      color: var(--muted);
    }

    .score-value {
      font-family: var(--sans);
      font-size: clamp(38px, 2.65vw, 50px);
      font-weight: 900;
      line-height: 0.88;
      letter-spacing: -0.055em;
    }

    .score-hot { color: var(--lime); }
    .score-warm { color: #d8ff7a; }
    .score-cool { color: var(--cyan); }

    .provision-btn {
      width: 132px;
      min-height: 38px;
      border-radius: 12px;
      border: 1px solid rgba(182,255,0,0.34);
      background: rgba(182,255,0,0.10);
      color: rgba(219,255,142,0.98);
      font-family: var(--mono);
      font-size: 8px;
      font-weight: 900;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      cursor: pointer;
    }

    .diagnosis-wrap {
      margin-top: 12px;
    }

    .diagnosis-label {
      margin: 0 0 5px;
      font-size: 8px;
      font-weight: 900;
      color: rgba(148,163,184,0.76);
    }

    .diagnosis-banner {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 4px;
      min-height: 38px;
      padding: 9px 12px;
      border: 1px solid rgba(182,255,0,0.30);
      border-radius: 12px;
      background:
        linear-gradient(90deg, rgba(182,255,0,0.14), rgba(182,255,0,0.035)),
        rgba(15,23,42,0.34);
      font-family: var(--mono);
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0.075em;
      color: rgba(219,255,142,0.98);
      text-transform: uppercase;
    }

    .diagnosis-sep {
      color: rgba(219,255,142,0.42);
    }

    .sparkline-wrap {
      margin-top: 12px;
      padding: 11px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(2,6,23,0.42);
    }

    .sparkline-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }

    .sparkline-label {
      font-size: 8px;
      font-weight: 900;
      color: var(--muted);
    }

    .sparkline-note {
      font-family: var(--mono);
      font-size: 9px;
      color: rgba(203,213,225,0.74);
    }

    svg.sparkline {
      display: block;
      width: 100%;
      height: 34px;
    }

    .sparkline-path {
      stroke-width: 2.4;
      stroke-linecap: round;
      stroke-linejoin: round;
      fill: none;
      filter: drop-shadow(0 0 3px rgba(182,255,0,0.25));
    }

    .metric-grid {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }

    .metric,
    .season-context-tile {
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 10px;
      background: rgba(15,23,42,0.46);
      min-width: 0;
    }

    .metric-label,
    .season-context-tile span {
      display: block;
      font-family: var(--mono);
      font-size: 8px;
      font-weight: 900;
      letter-spacing: 0.11em;
      color: var(--muted);
      text-transform: uppercase;
    }

    .metric-value,
    .season-context-tile strong {
      display: block;
      margin-top: 4px;
      font-family: var(--mono);
      font-size: 15px;
      font-weight: 900;
      color: var(--text);
      white-space: nowrap;
    }

    .value-edge,
    .value-ballistics,
    .value-apex,
    .value-pulse {
      color: var(--lime);
    }

    .season-context-strip {
      margin-top: 12px;
      padding: 10px;
      border: 1px solid rgba(34,211,238,0.20);
      border-radius: 14px;
      background:
        radial-gradient(circle at 50% 0%, rgba(34,211,238,0.07), transparent 62%),
        rgba(2,6,12,0.45);
    }

    .season-context-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
      font-size: 8px;
      font-weight: 900;
      color: rgba(148,163,184,0.82);
    }

    .season-context-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 6px;
    }

    .hitter-card .season-context-grid {
      grid-template-columns: repeat(5, minmax(0, 1fr));
    }

    .why-full {
      margin-top: 11px;
      padding-top: 10px;
      border-top: 1px solid rgba(255,255,255,0.07);
      font-size: 10px;
      line-height: 1.55;
      color: rgba(203,213,225,0.82);
    }

    @media (max-width: 1200px) {
      .boards {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 700px) {
      .topbar-inner,
      .app {
        width: min(100%, calc(100% - 16px));
      }

      .hero {
        grid-template-columns: 1fr;
      }

      .summary-card {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .section-head {
        flex-direction: column;
        align-items: stretch;
      }

      .player-top {
        grid-template-columns: 1fr;
      }

      .score-head,
      .score-meta {
        align-items: flex-start;
        text-align: left;
      }

      .provision-btn {
        width: 100%;
      }

      .metric-grid,
      .season-context-grid,
      .hitter-card .season-context-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

/* V2 typography refinement override: closer to live MLB Extraction */
.signal-wall-v2-typography-lock .player-name,
body .player-name {
  font-family: var(--sans) !important;
  font-size: clamp(19px, 1.28vw, 25px) !important;
  font-weight: 760 !important;
  line-height: 1.02 !important;
  letter-spacing: -0.038em !important;
  text-transform: uppercase !important;
  color: rgba(248,250,252,0.98) !important;
}

.signal-wall-v2-typography-lock .score-value,
body .score-value {
  font-family: var(--sans) !important;
  font-size: clamp(34px, 2.25vw, 46px) !important;
  font-weight: 860 !important;
  line-height: 0.9 !important;
  letter-spacing: -0.052em !important;
}

.signal-wall-v2-typography-lock .hero-title,
body .hero-title {
  font-weight: 620 !important;
  letter-spacing: -0.047em !important;
}

.signal-wall-v2-typography-lock .section-title,
body .section-title {
  font-size: clamp(22px, 1.7vw, 28px) !important;
  font-weight: 780 !important;
  letter-spacing: -0.042em !important;
}

.signal-wall-v2-typography-lock .diagnosis-banner,
body .diagnosis-banner {
  font-size: 8.5px !important;
  font-weight: 760 !important;
  letter-spacing: 0.065em !important;
}

.signal-wall-v2-typography-lock .rankline,
.signal-wall-v2-typography-lock .signal-line,
body .rankline,
body .signal-line {
  font-weight: 700 !important;
  letter-spacing: 0.09em !important;
}



/* V2 canvas density lock: match live MLB Extraction proportions */

body.signal-wall-v2-typography-lock .topbar-inner,

body.signal-wall-v2-typography-lock .app {

  width: min(1440px, calc(100% - 48px)) !important;

  max-width: 1440px !important;

}

body.signal-wall-v2-typography-lock .hero {

  grid-template-columns: minmax(0, 1fr) 360px !important;

  gap: 16px !important;

  margin-bottom: 16px !important;

}

body.signal-wall-v2-typography-lock .boards {

  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;

  gap: 16px !important;

}

body.signal-wall-v2-typography-lock .cards-grid {

  gap: 12px !important;

  padding: 12px !important;

}

body.signal-wall-v2-typography-lock .player-card {

  padding: 14px !important;

  border-radius: 18px !important;

}

body.signal-wall-v2-typography-lock .player-top {

  grid-template-columns: minmax(0, 1fr) 116px !important;

  gap: 12px !important;

}

body.signal-wall-v2-typography-lock .player-head {

  grid-template-columns: 38px minmax(0, 1fr) !important;

  gap: 10px !important;

}

body.signal-wall-v2-typography-lock .avatar {

  width: 38px !important;

  height: 38px !important;

  font-size: 10px !important;

}

body.signal-wall-v2-typography-lock .player-name {

  font-size: clamp(18px, 1.12vw, 23px) !important;

  font-weight: 760 !important;

  line-height: 0.98 !important;

  letter-spacing: -0.04em !important;

}

body.signal-wall-v2-typography-lock .score-value {

  font-size: clamp(34px, 2.05vw, 44px) !important;

  font-weight: 880 !important;

}

body.signal-wall-v2-typography-lock .provision-btn {

  width: 116px !important;

  min-height: 34px !important;

  font-size: 7.5px !important;

}

body.signal-wall-v2-typography-lock .diagnosis-banner {

  min-height: 34px !important;

  padding: 8px 10px !important;

  font-size: 8px !important;

}

body.signal-wall-v2-typography-lock .sparkline-wrap {

  margin-top: 10px !important;

  padding: 9px !important;

}

body.signal-wall-v2-typography-lock .metric-grid {

  margin-top: 10px !important;

  gap: 7px !important;

}

body.signal-wall-v2-typography-lock .metric,

body.signal-wall-v2-typography-lock .season-context-tile {

  padding: 8px !important;

}

body.signal-wall-v2-typography-lock .season-context-strip {

  margin-top: 10px !important;

  padding: 9px !important;

}

body.signal-wall-v2-typography-lock .why-full {

  font-size: 9.5px !important;

  line-height: 1.45 !important;

}



/* V2 MLB Extraction typography/style match */
body.signal-wall-v2-typography-lock {
  --v2-sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --v2-mono: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace;
}

body.signal-wall-v2-typography-lock .hero-title,
body.signal-wall-v2-typography-lock h1 {
  font-family: var(--v2-sans) !important;
  font-size: clamp(56px, 5.35vw, 82px) !important;
  font-weight: 760 !important;
  line-height: 0.92 !important;
  letter-spacing: -0.072em !important;
  text-transform: none !important;
  color: #f7f8fa !important;
  max-width: 820px !important;
}

body.signal-wall-v2-typography-lock .hero-copy,
body.signal-wall-v2-typography-lock .hero p {
  font-family: var(--v2-sans) !important;
  font-size: 17px !important;
  line-height: 1.55 !important;
  font-weight: 430 !important;
  letter-spacing: -0.015em !important;
  color: rgba(247, 248, 250, 0.72) !important;
  max-width: 760px !important;
}

body.signal-wall-v2-typography-lock .section-title,
body.signal-wall-v2-typography-lock .board-title,
body.signal-wall-v2-typography-lock h2 {
  font-family: var(--v2-sans) !important;
  font-size: clamp(25px, 1.65vw, 34px) !important;
  font-weight: 780 !important;
  line-height: 0.98 !important;
  letter-spacing: -0.055em !important;
  text-transform: uppercase !important;
}

body.signal-wall-v2-typography-lock .player-name {
  font-family: var(--v2-sans) !important;
  font-size: clamp(20px, 1.22vw, 26px) !important;
  font-weight: 820 !important;
  line-height: 0.95 !important;
  letter-spacing: -0.06em !important;
  text-transform: uppercase !important;
}

body.signal-wall-v2-typography-lock .score-value {
  font-family: var(--v2-sans) !important;
  font-size: clamp(42px, 2.85vw, 58px) !important;
  font-weight: 860 !important;
  line-height: 0.82 !important;
  letter-spacing: -0.07em !important;
  font-style: italic !important;
}

body.signal-wall-v2-typography-lock .eyebrow,
body.signal-wall-v2-typography-lock .section-kicker,
body.signal-wall-v2-typography-lock .card-kicker,
body.signal-wall-v2-typography-lock .player-meta,
body.signal-wall-v2-typography-lock .metric-label,
body.signal-wall-v2-typography-lock .season-context-label,
body.signal-wall-v2-typography-lock .diagnosis-label,
body.signal-wall-v2-typography-lock .provision-btn {
  font-family: var(--v2-mono) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.14em !important;
}

body.signal-wall-v2-typography-lock .hero,
body.signal-wall-v2-typography-lock .summary-card,
body.signal-wall-v2-typography-lock .board-panel,
body.signal-wall-v2-typography-lock .player-card {
  background:
    radial-gradient(circle at 10% 0%, rgba(160, 255, 55, 0.08), transparent 35%),
    linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012)),
    rgba(7, 8, 13, 0.82) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.06),
    0 18px 60px rgba(0,0,0,0.34) !important;
}

body.signal-wall-v2-typography-lock .player-card {
  padding: 18px !important;
}

body.signal-wall-v2-typography-lock .diagnosis-banner {
  border-radius: 12px !important;
  background: rgba(139, 255, 35, 0.12) !important;
  border: 1px solid rgba(166, 255, 52, 0.42) !important;
  box-shadow: inset 0 1px 0 rgba(166,255,52,0.12) !important;
}

body.signal-wall-v2-typography-lock .provision-btn {
  background: rgba(10, 11, 15, 0.88) !important;
  border: 1px solid rgba(72, 132, 255, 0.62) !important;
  color: #ffffff !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.08),
    0 0 22px rgba(47, 111, 255, 0.12) !important;
}

body.signal-wall-v2-typography-lock .metric-value,
body.signal-wall-v2-typography-lock .season-context-value {
  font-family: var(--v2-sans) !important;
  font-weight: 760 !important;
  letter-spacing: -0.035em !important;
}


/* V2 extraction-ledger hero/card alignment pass */
body.signal-wall-v2-typography-lock .topbar-inner,
body.signal-wall-v2-typography-lock .app {
  width: min(1320px, calc(100% - 48px)) !important;
  max-width: 1320px !important;
}

body.signal-wall-v2-typography-lock .hero {
  display: grid !important;
  grid-template-columns: minmax(0, 1.45fr) minmax(360px, 0.78fr) !important;
  gap: 22px !important;
  align-items: stretch !important;
  margin: 28px auto 22px !important;
}

body.signal-wall-v2-typography-lock .hero,
body.signal-wall-v2-typography-lock .hero-panel,
body.signal-wall-v2-typography-lock .hero-card {
  min-height: 300px !important;
}

body.signal-wall-v2-typography-lock .hero-title,
body.signal-wall-v2-typography-lock h1 {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  font-size: clamp(58px, 5.15vw, 82px) !important;
  font-weight: 720 !important;
  line-height: 0.91 !important;
  letter-spacing: -0.078em !important;
  text-transform: none !important;
  color: #f6f7f8 !important;
  max-width: 760px !important;
}

body.signal-wall-v2-typography-lock .hero-copy,
body.signal-wall-v2-typography-lock .hero p {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  font-size: 17px !important;
  line-height: 1.55 !important;
  font-weight: 420 !important;
  letter-spacing: -0.018em !important;
  color: rgba(246, 247, 248, 0.72) !important;
  max-width: 720px !important;
}

/* Convert V2 tiled stat card toward MLB Extraction single vertical card */
body.signal-wall-v2-typography-lock .summary-card,
body.signal-wall-v2-typography-lock .executive-card,
body.signal-wall-v2-typography-lock .stat-card,
body.signal-wall-v2-typography-lock .hero-stats {
  display: flex !important;
  flex-direction: column !important;
  justify-content: flex-start !important;
  gap: 18px !important;
  padding: 30px 34px !important;
  border-radius: 22px !important;
  min-height: 300px !important;
}

body.signal-wall-v2-typography-lock .summary-grid,
body.signal-wall-v2-typography-lock .stat-grid,
body.signal-wall-v2-typography-lock .hero-stat-grid {
  display: flex !important;
  flex-direction: column !important;
  gap: 14px !important;
}

body.signal-wall-v2-typography-lock .summary-tile,
body.signal-wall-v2-typography-lock .stat-tile,
body.signal-wall-v2-typography-lock .hero-stat {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-height: 0 !important;
}

body.signal-wall-v2-typography-lock .summary-tile span,
body.signal-wall-v2-typography-lock .stat-tile span,
body.signal-wall-v2-typography-lock .hero-stat span,
body.signal-wall-v2-typography-lock .summary-label,
body.signal-wall-v2-typography-lock .stat-label {
  display: block !important;
  font-family: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace !important;
  font-size: 11px !important;
  font-weight: 760 !important;
  line-height: 1 !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  color: rgba(225, 228, 238, 0.52) !important;
  margin-bottom: 5px !important;
}

body.signal-wall-v2-typography-lock .summary-tile strong,
body.signal-wall-v2-typography-lock .stat-tile strong,
body.signal-wall-v2-typography-lock .hero-stat strong,
body.signal-wall-v2-typography-lock .summary-value,
body.signal-wall-v2-typography-lock .stat-value {
  display: block !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  font-size: clamp(28px, 2.3vw, 42px) !important;
  font-weight: 760 !important;
  line-height: 0.95 !important;
  letter-spacing: -0.058em !important;
  color: #f6f7f8 !important;
}

/* Board title/card typography closer to MLB Extraction */
body.signal-wall-v2-typography-lock .section-title,
body.signal-wall-v2-typography-lock .board-title,
body.signal-wall-v2-typography-lock h2 {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  font-size: clamp(24px, 1.55vw, 31px) !important;
  font-weight: 740 !important;
  line-height: 0.98 !important;
  letter-spacing: -0.058em !important;
  text-transform: uppercase !important;
}

body.signal-wall-v2-typography-lock .boards {
  gap: 18px !important;
}

body.signal-wall-v2-typography-lock .board-panel {
  border-radius: 22px !important;
}

body.signal-wall-v2-typography-lock .player-card {
  padding: 18px !important;
  border-radius: 18px !important;
}

body.signal-wall-v2-typography-lock .player-name {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  font-size: clamp(21px, 1.35vw, 27px) !important;
  font-weight: 780 !important;
  line-height: 0.96 !important;
  letter-spacing: -0.065em !important;
  text-transform: uppercase !important;
}

body.signal-wall-v2-typography-lock .score-value {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  font-size: clamp(46px, 3vw, 60px) !important;
  font-weight: 820 !important;
  line-height: 0.82 !important;
  letter-spacing: -0.075em !important;
  font-style: italic !important;
}

body.signal-wall-v2-typography-lock .provision-btn {
  background: rgba(8, 9, 13, 0.9) !important;
  border: 1px solid rgba(67, 119, 255, 0.68) !important;
  color: #fff !important;
  border-radius: 10px !important;
  width: 132px !important;
  min-height: 42px !important;
  font-size: 8px !important;
  font-family: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace !important;
  letter-spacing: 0.12em !important;
}

body.signal-wall-v2-typography-lock .diagnosis-banner {
  min-height: 46px !important;
  padding: 12px 14px !important;
  border-radius: 12px !important;
}

body.signal-wall-v2-typography-lock .metric,
body.signal-wall-v2-typography-lock .season-context-tile {
  border-radius: 11px !important;
}

@media (max-width: 960px) {
  body.signal-wall-v2-typography-lock .hero {
    grid-template-columns: 1fr !important;
  }

  body.signal-wall-v2-typography-lock .boards {
    grid-template-columns: 1fr !important;
  }
}


/* V2 live-visual calibration: reduce oversized local V2 toward MLB Extraction production scale */
body.signal-wall-v2-typography-lock .topbar-inner,
body.signal-wall-v2-typography-lock .app {
  width: min(1180px, calc(100% - 64px)) !important;
  max-width: 1180px !important;
}

body.signal-wall-v2-typography-lock .hero {
  grid-template-columns: minmax(0, 1.56fr) minmax(310px, 0.74fr) !important;
  gap: 16px !important;
  margin: 30px auto 20px !important;
}

body.signal-wall-v2-typography-lock .hero,
body.signal-wall-v2-typography-lock .hero-panel,
body.signal-wall-v2-typography-lock .hero-card {
  min-height: 250px !important;
}

body.signal-wall-v2-typography-lock .hero-panel,
body.signal-wall-v2-typography-lock .hero-card,
body.signal-wall-v2-typography-lock .summary-card {
  padding: 26px 30px !important;
  border-radius: 18px !important;
}

body.signal-wall-v2-typography-lock .hero-title,
body.signal-wall-v2-typography-lock h1 {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  font-size: clamp(44px, 4.05vw, 62px) !important;
  font-weight: 560 !important;
  line-height: 0.96 !important;
  letter-spacing: -0.064em !important;
  text-transform: none !important;
  max-width: 680px !important;
  color: #f6f7f8 !important;
}

body.signal-wall-v2-typography-lock .hero-copy,
body.signal-wall-v2-typography-lock .hero p {
  font-size: 14.5px !important;
  line-height: 1.5 !important;
  font-weight: 400 !important;
  letter-spacing: -0.01em !important;
  max-width: 660px !important;
  color: rgba(246, 247, 248, 0.72) !important;
}

body.signal-wall-v2-typography-lock .summary-card,
body.signal-wall-v2-typography-lock .executive-card,
body.signal-wall-v2-typography-lock .stat-card,
body.signal-wall-v2-typography-lock .hero-stats {
  gap: 12px !important;
  padding: 26px 30px !important;
  min-height: 250px !important;
  justify-content: flex-start !important;
}

body.signal-wall-v2-typography-lock .summary-tile span,
body.signal-wall-v2-typography-lock .stat-tile span,
body.signal-wall-v2-typography-lock .hero-stat span,
body.signal-wall-v2-typography-lock .summary-label,
body.signal-wall-v2-typography-lock .stat-label {
  font-size: 9.5px !important;
  font-weight: 680 !important;
  line-height: 1 !important;
  letter-spacing: 0.17em !important;
  color: rgba(225, 228, 238, 0.52) !important;
  margin-bottom: 4px !important;
}

body.signal-wall-v2-typography-lock .summary-tile strong,
body.signal-wall-v2-typography-lock .stat-tile strong,
body.signal-wall-v2-typography-lock .hero-stat strong,
body.signal-wall-v2-typography-lock .summary-value,
body.signal-wall-v2-typography-lock .stat-value {
  font-size: clamp(23px, 1.65vw, 31px) !important;
  font-weight: 560 !important;
  line-height: 0.98 !important;
  letter-spacing: -0.05em !important;
}

body.signal-wall-v2-typography-lock .boards {
  gap: 15px !important;
}

body.signal-wall-v2-typography-lock .board-panel {
  border-radius: 18px !important;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01)),
    rgba(14, 15, 17, 0.86) !important;
}

body.signal-wall-v2-typography-lock .section-title,
body.signal-wall-v2-typography-lock .board-title,
body.signal-wall-v2-typography-lock h2 {
  font-size: clamp(20px, 1.24vw, 25px) !important;
  font-weight: 600 !important;
  line-height: 1 !important;
  letter-spacing: -0.045em !important;
}

body.signal-wall-v2-typography-lock .cards-grid {
  gap: 11px !important;
  padding: 11px !important;
}

body.signal-wall-v2-typography-lock .player-card {
  padding: 14px !important;
  border-radius: 15px !important;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.026), rgba(255,255,255,0.01)),
    rgba(15, 16, 18, 0.88) !important;
}

body.signal-wall-v2-typography-lock .player-top {
  grid-template-columns: minmax(0, 1fr) 112px !important;
  gap: 9px !important;
}

body.signal-wall-v2-typography-lock .player-head {
  grid-template-columns: 34px minmax(0, 1fr) !important;
  gap: 9px !important;
}

body.signal-wall-v2-typography-lock .avatar {
  width: 34px !important;
  height: 34px !important;
  font-size: 9px !important;
}

body.signal-wall-v2-typography-lock .player-name {
  font-size: clamp(17px, 0.96vw, 21px) !important;
  font-weight: 620 !important;
  line-height: 0.98 !important;
  letter-spacing: -0.052em !important;
}

body.signal-wall-v2-typography-lock .score-value {
  font-size: clamp(36px, 2.05vw, 44px) !important;
  font-weight: 680 !important;
  line-height: 0.86 !important;
  letter-spacing: -0.067em !important;
  font-style: italic !important;
}

body.signal-wall-v2-typography-lock .provision-btn {
  width: 112px !important;
  min-height: 34px !important;
  font-size: 7px !important;
  border-radius: 8px !important;
}

body.signal-wall-v2-typography-lock .diagnosis-banner {
  min-height: 35px !important;
  padding: 8px 10px !important;
  border-radius: 10px !important;
}

body.signal-wall-v2-typography-lock .sparkline-wrap {
  margin-top: 8px !important;
  padding: 8px !important;
}

body.signal-wall-v2-typography-lock .metric-grid {
  margin-top: 8px !important;
  gap: 6px !important;
}

body.signal-wall-v2-typography-lock .metric,
body.signal-wall-v2-typography-lock .season-context-tile {
  padding: 7px 8px !important;
  border-radius: 9px !important;
}

body.signal-wall-v2-typography-lock .season-context-strip {
  margin-top: 8px !important;
  padding: 8px !important;
}

body.signal-wall-v2-typography-lock .why-full {
  font-size: 8.5px !important;
  line-height: 1.38 !important;
}

body.signal-wall-v2-typography-lock .hero,
body.signal-wall-v2-typography-lock .summary-card,
body.signal-wall-v2-typography-lock .hero-panel,
body.signal-wall-v2-typography-lock .hero-card,
body.signal-wall-v2-typography-lock .board-panel,
body.signal-wall-v2-typography-lock .player-card {
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.045),
    0 16px 44px rgba(0,0,0,0.28) !important;
}

body.signal-wall-v2-typography-lock .hero,
body.signal-wall-v2-typography-lock .summary-card,
body.signal-wall-v2-typography-lock .hero-panel,
body.signal-wall-v2-typography-lock .hero-card {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.024), rgba(255,255,255,0.008)),
    rgba(8, 9, 12, 0.88) !important;
}


/* V2 summary-card content model: MLB Extraction vertical card */
body.signal-wall-v2-typography-lock .extraction-summary-card {
  display: flex !important;
  flex-direction: column !important;
  justify-content: flex-start !important;
  gap: 12px !important;
}

body.signal-wall-v2-typography-lock .summary-row {
  display: block !important;
}

body.signal-wall-v2-typography-lock .summary-row span {
  display: block !important;
  font-family: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace !important;
  font-size: 9.5px !important;
  font-weight: 680 !important;
  line-height: 1 !important;
  letter-spacing: 0.17em !important;
  text-transform: uppercase !important;
  color: rgba(225, 228, 238, 0.52) !important;
  margin-bottom: 4px !important;
}

body.signal-wall-v2-typography-lock .summary-row strong {
  display: block !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  font-size: clamp(23px, 1.65vw, 31px) !important;
  font-weight: 560 !important;
  line-height: 0.98 !important;
  letter-spacing: -0.05em !important;
  color: #f6f7f8 !important;
}

body.signal-wall-v2-typography-lock .summary-row .summary-window {
  font-size: clamp(16px, 1.18vw, 21px) !important;
  letter-spacing: -0.035em !important;
}

body.signal-wall-v2-typography-lock .summary-note {
  margin: 8px 0 0 !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  font-size: 12.5px !important;
  line-height: 1.45 !important;
  font-weight: 400 !important;
  letter-spacing: -0.01em !important;
  color: rgba(246, 247, 248, 0.66) !important;
}


/* V2 viewport overflow guard */
html,
body {
  max-width: 100% !important;
  overflow-x: hidden !important;
}

body.signal-wall-v2-typography-lock .topbar-inner,
body.signal-wall-v2-typography-lock .app,
body.signal-wall-v2-typography-lock .hero,
body.signal-wall-v2-typography-lock .boards {
  box-sizing: border-box !important;
}

body.signal-wall-v2-typography-lock .app {
  margin-left: auto !important;
  margin-right: auto !important;
}

body.signal-wall-v2-typography-lock .hero,
body.signal-wall-v2-typography-lock .boards {
  width: 100% !important;
}


/* V2 compact summary terminal note */
body.signal-wall-v2-typography-lock .summary-terminal-note {
  margin-top: 10px !important;
  padding-top: 14px !important;
  border-top: 1px solid rgba(255,255,255,0.08) !important;
  font-family: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace !important;
  font-size: 9.5px !important;
  line-height: 1.25 !important;
  font-weight: 760 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: rgba(166, 255, 52, 0.82) !important;
}


/* V2 hero final spacing polish */
body.signal-wall-v2-typography-lock .hero {
  gap: 14px !important;
}

body.signal-wall-v2-typography-lock .hero-card,
body.signal-wall-v2-typography-lock .summary-card {
  min-height: 238px !important;
}

body.signal-wall-v2-typography-lock .hero-panel,
body.signal-wall-v2-typography-lock .hero-card,
body.signal-wall-v2-typography-lock .summary-card {
  padding-top: 24px !important;
  padding-bottom: 24px !important;
}

body.signal-wall-v2-typography-lock .summary-card.extraction-summary-card {
  gap: 10px !important;
}

body.signal-wall-v2-typography-lock .summary-terminal-note {
  margin-top: 6px !important;
  padding-top: 11px !important;
}

  </style>
</head>

<body class="signal-wall-v2-typography-lock">
  <div class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="brand-mark"></div>
        <div>
          <div class="brand-kicker"><span class="brand-white">DIAMOND</span><span class="brand-blue">SIGNALS</span></div>
          <div class="brand-title">Signal Wall // Extraction Chassis</div>
        </div>
      </div>
      <div class="livebox">
        <div class="live-label">● LIVE</div>
        <div class="live-time">{{ generated_at }}</div>
      </div>
    </div>
  </div>

  {{ nav_html | safe }}
  {{ search_html | safe }}

  <main class="app">
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">Executive Terminal</div>
        <h1 class="hero-title">Today’s Signal Wall</h1>
        <p class="hero-copy">
          Live pitcher and hitter movement board built from the MLB Extraction Ledger chassis. Edge Score, SEAGER, BABIP, BB%, K%, BB/K, K/BB, and season-context command data remain preserved.
        </p>
      </div>

      <aside class="summary-card extraction-summary-card">
        <div class="summary-row">
          <span>Mode</span>
          <strong>EDGE</strong>
        </div>
        <div class="summary-row">
          <span>Pitchers</span>
          <strong>{{ pitchers|length }}</strong>
        </div>
        <div class="summary-row">
          <span>Hitters</span>
          <strong>{{ hitters|length }}</strong>
        </div>
        <div class="summary-row">
          <span>Window</span>
          <strong class="summary-window">LIVE</strong>
        </div>
        <div class="summary-note summary-terminal-note">
          SEASON_CONTEXT // SEAGER + BABIP ACTIVE
        </div>
      </aside>
    </section>

    <section class="boards">
      <section class="section">
        <div class="section-head">
          <div>
            <div class="section-kicker">Command Profile</div>
            <h2 class="section-title">Pitcher Signals</h2>
          </div>
          <div class="section-badge">Top {{ pitchers|length }}</div>
        </div>

        <div class="cards-grid">
          {% for row in pitchers %}
          {{ card(row) }}
          {% endfor %}
        </div>
      </section>

      <section class="section">
        <div class="section-head">
          <div>
            <div class="section-kicker">Plate Discipline</div>
            <h2 class="section-title">Hitter Signals</h2>
          </div>
          <div class="section-badge">Top {{ hitters|length }}</div>
        </div>

        <div class="cards-grid">
          {% for row in hitters %}
          {{ card(row) }}
          {% endfor %}
        </div>
      </section>
    </section>
  </main>

  <script src="/player-search.js"></script>
  <script src="/player-card-actions.js"></script>
</body>
</html>
'''

CARD = r'''
<article
  class="player-card js-player-card {{ row.player_type }}-card"
  data-player-id="{{ row.resolved_player_id }}"
  data-player-name="{{ row.player_name }}"
  data-player-type="{{ row.player_type }}"
  data-player-team="MLB"
  data-profile-url="{% if row.resolved_player_id %}/scout/{{ row.resolved_player_id }}/{% else %}#{% endif %}"
>
  <div class="player-top">
    <div class="player-head">
      <div class="avatar">{{ row.avatar }}</div>
      <div class="player-ident">
        <div class="rankline">{% if row.rank == 1 %}[ PRIMARY SIGNAL ]{% else %}#{{ row.rank }} {{ row.board_label }} SIGNAL{% endif %}</div>
        <h3 class="player-name">{{ row.player_name }}</h3>
        <div class="signal-line">{{ row.board_label }} // Live Edge Signal // {{ row.sample_note }}</div>
      </div>
    </div>

    <div class="scorebox">
      <div class="score-head">
        <div class="score-meta">
          <div class="score-label">Edge Score</div>
          <div class="score-value {{ row.score_class }}">{{ row.edge_score }}</div>
        </div>
        <button type="button" class="provision-btn js-add-to-roster" data-default-label="INITIATE TRACKING">INITIATE TRACKING</button>
      </div>
    </div>
  </div>

  <div class="diagnosis-wrap">
    <div class="diagnosis-label">Diagnosis</div>
    <div class="diagnosis-banner">
      <span>{{ row.diagnosis[0] }}</span>
      <span class="diagnosis-sep">//</span>
      <span>{{ row.diagnosis[1] }}</span>
      <span class="diagnosis-sep">//</span>
      <span>{{ row.diagnosis[2] }}</span>
    </div>
  </div>

  <div class="sparkline-wrap">
    <div class="sparkline-head">
      <div class="sparkline-label">{{ row.spark_label }}</div>
      <div class="sparkline-note">{{ row.sample_note }}</div>
    </div>
    <svg class="sparkline compact" viewBox="0 0 120 34" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="{{ row.player_type }}Gradient{{ row.rank }}" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#444444" stop-opacity="0.65"></stop>
          <stop offset="100%" stop-color="{{ row.gradient }}" stop-opacity="1"></stop>
        </linearGradient>
      </defs>
      <polyline class="sparkline-path" stroke="url(#{{ row.player_type }}Gradient{{ row.rank }})" points="{{ row.trend_points }}" />
    </svg>
  </div>

  <div class="metric-grid">
    <div class="metric">
      <div class="metric-label">{{ row.metric_1_label }}</div>
      <div class="metric-value value-edge">{{ row.metric_1 }}</div>
    </div>
    <div class="metric">
      <div class="metric-label">{{ row.metric_2_label }}</div>
      <div class="metric-value value-edge">{{ row.metric_2 }}</div>
    </div>
    <div class="metric">
      <div class="metric-label">{{ row.metric_3_label }}</div>
      <div class="metric-value value-edge">{{ row.metric_3 }}</div>
    </div>
  </div>

  <div class="season-context-strip">
    <div class="season-context-head">
      <span>[ {{ row.context_label }} // {{ row.context_season }} ]</span>
      <span>MLB_SEASON_CONTEXT</span>
    </div>
    <div class="season-context-grid">
      {% for label, value in row.context_tiles %}
      <div class="season-context-tile">
        <span>{{ label }}</span>
        <strong>{{ value }}</strong>
      </div>
      {% endfor %}
    </div>
  </div>

  <div class="why-full"><strong>Signal Thesis:</strong> {{ row.why }}</div>
</article>
'''

def render_html() -> str:
    payload = json.loads(SIGNALS_JSON.read_text(encoding="utf-8"))
    pitchers = normalize_rows(payload.get("top_pitchers") or [], "pitcher")
    hitters = normalize_rows(payload.get("top_hitters") or [], "hitter")

    nav_template = load_text(NAV_TEMPLATE_PATH)
    search_html = load_text(SEARCH_TEMPLATE_PATH)
    shell_styles = load_text(SHELL_STYLES_PATH)

    try:
        nav_html = Template(nav_template).render(active_nav="signal_wall") if nav_template else ""
    except Exception:
        nav_html = ""

    card_template = Template(CARD)

    def card(row):
        return card_template.render(row=row)

    return Template(HTML).render(
        generated_at=payload.get("generated_at") or datetime.now().isoformat(),
        pitchers=pitchers,
        hitters=hitters,
        nav_html=nav_html,
        search_html=search_html,
        shell_styles=shell_styles,
        card=card,
    )

def main() -> None:
    if not SIGNALS_JSON.exists():
        raise SystemExit("Missing dist/signals.json. Run dashboard/build_dashboard.py first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render_html(), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
