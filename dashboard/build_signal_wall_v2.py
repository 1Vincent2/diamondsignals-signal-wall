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
  <title>DiamondSignals // Signal Wall V2</title>
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
      font-weight: 900;
      color: var(--muted);
    }

    .player-name {
      margin: 4px 0 0;
      font-family: var(--sans);
      font-size: clamp(24px, 1.9vw, 34px);
      font-weight: 900;
      line-height: 0.94;
      letter-spacing: -0.055em;
      text-transform: uppercase;
      color: rgba(248,250,252,0.98);
    }

    .signal-line {
      margin-top: 7px;
      font-size: 9px;
      line-height: 1.45;
      color: rgba(148,163,184,0.86);
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
      font-size: clamp(42px, 3.2vw, 58px);
      font-weight: 950;
      line-height: 0.86;
      letter-spacing: -0.06em;
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
      font-weight: 900;
      letter-spacing: 0.08em;
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
  </style>
</head>

<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="brand-mark"></div>
        <div>
          <div class="brand-kicker"><span class="brand-white">DIAMOND</span><span class="brand-blue">SIGNALS</span></div>
          <div class="brand-title">Signal Wall V2 // Extraction Chassis</div>
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
        <h1 class="hero-title">Today’s Signal Wall V2</h1>
        <p class="hero-copy">
          Clean-room Signal Wall built from the MLB Extraction Ledger chassis. Pitchers and hitters remain split into forensic panes while preserving Edge Score, SEAGER, BABIP, BB%, K%, BB/K, K/BB, and season-context command data.
        </p>
      </div>

      <aside class="summary-card">
        <div class="summary-tile">
          <div class="summary-label">Pitchers</div>
          <div class="summary-value">{{ pitchers|length }}</div>
        </div>
        <div class="summary-tile">
          <div class="summary-label">Hitters</div>
          <div class="summary-value">{{ hitters|length }}</div>
        </div>
        <div class="summary-tile">
          <div class="summary-label">Mode</div>
          <div class="summary-value">LIVE</div>
        </div>
        <div class="summary-tile">
          <div class="summary-label">Source</div>
          <div class="summary-value" style="font-size:18px;line-height:1.1;">signals.json</div>
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
