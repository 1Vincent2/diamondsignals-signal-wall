from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import math
import os

import pandas as pd
from jinja2 import Template
from pybaseball import statcast
from supabase import create_client

from dashboard.lib.report_status import build_report_status


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DIST_DIR = REPO_ROOT / "dist"
IVB_DIR = DIST_DIR / "ivb-heat-map"
STATUS_DIR = DIST_DIR / "status"
IVB_HEAT_MAP_STATUS_PATH = STATUS_DIR / "ivb-heat-map.json"
TEMPLATES_DIR = BASE_DIR / "templates"

# IVB_HEAT_MAP_SHARED_NAV_PATH_V1
# Desktop uses the compact shared pro nav.
# Mobile/menu drawer keeps the legacy shell nav contract intact.
DESKTOP_NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav_v2.html").read_text(encoding="utf-8")
MOBILE_NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
FOOTER_TEMPLATE = (TEMPLATES_DIR / "shell_footer.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")

TIMEZONE_LABEL = "America/New_York"
LOOKBACK_DAYS = 7
MIN_FASTBALL_COUNT = 8
CLIMBER_THRESHOLD = 1.0
IVB_LAB_TABLE = "ivb_lab_daily"

IVB_HEAT_MAP_STATUS_MODE = "statcast_supabase_ivb_heat_map_dynamic_v1"
IVB_HEAT_MAP_PIPELINE_LAYERS = [
    "pybaseball_statcast_pitch_level_feed",
    "fastball_ivb_window",
    "velocity_bucket_ivb_baseline",
    "ivb_vs_avg_scoring",
    "dead_zone_detection",
    "supabase_lab_write_readback",
    "tracking_identity_payloads",
    "no_static_player_seed_fallback",
]


HTML_TEMPLATE = Template(
    r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals // IVB Heat Map</title>
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
      --orange: #fb923c;
      --purple: #a855f7;
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

    .topbar { position: sticky; top: 0; z-index: 50; background: rgba(8, 8, 8, 0.90); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(255,255,255,0.05); }
    .topbar-inner, .app { width: min(1180px, calc(100% - 24px)); margin: 0 auto; }
    .topbar-inner { min-height: 62px; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 0; }
    .brand { display: flex; align-items: center; gap: 10px; min-width: 0; }
    .brand-mark { width: 11px; height: 11px; border-radius: 999px; background: var(--lime-hot); box-shadow: 0 0 10px rgba(182,255,0,0.35); flex: 0 0 auto; }
    .brand-kicker { font-size: 10px; line-height: 1; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 800; margin-bottom: 4px; }
    .brand-white { color: var(--text); }
    .brand-blue { color: var(--blue); }
    .brand-title { font-size: 16px; line-height: 1.05; letter-spacing: -0.02em; font-weight: 800; }
    .livebox { text-align: right; }
    .live-label { display: inline-flex; align-items: center; gap: 7px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.16em; color: var(--lime-hot); font-weight: 800; margin-bottom: 4px; }
    .live-dot { width: 7px; height: 7px; border-radius: 999px; background: var(--lime-hot); box-shadow: 0 0 10px rgba(182,255,0,0.35); }
    .live-time { font-family: var(--mono); font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }

    .app { padding: 20px 0 36px; }
    .hero-card, .metric-card, .section-card, .leader-card {
      background: var(--card-radial);
      border: 0.5px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }
    .hero-card::before, .metric-card::before, .section-card::before, .leader-card::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      border-radius: inherit;
      padding: 0.5px;
      background: linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.01));
      -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.55;
    }

    .hero-card { padding: 22px; margin-bottom: 16px; }
    .eyebrow { font-size: 10px; line-height: 1; letter-spacing: 0.18em; text-transform: uppercase; color: var(--blue); font-weight: 800; margin-bottom: 10px; }
    .hero-title { margin: 0 0 10px; font-size: clamp(30px, 6vw, 56px); line-height: 0.95; letter-spacing: -0.04em; font-weight: 900; text-transform: uppercase; }
    .hero-copy { margin: 0; max-width: 800px; color: var(--soft); font-size: 14px; }

    .top-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px; }
    .metric-card { padding: 16px; }
    .metric-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); font-weight: 800; margin-bottom: 8px; }
    .metric-value { font-family: var(--mono); font-size: 28px; font-weight: 800; line-height: 1; }
    .metric-note { margin-top: 8px; font-size: 12px; color: var(--soft); }

    .main-grid { display: grid; grid-template-columns: 1.35fr 0.85fr; gap: 16px; }
    .section-card, .leader-card { padding: 16px; }

    .section-head, .leader-head {
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .section-kicker { font-size: 10px; line-height: 1; letter-spacing: 0.16em; text-transform: uppercase; color: var(--blue); font-weight: 800; margin-bottom: 7px; }
    .section-title { margin: 0; font-size: 20px; line-height: 1.02; letter-spacing: -0.03em; text-transform: uppercase; font-weight: 900; }
    .section-badge { font-family: var(--mono); font-size: 11px; color: var(--soft); border: 1px solid rgba(255,255,255,0.08); border-radius: 999px; padding: 7px 10px; background: rgba(255,255,255,0.02); }
    .section-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

.field-guide-trigger {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 36px;
  border-radius: 999px;
  border: 1px solid rgba(106,166,255,0.32);
  background: rgba(106,166,255,0.14);
  color: var(--text);
  padding: 0 14px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  cursor: pointer;
  box-shadow: 0 0 12px rgba(106,166,255,0.10);
}

    .heat-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .heat-card {
      border-radius: 16px;
      padding: 14px;
      min-height: 176px;
      position: relative;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.07);
      box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    }
    .heat-card.cold {
      background: linear-gradient(180deg, rgba(59,130,246,0.25) 0%, rgba(12,20,38,0.92) 100%);
      border-color: rgba(96,165,250,0.32);
    }
    .heat-card.neutral {
      background: linear-gradient(180deg, rgba(255,255,255,0.12) 0%, rgba(15,15,15,0.92) 100%);
      border-color: rgba(255,255,255,0.12);
    }
    .heat-card.hot {
      background: linear-gradient(180deg, rgba(251,146,60,0.30) 0%, rgba(168,85,247,0.35) 100%);
      border-color: rgba(192,132,252,0.30);
    }

    .heat-rank { font-family: var(--mono); font-size: 11px; color: var(--soft); margin-bottom: 8px; text-transform: uppercase; }
    .heat-name { font-size: 18px; line-height: 1.02; letter-spacing: -0.03em; font-weight: 900; margin: 0 0 6px; }
    .heat-meta { font-family: var(--mono); font-size: 11px; color: var(--soft); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; }

    .heat-header-tags {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }

    .heat-band {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 9px;
      font-family: var(--mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(0,0,0,0.18);
    }

    .heat-climber {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 9px;
      font-family: var(--mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border: 1px solid rgba(182,255,0,0.22);
      background: rgba(182,255,0,0.08);
      color: var(--lime-hot);
    }
    .heat-transition {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 9px;
      font-family: var(--mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border: 1px solid rgba(106,166,255,0.24);
      background: rgba(106,166,255,0.10);
      color: var(--blue);
    }

    .heat-transition.apex {
      border-color: rgba(168,85,247,0.28);
      background: rgba(168,85,247,0.12);
      color: #d8b4fe;
    }

    .heat-transition.exit {
      border-color: rgba(182,255,0,0.24);
      background: rgba(182,255,0,0.10);
      color: var(--lime-hot);
    }

    .heat-transition.enter-dead {
      border-color: rgba(239,68,68,0.24);
      background: rgba(239,68,68,0.10);
      color: #fca5a5;
    }
    .heat-values { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
    .heat-value-box {
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 12px;
      padding: 10px;
      background: rgba(0,0,0,0.16);
    }
    .heat-value-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.10em; color: var(--muted); font-weight: 800; margin-bottom: 5px; }
    .heat-value { font-family: var(--mono); font-size: 18px; font-weight: 800; }
    .heat-risk {
      margin-top: 10px;
      margin-bottom: 8px;
      font-size: 11px;
      line-height: 1.35;
      color: var(--red);
      font-family: var(--mono);
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .heat-brief { font-size: 12px; line-height: 1.45; color: var(--soft); }

    .leader-list { display: grid; gap: 10px; }
    .leader-row {
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 12px;
  background: rgba(255,255,255,0.02);
}

.leader-row.positive {
  border-color: rgba(182,255,0,0.18);
  background: rgba(182,255,0,0.05);
}

.leader-row.negative {
  border-color: rgba(239,68,68,0.18);
  background: rgba(239,68,68,0.05);
}

.leader-row.transition {
  border-color: rgba(168,85,247,0.18);
  background: rgba(168,85,247,0.06);
}

.leader-name { font-size: 15px; font-weight: 800; margin-bottom: 5px; }
.leader-sub { font-family: var(--mono); font-size: 11px; color: var(--soft); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 7px; }

.leader-delta {
  font-family: var(--mono);
  font-size: 18px;
  font-weight: 800;
  color: var(--lime-hot);
}

.leader-row.negative .leader-delta {
  color: var(--red);
}

.leader-row.transition .leader-delta {
  color: var(--purple);
}
    .lab-note {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid rgba(255,255,255,0.05);
      font-size: 12px;
      color: var(--tiny);
    }

    .field-guide-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.56);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.22s ease;
      z-index: 80;
    }

    .field-guide-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }

    .field-guide-modal {
      position: fixed;
      top: 50%;
      left: 50%;
      width: min(720px, calc(100vw - 24px));
      transform: translate(-50%, -46%);
      opacity: 0;
      pointer-events: none;
      transition: transform 0.24s ease, opacity 0.24s ease;
      z-index: 90;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 18px;
      background: linear-gradient(180deg, #121212 0%, #080808 100%);
      box-shadow: 0 20px 60px rgba(0,0,0,0.42);
      overflow: hidden;
    }

    .field-guide-modal.open {
      opacity: 1;
      pointer-events: auto;
      transform: translate(-50%, -50%);
    }

    .field-guide-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 18px 18px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.02);
    }

    .field-guide-kicker {
      font-size: 10px;
      line-height: 1;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--blue);
      font-weight: 800;
      margin-bottom: 8px;
    }

    .field-guide-title {
      margin: 0;
      font-size: 22px;
      line-height: 1.02;
      letter-spacing: -0.03em;
      text-transform: uppercase;
      font-weight: 900;
      color: var(--text);
    }

    .field-guide-close {
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

    .field-guide-body {
      padding: 18px;
      display: grid;
      gap: 14px;
    }

    .field-guide-card {
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
      padding: 14px;
    }

    .field-guide-term {
      margin: 0 0 8px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-family: var(--mono);
      font-weight: 800;
    }

    .field-guide-term.apex { color: var(--purple); }
    .field-guide-term.dead { color: var(--blue); }
    .field-guide-term.whiff { color: var(--lime-hot); }

    .field-guide-copy {
      margin: 0;
      color: var(--soft);
      font-size: 13px;
      line-height: 1.5;
    }

    .field-guide-note {
      border-top: 1px solid rgba(255,255,255,0.05);
      padding-top: 12px;
      color: var(--tiny);
      font-size: 12px;
      line-height: 1.45;
    }

    .heat-action-row {
      margin-top: 12px;
      display: flex;
      justify-content: flex-start;
    }

    .heat-provision-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      padding: 0 14px;
      border-radius: 12px;
      border: 1px solid rgba(96,165,250,0.32);
      background: rgba(37,99,235,0.95);
      color: white;
      font-family: var(--mono);
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      cursor: pointer;
      transition: 160ms ease;
      white-space: nowrap;
    }

    .heat-provision-btn:hover {
      background: rgba(59,130,246,1);
      border-color: rgba(96,165,250,0.40);
      box-shadow: 0 0 16px rgba(59,130,246,0.16);
      transform: translateY(-1px);
    }

    {{ shell_styles | safe }}

    @media (max-width: 980px) {
      .main-grid { grid-template-columns: 1fr; }
      .heat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .top-metrics { grid-template-columns: 1fr; }
    }

    @media (max-width: 640px) {
      .topbar-inner, .app { width: min(100%, calc(100% - 16px)); }
      .heat-grid { grid-template-columns: 1fr; }
      .hero-card, .metric-card, .section-card, .leader-card { padding-left: 16px; padding-right: 16px; }
      .field-guide-modal { width: min(100vw - 16px, 720px); }

      .heat-action-row {
        margin-top: 10px;
      }

      .heat-provision-btn {
        min-height: 32px;
        padding: 0 12px;
        border-radius: 8px;
        font-size: 8px;
        letter-spacing: 0.04em;
      }
    }

      /* IVB_HEAT_MAP_MOBILE_REFACTOR_V1 */
      @media screen and (max-width: 760px) {
        body {
          overflow-x: hidden !important;
        }

        .search-strip {
          display: none !important;
        }

        .app {
          width: min(100%, calc(100% - 28px)) !important;
          padding: 18px 0 52px !important;
        }

        .hero-card {
          padding: 18px 16px !important;
          margin: 16px 0 14px !important;
          border-radius: 22px !important;
        }

        .eyebrow {
          font-size: 9px !important;
          line-height: 1 !important;
          letter-spacing: 0.16em !important;
          margin-bottom: 10px !important;
        }

        .hero-title {
          font-size: 34px !important;
          line-height: 1.04 !important;
          letter-spacing: -0.045em !important;
          font-weight: 700 !important;
          text-transform: none !important;
          margin: 0 0 10px !important;
        }

        .hero-copy {
          font-size: 13px !important;
          line-height: 1.48 !important;
          color: rgba(226,232,240,0.72) !important;
          margin: 0 !important;
        }

        .top-metrics {
          display: none !important;
        }

        .main-grid {
          grid-template-columns: 1fr !important;
          gap: 14px !important;
        }

        .section-card,
        .leader-card {
          padding: 14px !important;
          border-radius: 22px !important;
        }

        .section-head,
        .leader-head {
          flex-direction: column !important;
          align-items: stretch !important;
          gap: 10px !important;
          margin-bottom: 14px !important;
          padding-bottom: 12px !important;
        }

        .section-kicker {
          font-size: 9px !important;
          line-height: 1 !important;
          letter-spacing: 0.16em !important;
        }

        .section-title {
          font-size: 21px !important;
          line-height: 1 !important;
          letter-spacing: -0.03em !important;
          font-weight: 760 !important;
          text-transform: uppercase !important;
        }

        .section-actions {
          width: 100% !important;
          display: grid !important;
          grid-template-columns: 1fr !important;
          gap: 8px !important;
        }

        .field-guide-trigger,
        .section-badge {
          width: 100% !important;
          min-height: 36px !important;
          display: flex !important;
          align-items: center !important;
          justify-content: center !important;
          text-align: center !important;
          font-size: 9px !important;
          line-height: 1.25 !important;
          letter-spacing: 0.12em !important;
          border-radius: 999px !important;
        }

        .heat-grid {
          grid-template-columns: 1fr !important;
          gap: 14px !important;
        }

        .heat-card {
          min-height: 0 !important;
          padding: 12px !important;
          border-radius: 20px !important;
        }

        .heat-rank {
          font-size: 8px !important;
          line-height: 1.15 !important;
          letter-spacing: 0.14em !important;
          margin-bottom: 7px !important;
        }

        .heat-name {
          font-size: 23px !important;
          line-height: 0.98 !important;
          letter-spacing: -0.04em !important;
          font-weight: 760 !important;
        }

        .heat-meta {
          font-size: 9px !important;
          line-height: 1.25 !important;
          letter-spacing: 0.12em !important;
          margin-bottom: 10px !important;
        }

        .heat-header-tags {
          gap: 6px !important;
          flex-wrap: wrap !important;
        }

        .heat-band,
        .heat-climber,
        .heat-transition {
          font-size: 8px !important;
          line-height: 1 !important;
          letter-spacing: 0.12em !important;
          padding: 6px 8px !important;
        }

        .heat-action-row {
          margin-top: 10px !important;
        }

        .heat-provision-btn {
          width: 100% !important;
          min-height: 38px !important;
          justify-content: center !important;
          text-align: center !important;
          font-size: 9px !important;
          letter-spacing: 0.14em !important;
          border-radius: 999px !important;
        }

        .heat-values {
          grid-template-columns: 1fr !important;
          gap: 8px !important;
          margin-top: 12px !important;
        }

        .heat-value-box {
          padding: 8px 9px !important;
          border-radius: 14px !important;
        }

        .heat-value-label {
          font-size: 8px !important;
          line-height: 1 !important;
          letter-spacing: 0.14em !important;
        }

        .heat-value {
          font-size: 16px !important;
          line-height: 18px !important;
        }

        .heat-risk {
          font-size: 9px !important;
          line-height: 1.25 !important;
          letter-spacing: 0.12em !important;
          padding: 8px 10px !important;
          border-radius: 14px !important;
        }

        .heat-brief,
        .lab-note {
          font-size: 12px !important;
          line-height: 1.45 !important;
        }

        .leader-row {
          padding: 10px !important;
          border-radius: 16px !important;
        }

        .leader-name {
          font-size: 14px !important;
          line-height: 1.1 !important;
        }

        .leader-sub,
        .leader-delta {
          font-size: 9px !important;
          line-height: 1.25 !important;
          letter-spacing: 0.12em !important;
        }

        .field-guide-modal {
          width: calc(100vw - 16px) !important;
          max-height: calc(100vh - 16px) !important;
          border-radius: 22px !important;
        }
      }

      /* IVB_MOBILE_TRANSPARENT_TRACKING_BUTTON_V1 */
      @media screen and (max-width: 760px) {
        .heat-provision-btn {
          background: rgba(2, 6, 23, 0.58) !important;
          border: 1px solid rgba(106, 166, 255, 0.42) !important;
          color: rgba(255, 255, 255, 0.92) !important;
          box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.06),
            0 0 14px rgba(59,130,246,0.10) !important;
        }

        .heat-provision-btn:hover {
          background: rgba(2, 6, 23, 0.76) !important;
          border-color: rgba(106, 166, 255, 0.60) !important;
          box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.08),
            0 0 18px rgba(59,130,246,0.16) !important;
          transform: none !important;
        }
      }


/* IVB_HEAT_MAP_TITLE_AUDIT_LINE_V1
   Desktop-only title-section audit line.
   Mobile layout, mobile menu, and Field Guide modal behavior intentionally untouched.
*/
.ivb-active-audit-line {
  display: none !important;
}

@media screen and (min-width: 981px) {
  .ivb-active-audit-line {
    display: block !important;
    margin-top: 18px !important;
    padding-top: 18px !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
    font-family: var(--sans) !important;
    font-size: 17px !important;
    line-height: 1.4 !important;
    color: rgba(226,232,240,0.76) !important;
  }

  .ivb-active-audit-line strong {
    color: var(--lime-hot) !important;
    font-weight: 900 !important;
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
          <div class="brand-title">LAB Terminal // IVB Heat Map</div>
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

  <div id="fieldGuideOverlay" class="field-guide-overlay" onclick="closeFieldGuide()"></div>

  <div id="fieldGuideModal" class="field-guide-modal" aria-hidden="true">
    <div class="field-guide-head">
      <div>
        <div class="field-guide-kicker">Tactical Field Guide</div>
        <h2 class="field-guide-title">How to Read the Heat</h2>
      </div>
      <button class="field-guide-close" type="button" onclick="closeFieldGuide()" aria-label="Close field guide">×</button>
    </div>

        <div class="field-guide-body">
      <div class="field-guide-card">
        <h3 class="field-guide-term whiff">IVB RAW</h3>
        <p class="field-guide-copy">
          Induced Vertical Break measures how much a fastball resists gravity due to backspin. Higher IVB means the pitch stays above the barrel longer. In this terminal, 18"+ marks the elite carry threshold.
          <br><strong>Result:</strong> Elevated whiff potential and weaker vertical contact.
        </p>
      </div>

      <div class="field-guide-card">
        <h3 class="field-guide-term whiff">IVB VS AVG</h3>
        <p class="field-guide-copy">
          A velo-bucket comparison showing how much a pitcher’s IVB beats or trails the league norm for that exact fastball speed band. Winning the bucket creates carry-driven deception because the ball moves better than the hitter expects for the velocity.
          <br><strong>Result:</strong> Hidden shape advantage and increased swing-miss deception.
        </p>
      </div>

      <div class="field-guide-card">
        <h3 class="field-guide-term apex">APEX RISE</h3>
        <p class="field-guide-copy">
          The elite 18"+ IVB tier. These fastballs create the visual illusion of rise because they drop less than the hitter’s brain expects. That mismatch leads to swing-unders and pop-up contact.
          <br><strong>Result:</strong> High whiff rate and premium carry profile.
        </p>
      </div>

      <div class="field-guide-card">
        <h3 class="field-guide-term dead">THE DEAD ZONE</h3>
        <p class="field-guide-copy">
          The 12"–15" IVB danger band. This flatter path intersects more directly with the hitter’s natural swing plane and behaves like a Barrel Magnet when shape quality is not strong enough to miss the barrel.
          <br><strong>Result:</strong> Elevated hard contact and home-run risk.
        </p>
      </div>

      <div class="field-guide-card">
        <h3 class="field-guide-term whiff">WHIFF PROB</h3>
        <p class="field-guide-copy">
          The terminal’s bottom-line translation layer for bat-missing expectation. It converts IVB and, later, VAA into a direct swing-and-miss forecast instead of forcing the user to interpret raw physics manually.
          <br><strong>Result:</strong> Faster identification of strikeout-friendly fastball shapes.
          <br><br>As VAA is integrated, Whiff Prob will become the most predictive metric in the terminal for identifying elite swing-and-miss talent.
        </p>
      </div>

      <div class="field-guide-card">
        <h3 class="field-guide-term whiff">CLIMBERS</h3>
        <p class="field-guide-copy">
          A pitch-shape breakout signal. A gain of roughly +1.5" or more in IVB indicates a meaningful mechanical or shape-level change that fantasy managers should treat as actionable, not cosmetic.
          <br><strong>Result:</strong> Early identification of emerging bat-missing arms.
        </p>
      </div>

      <div class="field-guide-card">
        <h3 class="field-guide-term whiff">STATUS KEY</h3>
        <p class="field-guide-copy">
          <strong>CLEAR:</strong> Signal Strength: High. The pitch has cleared the dangerous flat-path zone.
          <br><strong>Result:</strong> Safer fastball shape profile.
          <br><br><strong>COLD:</strong> Signal Strength: Danger. The pitch lacks vertical life and stays in the hitter’s path.
          <br><strong>Result:</strong> Elevated contact quality and home-run risk.
          <br><br><strong>MEDIUM:</strong> Signal Strength: Neutral. Bat-missing shape is present, but location or velocity still needs to carry part of the profile.
          <br><strong>Result:</strong> Moderate whiff utility.
          <br><br><strong>HIGH:</strong> Signal Strength: Premium. The shape profile is optimized for swing-and-miss.
          <br><strong>Result:</strong> Strong strikeout potential.
          <br><br><strong>LOW:</strong> Signal Strength: Weak. The shape does not independently generate enough bat-missing utility.
          <br><strong>Result:</strong> Contact-driven outcome risk.
        </p>
      </div>

      <div class="field-guide-note">
        VAA remains a pending upstream layer in this version. That is why VAA currently displays as <strong>--</strong> on the cards. This will be upgraded later through upstream metric refinement and can cleanly live in a future Supabase-backed LAB table or view.
      </div>
    </div>

      <div class="field-guide-card">
        <h3 class="field-guide-term dead">The Dead Zone</h3>
        <p class="field-guide-copy">
          Fastballs with 12"–15" IVB. This flatter shape sits directly in the swing path, creating a barrel-friendly entry profile. The result: louder contact, more home-run danger, and contact-risk traps.
        </p>
      </div>

      <div class="field-guide-card">
        <h3 class="field-guide-term whiff">Whiff Prob</h3>
        <p class="field-guide-copy">
          A proprietary DiamondSignals translation layer combining IVB rise and eventually VAA flatness. It converts pitch-shape data into a direct swing-and-miss expectation, independent of pure velocity.
        </p>
      </div>

      <div class="field-guide-note">
        VAA remains a pending upstream layer in this version. That is why VAA currently displays as <strong>--</strong> on the cards. This will be upgraded later through upstream metric refinement and can cleanly live in a future Supabase-backed LAB table or view.
      </div>
    </div>
  </div>

  <div class="app">
    <section class="hero-card">
      <div class="eyebrow">Pitch Shape Intelligence</div>
      <h1 class="hero-title">IVB Heat Map</h1>
      <p class="hero-copy">
        The X-Ray layer for fastball carry. This board tracks raw induced vertical break, shape relative to velocity norms, and the dead-zone risk profile where ride flattens into damage.
      </p>
      <p class="ivb-active-audit-line">
        <strong>Active Audit Layer:</strong> Click any player card to inspect the full performance audit.
      </p>
    </section>

    <section class="top-metrics">
      <article class="metric-card">
        <div class="metric-label">Field Tilt</div>
        <div class="metric-value">{{ field_tilt_pct }}%</div>
        <div class="metric-note">Share of tracked arms in the current window carrying elite 18"+ IVB.</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">Tracked Arms</div>
        <div class="metric-value">{{ tracked_pitchers }}</div>
        <div class="metric-note">Pitchers meeting the fastball sample threshold in the last {{ lookback_days }} days.</div>
      </article>
      <article class="metric-card">
        <div class="metric-label">Dead Zone Count</div>
        <div class="metric-value">{{ dead_zone_count }}</div>
        <div class="metric-note">Pitchers sitting in the 12"–15" carry band, where contact quality risk rises.</div>
      </article>
    </section>

    <section class="main-grid">
      <article class="section-card">
        <div class="section-head">
          <div>
            <div class="section-kicker">Gradient Map</div>
            <h2 class="section-title">Pitch Shape Board</h2>
          </div>
          <div class="section-actions">
            <button class="field-guide-trigger" type="button" onclick="openFieldGuide()">Field Guide</button>
            <div class="section-badge">Top {{ heat_cards|length }}</div>
          </div>
        </div>

        <div class="heat-grid">
          {% for row in heat_cards %}
          <article
            class="heat-card js-player-card {{ row.heat_class }}"
            data-player-id="{{ row.player_id }}"
            data-player-name="{{ row.player_name }}"
            data-player-type="pitcher"
            data-player-team="{{ row.team }}"
            data-profile-url="/scout/{{ row.player_id }}/"
            data-source-tag="IVB_HEAT_MAP"
          >
            <div class="heat-rank">#{{ loop.index }} // {{ row.band_label }}</div>
            <h3 class="heat-name">{{ row.player_name }}</h3>
            <div class="heat-meta">{{ row.team }} {% if row.velocity_bucket %}// {{ row.velocity_bucket }}{% endif %}</div>

                        <div class="heat-header-tags">
              <span class="heat-band">{{ row.heat_tag }}</span>
              {% if row.climber_flag %}
              <span class="heat-climber">CLIMBER</span>
              {% endif %}
              {% if row.transition_badge %}
                {% if row.transition_badge == "ENTERED APEX" %}
                <span class="heat-transition apex">{{ row.transition_badge }}</span>
                {% elif row.transition_badge == "EXITED DEAD ZONE" %}
                <span class="heat-transition exit">{{ row.transition_badge }}</span>
                {% elif row.transition_badge == "ENTERED DEAD ZONE" %}
                <span class="heat-transition enter-dead">{{ row.transition_badge }}</span>
                {% else %}
                <span class="heat-transition">{{ row.transition_badge }}</span>
                {% endif %}
              {% endif %}
            </div>

            <div class="heat-action-row">
              <button
                type="button"
                class="heat-provision-btn js-add-to-roster"
                data-default-label="INITIATE TRACKING"
                data-player-id="{{ row.player_id }}"
                data-player-name="{{ row.player_name }}"
                data-player-type="pitcher"
                data-player-team="{{ row.team }}"
                data-profile-url="/scout/{{ row.player_id }}/"
                data-source-tag="IVB_HEAT_MAP"
              >INITIATE TRACKING</button>
            </div>

            <div class="heat-values">
              <div class="heat-value-box">
                <div class="heat-value-label">IVB Raw</div>
                <div class="heat-value">{{ row.ivb_raw }}</div>
              </div>
              <div class="heat-value-box">
                <div class="heat-value-label">IVB vs Avg</div>
                <div class="heat-value">{{ row.ivb_vs_avg }}</div>
              </div>
              <div class="heat-value-box">
                <div class="heat-value-label">VAA</div>
                <div class="heat-value">{{ row.vaa }}</div>
              </div>
              <div class="heat-value-box">
                <div class="heat-value-label">Whiff Prob</div>
                <div class="heat-value">{{ row.whiff_probability }}</div>
              </div>
              <div class="heat-value-box">
                <div class="heat-value-label">Dead Zone</div>
                <div class="heat-value">{{ row.dead_zone_label }}</div>
              </div>
            </div>

            {% if row.contact_risk %}
            <div class="heat-risk">{{ row.contact_risk }}</div>
            {% endif %}
            <div class="heat-brief">{{ row.brief }}</div>
          </article>
          {% endfor %}
        </div>

        <div class="lab-note">
          Version 1 uses fastball IVB from current Statcast movement inputs. VAA is scaffolded until the dedicated calculation layer is added upstream.
        </div>
      </article>
      <aside class="leader-card">
        <div class="leader-head">
          <div>
            <div class="section-kicker">Movement Shift</div>
            <h2 class="section-title">Climbers</h2>
          </div>
          <div class="section-badge">Recent vs Prior</div>
        </div>

        <div class="leader-list">
          {% for row in climbers %}
          <div class="leader-row positive">
            <div class="leader-name">{{ row.player_name }}</div>
            <div class="leader-sub">{{ row.team }} // {{ row.recent_label }}</div>
            <div class="leader-delta">{{ row.delta_label }}</div>
          </div>
          {% endfor %}
        </div>

        <div class="lab-note">
          Climbers compare the most recent half of the rolling window to the prior half.
        </div>

        <div class="leader-head" style="margin-top: 18px;">
          <div>
            <div class="section-kicker">Movement Fade</div>
            <h2 class="section-title">Fallers</h2>
          </div>
          <div class="section-badge">Recent vs Prior</div>
        </div>

        <div class="leader-list">
          {% if fallers %}
            {% for row in fallers %}
            <div class="leader-row negative">
              <div class="leader-name">{{ row.player_name }}</div>
              <div class="leader-sub">{{ row.team }} // {{ row.recent_label }}</div>
              <div class="leader-delta">{{ row.delta_label }}</div>
            </div>
            {% endfor %}
          {% else %}
            <div class="leader-row">
              <div class="leader-name">No major fallers</div>
              <div class="leader-sub">LAB // Recent vs prior window</div>
              <div class="leader-delta">--</div>
            </div>
          {% endif %}
        </div>

        <div class="lab-note">
          Fallers flag arms losing carry relative to their prior window.
        </div>

        <div class="leader-head" style="margin-top: 18px;">
          <div>
            <div class="section-kicker">Threshold Break</div>
            <h2 class="section-title">Entered Apex Rise</h2>
          </div>
          <div class="section-badge">Zone Transition</div>
        </div>

        <div class="leader-list">
          {% if entered_apex %}
            {% for row in entered_apex %}
            <div class="leader-row transition">
              <div class="leader-name">{{ row.player_name }}</div>
              <div class="leader-sub">{{ row.team }} // {{ row.transition_label }}</div>
              <div class="leader-sub">{{ row.detail_label }}</div>
              <div class="leader-delta">{{ row.delta_label }}</div>
            </div>
            {% endfor %}
          {% else %}
            <div class="leader-row">
              <div class="leader-name">No new apex entries</div>
              <div class="leader-sub">LAB // Zone transition</div>
              <div class="leader-delta">--</div>
            </div>
          {% endif %}
        </div>

        <div class="lab-note">
          These arms crossed into the 18"+ elite carry tier during the recent window.
        </div>

        <div class="leader-head" style="margin-top: 18px;">
          <div>
            <div class="section-kicker">Zone Shift</div>
            <h2 class="section-title">Dead Zone Changes</h2>
          </div>
          <div class="section-badge">Entry / Exit</div>
        </div>

        <div class="leader-list">
          {% if zone_shift %}
            {% for row in zone_shift %}
            <div class="leader-row transition">
              <div class="leader-name">{{ row.player_name }}</div>
              <div class="leader-sub">{{ row.team }} // {{ row.transition_label }}</div>
              <div class="leader-sub">{{ row.detail_label }}</div>
              <div class="leader-delta">{{ row.delta_label }}</div>
            </div>
            {% endfor %}
          {% else %}
            <div class="leader-row">
              <div class="leader-name">No dead-zone transitions</div>
              <div class="leader-sub">LAB // Entry / exit</div>
              <div class="leader-delta">--</div>
            </div>
          {% endif %}
        </div>

        <div class="lab-note">
          Tracks arms entering or escaping the 12"–15" dead-zone band.
        </div>
      </aside>
    </section>

    {{ footer_html | safe }}
  </div>

  <script src="/player-search.js"></script>
  <script src="/player-card-actions.js"></script>
  <script>
    function openFieldGuide() {
      const overlay = document.getElementById("fieldGuideOverlay");
      const modal = document.getElementById("fieldGuideModal");
      if (!overlay || !modal) return;
      overlay.classList.add("open");
      modal.classList.add("open");
      modal.setAttribute("aria-hidden", "false");
    }

    function closeFieldGuide() {
      const overlay = document.getElementById("fieldGuideOverlay");
      const modal = document.getElementById("fieldGuideModal");
      if (!overlay || !modal) return;
      overlay.classList.remove("open");
      modal.classList.remove("open");
      modal.setAttribute("aria-hidden", "true");
    }

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeFieldGuide();
      }
    });
  </script>
</body>
</html>
"""
)


def safe_float(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None
    
def get_supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)

def format_signed(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}{suffix}"


def format_plain(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "--"
    return f"{value:.1f}{suffix}"


def velocity_bucket_label(v: float | None) -> str:
    if v is None:
        return ""
    low = int(math.floor(v / 2.0) * 2)
    high = low + 1
    return f"{low}-{high} mph"


def heat_class(ivb: float | None) -> str:
    if ivb is None:
        return "neutral"
    if 12.0 <= ivb <= 15.0:
        return "cold"
    if ivb >= 18.0:
        return "hot"
    return "neutral"


def band_label(ivb: float | None) -> str:
    if ivb is None:
        return "Unclassified"
    if 12.0 <= ivb <= 15.0:
        return "Dead Zone"
    if ivb >= 18.0:
        return "Apex Carry"
    return "Standard Carry"


def heat_tag(ivb: float | None) -> str:
    if ivb is None:
        return "NO SIGNAL"
    if 12.0 <= ivb <= 15.0:
        return "COLD // DEAD ZONE"
    if ivb >= 18.0:
        return "HOT // APEX RISE"
    return "NEUTRAL // MLB RANGE"


def whiff_probability_label(ivb: float | None, vaa: float | None, ivb_vs_avg: float | None) -> str:
    if ivb is None:
        return "LOW"

    # Dead zone should always suppress whiff expectation
    if 12.0 <= ivb <= 15.0:
        return "LOW"

    # Premium carry + flatter approach angle
    if ivb >= 18.0 and vaa is not None and vaa >= -4.5:
        return "HIGH"

    # Strong carry, but not quite flat enough to be elite
    if ivb >= 18.0 and vaa is not None and vaa < -4.5:
        return "MEDIUM"

    # Shape beating the velo bucket baseline
    if ivb >= 17.0 and ivb_vs_avg is not None and ivb_vs_avg >= 1.0:
        return "MEDIUM"

    return "LOW"


def contact_risk_label(ivb: float | None) -> str:
    if ivb is None:
        return ""
    if 12.0 <= ivb <= 15.0:
        return "BARREL MAGNET // CONTACT RISK"
    return ""


def build_brief(ivb: float | None, ivb_vs_avg: float | None, vaa: float | None) -> str:
    parts = []

    if ivb is not None:
        if 12.0 <= ivb <= 15.0:
            parts.append("Carry profile sits inside the dead zone.")
        elif ivb >= 18.0:
            parts.append("Fastball shape enters apex-rise territory.")
        else:
            parts.append("Movement reads in the standard carry band.")

    if ivb_vs_avg is not None:
        if ivb_vs_avg >= 1.5:
            parts.append("Shape is beating the velo bucket baseline.")
        elif ivb_vs_avg <= -1.0:
            parts.append("Ride trails the baseline for this velocity band.")

    if vaa is None:
        parts.append("VAA layer pending full upstream calculation.")
    else:
        parts.append(f"Approach angle checks in at {vaa:.1f}°.")

    return " ".join(parts)

def approx_vaa(row) -> float | None:
    try:
        y0 = 50.0
        plate_y = 17.0 / 12.0

        vy0 = float(row["vy0"])
        ay = float(row["ay"])
        vz0 = float(row["vz0"])
        az = float(row["az"])

        a = 0.5 * ay
        b = vy0
        c = y0 - plate_y

        disc = b * b - 4 * a * c
        if disc < 0:
            return None

        if abs(a) < 1e-9:
            if abs(b) < 1e-9:
                return None
            t = -c / b
            if t <= 0:
                return None
        else:
            t1 = (-b - math.sqrt(disc)) / (2 * a)
            t2 = (-b + math.sqrt(disc)) / (2 * a)
            ts = [t for t in (t1, t2) if t > 0]
            if not ts:
                return None
            t = min(ts)

        vy_plate = vy0 + ay * t
        vz_plate = vz0 + az * t

        if vy_plate == 0:
            return None

        return math.degrees(math.atan(vz_plate / abs(vy_plate)))
    except Exception:
        return None
    
def fetch_statcast_window(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Fetching Statcast from {start_date} to {end_date}...")
    print("This is a large query, it may take a moment to complete")
    raw = statcast(start_dt=start_date, end_dt=end_date)
    if raw is None or raw.empty:
        return pd.DataFrame()
    return raw.copy()


def build_ivb_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    pitches = raw.copy()

    pitches["release_speed"] = pd.to_numeric(pitches.get("release_speed"), errors="coerce")
    pitches["pfx_z"] = pd.to_numeric(pitches.get("pfx_z"), errors="coerce")
    pitches["game_date"] = pd.to_datetime(pitches.get("game_date"), errors="coerce")
    pitches["game_pk"] = pd.to_numeric(pitches.get("game_pk"), errors="coerce").astype("Int64")
    pitches["pitcher"] = pd.to_numeric(pitches.get("pitcher"), errors="coerce").astype("Int64")
    pitches["vx0"] = pd.to_numeric(pitches.get("vx0"), errors="coerce")
    pitches["vy0"] = pd.to_numeric(pitches.get("vy0"), errors="coerce")
    pitches["vz0"] = pd.to_numeric(pitches.get("vz0"), errors="coerce")
    pitches["ax"] = pd.to_numeric(pitches.get("ax"), errors="coerce")
    pitches["ay"] = pd.to_numeric(pitches.get("ay"), errors="coerce")
    pitches["az"] = pd.to_numeric(pitches.get("az"), errors="coerce")

    fastball_types = {"FF", "SI", "FC", "FA"}
    pitches["pitch_type"] = pitches.get("pitch_type").astype(str)
    pitches["is_fastball"] = pitches["pitch_type"].isin(fastball_types)
    pitches = pitches[pitches["is_fastball"] == True].copy()

    pitches["ivb_inches"] = pitches["pfx_z"] * 12.0
    pitches["vaa"] = pitches.apply(approx_vaa, axis=1)
    pitches = pitches.dropna(subset=["pitcher", "game_pk", "release_speed", "ivb_inches", "game_date"]).copy()
    if pitches.empty:
        return pd.DataFrame(), pd.DataFrame()

    pitches["velocity_bucket_floor"] = (pitches["release_speed"] // 2 * 2).astype("Int64")
    bucket_avgs = (
        pitches.groupby("velocity_bucket_floor", dropna=True)["ivb_inches"]
        .mean()
        .reset_index()
        .rename(columns={"ivb_inches": "bucket_ivb_avg"})
    )

    pitches = pitches.merge(bucket_avgs, on="velocity_bucket_floor", how="left")
    pitches["ivb_vs_avg"] = pitches["ivb_inches"] - pitches["bucket_ivb_avg"]
    pitches["dead_zone_flag"] = pitches["ivb_inches"].between(12.0, 15.0, inclusive="both")

    grouped = (
        pitches.groupby("pitcher", dropna=True)
                .agg(
            player_name=("player_name", "last"),
            team=("home_team", "last"),
            release_speed=("release_speed", "mean"),
            ivb_raw=("ivb_inches", "mean"),
            ivb_vs_avg=("ivb_vs_avg", "mean"),
            vaa=("vaa", "mean"),
            pitch_count=("ivb_inches", "size"),
            dead_zone_flag=("dead_zone_flag", "max"),
            velocity_bucket_floor=("velocity_bucket_floor", "last"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["pitch_count"] >= MIN_FASTBALL_COUNT].copy()
    grouped["team"] = grouped["team"].fillna("TEAM")
   

    pitches = pitches[pitches["pitcher"].isin(grouped["pitcher"])].copy()
    return grouped, pitches

def build_pitcher_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    if pitches.empty:
        return pd.DataFrame()

    appearances = (
        pitches.groupby(["pitcher", "game_pk"], dropna=True)
        .agg(
            game_date=("game_date", "max"),
            player_name=("player_name", "last"),
            team=("home_team", "last"),
            ivb_raw=("ivb_inches", "mean"),
            vaa=("vaa", "mean"),
            pitch_count=("ivb_inches", "size"),
        )
        .reset_index()
    )

    appearances = appearances.sort_values(["pitcher", "game_date", "game_pk"], ascending=[True, False, False]).copy()
    return appearances

def build_climbers(grouped: pd.DataFrame, pitches: pd.DataFrame) -> tuple[list[dict], set[int], dict[int, float]]:
    if grouped.empty or pitches.empty:
        return [], set(), {}

    appearances = build_pitcher_appearances(pitches)
    if appearances.empty:
        return [], set(), {}

    rows = []
    climber_ids = set()
    climber_delta_map = {}

    for pitcher_id, pitcher_apps in appearances.groupby("pitcher", sort=False):
        pitcher_apps = pitcher_apps.sort_values(["game_date", "game_pk"], ascending=[False, False]).reset_index(drop=True)
        if len(pitcher_apps) < 2:
            continue

        recent_ivb = safe_float(pitcher_apps.loc[0, "ivb_raw"])
        prior_ivb = safe_float(pitcher_apps.loc[1, "ivb_raw"])
        if recent_ivb is None or prior_ivb is None:
            continue

        delta = recent_ivb - prior_ivb

        try:
            pitcher_id_int = int(pitcher_id)
        except Exception:
            continue

        climber_delta_map[pitcher_id_int] = delta

        if delta > CLIMBER_THRESHOLD:
            climber_ids.add(pitcher_id_int)

        rows.append(
            {
                "pitcher": pitcher_id_int,
                "player_name": pitcher_apps.loc[0, "player_name"] or "Unknown",
                "team": pitcher_apps.loc[0, "team"] or "TEAM",
                "recent_label": "Last appearance vs prior",
                "delta": delta,
                "delta_label": format_signed(delta, '"'),
            }
        )

    rows = sorted(rows, key=lambda r: r["delta"], reverse=True)
    top_rows = rows[:6]

    display_rows = [
        {
            "player_name": row["player_name"],
            "team": row["team"],
            "recent_label": row["recent_label"],
            "delta_label": row["delta_label"],
        }
        for row in top_rows
    ]

    return display_rows, climber_ids, climber_delta_map

def build_fallers(grouped: pd.DataFrame, pitches: pd.DataFrame) -> list[dict]:
    if grouped.empty or pitches.empty:
        return []

    appearances = build_pitcher_appearances(pitches)
    if appearances.empty:
        return []

    rows = []

    for pitcher_id, pitcher_apps in appearances.groupby("pitcher", sort=False):
        pitcher_apps = pitcher_apps.sort_values(["game_date", "game_pk"], ascending=[False, False]).reset_index(drop=True)
        if len(pitcher_apps) < 2:
            continue

        recent_ivb = safe_float(pitcher_apps.loc[0, "ivb_raw"])
        prior_ivb = safe_float(pitcher_apps.loc[1, "ivb_raw"])
        if recent_ivb is None or prior_ivb is None:
            continue

        delta = recent_ivb - prior_ivb

        rows.append(
            {
                "player_name": pitcher_apps.loc[0, "player_name"] or "Unknown",
                "team": pitcher_apps.loc[0, "team"] or "TEAM",
                "recent_label": "Last appearance vs prior",
                "delta": delta,
                "delta_label": format_signed(delta, '"'),
            }
        )

    rows = sorted(rows, key=lambda r: r["delta"])
    top_rows = rows[:6]

    return [
        {
            "player_name": row["player_name"],
            "team": row["team"],
            "recent_label": row["recent_label"],
            "delta_label": row["delta_label"],
        }
        for row in top_rows
    ]

def build_zone_transitions(grouped: pd.DataFrame, pitches: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    if grouped.empty or pitches.empty:
        return [], []

    appearances = build_pitcher_appearances(pitches)
    if appearances.empty:
        return [], []

    entered_apex = []
    zone_shift = []

    for pitcher_id, pitcher_apps in appearances.groupby("pitcher", sort=False):
        pitcher_apps = pitcher_apps.sort_values(["game_date", "game_pk"], ascending=[False, False]).reset_index(drop=True)
        if len(pitcher_apps) < 2:
            continue

        recent_ivb = safe_float(pitcher_apps.loc[0, "ivb_raw"])
        prior_ivb = safe_float(pitcher_apps.loc[1, "ivb_raw"])
        if recent_ivb is None or prior_ivb is None:
            continue

        player_name = pitcher_apps.loc[0, "player_name"] or "Unknown"
        team = pitcher_apps.loc[0, "team"] or "TEAM"
        delta = recent_ivb - prior_ivb

        if prior_ivb < 18.0 and recent_ivb >= 18.0:
            entered_apex.append(
                {
                    "player_name": player_name,
                    "team": team,
                    "transition_label": "Entered Apex Rise",
                    "detail_label": f'{prior_ivb:.1f}" -> {recent_ivb:.1f}"',
                    "delta_label": format_signed(delta, '"'),
                    "delta": delta,
                }
            )

        if 12.0 <= prior_ivb <= 15.0 and not (12.0 <= recent_ivb <= 15.0):
            zone_shift.append(
                {
                    "player_name": player_name,
                    "team": team,
                    "transition_label": "Exited Dead Zone",
                    "detail_label": f'{prior_ivb:.1f}" -> {recent_ivb:.1f}"',
                    "delta_label": format_signed(delta, '"'),
                    "delta": delta,
                }
            )
        elif not (12.0 <= prior_ivb <= 15.0) and (12.0 <= recent_ivb <= 15.0):
            zone_shift.append(
                {
                    "player_name": player_name,
                    "team": team,
                    "transition_label": "Entered Dead Zone",
                    "detail_label": f'{prior_ivb:.1f}" -> {recent_ivb:.1f}"',
                    "delta_label": format_signed(delta, '"'),
                    "delta": delta,
                }
            )

    entered_apex = sorted(entered_apex, key=lambda r: r["delta"], reverse=True)[:6]
    zone_shift = sorted(zone_shift, key=lambda r: abs(r["delta"]), reverse=True)[:6]

    entered_apex = [
        {
            "player_name": row["player_name"],
            "team": row["team"],
            "transition_label": row["transition_label"],
            "detail_label": row["detail_label"],
            "delta_label": row["delta_label"],
        }
        for row in entered_apex
    ]

    zone_shift = [
        {
            "player_name": row["player_name"],
            "team": row["team"],
            "transition_label": row["transition_label"],
            "detail_label": row["detail_label"],
            "delta_label": row["delta_label"],
        }
        for row in zone_shift
    ]

    return entered_apex, zone_shift

def build_lab_rows(
    grouped: pd.DataFrame,
    climber_ids: set[int],
    climber_delta_map: dict[int, float],
    entered_apex_ids: set[int],
    entered_dead_zone_ids: set[int],
    exited_dead_zone_ids: set[int],
    report_date_value,
) -> list[dict]:
    if grouped.empty:
        return []

    rows = []

    for _, row in grouped.iterrows():
        player_id = row.get("pitcher")
        try:
            player_id_int = int(player_id)
        except Exception:
            continue

        ivb_raw = safe_float(row.get("ivb_raw"))
        ivb_vs_avg = safe_float(row.get("ivb_vs_avg"))
        vaa = safe_float(row.get("vaa"))
        avg_fastball_velo = safe_float(row.get("release_speed"))
        pitch_count = row.get("pitch_count")
        pitch_count_int = int(pitch_count) if pd.notna(pitch_count) else None

        dead_zone_flag = bool(row.get("dead_zone_flag"))
        contact_risk_flag = bool(contact_risk_label(ivb_raw))
        climber_delta = climber_delta_map.get(player_id_int)
        entered_apex_rise = player_id_int in entered_apex_ids
        entered_dead_zone = player_id_int in entered_dead_zone_ids
        exited_dead_zone = player_id_int in exited_dead_zone_ids

        rows.append(
            {
                "report_date": str(report_date_value),
                "player_id": player_id_int,
                "player_name": row.get("player_name", "Unknown Pitcher"),
                "team": row.get("team", "TEAM"),
                "pitch_count": pitch_count_int,
                "avg_fastball_velo": avg_fastball_velo,
                "ivb_raw": ivb_raw,
                "ivb_vs_avg": ivb_vs_avg,
                "dead_zone_flag": dead_zone_flag,
                "contact_risk_flag": contact_risk_flag,
                "whiff_probability": whiff_probability_label(ivb_raw, vaa, ivb_vs_avg),
                "climber_delta": climber_delta,
                "climber_flag": player_id_int in climber_ids,
                "entered_apex_rise": entered_apex_rise,
                "entered_dead_zone": entered_dead_zone,
                "exited_dead_zone": exited_dead_zone,
                "heat_band": band_label(ivb_raw),
                "vaa": vaa,
            }
        )

    return rows


def upsert_lab_rows(rows: list[dict]) -> None:
    if not rows:
        return

    client = get_supabase_client()
    client.table(IVB_LAB_TABLE).upsert(
        rows,
        on_conflict="report_date,player_id",
    ).execute()


def fetch_latest_lab_rows(report_date_value) -> list[dict]:
    client = get_supabase_client()
    response = (
        client.table(IVB_LAB_TABLE)
        .select("*")
        .eq("report_date", str(report_date_value))
        .order("ivb_vs_avg", desc=True)
        .limit(200)
        .execute()
    )
    return response.data or []


def lab_rows_to_cards(rows: list[dict]) -> list[dict]:
    cards = []

    for row in rows[:12]:
        ivb_raw = safe_float(row.get("ivb_raw"))
        ivb_vs_avg = safe_float(row.get("ivb_vs_avg"))
        vaa = safe_float(row.get("vaa"))

        transition_badge = ""
        if bool(row.get("entered_apex_rise")):
            transition_badge = "ENTERED APEX"
        elif bool(row.get("exited_dead_zone")):
            transition_badge = "EXITED DEAD ZONE"
        elif bool(row.get("entered_dead_zone")):
            transition_badge = "ENTERED DEAD ZONE"

        cards.append(
            {
                "player_id": row.get("player_id"),
                "player_name": row.get("player_name", "Unknown Pitcher"),
                "team": row.get("team", "TEAM"),
                "ivb_raw": format_plain(ivb_raw, '"'),
                "ivb_vs_avg": format_signed(ivb_vs_avg, '"'),
                "vaa": "--" if vaa is None else format_plain(vaa, "°"),
                "dead_zone_label": "COLD" if bool(row.get("dead_zone_flag")) else "CLEAR",
                "whiff_probability": row.get("whiff_probability") or "LOW",
                "climber_flag": bool(row.get("climber_flag")),
                "contact_risk": "BARREL MAGNET // CONTACT RISK" if bool(row.get("contact_risk_flag")) else "",
                "heat_class": heat_class(ivb_raw),
                "band_label": row.get("heat_band") or band_label(ivb_raw),
                "heat_tag": heat_tag(ivb_raw),
                "velocity_bucket": "",
                "transition_badge": transition_badge,
                "brief": build_brief(ivb_raw, ivb_vs_avg, vaa),
            }
        )

    return cards

def to_cards(grouped: pd.DataFrame, climber_ids: set[int]) -> list[dict]:
    if grouped.empty:
        return []

    ordered = grouped.sort_values(["ivb_vs_avg", "ivb_raw"], ascending=[False, False]).head(12).copy()
    rows = []

    for _, row in ordered.iterrows():
        pitcher_id = row.get("pitcher")
        ivb_raw = safe_float(row.get("ivb_raw"))
        ivb_delta = safe_float(row.get("ivb_vs_avg"))
        vaa = safe_float(row.get("vaa"))
        bucket_floor = safe_float(row.get("velocity_bucket_floor"))

        bucket_label = ""
        if bucket_floor is not None:
            bucket_label = velocity_bucket_label(bucket_floor)

        pitcher_id_int = None
        try:
            pitcher_id_int = int(pitcher_id)
        except Exception:
            pitcher_id_int = None

        rows.append(
            {
                "player_id": pitcher_id_int,
                "player_name": row.get("player_name", "Unknown Pitcher"),
                "team": row.get("team", "TEAM"),
                "ivb_raw": format_plain(ivb_raw, '"'),
                "ivb_vs_avg": format_signed(ivb_delta, '"'),
                "vaa": "--" if vaa is None else format_plain(vaa, "°"),
                "dead_zone_label": "COLD" if bool(row.get("dead_zone_flag")) else "CLEAR",
                "whiff_probability": whiff_probability_label(ivb_raw, vaa, ivb_delta),
                "climber_flag": pitcher_id_int in climber_ids if pitcher_id_int is not None else False,
                "contact_risk": contact_risk_label(ivb_raw),
                "heat_class": heat_class(ivb_raw),
                "band_label": band_label(ivb_raw),
                "heat_tag": heat_tag(ivb_raw),
                "velocity_bucket": bucket_label,
                "transition_badge": "",
                "brief": build_brief(ivb_raw, ivb_delta, vaa),
            }
        )

    return rows


def build_zone_transition_id_sets(pitches: pd.DataFrame) -> tuple[set[int], set[int], set[int]]:
    entered_apex_ids: set[int] = set()
    entered_dead_zone_ids: set[int] = set()
    exited_dead_zone_ids: set[int] = set()

    if pitches.empty:
        return entered_apex_ids, entered_dead_zone_ids, exited_dead_zone_ids

    appearances = build_pitcher_appearances(pitches)
    if appearances.empty:
        return entered_apex_ids, entered_dead_zone_ids, exited_dead_zone_ids

    for pitcher_id, pitcher_apps in appearances.groupby("pitcher", sort=False):
        pitcher_apps = pitcher_apps.sort_values(["game_date", "game_pk"], ascending=[False, False]).reset_index(drop=True)
        if len(pitcher_apps) < 2:
            continue

        recent_ivb = safe_float(pitcher_apps.loc[0, "ivb_raw"])
        prior_ivb = safe_float(pitcher_apps.loc[1, "ivb_raw"])
        if recent_ivb is None or prior_ivb is None:
            continue

        try:
            pitcher_id_int = int(pitcher_id)
        except Exception:
            continue

        if prior_ivb < 18.0 and recent_ivb >= 18.0:
            entered_apex_ids.add(pitcher_id_int)

        if 12.0 <= prior_ivb <= 15.0 and not (12.0 <= recent_ivb <= 15.0):
            exited_dead_zone_ids.add(pitcher_id_int)
        elif not (12.0 <= prior_ivb <= 15.0) and (12.0 <= recent_ivb <= 15.0):
            entered_dead_zone_ids.add(pitcher_id_int)

    return entered_apex_ids, entered_dead_zone_ids, exited_dead_zone_ids



def write_ivb_heat_map_status(
    *,
    build_started_at: str,
    build_finished_at: str,
    source_updated_at: str,
    heat_cards,
    climbers,
    dead_zone_count,
    raw_row_count: int,
    grouped_pitcher_count: int,
    lab_row_count: int,
    latest_lab_row_count: int,
    used_lab_readback: bool,
    used_direct_grouped_fallback: bool,
) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

    tile_count = int(len(heat_cards)) if heat_cards is not None else 0
    climber_count = int(len(climbers)) if climbers is not None else 0
    dead_zone_count = int(dead_zone_count or 0)
    degraded = tile_count == 0 or raw_row_count == 0 or grouped_pitcher_count == 0
    used_fallback = bool(used_direct_grouped_fallback or tile_count == 0)

    status_payload = build_report_status(
        "ivb_heat_map",
        build_success=not degraded,
        threshold_minutes=2880,
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        source_updated_at=source_updated_at,
        section_counts={
            "ivb_tiles": tile_count,
            "ivb_climbers": climber_count,
            "dead_zone": dead_zone_count,
            "raw_statcast_rows": int(raw_row_count),
            "grouped_pitchers": int(grouped_pitcher_count),
            "lab_rows_written": int(lab_row_count),
            "lab_rows_read": int(latest_lab_row_count),
        },
        used_fallback=used_fallback,
        degraded=degraded,
        notes=[
            (
                f"IVB Heat Map built with {tile_count} active tiles, {climber_count} climbers, "
                f"{dead_zone_count} dead-zone flags, {raw_row_count} raw Statcast rows, "
                f"{grouped_pitcher_count} grouped pitchers, {lab_row_count} lab rows written, "
                f"and {latest_lab_row_count} lab rows read."
            ),
            f"IVB source path: {'Supabase lab readback' if used_lab_readback else 'direct grouped Statcast fallback'}.",
        ],
    )

    status_payload["mode"] = IVB_HEAT_MAP_STATUS_MODE
    status_payload["pipeline_layers"] = IVB_HEAT_MAP_PIPELINE_LAYERS

    IVB_HEAT_MAP_STATUS_PATH.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote IVB Heat Map status -> {IVB_HEAT_MAP_STATUS_PATH}")


def write_ivb_heat_map() -> None:
    build_started_at = datetime.now(timezone.utc).isoformat()
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    IVB_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    raw = fetch_statcast_window(str(start_date), str(end_date))
    grouped, pitches = build_ivb_dataset(raw)

    raw_row_count = int(len(raw)) if raw is not None else 0
    grouped_pitcher_count = int(len(grouped)) if grouped is not None else 0

    if pitches is not None and not pitches.empty and "game_date" in pitches.columns:
        latest_game_date = pitches["game_date"].max()
        if pd.notna(latest_game_date):
            # Statcast game_date is date-only, not event-time precise.
            # Treat the latest available game date as end-of-day UTC so freshness
            # does not falsely degrade a healthy daily feed at midnight.
            source_updated_at = (
                latest_game_date.to_pydatetime()
                .replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc)
                .isoformat()
            )
        else:
            source_updated_at = build_started_at
    else:
        source_updated_at = build_started_at

    tracked_pitchers = int(len(grouped))
    dead_zone_count = int(grouped["dead_zone_flag"].fillna(False).sum()) if not grouped.empty else 0
    elite_count = int((grouped["ivb_raw"].fillna(-999) >= 18.0).sum()) if not grouped.empty else 0
    field_tilt_pct = 0 if tracked_pitchers == 0 else round((elite_count / tracked_pitchers) * 100)

    climbers, climber_ids, climber_delta_map = build_climbers(grouped, pitches)
    fallers = build_fallers(grouped, pitches)
    entered_apex, zone_shift = build_zone_transitions(grouped, pitches)
    entered_apex_ids, entered_dead_zone_ids, exited_dead_zone_ids = build_zone_transition_id_sets(pitches)

    lab_rows = build_lab_rows(
        grouped,
        climber_ids,
        climber_delta_map,
        entered_apex_ids,
        entered_dead_zone_ids,
        exited_dead_zone_ids,
        end_date,
    )
    lab_row_count = int(len(lab_rows))
    upsert_lab_rows(lab_rows)

    latest_lab_rows = fetch_latest_lab_rows(end_date)
    latest_lab_row_count = int(len(latest_lab_rows))
    used_lab_readback = latest_lab_row_count > 0
    used_direct_grouped_fallback = not used_lab_readback

    heat_cards = lab_rows_to_cards(latest_lab_rows) if used_lab_readback else to_cards(grouped, climber_ids)

    if not climbers:
        climbers = [
            {
                "player_name": "No signal yet",
                "team": "LAB",
                "recent_label": "Awaiting additional data",
                "delta_label": "--",
            }
        ]

    if not fallers:
        fallers = [
            {
                "player_name": "No major fallers",
                "team": "LAB",
                "recent_label": "Last appearance vs prior",
                "delta_label": "--",
            }
        ]

    html = HTML_TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        nav_html=(
            Template(DESKTOP_NAV_TEMPLATE).render(active_nav="ivb_heat_map")
            + "\n"
            + Template(MOBILE_NAV_TEMPLATE).render(active_nav="ivb_heat_map")
        ),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
        field_tilt_pct=field_tilt_pct,
        tracked_pitchers=tracked_pitchers,
        dead_zone_count=dead_zone_count,
        heat_cards=heat_cards,
        climbers=climbers,
        fallers=fallers,
        entered_apex=entered_apex,
        zone_shift=zone_shift,
        lookback_days=LOOKBACK_DAYS,
    )

    (IVB_DIR / "index.html").write_text(html, encoding="utf-8")
    print("Wrote dist/ivb-heat-map/index.html")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "source_window_start": str(start_date),
        "source_window_end": str(end_date),
        "source_updated_at": source_updated_at,
        "raw_row_count": raw_row_count,
        "grouped_pitcher_count": grouped_pitcher_count,
        "lab_row_count": lab_row_count,
        "latest_lab_row_count": latest_lab_row_count,
        "used_lab_readback": used_lab_readback,
        "used_direct_grouped_fallback": used_direct_grouped_fallback,
        "field_tilt_pct": field_tilt_pct,
        "tracked_pitchers": tracked_pitchers,
        "dead_zone_count": dead_zone_count,
        "heat_cards": heat_cards,
        "climbers": climbers,
        "fallers": fallers,
        "entered_apex": entered_apex,
        "zone_shift": zone_shift,
    }
    (DIST_DIR / "ivb_heat_map.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote dist/ivb_heat_map.json")

    build_finished_at = datetime.now(timezone.utc).isoformat()
    write_ivb_heat_map_status(
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        source_updated_at=source_updated_at,
        heat_cards=heat_cards,
        climbers=climbers,
        dead_zone_count=dead_zone_count,
        raw_row_count=raw_row_count,
        grouped_pitcher_count=grouped_pitcher_count,
        lab_row_count=lab_row_count,
        latest_lab_row_count=latest_lab_row_count,
        used_lab_readback=used_lab_readback,
        used_direct_grouped_fallback=used_direct_grouped_fallback,
    )


if __name__ == "__main__":
    write_ivb_heat_map()