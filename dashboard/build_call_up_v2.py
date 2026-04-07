from __future__ import annotations

from datetime import datetime
from pathlib import Path

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
SCORE_VERSION = "EDGE_v2.0"

def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / std


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
    hitters["source_badge"] = SOURCE_BADGE
    hitters["score_version"] = SCORE_VERSION

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

    return pitchers.sort_values(["edge_score", "bf"], ascending=[False, False]).reset_index(drop=True)
    return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / std

def safe_name(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return "Unknown"
    return " ".join(part.capitalize() for part in text.split())



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
          Clean rebuild. Shared shell. Shared nav. Default windows locked to 72 HR and 14 DAY.
          Legacy layout and variant complexity intentionally removed.
        </p>
      </div>

      <div class="summary-card">
        <div>
          <div class="summary-label">Window</div>
          <div class="summary-value">72 HR</div>
        </div>
        <div>
          <div class="summary-label">Mode</div>
          <div class="summary-value">AAA</div>
        </div>
        <div>
          <div class="summary-label">Timezone</div>
          <div class="summary-value">{{ timezone_label }}</div>
        </div>
      </div>
    </section>

          <section class="section-card">
        <div class="tabs">
          <div class="tab active">72 HR</div>
          <div class="tab">14 DAY</div>
        </div>

        <h2 class="section-title">Promotion Watch Board</h2>

        {% if total_signals == 0 %}
        <div class="placeholder">
          No live AAA promotion-watch signals available yet.
        </div>
        {% else %}
        <div style="display:grid; gap:18px;">
          <div>
            <div class="eyebrow" style="margin-bottom:10px;">Hitters</div>
            <div style="display:grid; gap:12px;">
              {% for row in hitters %}
              <article style="border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:16px; background:rgba(255,255,255,0.02);">
                <div style="display:flex; justify-content:space-between; gap:16px; align-items:flex-start; flex-wrap:wrap;">
                  <div>
                    <div style="font-size:20px; font-weight:800; letter-spacing:-0.03em;">{{ row.player_name }}</div>
                    <div style="margin-top:4px; font-family:var(--mono); font-size:11px; color:var(--soft); text-transform:uppercase; letter-spacing:0.08em;">
                      {{ row.signal_type }} • {{ row.sample_note }}
                    </div>
                  </div>
                  <div style="text-align:right;">
                    <div class="summary-label">Signal Score</div>
                    <div style="font-size:24px; font-weight:800; line-height:1;">{{ row.edge_score }}</div>
                  </div>
                </div>

                <div style="margin-top:12px; color:var(--soft); font-size:14px; line-height:1.55;">
                  {{ row.why }}
                </div>

                <div style="margin-top:14px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px;">
                  <div style="border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:10px; background:rgba(255,255,255,0.02);">
                    <div class="summary-label">{{ row.metric_1_label }}</div>
                    <div style="margin-top:6px; font-size:18px; font-weight:800;">{{ row.metric_1 }}</div>
                  </div>
                  <div style="border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:10px; background:rgba(255,255,255,0.02);">
                    <div class="summary-label">{{ row.metric_2_label }}</div>
                    <div style="margin-top:6px; font-size:18px; font-weight:800;">{{ row.metric_2 }}</div>
                  </div>
                  <div style="border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:10px; background:rgba(255,255,255,0.02);">
                    <div class="summary-label">{{ row.metric_3_label }}</div>
                    <div style="margin-top:6px; font-size:18px; font-weight:800;">{{ row.metric_3 }}</div>
                  </div>
                </div>
              </article>
              {% endfor %}
            </div>
          </div>

          <div>
            <div class="eyebrow" style="margin-bottom:10px;">Pitchers</div>
            <div style="display:grid; gap:12px;">
              {% for row in pitchers %}
              <article style="border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:16px; background:rgba(255,255,255,0.02);">
                <div style="display:flex; justify-content:space-between; gap:16px; align-items:flex-start; flex-wrap:wrap;">
                  <div>
                    <div style="font-size:20px; font-weight:800; letter-spacing:-0.03em;">{{ row.player_name }}</div>
                    <div style="margin-top:4px; font-family:var(--mono); font-size:11px; color:var(--soft); text-transform:uppercase; letter-spacing:0.08em;">
                      {{ row.signal_type }} • {{ row.sample_note }}
                    </div>
                  </div>
                  <div style="text-align:right;">
                    <div class="summary-label">Signal Score</div>
                    <div style="font-size:24px; font-weight:800; line-height:1;">{{ row.edge_score }}</div>
                  </div>
                </div>

                <div style="margin-top:12px; color:var(--soft); font-size:14px; line-height:1.55;">
                  {{ row.why }}
                </div>

                <div style="margin-top:14px; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px;">
                  <div style="border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:10px; background:rgba(255,255,255,0.02);">
                    <div class="summary-label">{{ row.metric_1_label }}</div>
                    <div style="margin-top:6px; font-size:18px; font-weight:800;">{{ row.metric_1 }}</div>
                  </div>
                  <div style="border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:10px; background:rgba(255,255,255,0.02);">
                    <div class="summary-label">{{ row.metric_2_label }}</div>
                    <div style="margin-top:6px; font-size:18px; font-weight:800;">{{ row.metric_2 }}</div>
                  </div>
                  <div style="border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:10px; background:rgba(255,255,255,0.02);">
                    <div class="summary-label">{{ row.metric_3_label }}</div>
                    <div style="margin-top:6px; font-size:18px; font-weight:800;">{{ row.metric_3 }}</div>
                  </div>
                </div>
              </article>
              {% endfor %}
            </div>
          </div>
        </div>
        {% endif %}
      </section>

    {{ footer_html | safe }}
  </div>
</body>
</html>
"""
)
def fetch_latest_aaa_weekly_signal_base() -> pd.DataFrame:
    from supabase import create_client
    import os

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
        return pd.DataFrame()

    latest_week = rows[0]["week_start"]

    data_resp = (
        sb.table("milb_aaa_weekly_signal_base")
        .select("*")
        .eq("week_start", latest_week)
        .execute()
    )

    data = data_resp.data or []
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def load_source_frame() -> pd.DataFrame:
    return fetch_latest_aaa_weekly_signal_base()


def render_html() -> str:
    df = load_source_frame()

    hitters = build_aaa_hitter_promotion_watch(df).head(6) if not df.empty else pd.DataFrame()
    pitchers = build_aaa_pitcher_promotion_watch(df).head(6) if not df.empty else pd.DataFrame()

    total_signals = len(hitters) + len(pitchers)

    return HTML_TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        timezone_label=TIMEZONE_LABEL,
        nav_html=Template(NAV_TEMPLATE).render(active_nav="promotion_watch"),
        search_html=SEARCH_TEMPLATE,
        footer_html=FOOTER_TEMPLATE,
        shell_styles=SHELL_STYLES_TEMPLATE,
        total_signals=total_signals,
        hitters=hitters.to_dict(orient="records"),
        pitchers=pitchers.to_dict(orient="records"),
    )
def main() -> None:
    CALL_UP_DIR.mkdir(parents=True, exist_ok=True)
    html = render_html()
    output_path = CALL_UP_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")

if __name__ == "__main__":
    main()