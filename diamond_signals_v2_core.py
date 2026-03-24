#!/usr/bin/env python3
import os
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from jinja2 import Template
from pybaseball import statcast

DIST_DIR = Path("dist")
DIST_DIR.mkdir(parents=True, exist_ok=True)

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "65"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
SITE_URL = os.getenv("SITE_URL", "").strip()
TIMEZONE_LABEL = os.getenv("TIMEZONE_LABEL", "America/New_York")

TODAY = date.today()
START_DATE = TODAY - timedelta(days=28)
END_DATE = TODAY


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s.mean()) / std


def safe_name(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    return str(value).strip()


def markdown_escape(text: str) -> str:
    specials = r"_*[]()~`>#+-=|{}.!"
    out = str(text)
    for ch in specials:
        out = out.replace(ch, f"\\{ch}")
    return out


def fetch_statcast_window(start_dt: date, end_dt: date) -> pd.DataFrame:
    print(f"Fetching Statcast from {start_dt} to {end_dt}...")
    df = statcast(start_dt=start_dt.strftime("%Y-%m-%d"), end_dt=end_dt.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        raise RuntimeError("Statcast returned no data for the requested window.")
    return df


def build_hitter_signals(df: pd.DataFrame) -> pd.DataFrame:
    hitters = df.copy()
    bbe = hitters[hitters["launch_speed"].notna()].copy()
    if bbe.empty:
        return pd.DataFrame()

    bbe["game_date"] = pd.to_datetime(bbe["game_date"])
    bbe["is_recent"] = bbe["game_date"] >= pd.Timestamp(TODAY - timedelta(days=7))
    bbe["is_baseline"] = (bbe["game_date"] < pd.Timestamp(TODAY - timedelta(days=7))) & (
        bbe["game_date"] >= pd.Timestamp(TODAY - timedelta(days=28))
    )

    bbe["barrel_like"] = (
        (pd.to_numeric(bbe["launch_speed"], errors="coerce") >= 98) &
        (pd.to_numeric(bbe["launch_angle"], errors="coerce").between(26, 30, inclusive="both"))
    ).astype(int)

    recent = (
        bbe[bbe["is_recent"]]
        .groupby(["batter", "player_name"], dropna=False)
        .agg(
            recent_bbe=("launch_speed", "size"),
            recent_ev=("launch_speed", "mean"),
            recent_max_ev=("launch_speed", "max"),
            recent_barrel_rate=("barrel_like", "mean"),
        )
        .reset_index()
    )

    baseline = (
        bbe[bbe["is_baseline"]]
        .groupby(["batter", "player_name"], dropna=False)
        .agg(
            baseline_bbe=("launch_speed", "size"),
            baseline_ev=("launch_speed", "mean"),
            baseline_max_ev=("launch_speed", "max"),
            baseline_barrel_rate=("barrel_like", "mean"),
        )
        .reset_index()
    )

    merged = recent.merge(
        baseline,
        on=["batter", "player_name"],
        how="left",
        suffixes=("", "_base")
    )

    merged = merged[merged["recent_bbe"] >= 4].copy()
    merged["baseline_bbe"] = merged["baseline_bbe"].fillna(0)

    merged["ev_delta"] = merged["recent_ev"] - merged["baseline_ev"].fillna(merged["recent_ev"])
    merged["barrel_rate_delta"] = merged["recent_barrel_rate"] - merged["baseline_barrel_rate"].fillna(merged["recent_barrel_rate"])
    merged["quality_index"] = (
        0.50 * zscore(merged["recent_ev"]) +
        0.25 * zscore(merged["recent_max_ev"]) +
        0.25 * zscore(merged["recent_barrel_rate"])
    )
    merged["delta_index"] = (
        0.65 * zscore(merged["ev_delta"]) +
        0.35 * zscore(merged["barrel_rate_delta"])
    )
    merged["edge_score"] = (
        50 +
        18 * merged["quality_index"] +
        14 * merged["delta_index"] +
        4 * zscore(merged["recent_bbe"])
    ).clip(1, 99).round(1)

    merged["player_name"] = merged["player_name"].apply(safe_name)
    merged["signal_type"] = "Hitter"
    merged["why"] = merged.apply(
        lambda r: (
            f"Avg EV {r['recent_ev']:.1f} mph "
            f"({r['ev_delta']:+.1f} vs baseline), "
            f"barrel-like rate {100 * r['recent_barrel_rate']:.1f}%."
        ),
        axis=1
    )
    merged["metric_1"] = merged["recent_ev"].round(1)
    merged["metric_1_label"] = "Avg EV"
    merged["metric_2"] = (100 * merged["recent_barrel_rate"]).round(1)
    merged["metric_2_label"] = "Barrel-like %"
    merged["metric_3"] = merged["recent_max_ev"].round(1)
    merged["metric_3_label"] = "Max EV"

    return merged.sort_values("edge_score", ascending=False).reset_index(drop=True)


def build_pitcher_signals(df: pd.DataFrame) -> pd.DataFrame:
    pitchers = df.copy()
    pitchers["game_date"] = pd.to_datetime(pitchers["game_date"])
    pitchers["is_whiff"] = pitchers["description"].isin(["swinging_strike", "swinging_strike_blocked"]).astype(int)

    fastballs = {"FF", "FT", "SI", "FC"}
    pitchers["is_fastball"] = pitchers["pitch_type"].isin(fastballs).astype(int)

    pitchers["is_recent"] = pitchers["game_date"] >= pd.Timestamp(TODAY - timedelta(days=7))
    pitchers["is_baseline"] = (pitchers["game_date"] < pd.Timestamp(TODAY - timedelta(days=7))) & (
        pitchers["game_date"] >= pd.Timestamp(TODAY - timedelta(days=28))
    )

    recent = (
        pitchers[pitchers["is_recent"]]
        .groupby(["pitcher", "player_name"], dropna=False)
        .agg(
            recent_pitches=("pitch_type", "size"),
            recent_whiff_rate=("is_whiff", "mean"),
            recent_fb_velo=("release_speed", lambda s: pd.to_numeric(s[pitchers.loc[s.index, "is_fastball"] == 1], errors="coerce").mean()),
            recent_extension=("release_extension", "mean"),
        )
        .reset_index()
    )

    baseline = (
        pitchers[pitchers["is_baseline"]]
        .groupby(["pitcher", "player_name"], dropna=False)
        .agg(
            baseline_pitches=("pitch_type", "size"),
            baseline_whiff_rate=("is_whiff", "mean"),
            baseline_fb_velo=("release_speed", lambda s: pd.to_numeric(s[pitchers.loc[s.index, "is_fastball"] == 1], errors="coerce").mean()),
            baseline_extension=("release_extension", "mean"),
        )
        .reset_index()
    )

    merged = recent.merge(baseline, on=["pitcher", "player_name"], how="left")
    merged = merged[merged["recent_pitches"] >= 40].copy()

    merged["velo_delta"] = merged["recent_fb_velo"] - merged["baseline_fb_velo"].fillna(merged["recent_fb_velo"])
    merged["whiff_delta"] = merged["recent_whiff_rate"] - merged["baseline_whiff_rate"].fillna(merged["recent_whiff_rate"])
    merged["extension_delta"] = merged["recent_extension"] - merged["baseline_extension"].fillna(merged["recent_extension"])

    merged["quality_index"] = (
        0.50 * zscore(merged["recent_whiff_rate"]) +
        0.30 * zscore(merged["recent_fb_velo"]) +
        0.20 * zscore(merged["recent_extension"])
    )
    merged["delta_index"] = (
        0.50 * zscore(merged["velo_delta"]) +
        0.35 * zscore(merged["whiff_delta"]) +
        0.15 * zscore(merged["extension_delta"])
    )
    merged["edge_score"] = (
        50 +
        16 * merged["quality_index"] +
        16 * merged["delta_index"] +
        3 * zscore(merged["recent_pitches"])
    ).clip(1, 99).round(1)

    merged["player_name"] = merged["player_name"].apply(safe_name)
    merged["signal_type"] = "Pitcher"
    merged["why"] = merged.apply(
        lambda r: (
            f"Whiff rate {100 * r['recent_whiff_rate']:.1f}% "
            f"({100 * r['whiff_delta']:+.1f} pts vs baseline), "
            f"FB velo {r['recent_fb_velo']:.1f} mph ({r['velo_delta']:+.1f})."
        ),
        axis=1
    )
    merged["metric_1"] = (100 * merged["recent_whiff_rate"]).round(1)
    merged["metric_1_label"] = "Whiff %"
    merged["metric_2"] = merged["recent_fb_velo"].round(1)
    merged["metric_2_label"] = "FB Velo"
    merged["metric_3"] = merged["recent_extension"].round(2)
    merged["metric_3_label"] = "Extension"

    return merged.sort_values("edge_score", ascending=False).reset_index(drop=True)


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
        body += f"\n[Open Signal Wall]({markdown_escape(SITE_URL)})"
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


HTML_TEMPLATE = Template("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DiamondSignals — Daily Signal Wall</title>
  <style>
    :root {
      --bg: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #e2e8f0;
      --soft: #f8fafc;
      --accent: #1d4ed8;
      --gold: #b45309;
      --shadow: 0 10px 30px rgba(2, 8, 23, 0.08);
      --radius: 18px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Inter, Arial, sans-serif;
      line-height: 1.4;
    }
    .wrap {
      width: min(1180px, calc(100% - 28px));
      margin: 0 auto;
      padding: 24px 0 56px;
    }
    .hero {
      display: grid;
      gap: 14px;
      padding: 8px 0 24px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 24px;
    }
    .eyebrow {
      font-size: 12px;
      letter-spacing: .16em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 800;
    }
    h1 {
      margin: 0;
      font-size: clamp(30px, 6vw, 52px);
      line-height: .98;
      letter-spacing: -0.03em;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      max-width: 780px;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 4px;
    }
    .pill {
      border: 1px solid var(--line);
      background: var(--soft);
      color: var(--ink);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 700;
    }
    .grid {
      display: grid;
      gap: 18px;
      grid-template-columns: 1fr;
    }
    @media (min-width: 960px) {
      .grid { grid-template-columns: 1fr 1fr; }
    }
    .panel {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel-head {
      padding: 18px 18px 14px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(to bottom, #fff, #fbfdff);
    }
    .panel-title {
      margin: 0;
      font-size: 18px;
      letter-spacing: -0.02em;
    }
    .panel-sub {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .cards {
      display: grid;
      gap: 0;
    }
    .card {
      padding: 16px 18px;
      border-top: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }
    .topline {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }
    .rankname {
      display: grid;
      gap: 4px;
    }
    .rank {
      font-size: 12px;
      color: var(--muted);
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .12em;
    }
    .name {
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.02em;
      line-height: 1.05;
    }
    .score {
      min-width: 84px;
      text-align: right;
    }
    .score-label {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .12em;
      font-weight: 800;
    }
    .score-value {
      font-size: 28px;
      line-height: 1;
      font-weight: 900;
      letter-spacing: -0.03em;
    }
    .metric-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }
    .metric {
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
    }
    .metric-label {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .10em;
      font-weight: 800;
    }
    .metric-value {
      margin-top: 3px;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }
    .why {
      color: var(--ink);
      font-size: 14px;
    }
    .footer {
      margin-top: 26px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">DiamondSignals // Daily Trigger</div>
      <h1>Signal Wall</h1>
      <p class="sub">
        A mobile-first ranking board for today’s strongest MLB trigger candidates.
        Edge Scores emphasize recent quality plus short-term acceleration against baseline.
      </p>
      <div class="meta">
        <span class="pill">Updated: {{ generated_at }}</span>
        <span class="pill">Window: last 28 days</span>
        <span class="pill">Threshold: {{ threshold }}</span>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <div class="panel-head">
          <h2 class="panel-title">Top 5 Pitchers</h2>
          <p class="panel-sub">Whiff pressure, fastball life, and recent lift vs baseline.</p>
        </div>
        <div class="cards">
          {% for row in pitchers %}
          <article class="card">
            <div class="topline">
              <div class="rankname">
                <div class="rank">#{{ loop.index }} Pitcher</div>
                <div class="name">{{ row.player_name }}</div>
              </div>
              <div class="score">
                <div class="score-label">Edge Score</div>
                <div class="score-value">{{ row.edge_score }}</div>
              </div>
            </div>
            <div class="metric-row">
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

      <div class="panel">
        <div class="panel-head">
          <h2 class="panel-title">Top 5 Hitters</h2>
          <p class="panel-sub">Contact quality, max damage, and recent acceleration vs baseline.</p>
        </div>
        <div class="cards">
          {% for row in hitters %}
          <article class="card">
            <div class="topline">
              <div class="rankname">
                <div class="rank">#{{ loop.index }} Hitter</div>
                <div class="name">{{ row.player_name }}</div>
              </div>
              <div class="score">
                <div class="score-label">Edge Score</div>
                <div class="score-value">{{ row.edge_score }}</div>
              </div>
            </div>
            <div class="metric-row">
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

    <section class="footer">
      DiamondSignals Signal Wall • Generated during Netlify build • {{ timezone_label }}
    </section>
  </div>
</body>
</html>
""")


def render_html(pitchers: pd.DataFrame, hitters: pd.DataFrame) -> str:
    return HTML_TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        threshold=f"{ALERT_THRESHOLD:.0f}+",
        timezone_label=TIMEZONE_LABEL,
        pitchers=pitchers.to_dict(orient="records"),
        hitters=hitters.to_dict(orient="records"),
    )


def main() -> None:
    raw = fetch_statcast_window(START_DATE, END_DATE)

    hitter_signals = build_hitter_signals(raw)
    pitcher_signals = build_pitcher_signals(raw)

    if hitter_signals.empty and pitcher_signals.empty:
        raise RuntimeError("No hitter or pitcher signals were produced.")

    top_hitters = hitter_signals.head(5).copy()
    top_pitchers = pitcher_signals.head(5).copy()

    combined_alerts = pd.concat([top_pitchers, top_hitters], ignore_index=True)
    combined_alerts = combined_alerts.sort_values("edge_score", ascending=False).reset_index(drop=True)

    html = render_html(top_pitchers, top_hitters)
    output_path = DIST_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path}")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "top_pitchers": top_pitchers[["player_name", "edge_score"]].to_dict(orient="records"),
        "top_hitters": top_hitters[["player_name", "edge_score"]].to_dict(orient="records"),
    }
    (DIST_DIR / "signals.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote dist/signals.json")

    send_telegram_alerts(combined_alerts)


if __name__ == "__main__":
    main()