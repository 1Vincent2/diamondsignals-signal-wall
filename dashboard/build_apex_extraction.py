from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


OUT_DIR = Path("dist/apex-extraction")
OUT_JSON = OUT_DIR / "apex_extraction.json"
OUT_HTML = OUT_DIR / "index.html"


def score_apex_candidate(row: dict) -> dict:
    physical = float(row.get("physical_shift_score", 0))
    vision = float(row.get("vision_delta_score", 0))
    market = float(row.get("market_latency_score", 0))

    physical_shift = physical >= 70
    vision_delta = vision >= 60
    market_latency = market >= 65

    trigger_count = sum([physical_shift, vision_delta, market_latency])

    apex_score = round((physical * 0.45) + (vision * 0.25) + (market * 0.30), 1)

    if physical_shift and vision_delta and market_latency:
        verdict = "APEX EXTRACTION"
    elif physical_shift and market_latency:
        verdict = "SUBSURFACE WATCH"
    elif physical_shift:
        verdict = "PHYSICAL SHIFT"
    else:
        verdict = "NO SIGNAL"

    return {
        **row,
        "apex_score": apex_score,
        "trigger_count": trigger_count,
        "physical_shift": physical_shift,
        "vision_delta": vision_delta,
        "market_latency": market_latency,
        "verdict": verdict,
    }


def demo_candidates() -> list[dict]:
    return [
        {
            "player_id": "demo-bat-001",
            "name": "PLAYER X",
            "team": "MLB",
            "role": "BAT",
            "signal_family": "APEX BAT",
            "physical_shift_score": 82,
            "vision_delta_score": 68,
            "market_latency_score": 78,
            "primary_signal": "95th percentile EV rising while market attention remains delayed",
            "supporting_metric": "xwOBA-wOBA gap > .060",
            "market_note": "Roster attention lagging physical ceiling",
            "action": "ADD",
            "forensic_metrics": [
                {
                    "category": "Physical",
                    "code": "LA_CONSISTENCY",
                    "label": "LA Consistency",
                    "value": "Surgical",
                    "purpose": "Proves the swing plane is stable enough to convert impact into damage."
                },
                {
                    "category": "Vision",
                    "code": "PULL_SIDE_AIR",
                    "label": "Pull-Side Air %",
                    "value": "+10%",
                    "purpose": "Predicts the home-run explosion before the box score catches up."
                },
                {
                    "category": "Market",
                    "code": "HIGH_STAKES_DELTA",
                    "label": "High-Stakes Delta",
                    "value": "Public Late",
                    "purpose": "Identifies the value gap between sharp signal and public ownership."
                }
            ],
        },
        {
            "player_id": "demo-arm-001",
            "name": "PLAYER Y",
            "team": "MLB",
            "role": "SP",
            "signal_family": "APEX ARM",
            "physical_shift_score": 86,
            "vision_delta_score": 63,
            "market_latency_score": 72,
            "primary_signal": "iVB carry and extension moving above baseline",
            "supporting_metric": "xERA gap suggests public ERA is stale",
            "market_note": "Surface results masking improving physics",
            "action": "WATCH",
            "forensic_metrics": [
                {
                    "category": "Physical",
                    "code": "SSW_DEVIATION",
                    "label": "SSW Deviation",
                    "value": ">15°",
                    "purpose": "Detects invisible movement created by seam-shifted wake."
                },
                {
                    "category": "Physical",
                    "code": "RELEASE_STABILITY",
                    "label": "Release Stability",
                    "value": "Locked",
                    "purpose": "Confirms repeatable mechanics behind the physics shift."
                },
                {
                    "category": "Market",
                    "code": "HIGH_STAKES_DELTA",
                    "label": "High-Stakes Delta",
                    "value": "Public Late",
                    "purpose": "Identifies the value gap between sharp signal and public ownership."
                }
            ],
        },
    ]


def build_payload() -> dict:
    rows = [score_apex_candidate(row) for row in demo_candidates()]
    rows = sorted(rows, key=lambda r: r["apex_score"], reverse=True)

    bats = [r for r in rows if r["role"] == "BAT"]
    arms = [r for r in rows if r["role"] in {"SP", "RP", "P"}]

    return {
        "report": "Apex Extraction",
        "subtitle": "Subsurface MLB Breakout Ledger",
        "version": "apex_extraction_v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "demo_scaffold",
        "logic": {
            "apex_score": "physical_shift_score*0.45 + vision_delta_score*0.25 + market_latency_score*0.30",
            "apex_trigger": "physical_shift >= 70 AND vision_delta >= 60 AND market_latency >= 65",
        },
        "counts": {
            "total": len(rows),
            "bats": len(bats),
            "arms": len(arms),
        },
        "top_signals": rows,
        "apex_bats": bats,
        "apex_arms": arms,
    }


def render_signal_card(row: dict) -> str:
    score = row.get("apex_score", 0)
    verdict = row.get("verdict", "NO SIGNAL")
    action = row.get("action", "WATCH")

    forensic_metrics = row.get("forensic_metrics", [])
    forensic_html = "\n".join(
        f"""
        <div class="forensic-chip">
          <span>{metric.get("category", "")}</span>
          <strong>{metric.get("label", "")}</strong>
          <em>{metric.get("value", "")}</em>
          <p>{metric.get("purpose", "")}</p>
        </div>
        """
        for metric in forensic_metrics
    )

    return f"""
      <article class="apex-card">
        <div class="card-top">
          <div>
            <div class="kicker">{row.get("signal_family", "APEX")}</div>
            <h2>{row.get("name", "UNKNOWN")}</h2>
            <div class="meta">{row.get("team", "MLB")} // {row.get("role", "ASSET")}</div>
          </div>
          <div class="score-box">
            <div class="score">{score}</div>
            <div class="score-label">APEX</div>
          </div>
        </div>

        <div class="verdict">{verdict}</div>

        <div class="grid">
          <div><span>PHYSICAL</span><strong>{row.get("physical_shift_score", 0)}</strong></div>
          <div><span>VISION</span><strong>{row.get("vision_delta_score", 0)}</strong></div>
          <div><span>MARKET</span><strong>{row.get("market_latency_score", 0)}</strong></div>
        </div>

        <div class="proof">
          <div class="proof-label">PRIMARY SIGNAL</div>
          <p>{row.get("primary_signal", "")}</p>
        </div>

        <div class="proof">
          <div class="proof-label">SUPPORTING METRIC</div>
          <p>{row.get("supporting_metric", "")}</p>
        </div>

        <div class="forensic-strip">
          {forensic_html}
        </div>

        <div class="market-note">{row.get("market_note", "")}</div>

        <div class="action-row">
          <span>COMMAND</span>
          <strong>{action}</strong>
        </div>

        <a class="provision-btn" href="/watchlist/?player_id={row.get("player_id", "")}&source=apex-extraction">
          PROVISION TO WATCHLIST
        </a>
      </article>
    """


def render_html(payload: dict) -> str:
    top_cards = "\n".join(render_signal_card(row) for row in payload["top_signals"])
    generated = payload.get("generated_at", "")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Apex Extraction // DiamondSignals</title>
  <style>
    :root {{
      --bg: #05070a;
      --panel: #0d1319;
      --panel2: #111a21;
      --line: rgba(255, 255, 255, 0.16);
      --text: #f6fbff;
      --muted: #81909d;
      --cyan: #ffffff;
      --emerald: #ffffff;
      --lime: #ffffff;
      --danger: #ff7a1a;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      background:
        radial-gradient(circle at 15% 0%, rgba(255, 255, 255, 0.10), transparent 32%),
        radial-gradient(circle at 88% 12%, rgba(160, 160, 160, 0.10), transparent 34%),
        var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
    }}

    .shell {{
      width: min(1180px, calc(100% - 28px));
      margin: 0 auto;
      padding: 34px 0 60px;
    }}

    .hero {{
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(13,19,25,.96), rgba(7,10,14,.98));
      border-radius: 28px;
      padding: 30px;
      box-shadow: 0 24px 80px rgba(0,0,0,.35), 0 0 0 1px rgba(255,122,26,.08);
    }}

    .eyebrow, .kicker, .proof-label, .action-row span, .score-label {{
      font-family: Menlo, Consolas, "Courier New", monospace;
      letter-spacing: 2px;
      text-transform: uppercase;
      font-weight: 800;
    }}

    .eyebrow {{
      color: var(--emerald);
      font-size: 12px;
      margin-bottom: 16px;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(38px, 7vw, 78px);
      line-height: .9;
      letter-spacing: -3px;
    }}

    .subhead {{
      margin-top: 18px;
      max-width: 790px;
      color: #c9d4dd;
      font-size: 18px;
      line-height: 1.55;
    }}

    .status-row {{
      margin-top: 24px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}

    .status-pill {{
      border: 1px solid rgba(255,255,255,.08);
      background: rgba(255,255,255,.035);
      border-radius: 18px;
      padding: 14px;
      font-family: Menlo, Consolas, "Courier New", monospace;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      line-height: 1.45;
    }}

    .status-pill strong {{
      display: block;
      color: var(--text);
      font-size: 18px;
      margin-top: 6px;
    }}

    .section-title {{
      margin: 34px 0 16px;
      font-family: Menlo, Consolas, "Courier New", monospace;
      color: var(--lime);
      letter-spacing: 3px;
      text-transform: uppercase;
      font-size: 13px;
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}

    .apex-card {{
      border: 1px solid rgba(255,255,255,.16);
      border-top: 1px solid var(--emerald);
      background: linear-gradient(180deg, rgba(17,26,33,.98), rgba(9,13,18,.98));
      border-radius: 24px;
      padding: 22px;
    }}

    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
    }}

    .kicker {{
      color: var(--emerald);
      font-size: 11px;
      margin-bottom: 8px;
    }}

    h2 {{
      margin: 0;
      font-size: 30px;
      line-height: 1;
      letter-spacing: -1px;
    }}

    .meta {{
      margin-top: 8px;
      color: var(--muted);
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 12px;
      letter-spacing: 2px;
    }}

    .score-box {{
      text-align: right;
      min-width: 88px;
    }}

    .score {{
      font-family: Arial, Helvetica, sans-serif;
      font-size: 52px;
      line-height: 50px;
      font-weight: 900;
      font-style: italic;
      letter-spacing: -2px;
      color: var(--danger);
    }}

    .score-label {{
      margin-top: 6px;
      color: #a0a0a0;
      font-size: 10px;
    }}

    .verdict {{
      margin-top: 18px;
      border: 1px solid rgba(255,255,255,.28);
      border-left: 3px solid #ffffff;
      background: rgba(255,255,255,.055);
      border-radius: 16px;
      padding: 13px 14px;
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 13px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #ffffff;
      font-weight: 900;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0,1fr));
      gap: 10px;
      margin-top: 16px;
    }}

    .grid div {{
      border: 1px solid rgba(255,255,255,.08);
      background: rgba(255,255,255,.035);
      border-radius: 14px;
      padding: 12px;
    }}

    .grid span {{
      display: block;
      color: var(--muted);
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 10px;
      letter-spacing: 2px;
    }}

    .grid strong {{
      display: block;
      margin-top: 6px;
      font-size: 22px;
      color: var(--danger);
    }}

    .proof {{
      margin-top: 16px;
      border-top: 1px solid rgba(255,255,255,.08);
      padding-top: 14px;
    }}

    .proof-label {{
      color: #a0a0a0;
      font-size: 10px;
    }}

    .proof p {{
      margin: 8px 0 0;
      color: #d4dde5;
      line-height: 1.45;
    }}

    .market-note {{
      margin-top: 16px;
      color: #9cabb6;
      font-size: 14px;
      line-height: 1.45;
    }}

    .forensic-strip {{
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}

    .forensic-chip {{
      border: 1px solid rgba(255,255,255,.12);
      background: rgba(255,255,255,.035);
      border-radius: 14px;
      padding: 12px;
      min-height: 122px;
    }}

    .forensic-chip span {{
      display: block;
      color: #a0a0a0;
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 9px;
      line-height: 12px;
      letter-spacing: 2px;
      text-transform: uppercase;
      font-weight: 900;
    }}

    .forensic-chip strong {{
      display: block;
      margin-top: 7px;
      color: #ffffff;
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 11px;
      line-height: 15px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
    }}

    .forensic-chip em {{
      display: block;
      margin-top: 7px;
      color: var(--danger);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 18px;
      line-height: 20px;
      font-style: normal;
      font-weight: 900;
    }}

    .forensic-chip p {{
      margin: 8px 0 0;
      color: #8f9aa6;
      font-size: 12px;
      line-height: 1.35;
    }}

    .action-row {{
      margin-top: 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-top: 1px solid rgba(255,255,255,.08);
      padding-top: 14px;
    }}

    .action-row span {{
      color: var(--muted);
      font-size: 10px;
    }}

    .action-row strong {{
      color: var(--danger);
      font-family: Menlo, Consolas, "Courier New", monospace;
      letter-spacing: 2px;
    }}

    .provision-btn {{
      display: inline-block;
      margin-top: 16px;
      padding: 13px 16px;
      border-radius: 14px;
      border: 1px solid rgba(255,122,26,.42);
      background: rgba(255,122,26,.08);
      color: #ffffff;
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 11px;
      line-height: 14px;
      letter-spacing: 2px;
      text-transform: uppercase;
      font-weight: 900;
      text-decoration: none;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.08);
    }}

    .provision-btn:hover {{
      border-color: rgba(255,122,26,.72);
      background: rgba(255,122,26,.14);
    }}

    @media (max-width: 760px) {{
      .hero {{ padding: 22px; }}
      .status-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .cards {{ grid-template-columns: 1fr; }}
      .card-top {{ align-items: flex-start; }}
      .forensic-strip {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">DIAMONDSIGNALS // SUBSURFACE BREAKOUT LEDGER</div>
      <h1>APEX EXTRACTION</h1>
      <p class="subhead">
        Physical shift first. Vision and discipline confirm it. Market latency makes it actionable.
        This surface identifies MLB players whose underlying profile is moving before public pricing catches up.
      </p>

      <div class="status-row">
        <div class="status-pill">Report Status<strong>{payload.get("status", "")}</strong></div>
        <div class="status-pill">Candidates<strong>{payload["counts"]["total"]}</strong></div>
        <div class="status-pill">Apex Bats<strong>{payload["counts"]["bats"]}</strong></div>
        <div class="status-pill">Apex Arms<strong>{payload["counts"]["arms"]}</strong></div>
      </div>
    </section>

    <div class="section-title">&gt;_ DECLASSIFIED APEX SIGNALS // GENERATED {generated}</div>

    <section class="cards">
      {top_cards}
    </section>
  </main>
</body>
</html>"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote {OUT_JSON} with {payload['counts']['total']} Apex candidates")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
