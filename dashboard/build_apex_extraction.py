from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


OUT_DIR = Path("dist/apex-extraction")
OUT_JSON = OUT_DIR / "apex_extraction.json"
OUT_HTML = OUT_DIR / "index.html"
DATA_DIR = Path("dist")


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).replace("%", "").replace('"', "").replace("°", "").replace("ft", "").strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def load_json(path: Path, fallback):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to load {path}: {exc}")
    return fallback


def player_index_by_id() -> dict:
    data = load_json(DATA_DIR / "player_index.json", {"players": []})
    return {str(row.get("player_id")): row for row in data.get("players", []) if row.get("player_id")}


def display_name_from_index(player_id: str, fallback: str = "UNKNOWN") -> str:
    idx = player_index_by_id()
    row = idx.get(str(player_id), {})
    return (row.get("full_name") or fallback or "UNKNOWN").upper()



def score_apex_candidate(row: dict) -> dict:
    physical = float(row.get("physical_shift_score", 0))
    vision = float(row.get("vision_delta_score", 0))
    market = float(row.get("market_latency_score", 0))

    physical_shift = physical >= 70
    vision_delta = vision >= 60
    market_latency = market >= 65

    trigger_count = sum([physical_shift, vision_delta, market_latency])

    apex_score = round((physical * 0.45) + (vision * 0.25) + (market * 0.30), 1)

    if trigger_count == 3:
        verdict = "APEX EXTRACTION"
    elif physical_shift and market_latency:
        verdict = "APEX WATCH"
    elif trigger_count == 2:
        verdict = "SUBSURFACE BREAKOUT"
    elif trigger_count == 1:
        verdict = "PHYSICAL WATCH"
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



def build_apex_arms_from_existing_reports() -> list[dict]:
    stuff = load_json(DATA_DIR / "stuff_disruption_feed.json", {"cards": []}).get("cards", [])
    ivb = load_json(DATA_DIR / "ivb_heat_map.json", {"heat_cards": []}).get("heat_cards", [])
    velocity = load_json(DATA_DIR / "velocity_decay_monitor.json", {"cards": []}).get("cards", [])

    ivb_by_name = {str(row.get("player_name", "")).lower(): row for row in ivb}
    velocity_by_name = {str(row.get("player_name", "")).lower(): row for row in velocity}

    candidates = []

    for row in stuff[:24]:
        name = row.get("player_name", "UNKNOWN")
        key = str(name).lower()
        ivb_row = ivb_by_name.get(key, {})
        velo_row = velocity_by_name.get(key, {})

        disruption = safe_float(row.get("disruption_score"))
        ivb_delta = safe_float(row.get("ivb_delta"))
        vaa_delta = safe_float(row.get("vaa_delta"))
        movement_delta = safe_float(row.get("movement_delta"))
        active_spin_delta = safe_float(row.get("active_spin_delta"))

        physical = clamp((disruption * 0.55) + max(ivb_delta, 0) * 6 + abs(min(vaa_delta, 0)) * 10 + max(movement_delta, 0) * 2)
        vision = clamp(58 + max(ivb_delta, 0) * 3 + abs(min(vaa_delta, 0)) * 8)
        market = clamp(62 + (8 if row.get("apex_tier") in {"S-TIER", "A-TIER"} else 0) + (6 if ivb_row.get("transition_badge") else 0))

        primary_signal = row.get("analysis") or "Pitch-shape disruption detected across the recent window."
        supporting_metric = f"iVB Delta {row.get('ivb_delta_label', '')} // VAA Delta {row.get('vaa_delta_label', '')}".strip()
        market_note = "Public results may be lagging the shape-change signal."

        forensic_metrics = [
            {
                "category": "Geometry",
                "code": "VAA_DELTA",
                "label": "VAA Delta",
                "value": row.get("vaa_delta_label") or f"{vaa_delta:+.1f}°",
                "purpose": "Tracks whether the fastball is entering the zone on a flatter, more damaging plane."
            },
            {
                "category": "Deception",
                "code": "SSW_PROXY",
                "label": "Movement Deviation",
                "value": row.get("movement_delta_label") or f"{movement_delta:+.1f}\"",
                "purpose": "Proxy for late movement or seam-driven deception before direct SSW is wired."
            },
            {
                "category": "Market",
                "code": "HIGH_STAKES_DELTA",
                "label": "High-Stakes Delta",
                "value": "Public Late",
                "purpose": "Identifies the value gap between physical signal and public attention."
            },
        ]

        candidates.append({
            "player_id": str(row.get("player_id", name)),
            "name": str(name).upper(),
            "team": row.get("team", "MLB"),
            "role": "SP",
            "signal_family": "APEX ARM",
            "physical_shift_score": round(physical, 1),
            "vision_delta_score": round(vision, 1),
            "market_latency_score": round(market, 1),
            "primary_signal": primary_signal,
            "supporting_metric": supporting_metric,
            "market_note": market_note,
            "action": "WATCH" if market < 72 else "ADD",
            "forensic_metrics": forensic_metrics,
        })

    return candidates


def build_apex_bats_from_scout_metrics() -> list[dict]:
    scout = load_json(DATA_DIR / "scout_metrics.json", {"players": {}}).get("players", {})
    idx = player_index_by_id()
    candidates = []

    for player_id, row in scout.items():
        if row.get("player_type") != "hitter":
            continue

        ballistics = row.get("ballistics", {})
        movement = row.get("movement", {})
        results = row.get("results", {})

        max_ev = safe_float(ballistics.get("value_2"))
        hard_hit = safe_float(ballistics.get("value_3"))
        diamond_delta = safe_float(ballistics.get("value_4"))
        sweet_spot = safe_float(movement.get("value_1"))
        barrel = safe_float(movement.get("value_2"))
        launch_angle = safe_float(movement.get("value_3"))
        xba = safe_float(movement.get("value_4"))
        avg = safe_float(results.get("value_1"))
        k_rate = safe_float(results.get("value_2"))
        signal = str(results.get("value_4") or "")

        optimized_la = 15 <= launch_angle <= 25
        physical = clamp(
            42
            + max(0, max_ev - 103) * 4
            + hard_hit * 0.45
            + barrel * 1.4
            + (12 if optimized_la else 0)
        )
        vision = clamp(
            50
            + max(0, 18 - k_rate) * 1.2
            + sweet_spot * 0.35
            + (8 if optimized_la else 0)
        )
        market = clamp(
            50
            + max(0, diamond_delta) * 90
            + (10 if signal == "GOLD BUY" else 0)
            + (8 if xba - avg > 0.06 else 0)
        )

        if physical < 62 and market < 62:
            continue

        player = idx.get(str(player_id), {})
        name = (player.get("full_name") or f"PLAYER {player_id}").upper()
        team = player.get("team") or player.get("team_name") or "MLB"

        forensic_metrics = [
            {
                "category": "Physics",
                "code": "DHH_PROXY",
                "label": "Dynamic Hard-Hit",
                "value": f"{max_ev:.1f} mph",
                "purpose": "Proxy for flight-optimized damage while true DHH is being wired."
            },
            {
                "category": "Vision",
                "code": "LA_CONSISTENCY",
                "label": "LA Consistency",
                "value": "Surgical" if optimized_la else f"{launch_angle:.1f}°",
                "purpose": "Shows whether the swing plane is converting force into flight."
            },
            {
                "category": "Market",
                "code": "HIGH_STAKES_DELTA",
                "label": "High-Stakes Delta",
                "value": f"{diamond_delta:+.3f}",
                "purpose": "Identifies the value gap between expected quality and public results."
            },
        ]

        candidates.append({
            "player_id": str(player_id),
            "name": name,
            "team": team,
            "role": "BAT",
            "signal_family": "APEX BAT",
            "physical_shift_score": round(physical, 1),
            "vision_delta_score": round(vision, 1),
            "market_latency_score": round(market, 1),
            "primary_signal": row.get("briefing") or "Underlying contact quality is ahead of surface results.",
            "supporting_metric": f"Max EV {max_ev:.1f} mph // xBA {xba:.3f} vs AVG {avg:.3f}",
            "market_note": "Surface production may be lagging the underlying contact profile.",
            "action": "ADD" if market >= 72 and physical >= 70 else "WATCH",
            "forensic_metrics": forensic_metrics,
        })

    return candidates


def real_candidates() -> list[dict]:
    arms = build_apex_arms_from_existing_reports()
    bats = build_apex_bats_from_scout_metrics()

    arms = sorted(arms, key=lambda r: (r["physical_shift_score"] + r["market_latency_score"]), reverse=True)[:8]
    bats = sorted(bats, key=lambda r: (r["physical_shift_score"] + r["market_latency_score"]), reverse=True)[:8]

    return bats + arms


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
    source_rows = real_candidates()
    if not source_rows:
        source_rows = demo_candidates()

    rows = [score_apex_candidate(row) for row in source_rows]
    rows = sorted(rows, key=lambda r: r["apex_score"], reverse=True)

    bats = [r for r in rows if r["role"] == "BAT"]
    arms = [r for r in rows if r["role"] in {"SP", "RP", "P"}]

    return {
        "report": "Apex Extraction",
        "subtitle": "Subsurface MLB Breakout Ledger",
        "version": "apex_extraction_v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "real_data_v0.2",
        "logic": {
            "apex_score": "physical_shift_score*0.45 + vision_delta_score*0.25 + market_latency_score*0.30",
            "apex_trigger": "3 clusters = APEX EXTRACTION; 2 clusters = SUBSURFACE BREAKOUT/APEX WATCH; 1 cluster = PHYSICAL WATCH",
            "cluster_gate": "Alert only when at least two clusters fire simultaneously; highest tier requires all three clusters.",
        },
        "technical_array": [
            {
                "metric": "VAA Above Expected",
                "code": "VAA_DELTA",
                "category": "Pitching Geometry",
                "trigger": "VAA_Delta > 0.2 degrees over a 2-start window",
                "shark_meaning": "The ball is disappearing at the top of the zone."
            },
            {
                "metric": "SSW Deviation",
                "code": "SSW_DEVIATION",
                "category": "Pitching Deception",
                "trigger": "Observed movement vs spin-based movement deviation > 15%",
                "shark_meaning": "The ball is moving in ways the human brain cannot calculate."
            },
            {
                "metric": "Whiff-per-Swing by Zone Quadrant",
                "code": "SHADOW_ZONE_WHIFF_REDUCTION",
                "category": "Hitter Vision",
                "trigger": "Shadow-zone whiff reduction > 10%",
                "shark_meaning": "The hitter has downloaded the strike zone."
            },
            {
                "metric": "Dynamic Hard-Hit",
                "code": "DHH_SCORE",
                "category": "Hitter Physics",
                "trigger": "DHH score increase > 15%",
                "shark_meaning": "The power is being optimized for flight, not just force."
            }
        ],
        "counts": {
            "total": len(rows),
            "bats": len(bats),
            "arms": len(arms),
        },
        "top_signals": rows,
        "apex_bats": bats,
        "apex_arms": arms,
    }


def apex_heat_class(score: float) -> str:
    try:
        score = float(score)
    except Exception:
        score = 0.0

    if score >= 90:
        return "apex-hot"
    if score >= 80:
        return "apex-edge"
    if score >= 70:
        return "apex-watch"
    if score >= 60:
        return "apex-neutral"
    return "apex-dormant"


def render_signal_card(row: dict) -> str:
    score = row.get("apex_score", 0)
    heat_class = apex_heat_class(score)
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
      <article class="apex-card {heat_class}">
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
    bat_cards = "\n".join(render_signal_card(row) for row in payload.get("apex_bats", []))
    arm_cards = "\n".join(render_signal_card(row) for row in payload.get("apex_arms", []))
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
      color: var(--heat, var(--danger));
      text-shadow: 0 0 18px var(--heat-glow, rgba(255,122,26,.18));
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
      color: var(--heat, var(--danger));
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
      color: var(--heat, var(--danger));
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
      color: var(--heat, var(--danger));
      font-family: Menlo, Consolas, "Courier New", monospace;
      letter-spacing: 2px;
    }}

    .apex-card.apex-hot {{
      --heat: #ffffff;
      --heat-glow: rgba(255,215,0,.28);
      border: 2px solid rgba(255,122,26,.72);
      box-shadow: 0 0 0 1px rgba(255,255,255,.10), 0 20px 70px rgba(255,122,26,.10);
    }}

    .apex-card.apex-hot .score-label::after {{
      content: " // PURE SIGNAL";
      color: #ffd700;
    }}

    .apex-card.apex-edge {{
      --heat: #ff8c00;
      --heat-glow: rgba(255,140,0,.24);
      border-color: rgba(255,140,0,.42);
      border-top-color: #ff8c00;
      box-shadow: 0 0 0 1px rgba(255,140,0,.08);
    }}

    .apex-card.apex-watch {{
      --heat: #b86500;
      --heat-glow: rgba(184,101,0,.18);
      border-color: rgba(184,101,0,.24);
      border-top-color: #b86500;
    }}

    .apex-card.apex-neutral {{
      --heat: #a0a0a0;
      --heat-glow: rgba(160,160,160,.08);
      border-color: rgba(160,160,160,.12);
      border-top-color: rgba(160,160,160,.42);
    }}

    .apex-card.apex-dormant {{
      --heat: #4a4a4a;
      --heat-glow: rgba(74,74,74,.04);
      opacity: .60;
      border-color: rgba(74,74,74,.18);
      border-top-color: rgba(74,74,74,.42);
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

    <div class="section-title">&gt;_ APEX BATS // SUBSURFACE POWER OPTIMIZATION // GENERATED {generated}</div>
    <section class="cards">
      {bat_cards}
    </section>

    <div class="section-title">&gt;_ APEX ARMS // PITCHING GEOMETRY + DECEPTION ARRAY</div>
    <section class="cards">
      {arm_cards}
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
