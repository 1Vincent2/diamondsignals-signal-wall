from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from jinja2 import Template


OUT_DIR = Path("dist/apex-extraction")
OUT_JSON = OUT_DIR / "apex_extraction.json"
OUT_HTML = OUT_DIR / "index.html"

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SEARCH_TEMPLATE = (TEMPLATES_DIR / "shell_search.html").read_text(encoding="utf-8")
SHELL_STYLES_TEMPLATE = (TEMPLATES_DIR / "shell_styles.css").read_text(encoding="utf-8")
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

        tier_bonus = 12 if row.get("apex_tier") == "S-TIER" else 8 if row.get("apex_tier") == "A-TIER" else 0
        transition_bonus = 8 if ivb_row.get("transition_badge") else 0
        shape_alert_bonus = 8 if "APEX" in str(row.get("primary_alert", "")).upper() else 0
        whiff_bonus = 6 if str(ivb_row.get("whiff_probability", "")).upper() == "HIGH" else 0

        # Apex Arms should reward simultaneous geometry + deception signals.
        # Stuff disruption is the base; iVB, VAA, movement and transition badges lift the signal into Apex territory.
        physical = clamp(
            20
            + disruption * 0.62
            + max(ivb_delta, 0) * 7.5
            + abs(vaa_delta) * 7.0
            + max(movement_delta, 0) * 2.4
            + tier_bonus
            + transition_bonus
            + shape_alert_bonus
        )

        vision = clamp(
            48
            + max(ivb_delta, 0) * 4.0
            + abs(vaa_delta) * 9.0
            + max(active_spin_delta, 0) * 12.0
            + whiff_bonus
            + (6 if row.get("sample_note") else 0)
        )

        market = clamp(
            58
            + tier_bonus
            + transition_bonus
            + shape_alert_bonus
            + (6 if row.get("card_class") == "apex-top" else 0)
        )

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
        return "apex-elite"
    if score >= 85:
        return "apex-hot"
    if score >= 78:
        return "apex-warm"
    return "apex-cool"


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
      <article
        class="apex-card {heat_class}"
        id="apex-player-{row.get("player_id", "")}"
        data-player-id="{row.get("player_id", "")}"
        data-player-name="{row.get("name", "")}"
        data-player-team="{row.get("team", "MLB")}"
        data-player-role="{row.get("role", "ASSET")}"
        data-signal-family="{row.get("signal_family", "APEX")}"
        data-apex-score="{score}"
        data-physical="{row.get("physical_shift_score", 0)}"
        data-vision="{row.get("vision_delta_score", 0)}"
        data-market="{row.get("market_latency_score", 0)}"
        data-verdict="{verdict}"
        data-action="{action}"
        data-supporting-metric="{row.get("supporting_metric", "")}"
      >
        <div class="card-top">
          <div>
            <div class="kicker">{row.get("signal_family", "APEX")}</div>
            <h2>{row.get("name", "UNKNOWN")}</h2>
            <div class="meta">{row.get("team", "MLB")} // {row.get("role", "ASSET")}</div>
          </div>
          <div class="score-box">
            <div class="score-label">EXTRACTION SCORE</div>
            <div class="score">{score}</div>
          </div>
        </div>

        <div class="diagnosis-label">DIAGNOSIS</div>
        <div class="verdict">
          <div class="verdict-text">[ {verdict} ]</div>
          <div class="verdict-rail">
            <span></span>
          </div>
        </div>

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

        <button
          type="button"
          class="provision-btn js-add-to-roster"
          data-player-id="{row.get("player_id", "")}"
          data-player-name="{row.get("name", "")}"
          data-player-team="{row.get("team", "MLB")}"
          data-source-tag="APEX_EXTRACTION"
        >
          INITIATE TRACKING
        </button>
      </article>
    """


def render_html(payload: dict) -> str:
    bat_cards = "\n".join(render_signal_card(row) for row in payload.get("apex_bats", []))
    arm_cards = "\n".join(render_signal_card(row) for row in payload.get("apex_arms", []))
    generated = payload.get("generated_at", "")
    try:
        generated_label = datetime.fromisoformat(str(generated).replace("Z", "+00:00")).strftime("%Y-%m-%d %I:%M %p")
    except Exception:
        generated_label = str(generated)
    nav_html = Template(NAV_TEMPLATE).render(active_nav="apex_extraction")
    search_html = Template(SEARCH_TEMPLATE).render()
    shell_styles = SHELL_STYLES_TEMPLATE

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Apex Extraction // DiamondSignals</title>
  <style>
    {shell_styles}

    :root {{
      --bg: #05070a;
      --panel: #0d1319;
      --panel2: #111a21;
      --line: rgba(255, 255, 255, 0.16);
      --text: #f6fbff;
      --muted: #81909d;
      --cyan: #4ea3ff;
      --emerald: #ccff00;
      --lime: #ccff00;
      --lime-hot: #ccff00;
      --brand-blue: #4ea3ff;
      --danger: #ff7a1a;
      --mono: "JetBrains Mono", "Roboto Mono", "SFMono-Regular", Menlo, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      background:
        radial-gradient(circle at 15% 0%, rgba(255, 255, 255, 0.10), transparent 32%),
        radial-gradient(circle at 88% 12%, rgba(160, 160, 160, 0.10), transparent 34%),
        var(--bg);
      color: var(--text);
      font-family: var(--sans);
    }}


    .topbar {{
      border-bottom: 1px solid rgba(255,255,255,.08);
      background:
        radial-gradient(circle at 12% 0%, rgba(255,255,255,.08), transparent 28%),
        rgba(5,7,10,.72);
      backdrop-filter: blur(18px);
      position: relative;
      z-index: 20;
    }}

    .topbar-inner {{
      width: min(1180px, calc(100% - 28px));
      margin: 0 auto;
      min-height: 74px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }}

    .brand {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .brand-mark {{
      font-family: var(--mono);
      font-size: 13px;
      font-weight: 900;
      letter-spacing: .22em;
      color: var(--text);
      text-transform: uppercase;
    }}

    .brand-mark span {{
      color: var(--lime);
      text-shadow: 0 0 12px rgba(255,255,255,.14);
    }}

    .brand-sub {{
      font-family: var(--mono);
      font-size: 10px;
      color: var(--muted);
      letter-spacing: .14em;
      text-transform: uppercase;
    }}

    .livebox {{
      text-align: right;
      font-family: var(--mono);
      text-transform: uppercase;
      letter-spacing: .10em;
    }}

    .live-label {{
      display: inline-flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      font-size: 10px;
      color: var(--text);
      font-weight: 900;
    }}

    .live-dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--lime);
      box-shadow: 0 0 12px rgba(255,255,255,.18);
      display: inline-block;
    }}

    .live-time {{
      margin-top: 5px;
      font-size: 10px;
      color: var(--muted);
    }}



    /* === APEX INSTITUTIONAL TOPBAR OVERRIDE === */
    @keyframes dsApexLivePulse {{
      0% {{
        transform: scale(0.9);
        opacity: 0.78;
        box-shadow:
          0 0 0 0 rgba(204,255,0,0.55),
          0 0 10px rgba(204,255,0,0.50),
          0 0 18px rgba(204,255,0,0.26);
      }}

      50% {{
        transform: scale(1.22);
        opacity: 1;
        box-shadow:
          0 0 0 10px rgba(204,255,0,0),
          0 0 18px rgba(204,255,0,0.80),
          0 0 30px rgba(204,255,0,0.38);
      }}

      100% {{
        transform: scale(0.9);
        opacity: 0.78;
        box-shadow:
          0 0 0 0 rgba(204,255,0,0),
          0 0 10px rgba(204,255,0,0.50),
          0 0 18px rgba(204,255,0,0.26);
      }}
    }}

    .topbar {{
      position: relative;
      z-index: 40;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      background:
        radial-gradient(circle at 12% 0%, rgba(204,255,0,0.10), transparent 24%),
        radial-gradient(circle at 50% -10%, rgba(78,163,255,0.12), transparent 28%),
        rgba(4,6,9,0.94);
      backdrop-filter: blur(18px);
    }}

    .topbar-inner {{
      width: min(1180px, calc(100% - 28px));
      margin: 0 auto;
      min-height: 86px;
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 18px;
    }}

    .brand-lockup {{
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }}

    .brand-pulse {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      background: #ccff00;
      flex: 0 0 auto;
      animation: dsApexLivePulse 1.15s ease-in-out infinite;
    }}

    .brand-copy {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 0;
    }}

    .brand-mark {{
      font-family: var(--mono);
      font-size: 16px;
      font-weight: 900;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: #ffffff;
      line-height: 1;
    }}

    .brand-mark span {{
      color: #4ea3ff;
      text-shadow: 0 0 14px rgba(78,163,255,0.22);
    }}

    .brand-sub {{
      font-family: var(--sans);
      font-size: 13px;
      font-weight: 850;
      letter-spacing: -0.02em;
      color: #f4f7fb;
      opacity: 0.98;
    }}

    .topbar-center {{
      display: flex;
      justify-content: center;
      align-items: center;
    }}

    .topbar .field-guide-pill {{
      position: static;
      inset: auto;
      appearance: none;
      border: 1px solid rgba(204,255,0,0.24);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
      color: #f5f7fb;
      border-radius: 999px;
      padding: 15px 26px;
      min-height: 56px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      cursor: pointer;
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.04),
        0 0 0 1px rgba(204,255,0,0.04),
        0 0 18px rgba(204,255,0,0.06);
      transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }}

    .topbar .field-guide-pill:hover {{
      transform: translateY(-1px);
      border-color: rgba(204,255,0,0.42);
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.05),
        0 0 22px rgba(204,255,0,0.14);
    }}

    .field-guide-icon {{
      font-size: 15px;
      color: #e8edf3;
      opacity: 0.96;
    }}

    .livebox {{
      justify-self: end;
      text-align: right;
    }}

    .live-label {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.14em;
      color: #ccff00;
      text-transform: uppercase;
    }}

    .live-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #ccff00;
      display: inline-block;
      animation: dsApexLivePulse 1.15s ease-in-out infinite;
    }}

    .live-time {{
      margin-top: 6px;
      font-family: var(--mono);
      font-size: 11px;
      color: #9aa5b5;
      letter-spacing: 0.10em;
    }}

    .topnav {{
      border-top: 1px solid rgba(255,255,255,0.05);
      border-bottom: 1px solid rgba(255,255,255,0.07);
      background:
        linear-gradient(90deg, rgba(18,22,28,0.96), rgba(10,12,16,0.96));
    }}

    .topnav-link {{
      border: 1px solid rgba(255,255,255,0.08);
      background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012));
    }}

    .topnav-link.active {{
      color: var(--text);
      border-color: rgba(204,255,0,0.34);
      box-shadow:
        0 0 0 1px rgba(204,255,0,0.07),
        0 0 18px rgba(204,255,0,0.12);
    }}

    .topnav-tag {{
      color: #ccff00;
    }}

    .search-strip {{
      border-bottom: 1px solid rgba(255,255,255,0.06);
      background:
        linear-gradient(90deg, rgba(20,24,30,0.94), rgba(12,15,19,0.94));
    }}

    .player-search-input {{
      border: 1px solid rgba(204,255,0,0.24);
      background: rgba(255,255,255,0.07);
      box-shadow: 0 0 10px rgba(204,255,0,0.06);
    }}

    .player-search-input:focus {{
      border-color: rgba(204,255,0,0.48);
      background: rgba(255,255,255,0.09);
      box-shadow:
        0 0 0 3px rgba(204,255,0,0.10),
        0 0 14px rgba(204,255,0,0.14);
    }}

    @media (max-width: 900px) {{
      .topbar-inner {{
        grid-template-columns: 1fr;
        justify-items: start;
        gap: 14px;
        padding: 16px 0;
      }}

      .topbar-center {{
        width: 100%;
        justify-content: flex-start;
      }}

      .livebox {{
        justify-self: start;
        text-align: left;
      }}

      .topbar .field-guide-pill {{
        min-height: 48px;
        padding: 12px 18px;
      }}
    }}
    /* === END APEX INSTITUTIONAL TOPBAR OVERRIDE === */


    .app {{
      width: min(1180px, calc(100% - 28px));
      margin: 0 auto;
      padding: 28px 0 56px;
    }}

    .hero {{
      display: grid;
      grid-template-columns: 1.25fr 0.75fr;
      gap: 18px;
      margin-bottom: 20px;
    }}

    .hero-card,
    .summary-card {{
      background: var(--card-radial);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}

    .hero-card {{
      padding: 24px 24px 22px;
      min-width: 0;
    }}

    .summary-card {{
      padding: 18px;
      display: grid;
      gap: 14px;
      align-content: start;
    }}

    .summary-label {{
      font-family: var(--mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--tiny);
    }}

    .summary-value {{
      font-size: 28px;
      line-height: 1;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}

    .status-note {{
      margin-top: 6px;
      color: #aab3bd;
      font-size: 15px;
      line-height: 1.55;
      font-weight: 700;
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

    .hero-title {{
      margin: 0;
      font-size: clamp(32px, 6vw, 58px);
      line-height: 0.95;
      letter-spacing: -0.05em;
      text-transform: uppercase;
      font-weight: 900;
    }}

    .hero-sub {{
      margin: 14px 0 0;
      max-width: 58ch;
      color: var(--soft);
      font-size: 14px;
      line-height: 1.65;
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

    .section-head {{
      margin: 22px 0 12px;
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
    }}

    .section-kicker {{
      font-family: var(--mono);
      color: #6ea8ff;
      font-size: 11px;
      line-height: 1;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      font-weight: 800;
      margin-bottom: 8px;
    }}

    .section-heading {{
      margin: 0;
      color: var(--text);
      font-family: var(--sans);
      font-size: 24px;
      line-height: 1;
      letter-spacing: -0.03em;
      text-transform: uppercase;
      font-weight: 900;
    }}

    .section-title {{
      margin: 20px 0 10px;
      font-family: Menlo, Consolas, "Courier New", monospace;
      color: var(--lime);
      letter-spacing: 2.2px;
      text-transform: uppercase;
      font-size: 11px;
      line-height: 1.35;
    }}


    .apex-elite {{
      --heat: #ffffff;
      --heat-glow: rgba(255, 255, 255, .30);
      --card-accent: rgba(255, 255, 255, .68);
      --rail-accent: #ffffff;
    }}

    .apex-hot {{
      --heat: #f5c451;
      --heat-glow: rgba(245, 196, 81, .20);
      --card-accent: rgba(245, 196, 81, .42);
      --rail-accent: #f5c451;
    }}

    .apex-warm {{
      --heat: #a87945;
      --heat-glow: rgba(168, 121, 69, .11);
      --card-accent: rgba(168, 121, 69, .24);
      --rail-accent: #a87945;
    }}

    .apex-cool {{
      --heat: #8fa3b8;
      --heat-glow: rgba(143, 163, 184, .10);
      --card-accent: rgba(143, 163, 184, .22);
      --rail-accent: #8fa3b8;
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}

    .apex-card {{
      position: relative;
      border: 1px solid rgba(255,255,255,.10);
      border-top: 1px solid var(--card-accent, rgba(255,255,255,.22));
      background: linear-gradient(180deg, rgba(17,20,24,.96), rgba(8,10,13,.98));
      border-radius: 22px;
      padding: 18px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.035);
      scroll-margin-top: 112px;
    }}
      50% {{
        transform: translateY(-2px);
        box-shadow:
          0 0 0 1px rgba(204,255,0,.52),
          0 0 44px rgba(255,122,26,.36),
          0 0 70px rgba(204,255,0,.18),
          inset 0 1px 0 rgba(255,255,255,.10);
      }}
    }}


    .apex-card.target-signal-card {{
      border-color: rgba(204,255,0,.72);
      border-top-color: rgba(204,255,0,.95);
      box-shadow:
        0 0 0 1px rgba(204,255,0,.30),
        0 0 42px rgba(204,255,0,.20),
        0 0 68px rgba(255,122,26,.16),
        inset 0 1px 0 rgba(255,255,255,.10);
      animation: apexTargetPulse 1.25s ease-in-out 0s 7;
    }}

    .apex-card.target-signal-card::before {{
      content: "EXTRACTED FROM EMAIL";
      position: absolute;
      right: 16px;
      top: -13px;
      z-index: 5;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid rgba(204,255,0,.55);
      background: rgba(9,12,16,.96);
      color: #ccff00;
      font-family: var(--mono);
      font-size: 9px;
      line-height: 1;
      letter-spacing: .16em;
      font-weight: 900;
      text-transform: uppercase;
      box-shadow: 0 0 22px rgba(204,255,0,.24);
    }}

    @keyframes apexTargetPulse {{
      0%, 100% {{
        transform: translateY(0);
        box-shadow:
          0 0 0 1px rgba(204,255,0,.28),
          0 0 30px rgba(204,255,0,.16),
          inset 0 1px 0 rgba(255,255,255,.08);
      }}
      50% {{
        transform: translateY(-3px);
        box-shadow:
          0 0 0 1px rgba(204,255,0,.72),
          0 0 54px rgba(204,255,0,.30),
          0 0 84px rgba(255,122,26,.18),
          inset 0 1px 0 rgba(255,255,255,.12);
      }}
    }}

    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }}

    .kicker {{
      color: var(--emerald);
      font-size: 10px;
      margin-bottom: 6px;
    }}

    h2 {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 27px;
      line-height: 1.02;
      letter-spacing: -0.035em;
      font-weight: 800;
    }}

    .meta {{
      margin-top: 6px;
      color: var(--muted);
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 11px;
      line-height: 1.25;
      letter-spacing: 2px;
    }}

    .score-box {{
      text-align: right;
      min-width: 118px;
      align-self: flex-start;
    }}

    .score {{
      display: inline-block;
      font-family: var(--sans);
      font-size: 50px;
      line-height: 0.9;
      font-weight: 900;
      font-style: italic;
      letter-spacing: -0.055em;
      color: var(--heat, #ffffff);
      text-shadow: 0 0 12px var(--heat-glow, rgba(255,255,255,.12));
      transform: skewX(-7deg);
      transform-origin: right center;
    }}

    .score-label {{
      margin-bottom: 8px;
      color: #8f96a1;
      font-size: 11px;
      line-height: 13px;
      letter-spacing: 2.5px;
      text-align: right;
    }}

    .verdict {{
      margin-top: 18px;
      border: 1px solid rgba(255,255,255,.22);
      border-left: 1px solid rgba(255,255,255,.42);
      background: rgba(255,255,255,.045);
      border-radius: 16px;
      padding: 13px 14px;
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 12px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #f5f5f5;
      font-weight: 800;
    }}

    .diagnosis-label {{
      margin-top: 16px;
      margin-bottom: 8px;
      color: #8f96a1;
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 11px;
      line-height: 13px;
      letter-spacing: 2.5px;
      text-transform: uppercase;
      font-weight: 900;
    }}

    .verdict {{
      margin-top: 0;
      border: 1px solid rgba(255,255,255,.18);
      background: linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.025));
      border-radius: 18px;
      padding: 0;
      overflow: hidden;
      font-family: Menlo, Consolas, "Courier New", monospace;
      text-transform: uppercase;
      color: #f5f5f5;
      font-weight: 800;
    }}

    .verdict-text {{
      padding: 13px 16px;
      font-size: 13px;
      line-height: 16px;
      letter-spacing: 2px;
      border-bottom: 1px solid rgba(255,255,255,.10);
    }}

    .verdict-rail {{
      height: 44px;
      padding: 0 18px;
      display: flex;
      align-items: center;
      background: rgba(255,255,255,.018);
    }}

    .verdict-rail span {{
      display: block;
      width: 100%;
      height: 3px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(255,255,255,.04), var(--heat, #ff8c00));
      box-shadow: 0 0 16px var(--heat-glow, rgba(255,122,26,.14));
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0,1fr));
      gap: 9px;
      margin-top: 12px;
    }}

    .grid div {{
      border: 1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.035);
      border-radius: 13px;
      padding: 10px 12px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
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
      margin-top: 5px;
      font-family: var(--sans);
      font-size: 20px;
      line-height: 22px;
      font-weight: 900;
      letter-spacing: -0.04em;
      color: var(--heat, var(--danger));
    }}

    .proof {{
      margin-top: 16px;
      border-top: 1px solid rgba(255,255,255,.08);
      padding-top: 14px;
    }}

    .proof-label {{
      color: #a0a0a0;
      font-size: 9px;
      letter-spacing: 2px;
    }}

    .proof p {{
      margin: 6px 0 0;
      color: #d4dde5;
      font-size: 12.5px;
      line-height: 1.42;
    }}

    .market-note {{
      margin-top: 11px;
      color: #9cabb6;
      font-size: 12.5px;
      line-height: 1.4;
    }}

    .forensic-strip {{
      margin-top: 11px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 9px;
    }}

    .forensic-chip {{
      border: 1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.03);
      border-radius: 13px;
      padding: 10px;
      min-height: 104px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
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
      font-weight: 800;
      letter-spacing: -0.015em;
    }}

    .forensic-chip p {{
      margin: 8px 0 0;
      color: #8f9aa6;
      font-size: 12px;
      line-height: 1.35;
    }}

    .action-row {{
      margin-top: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-top: 1px solid rgba(255,255,255,.08);
      padding-top: 11px;
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
.provision-btn {{
      display: inline-block;
      margin-top: 16px;
      padding: 12px 15px;
      border-radius: 10px;
      border: 1px solid rgba(255,122,26,.28);
      background: rgba(8,12,18,.66);
      color: #ffffff;
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 10px;
      line-height: 13px;
      letter-spacing: 1.7px;
      text-transform: uppercase;
      font-weight: 800;
      text-decoration: none;
      box-shadow: inset 0 1px 0 rgba(255,255,255,.045);
    }}

    .provision-btn:hover {{
      border-color: rgba(255,122,26,.46);
      background: rgba(255,122,26,.075);
    }}

    @media (max-width: 760px) {{

      .hero {{ grid-template-columns: 1fr; }}
      .summary-card {{ padding: 14px; }}
      .hero-title {{
        font-size: clamp(28px, 10vw, 40px);
        line-height: 0.98;
      }}
      .hero-sub {{
        margin-top: 10px;
        font-size: 13px;
        line-height: 1.55;
      }}

      .cards {{ grid-template-columns: 1fr; }}
      .card-top {{ align-items: flex-start; }}
      .forensic-strip {{ grid-template-columns: 1fr; }}
    }}
  

    .field-guide-pill {{
      position: fixed;
      right: max(24px, calc((100vw - 1120px) / 2 + 18px));
      bottom: 24px;
      z-index: 80;
      border: 1px solid rgba(255,255,255,.16);
      background: rgba(8,11,15,.92);
      color: #ffffff;
      border-radius: 999px;
      padding: 11px 14px;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: .16em;
      font-weight: 900;
      text-transform: uppercase;
      cursor: pointer;
      box-shadow: 0 0 26px rgba(245,196,81,.14);
    }}

    .field-guide-pill:hover {{
      border-color: rgba(204,255,0,.42);
      color: #ccff00;
    }}

    .field-guide-backdrop {{
      position: fixed;
      inset: 0;
      z-index: 90;
      background: rgba(0,0,0,.54);
      opacity: 0;
      pointer-events: none;
      transition: opacity .22s ease;
    }}

    .field-guide-drawer {{
      position: fixed;
      top: 0;
      right: 0;
      z-index: 91;
      width: min(430px, 92vw);
      height: 100vh;
      overflow: auto;
      padding: 28px;
      background:
        linear-gradient(180deg, rgba(18,22,28,.98), rgba(5,8,12,.99));
      border-left: 1px solid rgba(255,255,255,.14);
      box-shadow: -24px 0 80px rgba(0,0,0,.55), -4px 0 30px rgba(255,122,26,.10);
      transform: translateX(105%);
      transition: transform .26s ease;
    }}

    body.field-guide-open .field-guide-backdrop {{
      opacity: 1;
      pointer-events: auto;
    }}

    body.field-guide-open .field-guide-drawer {{
      transform: translateX(0);
    }}

    .field-guide-top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 22px;
    }}

    .field-guide-kicker {{
      color: #ff9a3d;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: .2em;
      text-transform: uppercase;
      font-weight: 900;
      margin-bottom: 8px;
    }}

    .field-guide-title {{
      margin: 0;
      color: #fff;
      font-family: var(--sans);
      font-size: 30px;
      line-height: .94;
      letter-spacing: -.05em;
      text-transform: uppercase;
      font-weight: 950;
    }}

    .field-guide-close {{
      border: 1px solid rgba(255,255,255,.18);
      background: rgba(255,255,255,.04);
      color: #fff;
      border-radius: 999px;
      width: 34px;
      height: 34px;
      cursor: pointer;
      font-size: 18px;
      line-height: 1;
    }}

    .guide-item {{
      border-top: 1px solid rgba(255,255,255,.10);
      padding: 15px 0;
    }}

    .guide-item strong {{
      display: block;
      color: #fff;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: .18em;
      text-transform: uppercase;
      margin-bottom: 7px;
    }}

    .guide-item p {{
      margin: 0;
      color: #aeb8c4;
      font-size: 13px;
      line-height: 1.5;
    }}

    @media (max-width: 700px) {{
      .field-guide-pill {{
      position: fixed;
      right: max(24px, calc((100vw - 1120px) / 2 + 18px));
      bottom: 24px;
      z-index: 80;
      border: 1px solid rgba(255,255,255,.16);
      background: rgba(8,11,15,.92);
      color: #ffffff;
      border-radius: 999px;
      padding: 11px 14px;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: .16em;
      font-weight: 900;
      text-transform: uppercase;
      cursor: pointer;
      box-shadow: 0 0 26px rgba(245,196,81,.14);
    }}
      .field-guide-drawer {{
        padding: 22px;
      }}
    }}


    /* FINAL APEX HEADER NORMALIZATION */
    .topbar {{
      position: sticky !important;
      top: 0 !important;
      z-index: 50 !important;
      background: rgba(8, 8, 8, 0.90) !important;
      backdrop-filter: blur(10px) !important;
      border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    }}

    .topbar-inner {{
      width: min(1180px, calc(100% - 24px)) !important;
      margin: 0 auto !important;
      min-height: 62px !important;
      display: flex !important;
      align-items: center !important;
      justify-content: space-between !important;
      gap: 12px !important;
      padding: 12px 0 !important;
    }}

    .topbar .brand {{
      display: flex !important;
      align-items: center !important;
      flex-direction: row !important;
      gap: 10px !important;
      min-width: 0 !important;
    }}

    .topbar .brand-mark {{
      width: 14px !important;
      height: 14px !important;
      min-width: 14px !important;
      min-height: 14px !important;
      border-radius: 999px !important;
      background: #ccff00 !important;
      animation: dsLivePulse 1.15s ease-in-out infinite !important;
      box-shadow: 0 0 0 0 rgba(204,255,0,0.55), 0 0 12px rgba(204,255,0,0.55), 0 0 22px rgba(204,255,0,0.28) !important;
      flex: 0 0 auto !important;
      font-size: 0 !important;
      line-height: 0 !important;
    }}

    .topbar .brand-kicker {{
      font-family: var(--mono) !important;
      font-size: 10px !important;
      line-height: 1 !important;
      letter-spacing: 0.18em !important;
      text-transform: uppercase !important;
      font-weight: 800 !important;
      margin-bottom: 4px !important;
    }}

    .topbar .brand-white {{ color: var(--text) !important; }}
    .topbar .brand-blue {{ color: var(--brand-blue, #4ea3ff) !important; }}

    .topbar .brand-title {{
      font-family: var(--sans) !important;
      font-size: 16px !important;
      line-height: 1.05 !important;
      letter-spacing: -0.02em !important;
      font-weight: 800 !important;
      color: var(--text) !important;
      text-transform: none !important;
    }}

    .topbar .livebox {{
      justify-self: auto !important;
      text-align: right !important;
    }}

    .topbar .live-label {{
      font-family: var(--mono) !important;
      font-size: 11px !important;
      letter-spacing: 0.14em !important;
      text-transform: uppercase !important;
      color: var(--soft, #8a95a3) !important;
      display: inline-flex !important;
      align-items: center !important;
      gap: 8px !important;
    }}

    .topbar .live-dot {{
      width: 14px !important;
      height: 14px !important;
      min-width: 14px !important;
      min-height: 14px !important;
      border-radius: 999px !important;
      background: #ccff00 !important;
      animation: dsLivePulse 1.15s ease-in-out infinite !important;
      box-shadow: 0 0 0 0 rgba(204,255,0,0.55), 0 0 12px rgba(204,255,0,0.55), 0 0 22px rgba(204,255,0,0.28) !important;
    }}

    .topbar-center {{
      display: none !important;
    }}

    .field-guide-pill {{
      position: fixed !important;
      right: 24px !important;
      bottom: 24px !important;
      z-index: 90 !important;
    }}

    @media (max-width: 640px) {{
      .topbar-inner {{
        width: min(100%, calc(100% - 16px)) !important;
      }}

      .topbar .brand-title {{
        font-size: 14px !important;
      }}

      .field-guide-pill {{
        right: 14px !important;
        bottom: 14px !important;
      }}
    }}

    /* FINAL APEX BRAND FONT NORMALIZATION */
    .topbar .brand-kicker,
    .topbar .brand-white,
    .topbar .brand-blue {{
      font-family: var(--sans) !important;
      font-weight: 900 !important;
      letter-spacing: .18em !important;
      line-height: 1 !important;
    }}

    /* FINAL APEX FIELD GUIDE POSITION NORMALIZATION */
    .field-guide-pill {{
      position: fixed !important;
      right: max(22px, calc((100vw - 1180px) / 2 + 22px)) !important;
      bottom: 22px !important;
      z-index: 95 !important;
    }}

    @media (max-width: 1280px) {{
      .field-guide-pill {{
        right: 22px !important;
        bottom: 22px !important;
      }}
    }}

    @media (max-width: 640px) {{
      .field-guide-pill {{
        right: 14px !important;
        bottom: 14px !important;
      }}
    }}

  </style>
</head>
<body>
  <div class="field-guide-backdrop" data-close-field-guide></div>
  <aside class="field-guide-drawer" aria-label="Apex Extraction Field Guide">
    <div class="field-guide-top">
      <div>
        <div class="field-guide-kicker">Operator Guide</div>
        <h2 class="field-guide-title">Apex Extraction</h2>
      </div>
      <button class="field-guide-close" type="button" data-close-field-guide aria-label="Close Field Guide">×</button>
    </div>

    <div class="guide-item">
      <strong>Apex Score</strong>
      <p>Weighted extraction score built from Physical, Vision, and Market Latency. It is the top-line conviction score, not a single raw stat.</p>
    </div>
    <div class="guide-item">
      <strong>Physical</strong>
      <p>Measures underlying force or shape change: exit velocity, contact authority, pitch movement, carry, extension, or comparable physical traits.</p>
    </div>
    <div class="guide-item">
      <strong>Vision</strong>
      <p>Measures whether the player is converting the physical trait into usable skill: swing plane, zone control, whiff behavior, command, or deception.</p>
    </div>
    <div class="guide-item">
      <strong>Market</strong>
      <p>Measures the gap between the player’s underlying signal and public recognition. High market latency means the market may still be late.</p>
    </div>
    <div class="guide-item">
      <strong>Heat Fade</strong>
      <p>Orange is reserved for higher-conviction Apex profiles. As score degrades, cards cool from orange into bronze and steel.</p>
    </div>
    <div class="guide-item">
      <strong>Provision To Watchlist</strong>
      <p>Stages the player for roster surveillance. This is the action layer between signal discovery and roster execution.</p>
    </div>
  </aside>
  <div class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <div class="brand-mark"></div>
        <div class="brand-text">
          <div class="brand-kicker"><span class="brand-white">DIAMOND</span><span class="brand-blue">SIGNALS</span></div>
          <div class="brand-title">Apex Extraction // Institutional Edge</div>
        </div>
      </div>

      <div class="livebox">
        <div class="live-label"><span class="live-dot"></span>LIVE</div>
        <div class="live-time">{generated_label}</div>
      </div>
    </div>
  </div>

  <button
    class="field-guide-pill"
    type="button"
    data-open-field-guide
    aria-label="Open Apex Extraction Field Guide"
  >
    <span class="field-guide-icon">ⓘ</span>
    <span>Field Guide</span>
  </button>

  {nav_html}
  {search_html}
  <div class="app">
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">DIAMONDSIGNALS // SUBSURFACE BREAKOUT LEDGER</div>
        <h1 class="hero-title">APEX EXTRACTION</h1>
        <p class="hero-sub">
          Physical shift first. Vision and discipline confirm it. Market latency makes it actionable.
          This surface identifies MLB players whose underlying profile is moving before public pricing catches up.
        </p>
      </div>

      <aside class="summary-card">
        <div>
          <div class="summary-label">Mode</div>
          <div class="summary-value">APEX</div>
        </div>
        <div>
          <div class="summary-label">Candidates</div>
          <div class="summary-value">{payload["counts"]["total"]}</div>
        </div>
        <div>
          <div class="summary-label">Apex Bats</div>
          <div class="summary-value">{payload["counts"]["bats"]}</div>
        </div>
        <div>
          <div class="summary-label">Apex Arms</div>
          <div class="summary-value">{payload["counts"]["arms"]}</div>
        </div>
        <div class="status-note">Cluster-gated extraction surface. Heat fades as conviction decays.</div>
      </aside>
    </section>

    <div class="section-head">
      <div>
        <div class="section-kicker">LATENT ALPHA</div>
        <h2 class="section-heading">HITTER EXTRACTIONS</h2>
      </div>
      <div class="section-title">&gt;_ APEX BATS // GENERATED {generated}</div>
    </div>
    <section class="cards">
      {bat_cards}
    </section>

    <div class="section-head">
      <div>
        <div class="section-kicker">APEX ARM ARRAY</div>
        <h2 class="section-heading">PITCHER EXTRACTIONS</h2>
      </div>
      <div class="section-title">&gt;_ PITCHING GEOMETRY + DECEPTION ARRAY</div>
    </div>
    <section class="cards">
      {arm_cards}
    </section>
  </div>



  <script src="/player-search.js"></script>
  <script>
    (function () {{
      const params = new URLSearchParams(window.location.search);
      const targetPlayerId = (params.get("player") || "").trim();
      const targetPlayerName = (params.get("player_name") || params.get("name") || "").trim().toUpperCase();

      if (!targetPlayerId && !targetPlayerName) return;

      function normalize(value) {{
        return String(value || "")
          .trim()
          .toUpperCase()
          .replace(/\s+/g, " ");
      }}

      function findTargetCard() {{
        const cards = Array.from(document.querySelectorAll(".apex-card"));

        if (targetPlayerId) {{
          const byId = cards.find((card) => String(card.dataset.playerId || "").trim() === targetPlayerId);
          if (byId) return byId;
        }}

        if (targetPlayerName) {{
          const wanted = normalize(targetPlayerName);
          const exact = cards.find((card) => normalize(card.dataset.playerName) === wanted);
          if (exact) return exact;

          return cards.find((card) => normalize(card.dataset.playerName).includes(wanted) || wanted.includes(normalize(card.dataset.playerName)));
        }}

        return null;
      }}

      function activateTarget() {{
        const card = findTargetCard();
        if (!card) return;

        card.classList.remove("target-signal-card");
        void card.offsetWidth;
        card.classList.add("target-signal-card");

        setTimeout(() => {{
          card.scrollIntoView({{
            behavior: "smooth",
            block: "center"
          }});
        }}, 300);
      }}

      if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", activateTarget);
      }} else {{
        activateTarget();
      }}
    }})();
  </script>


  <script>
    (function () {{
      const openBtn = document.querySelector("[data-open-field-guide]");
      const closeEls = document.querySelectorAll("[data-close-field-guide]");

      if (!openBtn) return;

      openBtn.addEventListener("click", () => {{
        document.body.classList.add("field-guide-open");
      }});

      closeEls.forEach((el) => {{
        el.addEventListener("click", () => {{
          document.body.classList.remove("field-guide-open");
        }});
      }});

      document.addEventListener("keydown", (event) => {{
        if (event.key === "Escape") {{
          document.body.classList.remove("field-guide-open");
        }}
      }});
    }})();
  </script>


  <script>
    (function () {{
      const cards = Array.from(document.querySelectorAll(".js-player-action-card"));
      const closeEls = document.querySelectorAll("[data-close-player-action]");

      const fields = {{
        meta: document.getElementById("paMeta"),
        name: document.getElementById("paName"),
        score: document.getElementById("paScore"),
        physical: document.getElementById("paPhysical"),
        vision: document.getElementById("paVision"),
        market: document.getElementById("paMarket"),
        verdict: document.getElementById("paVerdict"),
        metric: document.getElementById("paMetric"),
        action: document.getElementById("paAction"),
        provision: document.getElementById("paProvision"),
        dossier: document.getElementById("paDossier"),
        copy: document.getElementById("paCopy")
      }};

      function openDrawer(card) {{
        const d = card.dataset;
        const playerId = d.playerId || "";
        const playerName = d.playerName || "PLAYER";

        fields.meta.textContent = `${{d.playerTeam || "MLB"}} // ${{d.playerRole || "ASSET"}} // ${{d.signalFamily || "APEX"}}`;
        fields.name.textContent = playerName;
        fields.score.textContent = d.apexScore || "--";
        fields.physical.textContent = d.physical || "--";
        fields.vision.textContent = d.vision || "--";
        fields.market.textContent = d.market || "--";
        fields.verdict.textContent = d.verdict || "--";
        fields.metric.textContent = d.supportingMetric || "--";
        fields.action.textContent = d.action || "--";

        fields.provision.href = `https://app.diamondsignals.ai/auth?next=/watchlist&player_id=${{encodeURIComponent(playerId)}}&source=apex-extraction`;
        fields.dossier.href = `/player/${{encodeURIComponent(playerId)}}/`;

        fields.copy.onclick = async () => {{
          const copyText = `${{playerName}} // Apex Score ${{d.apexScore || "--"}} // ${{d.verdict || ""}} // ${{d.supportingMetric || ""}}`;
          try {{
            await navigator.clipboard.writeText(copyText);
            fields.copy.textContent = "COPIED";
            setTimeout(() => fields.copy.textContent = "COPY SIGNAL", 1100);
          }} catch (err) {{
            fields.copy.textContent = "COPY FAILED";
            setTimeout(() => fields.copy.textContent = "COPY SIGNAL", 1100);
          }}
        }};

        document.body.classList.add("player-action-open");
      }}

      cards.forEach((card) => {{
        card.addEventListener("click", (event) => {{
          if (event.target.closest("a, button")) return;
          openDrawer(card);
        }});
      }});

      closeEls.forEach((el) => {{
        el.addEventListener("click", () => document.body.classList.remove("player-action-open"));
      }});

      document.addEventListener("keydown", (event) => {{
        if (event.key === "Escape") {{
          document.body.classList.remove("player-action-open");
        }}
      }});
    }})();
  </script>

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
