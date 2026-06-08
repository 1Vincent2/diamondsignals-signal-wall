from __future__ import annotations

import json
from pathlib import Path
try:
    from dashboard.lib.report_status import build_report_status, utc_now_iso
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dashboard.lib.report_status import build_report_status, utc_now_iso
from datetime import datetime, timezone

from jinja2 import Template


OUT_DIR = Path("dist/apex-extraction")
OUT_JSON = OUT_DIR / "apex_extraction.json"
OUT_HTML = OUT_DIR / "index.html"
STATUS_DIR = Path("dist/status")
APEX_EXTRACTION_STATUS_PATH = STATUS_DIR / "apex-extraction.json"

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
NAV_TEMPLATE = (TEMPLATES_DIR / "shell_nav.html").read_text(encoding="utf-8")
SHELL_NAV_V2_TEMPLATE = (TEMPLATES_DIR / "shell_nav_v2.html").read_text(encoding="utf-8")
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
          data-profile-url="/scout/{row.get("player_id", "")}/"
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
          <div class="audit-row">
            <a class="audit-link" href="/scout/{row.get("player_id", "")}/">PERFORMANCE AUDIT</a>
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
    nav_v2_html = Template(SHELL_NAV_V2_TEMPLATE).render(active_nav="apex_extraction")
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



    .audit-row {{
      margin-top: 14px;
      display: flex;
      align-items: center;
      justify-content: flex-start;
    }}

    .audit-link {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: rgba(148, 163, 184, 0.86);
      text-decoration: none;
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(2, 6, 23, 0.34);
      transition: border-color 160ms ease, color 160ms ease, background 160ms ease;
    }}

    .audit-link::before {{
      content: "↳";
      color: rgba(251, 146, 60, 0.78);
      font-size: 12px;
      letter-spacing: 0;
    }}

    .audit-link:hover {{
      color: rgba(255, 255, 255, 0.94);
      border-color: rgba(251, 146, 60, 0.46);
      background: rgba(251, 146, 60, 0.08);
    }}

    .provision-btn {{
          width: 100% !important;
          text-align: center !important;
          font-size: 9px !important;
          letter-spacing: 0.14em !important;
        }}

        .field-guide-pill {{
          right: 14px !important;
          bottom: 14px !important;
          padding: 10px 12px !important;
          font-size: 9px !important;
          letter-spacing: 0.14em !important;
        }}

        .field-guide-drawer {{
          width: calc(100vw - 16px) !important;
          padding: 22px !important;
        }}
      }}

      /* APEX_HIDE_MOBILE_SUMMARY_GRID_V1 */
      @media screen and (max-width: 760px) {{
        .hero .summary-card {{
          display: none !important;
        }}
      }}

      /* APEX_HIDE_MOBILE_SEARCH_STRIP_V1 */
      @media screen and (max-width: 760px) {{
        .search-strip {{
          display: none !important;
        }}
      }}

      /* APEX_MOBILE_TYPOGRAPHY_ALIGN_V1 */
      @media screen and (max-width: 760px) {{
        .hero-title {{
          text-transform: none !important;
          font-size: 34px !important;
          line-height: 1.04 !important;
          letter-spacing: -0.045em !important;
          font-weight: 650 !important;
        }}

        .section-kicker {{
          font-size: 10px !important;
          line-height: 1 !important;
          letter-spacing: 0.14em !important;
          font-weight: 800 !important;
        }}

        .section-heading {{
          font-size: 24px !important;
          line-height: 1.02 !important;
          letter-spacing: -0.035em !important;
          font-weight: 760 !important;
          text-transform: uppercase !important;
        }}

        .section-title {{
          font-size: 10px !important;
          line-height: 1.35 !important;
          letter-spacing: 0.16em !important;
          font-weight: 800 !important;
        }}

        .apex-card {{
          border-radius: 22px !important;
          padding: 14px !important;
        }}

        .kicker {{
          font-size: 9px !important;
          line-height: 1.2 !important;
          letter-spacing: 0.16em !important;
          font-weight: 800 !important;
        }}

        .name {{
          font-size: 27px !important;
          line-height: 0.98 !important;
          letter-spacing: -0.045em !important;
          font-weight: 760 !important;
        }}

        .meta {{
          font-size: 9px !important;
          line-height: 1.25 !important;
          letter-spacing: 0.16em !important;
        }}

        .score-label,
        .diagnosis-label,
        .proof-label {{
          font-size: 9px !important;
          line-height: 1 !important;
          letter-spacing: 0.16em !important;
          font-weight: 800 !important;
        }}

        .score {{
          font-size: 48px !important;
          line-height: 0.86 !important;
          letter-spacing: -0.06em !important;
          font-weight: 850 !important;
        }}

        .verdict-text {{
          font-size: 10px !important;
          line-height: 1.35 !important;
          letter-spacing: 0.14em !important;
          padding: 10px 12px !important;
        }}

        .grid {{
          grid-template-columns: 1fr !important;
          gap: 8px !important;
        }}

        .grid div {{
          padding: 9px 10px !important;
        }}

        .grid span {{
          font-size: 9px !important;
          line-height: 1 !important;
          letter-spacing: 0.16em !important;
        }}

        .grid strong {{
          font-size: 18px !important;
          line-height: 20px !important;
        }}
      }}
        .hero-sub {{
          margin-top: 0.5rem !important;
        }}
      }}
        .section-heading {{
          font-size: 21px !important;
          line-height: 1 !important;
          letter-spacing: -0.03em !important;
          font-weight: 760 !important;
        }}

        .section-title {{
          font-size: 9px !important;
          line-height: 1.25 !important;
          letter-spacing: 0.14em !important;
        }}

        .apex-card {{
          padding: 12px !important;
          border-radius: 20px !important;
        }}

        .kicker {{
          font-size: 8px !important;
          line-height: 1.15 !important;
          letter-spacing: 0.14em !important;
        }}

        .name {{
          font-size: 23px !important;
          line-height: 0.98 !important;
          letter-spacing: -0.04em !important;
          font-weight: 760 !important;
        }}

        .meta {{
          font-size: 10px !important;
          line-height: 1.2 !important;
          letter-spacing: 0.13em !important;
        }}

        .score-label,
        .diagnosis-label,
        .proof-label {{
          font-size: 8px !important;
          line-height: 1 !important;
          letter-spacing: 0.14em !important;
        }}

        .score {{
          font-size: 38px !important;
          line-height: 0.86 !important;
          letter-spacing: -0.055em !important;
        }}

        .verdict-text {{
          font-size: 9px !important;
          line-height: 1.25 !important;
          letter-spacing: 0.13em !important;
          padding: 9px 10px !important;
        }}

        .grid div {{
          padding: 8px 9px !important;
        }}

        .grid span {{
          font-size: 8px !important;
          line-height: 1 !important;
          letter-spacing: 0.14em !important;
        }}

        .grid strong {{
          font-size: 12px !important;
          line-height: 18px !important;
        }}

        .proof-copy,
        .primary-signal,
        .apex-card p {{
          font-size: 12px !important;
          line-height: 1.45 !important;
        }}
      }}

      /* APEX_MOBILE_FINAL_CARD_DENSITY_V2 */
      @media screen and (max-width: 760px) {{
        section.cards {{
          gap: 8px !important;
        }}

        section.cards .apex-card {{
          padding: 12px !important;
          border-radius: 20px !important;
        }}

        section.cards .apex-card .kicker {{
          font-size: 8px !important;
          line-height: 1.1 !important;
          letter-spacing: 0.14em !important;
          font-weight: 800 !important;
        }}

        section.cards .apex-card .name {{
          font-size: 22px !important;
          line-height: 0.96 !important;
          letter-spacing: -0.045em !important;
          font-weight: 760 !important;
        }}

        section.cards .apex-card .meta {{
          font-size: 10px !important;
          line-height: 1.2 !important;
          letter-spacing: 0.12em !important;
        }}

        section.cards .apex-card .score-label,
        section.cards .apex-card .diagnosis-label,
        section.cards .apex-card .proof-label {{
          font-size: 8px !important;
          line-height: 1 !important;
          letter-spacing: 0.14em !important;
        }}

        section.cards .apex-card .score {{
          font-size: 36px !important;
          line-height: 0.84 !important;
          letter-spacing: -0.06em !important;
          font-weight: 850 !important;
        }}

        section.cards .apex-card .verdict-text {{
          font-size: 8px !important;
          line-height: 1.2 !important;
          letter-spacing: 0.12em !important;
          padding: 8px 9px !important;
        }}

        section.cards .apex-card .grid {{
          grid-template-columns: 1fr !important;
          gap: 7px !important;
        }}

        section.cards .apex-card .grid div {{
          padding: 8px 9px !important;
        }}

        section.cards .apex-card .grid span {{
          font-size: 8px !important;
          line-height: 1 !important;
          letter-spacing: 0.13em !important;
        }}

        section.cards .apex-card .grid strong {{
          font-size: 15px !important;
          line-height: 17px !important;
        }}

        section.cards .apex-card p,
        section.cards .apex-card .proof-copy,
        section.cards .apex-card .primary-signal {{
          font-size: 11px !important;
          line-height: 1.38 !important;
        }}
      }}












      /* APEX_MOBILE_GRID_LOCK_V1
         Mobile-only containment: player cards and card detail metrics must collapse to one column.
         Does not touch desktop or global shell nav. */
      @media screen and (max-width: 760px) {{
        html,
        body {{
          max-width: 100% !important;
          overflow-x: hidden !important;
        }}

        .app {{
          width: min(100%, calc(100vw - 24px)) !important;
          max-width: calc(100vw - 24px) !important;
          overflow-x: hidden !important;
        }}

        .hero {{
          grid-template-columns: 1fr !important;
          width: 100% !important;
          max-width: 100% !important;
        }}

        section.cards,
        .cards {{
          display: grid !important;
          grid-template-columns: 1fr !important;
          width: 100% !important;
          max-width: 100% !important;
          overflow-x: hidden !important;
        }}

        section.cards .apex-card,
        .apex-card {{
          width: 100% !important;
          max-width: 100% !important;
          min-width: 0 !important;
          overflow: hidden !important;
        }}

        section.cards .apex-card .grid,
        .apex-card .grid,
        .grid {{
          display: grid !important;
          grid-template-columns: 1fr !important;
          width: 100% !important;
          max-width: 100% !important;
          min-width: 0 !important;
        }}

        section.cards .apex-card .forensic-strip,
        .apex-card .forensic-strip,
        .forensic-strip {{
          grid-template-columns: 1fr !important;
          width: 100% !important;
          max-width: 100% !important;
          min-width: 0 !important;
        }}

        .grid > *,
        .forensic-strip > * {{
          min-width: 0 !important;
          max-width: 100% !important;
        }}
      }}


      /* APEX_DESKTOP_FLOATING_FIELD_GUIDE_PILL_V1
         Desktop-only fix: make standalone Apex Field Guide match Signal Wall floating pill behavior.
         Mobile intentionally untouched. */
      @media screen and (min-width: 761px) {{
        .field-guide-pill {{
          position: fixed !important;
          right: max(24px, calc((100vw - 1180px) / 2 + 24px)) !important;
          bottom: 24px !important;
          top: auto !important;
          left: auto !important;
          z-index: 2147483002 !important;
          appearance: none !important;
          -webkit-appearance: none !important;
          display: inline-flex !important;
          align-items: center !important;
          justify-content: center !important;
          gap: 10px !important;
          min-height: 46px !important;
          padding: 0 18px !important;
          border-radius: 999px !important;
          border: 1px solid rgba(204,255,0,0.38) !important;
          background:
            radial-gradient(circle at 18% 50%, rgba(204,255,0,0.14), transparent 42%),
            rgba(7,10,14,0.92) !important;
          color: #ffffff !important;
          box-shadow:
            0 0 22px rgba(204,255,0,0.10),
            inset 0 1px 0 rgba(255,255,255,0.08) !important;
          backdrop-filter: blur(14px) !important;
          font-family: var(--mono, "JetBrains Mono", "Roboto Mono", monospace) !important;
          font-size: 11px !important;
          font-weight: 900 !important;
          letter-spacing: 0.16em !important;
          line-height: 1 !important;
          text-transform: uppercase !important;
          cursor: pointer !important;
          white-space: nowrap !important;
        }}

        .field-guide-pill:hover {{
          transform: translateY(-1px) !important;
          border-color: rgba(204,255,0,0.58) !important;
          box-shadow:
            0 0 28px rgba(204,255,0,0.18),
            inset 0 1px 0 rgba(255,255,255,0.10) !important;
        }}

        .field-guide-icon {{
          color: #ccff00 !important;
          font-size: 14px !important;
          line-height: 1 !important;
        }}
      }}






      /* APEX_MOBILE_MENU_MATCHED_FIELD_GUIDE_V3
         Mobile/tablet fix: remove Field Guide from topbar rail and restyle it to match the mobile Menu pill.
         Desktop Field Guide remains controlled by APEX_DESKTOP_FLOATING_FIELD_GUIDE_PILL_V1. */
      @media screen and (max-width: 900px) {{
        .topbar-center {{
          display: block !important;
          width: 0 !important;
          height: 0 !important;
          min-height: 0 !important;
          padding: 0 !important;
          margin: 0 !important;
          overflow: visible !important;
        }}

        .topbar .field-guide-pill,
        .field-guide-pill {{
          position: fixed !important;
          right: 14px !important;
          bottom: calc(16px + env(safe-area-inset-bottom)) !important;
          top: auto !important;
          left: auto !important;
          z-index: 2147483002 !important;

          width: auto !important;
          max-width: calc(100vw - 28px) !important;
          min-width: 0 !important;
          min-height: 42px !important;
          padding: 0 16px !important;

          display: inline-flex !important;
          align-items: center !important;
          justify-content: center !important;
          gap: 8px !important;

          border-radius: 999px !important;
          border: 1px solid rgba(204,255,0,0.58) !important;
          background:
            radial-gradient(circle at 24% 50%, rgba(204,255,0,0.24), transparent 38%),
            linear-gradient(180deg, rgba(8,12,16,0.98), rgba(2,4,7,0.99)) !important;
          color: #f8fafc !important;

          box-shadow:
            0 0 0 1px rgba(204,255,0,0.10),
            0 0 18px rgba(204,255,0,0.24),
            0 0 34px rgba(204,255,0,0.12),
            inset 0 1px 0 rgba(255,255,255,0.10) !important;

          backdrop-filter: blur(16px) !important;
          -webkit-backdrop-filter: blur(16px) !important;

          font-family: var(--mono, "JetBrains Mono", "Roboto Mono", monospace) !important;
          font-size: 9px !important;
          font-weight: 900 !important;
          letter-spacing: 0.14em !important;
          line-height: 1 !important;
          text-transform: uppercase !important;
          white-space: nowrap !important;
          cursor: pointer !important;
          transform: none !important;
        }}

        .topbar .field-guide-pill:hover,
        .field-guide-pill:hover {{
          transform: none !important;
          border-color: rgba(204,255,0,0.70) !important;
        }}

        .topbar .field-guide-icon,
        .field-guide-icon {{
          color: #ccff00 !important;
          font-size: 12px !important;
          line-height: 1 !important;
          text-shadow:
            0 0 12px rgba(204,255,0,0.72),
            0 0 24px rgba(204,255,0,0.38) !important;
        }}

        .field-guide-drawer {{
          width: min(100vw, 430px) !important;
          max-width: 100vw !important;
        }}
      }}




      /* APEX_FIELD_GUIDE_DRAWER_CLOSE_FIX_V1
         Removes inherited white button artifact and pins close control top-right inside drawer. */
      .field-guide-top {{
        position: relative !important;
        padding-right: 58px !important;
      }}

      .field-guide-close {{
        position: absolute !important;
        top: 18px !important;
        right: 18px !important;
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        min-height: 36px !important;

        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;

        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 999px !important;
        background: rgba(255,255,255,0.035) !important;
        color: rgba(255,255,255,0.90) !important;

        box-shadow: none !important;
        appearance: none !important;
        -webkit-appearance: none !important;

        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 24px !important;
        font-weight: 300 !important;
        line-height: 1 !important;
        letter-spacing: 0 !important;
        text-transform: none !important;
        cursor: pointer !important;
      }}

      .field-guide-close:hover {{
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(204,255,0,0.34) !important;
        color: #ffffff !important;
      }}

      @media screen and (max-width: 900px) {{
        .field-guide-top {{
          padding-right: 54px !important;
        }}

        .field-guide-close {{
          top: 16px !important;
          right: 16px !important;
          width: 34px !important;
          height: 34px !important;
          min-width: 34px !important;
          min-height: 34px !important;
          font-size: 23px !important;
        }}
      }}



      /* APEX_DESKTOP_TOPBAR_LOCKUP_ALIGN_V2
         Desktop-only: align Apex logo lockup with nav rail, put green dot left of logo,
         keep SIGNALS blue, preserve mobile layout and mobile drawer untouched. */
      @media screen and (min-width: 761px) {{
        .topbar-inner {{
          width: min(1180px, calc(100% - 48px)) !important;
          margin: 0 auto !important;
          min-height: 148px !important;
          display: grid !important;
          grid-template-columns: minmax(0, 1fr) auto !important;
          align-items: center !important;
          gap: 28px !important;
        }}

        .topbar .brand {{
          display: flex !important;
          flex-direction: row !important;
          align-items: center !important;
          justify-content: flex-start !important;
          gap: 16px !important;
          min-width: 0 !important;
          transform: none !important;
        }}

        .topbar .brand-mark {{
          width: 20px !important;
          height: 20px !important;
          min-width: 20px !important;
          min-height: 20px !important;
          border-radius: 999px !important;
          background: #b6ff00 !important;
          box-shadow:
            0 0 0 1px rgba(182,255,0,0.30),
            0 0 20px rgba(182,255,0,0.82),
            0 0 42px rgba(182,255,0,0.38) !important;
          animation: dsApexLivePulse 1.15s ease-in-out infinite !important;
          flex: 0 0 auto !important;
          margin: 0 !important;
        }}

        .topbar .brand-text {{
          display: flex !important;
          flex-direction: column !important;
          align-items: flex-start !important;
          justify-content: center !important;
          gap: 5px !important;
          min-width: 0 !important;
        }}

        .topbar .brand-kicker {{
          display: inline-flex !important;
          align-items: baseline !important;
          gap: 0 !important;
          font-family: var(--sans) !important;
          font-size: 22px !important;
          line-height: 1 !important;
          letter-spacing: -0.02em !important;
          font-weight: 950 !important;
          text-transform: uppercase !important;
        }}

        .topbar .brand-kicker .brand-white {{
          color: #f8fafc !important;
        }}

        .topbar .brand-kicker .brand-blue {{
          color: #4ea3ff !important;
          text-shadow:
            0 0 10px rgba(78,163,255,0.25),
            0 0 22px rgba(78,163,255,0.16) !important;
        }}

        .topbar .brand-title {{
          font-family: var(--sans) !important;
          font-size: 18px !important;
          line-height: 1.12 !important;
          letter-spacing: -0.025em !important;
          font-weight: 850 !important;
          color: #f8fafc !important;
          text-transform: none !important;
        }}

        .topbar .livebox {{
          justify-self: end !important;
          min-width: 210px !important;
          text-align: left !important;
        }}

        .ds-pro-desktop-nav-inner {{
          width: min(1180px, calc(100% - 48px)) !important;
          margin: 0 auto !important;
        }}

        .ds-pro-desktop-nav-links {{
          justify-content: flex-start !important;
        }}
      }}

      /* APEX_FIELD_GUIDE_CLOSED_STATE_LOCK_V1
         Drawer/backdrop hidden by default; opens only when body.field-guide-open is present. */
      .field-guide-backdrop {{
        position: fixed !important;
        inset: 0 !important;
        z-index: 2147483003 !important;
        background: rgba(0,0,0,0.62) !important;
        opacity: 0 !important;
        pointer-events: none !important;
        transition: opacity 180ms ease !important;
      }}

      body.field-guide-open .field-guide-backdrop {{
        opacity: 1 !important;
        pointer-events: auto !important;
      }}

      .field-guide-drawer {{
        position: fixed !important;
        top: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        left: auto !important;
        width: min(92vw, 430px) !important;
        max-width: 92vw !important;
        height: 100dvh !important;
        z-index: 2147483004 !important;
        transform: translateX(104%) !important;
        overflow-y: auto !important;
        -webkit-overflow-scrolling: touch !important;
        transition: transform 220ms ease !important;
        background:
          radial-gradient(circle at 20% 0%, rgba(204,255,0,0.10), transparent 34%),
          linear-gradient(180deg, rgba(10,13,18,0.98), rgba(3,5,8,0.99)) !important;
        border-left: 1px solid rgba(204,255,0,0.18) !important;
        box-shadow: -24px 0 70px rgba(0,0,0,0.62) !important;
      }}

      body.field-guide-open .field-guide-drawer {{
        transform: translateX(0) !important;
      }}

      body.field-guide-open {{
        overflow: hidden !important;
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

  {nav_v2_html}
  {nav_html}
  {search_html}
  <div class="app">
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">DIAMONDSIGNALS // SUBSURFACE BREAKOUT LEDGER</div>
        <h1 class="hero-title">Apex Extraction</h1>
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
  <script src="/player-card-actions.js"></script>
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

        fields.provision.href = "/watch-list/";
        fields.provision.dataset.playerId = playerId;
        fields.provision.dataset.playerName = playerName;
        fields.provision.dataset.playerTeam = d.playerTeam || "MLB";
        fields.provision.dataset.playerType = d.playerRole || "";
        fields.provision.dataset.profileUrl = `/scout/${{encodeURIComponent(playerId)}}/`;
        fields.provision.dataset.sourceTag = "apex-extraction";
        fields.dossier.href = `/scout/${{encodeURIComponent(playerId)}}/`;

        fields.provision.onclick = (event) => {{
          event.preventDefault();

          const stored = (() => {{
            try {{
              const raw = window.localStorage.getItem("diamondsignals_watch_list_v1");
              const parsed = raw ? JSON.parse(raw) : [];
              return Array.isArray(parsed) ? parsed : [];
            }} catch (err) {{
              return [];
            }}
          }})();

          const nextPlayer = {{
            playerId,
            playerName,
            playerType: d.playerRole || "",
            team: d.playerTeam || "MLB",
            profileUrl: `/scout/${{encodeURIComponent(playerId)}}/`,
            sourceTag: "APEX_EXTRACTION",
            savedAt: new Date().toISOString()
          }};

          const existingIndex = stored.findIndex((p) => {{
            if (nextPlayer.playerId && p.playerId) return String(p.playerId) === String(nextPlayer.playerId);
            return String(p.playerName || "").toLowerCase() === String(nextPlayer.playerName || "").toLowerCase();
          }});

          if (existingIndex >= 0) {{
            stored[existingIndex] = {{ ...stored[existingIndex], ...nextPlayer }};
          }} else {{
            stored.push(nextPlayer);
          }}

          try {{
            window.localStorage.setItem("diamondsignals_watch_list_v1", JSON.stringify(stored));
          }} catch (err) {{
            // Continue to Tracking Radar even if localStorage is unavailable.
          }}

          window.location.href = "/watch-list/";
        }};

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



def write_apex_extraction_status(payload: dict, *, build_started_at: str, build_finished_at: str) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

    counts = payload.get("counts", {}) or {}
    apex_bats = payload.get("apex_bats", []) or []
    apex_arms = payload.get("apex_arms", []) or []
    total = int(counts.get("total") or (len(apex_bats) + len(apex_arms)))

    section_counts = {
        "total_candidates": total,
        "apex_bats": int(counts.get("bats") or len(apex_bats)),
        "apex_arms": int(counts.get("arms") or len(apex_arms)),
    }

    missing_player_ids = []
    missing_names = []

    for row in apex_bats + apex_arms:
        if not row.get("player_id"):
            missing_player_ids.append(str(row.get("name") or "UNKNOWN"))
        if not row.get("name"):
            missing_names.append(str(row.get("player_id") or "UNKNOWN"))

    errors = []
    notes = []

    if total <= 0:
        errors.append("Apex Extraction produced zero candidates.")

    if missing_player_ids:
        errors.append(f"Missing player_id for {len(missing_player_ids)} Apex candidates.")

    if missing_names:
        errors.append(f"Missing player name for {len(missing_names)} Apex candidates.")

    notes.append(f"Apex Extraction built with {total} candidates.")
    notes.append("Performance Audit links are generated from canonical player_id values.")
    notes.append("Apex uses the shared Performance Audit evidence layer while preserving separate Apex selection/scoring logic.")

    status_payload = build_report_status(
        "apex-extraction",
        build_success=(len(errors) == 0),
        threshold_minutes=1440,
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        source_updated_at=payload.get("generated_at"),
        section_counts=section_counts,
        degraded=False,
        errors=errors,
        notes=notes,
    )

    status_payload["mode"] = "real_data_v0.2_hardened"
    status_payload["surface"] = "Apex Extraction"
    status_payload["hardening_notes"] = [
        "Shares canonical player identity / Performance Audit route pattern with MLB Extraction.",
        "Maintains independent Apex scoring and candidate selection logic.",
        "Audit-link integrity is covered by dashboard/audit_apex_extraction_links.py.",
    ]

    APEX_EXTRACTION_STATUS_PATH.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote Apex Extraction status -> {APEX_EXTRACTION_STATUS_PATH}")

def main() -> None:
    build_started_at = utc_now_iso()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_HTML.write_text(render_html(payload), encoding="utf-8")

    build_finished_at = utc_now_iso()
    write_apex_extraction_status(
        payload,
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
    )

    print(f"Wrote {OUT_JSON} with {payload['counts']['total']} Apex candidates")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
