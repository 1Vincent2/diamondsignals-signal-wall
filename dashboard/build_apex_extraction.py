from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


OUT_DIR = Path("dist/apex-extraction")
OUT_JSON = OUT_DIR / "apex_extraction.json"


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON} with {payload['counts']['total']} Apex candidates")


if __name__ == "__main__":
    main()
