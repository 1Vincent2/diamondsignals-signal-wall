#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from jinja2 import Template

from dashboard.lib.report_status import build_report_status

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DIST_DIR = REPO_ROOT / "dist"
HTML_DIR = DIST_DIR / "waiver-wire"
HTML_PATH = HTML_DIR / "index.html"
JSON_PATH = DIST_DIR / "waiver_wire.json"
STATUS_DIR = DIST_DIR / "status"
WAIVER_WIRE_STATUS_PATH = STATUS_DIR / "waiver-wire.json"

TEMPLATE_PATH = BASE_DIR / "templates" / "waiver_wire.html"
SHELL_STYLES_PATH = BASE_DIR / "templates" / "shell_styles.css"
SHELL_NAV_PATH = BASE_DIR / "templates" / "shell_nav.html"


def build_assets() -> list[dict]:
    """
    V1 uses editable static intelligence cards.
    Later this can be wired to roster %, market-attention feed, Statcast deltas,
    pitch-shape drift, Stuff+ movement, and role-opportunity feeds.
    """
    return [
        {
            "player_name": "Luis L. Ortiz",
            "team": "CLE",
            "position": "SP/RP",
            "rostered_pct": 18,
            "command": "PRIORITY WAIVER CLAIM",
            "command_class": "",
            "market_status": "UNPRICED MOVEMENT CHANGE",
            "surface_profile": (
                "Traditional fantasy managers still see a volatile arm with uneven surface outcomes. "
                "That keeps the acquisition cost suppressed."
            ),
            "forensic_trigger": (
                "Recent pitch-shape indicators point toward a more stable attack path: improved fastball-plane utility, "
                "better extension signal, and a cleaner secondary tunnel profile."
            ),
            "verdict": (
                "Acquire before the next visible box-score event. This is the exact mid-May window where physics changes "
                "can reprice faster than public ownership reacts."
            ),
            "metrics": [
                {"label": "Ownership Gate", "value": "≤20%"},
                {"label": "Signal Window", "value": "72H"},
                {"label": "Asset Type", "value": "Arm"},
                {"label": "Market Defect", "value": "ERA Lag"},
                {"label": "Command", "value": "Claim"},
                {"label": "Risk", "value": "Med"},
            ],
        },
        {
            "player_name": "Hayden Birdsong",
            "team": "SF",
            "position": "SP",
            "rostered_pct": 14,
            "command": "STASH / MONITOR",
            "command_class": "stash",
            "market_status": "ROLE + STUFF WATCH",
            "surface_profile": (
                "The public market has not fully priced the volatility because the role and innings path remain unstable."
            ),
            "forensic_trigger": (
                "Raw pitch traits remain extraction-worthy when command tightens. This profile belongs on the open-market radar "
                "before a cleaner outing forces attention."
            ),
            "verdict": (
                "Do not chase blindly, but provision to watchlist or stash in deeper formats. The upside is visible before the "
                "mainstream ownership curve moves."
            ),
            "metrics": [
                {"label": "Ownership Gate", "value": "≤15%"},
                {"label": "Signal Window", "value": "96H"},
                {"label": "Asset Type", "value": "Arm"},
                {"label": "Market Defect", "value": "Role Fog"},
                {"label": "Command", "value": "Stash"},
                {"label": "Risk", "value": "High"},
            ],
        },
        {
            "player_name": "Ben Rice",
            "team": "NYY",
            "position": "C/1B",
            "rostered_pct": 19,
            "command": "MONITOR BAT-PATH STABILITY",
            "command_class": "monitor",
            "market_status": "HITTER VOLATILITY COMPRESSION",
            "surface_profile": (
                "Casual managers see playing-time uncertainty and category volatility. That suppresses ownership despite a useful "
                "power-path profile."
            ),
            "forensic_trigger": (
                "The hitter version of this surface watches swing-decision stability, damage-zone contact, and volatility compression "
                "before the stat line confirms."
            ),
            "verdict": (
                "Monitor as the hitter-side proof case. If role clarity improves while bat-path stability holds, the wire price can "
                "disappear quickly."
            ),
            "metrics": [
                {"label": "Ownership Gate", "value": "≤20%"},
                {"label": "Signal Window", "value": "72H"},
                {"label": "Asset Type", "value": "Bat"},
                {"label": "Market Defect", "value": "Role Lag"},
                {"label": "Command", "value": "Track"},
                {"label": "Risk", "value": "Med"},
            ],
        },
    ]



def write_waiver_wire_status(
    *,
    build_started_at: str,
    build_finished_at: str,
    assets: list[dict],
) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)

    asset_count = int(len(assets)) if assets is not None else 0
    degraded = asset_count == 0

    command_counts = {}
    for row in assets or []:
        command = str(row.get("command") or "UNKNOWN").strip() or "UNKNOWN"
        command_counts[command] = command_counts.get(command, 0) + 1

    status_payload = build_report_status(
        "waiver_wire",
        build_success=True,
        threshold_minutes=240,
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        source_updated_at=build_finished_at,
        section_counts={
            "waiver_assets": asset_count,
            "command_groups": len(command_counts),
        },
        degraded=degraded,
        notes=[
            f"Waiver Wire Open Market built with {asset_count} active open-market assets."
        ],
    )

    status_payload["command_counts"] = command_counts
    status_payload["mode"] = "static_editable_v1"

    WAIVER_WIRE_STATUS_PATH.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote Waiver Wire status -> {WAIVER_WIRE_STATUS_PATH}")


def render() -> None:
    build_started_at = datetime.now(timezone.utc).isoformat()
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    assets = build_assets()
    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    shell_nav = Template(SHELL_NAV_PATH.read_text(encoding="utf-8")).render(active_nav="waiver_wire")

    HTML_PATH.write_text(
        template.render(
            assets=assets,
            generated_at=generated_at,
            shell_styles=SHELL_STYLES_PATH.read_text(encoding="utf-8"),
            shell_nav=shell_nav,
        ),
        encoding="utf-8",
    )

    print(f"Wrote waiver wire surface -> {HTML_PATH}")
    print(f"Assets rendered: {len(assets)}")

    payload = {
        "generated_at": generated_at,
        "mode": "static_editable_v1",
        "assets": assets,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote waiver wire payload -> {JSON_PATH}")

    build_finished_at = datetime.now(timezone.utc).isoformat()
    write_waiver_wire_status(
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        assets=assets,
    )


def main() -> None:
    render()


if __name__ == "__main__":
    main()
