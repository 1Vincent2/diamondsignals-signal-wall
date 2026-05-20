#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from jinja2 import Environment, FileSystemLoader, Template

try:
    from dashboard.lib.report_status import build_report_status
    from dashboard.lib.player_operational_status import apply_operational_status
    from dashboard.lib.player_identity import (
        load_canonical_player_universe,
        build_name_lookup,
        resolve_player_identity,
    )
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dashboard.lib.report_status import build_report_status
    from dashboard.lib.player_operational_status import apply_operational_status
    from dashboard.lib.player_identity import (
        load_canonical_player_universe,
        build_name_lookup,
        resolve_player_identity,
    )

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

CANONICAL_PLAYERS = load_canonical_player_universe()
CANONICAL_NAME_LOOKUP = build_name_lookup(CANONICAL_PLAYERS)

WAIVER_IDENTITY_OVERRIDES = {
    ("luis l. ortiz", "cle"): "682847",
    ("hayden birdsong", "sf"): "806185",
    ("aj smith-shawver", "atl"): "700363",
}




def build_assets() -> list[dict]:
    """
    V1 uses editable static intelligence cards.
    Later this can be wired to roster %, market-attention feed, Statcast deltas,
    pitch-shape drift, Stuff+ movement, KDE, role-opportunity feeds, and live availability.
    """
    def asset(
        player_name: str,
        team: str,
        position: str,
        rostered_pct: int,
        command: str,
        command_class: str,
        market_status: str,
        surface_profile: str,
        forensic_trigger: str,
        verdict: str,
        ownership_gate: str,
        signal_window: str,
        asset_type: str,
        market_defect: str,
        command_metric: str,
        risk: str,
        player_id: str = "",
        audit_slug: str = "",
        deployment_state: str = "DEPLOYMENT_CLEAR",
        deployment_label: str = "DEPLOYMENT CLEAR",
        operational_status: str = "ACTIVE",
        status_reason: str = "Status clear at build time",
        card_action: str = "OPEN PERFORMANCE AUDIT",
        visibility_state: str = "primary",
        status_source: str = "default_active",
    ) -> dict:
        identity = resolve_player_identity(
            player_id=player_id,
            player_name=player_name,
            team=team,
            players=CANONICAL_PLAYERS,
            name_lookup=CANONICAL_NAME_LOOKUP,
        )

        resolved_player_id = str(identity.get("player_id") or "").strip()
        player_name = identity.get("player_name") or player_name
        team = identity.get("team") or team
        position = identity.get("position") or position

        override_key = (str(player_name).strip().lower(), str(team).strip().lower())
        if not resolved_player_id and override_key in WAIVER_IDENTITY_OVERRIDES:
            resolved_player_id = WAIVER_IDENTITY_OVERRIDES[override_key]

        safe_slug = audit_slug or player_name.lower().replace(".", "").replace(" ", "-")
        scout_url = str(identity.get("scout_url") or "").strip()
        if scout_url == "#":
            scout_url = ""

        if resolved_player_id:
            scout_url = f"/scout/{resolved_player_id}/"

        watchlist_url = (
            "https://app.diamondsignals.ai/watchlist"
            f"?player_id={resolved_player_id}"
            f"&player_name={player_name.replace(' ', '%20')}"
            "&source=waiver-wire"
        )

        return {
            "player_name": player_name,
            "player_id": resolved_player_id,
            "audit_slug": safe_slug,
            "audit_url": scout_url or watchlist_url,
            "watchlist_url": watchlist_url,
            "has_scout_page": bool(scout_url),
            "identity_source": identity.get("identity_source"),
            "headshot_url": identity.get("headshot_url"),
            "scout_url": scout_url,
            "resolved_player_id": resolved_player_id,
            "team": team,
            "position": position,
            "rostered_pct": rostered_pct,
            "command": command,
            "command_class": command_class,
            "market_status": market_status,
            "surface_profile": surface_profile,
            "forensic_trigger": forensic_trigger,
            "verdict": verdict,
            "deployment_state": deployment_state,
            "deployment_label": deployment_label,
            "operational_status": operational_status,
            "status_reason": status_reason,
            "card_action": card_action,
            "visibility_state": visibility_state,
            "status_source": status_source,
            "search_blob": " ".join([player_name, team, position, command, market_status, deployment_label, operational_status]).lower(),
            "metrics": [
                {"label": "Ownership Gate", "value": ownership_gate},
                {"label": "Signal Window", "value": signal_window},
                {"label": "Asset Type", "value": asset_type},
                {"label": "Market Defect", "value": market_defect},
                {"label": "Command", "value": command_metric},
                {"label": "Risk", "value": risk},
                {"label": "Status Source", "value": status_source},
            ],
        }

    return [
        asset(
            "Luis L. Ortiz",
            "CLE",
            "SP/RP",
            18,
            "PRIORITY WAIVER CLAIM",
            "",
            "UNPRICED MOVEMENT CHANGE",
            "Traditional fantasy managers still see a volatile arm with uneven surface outcomes. That keeps the acquisition cost suppressed.",
            "Recent pitch-shape indicators point toward a more stable attack path: improved fastball-plane utility, better extension signal, and a cleaner secondary tunnel profile.",
            "Acquire before the next visible box-score event. This is the exact window where physics changes can reprice faster than public ownership reacts.",
            "≤20%",
            "72H",
            "Arm",
            "ERA Lag",
            "Claim",
            "Med",
        ),
        asset(
            "Hayden Birdsong",
            "SF",
            "SP",
            14,
            "STASH / MONITOR",
            "stash",
            "ROLE + STUFF WATCH",
            "The public market has not fully priced the volatility because the role and innings path remain unstable.",
            "Raw pitch traits remain extraction-worthy when command tightens. This profile belongs on the open-market radar before a cleaner outing forces attention.",
            "Do not chase blindly, but provision to watchlist or stash in deeper formats. The upside is visible before the mainstream ownership curve moves.",
            "≤15%",
            "96H",
            "Arm",
            "Role Fog",
            "Stash",
            "High",
        ),
        asset(
            "Ben Rice",
            "NYY",
            "C/1B",
            19,
            "MONITOR BAT-PATH STABILITY",
            "monitor",
            "HITTER VOLATILITY COMPRESSION",
            "Casual managers see playing-time uncertainty and category volatility. That suppresses ownership despite a useful power-path profile.",
            "The hitter version of this surface watches swing-decision stability, damage-zone contact, and volatility compression before the stat line confirms.",
            "Monitor as the hitter-side proof case. If role clarity improves while bat-path stability holds, the wire price can disappear quickly.",
            "≤20%",
            "72H",
            "Bat",
            "Role Lag",
            "Track",
            "Med",
        ),
        asset(
            "Edward Cabrera",
            "MIA",
            "SP",
            21,
            "VOLATILITY CLAIM",
            "stash",
            "STUFF OUTPACES TRUST",
            "The market still treats the profile as a command-risk headache, which keeps the acquisition cost below the raw arsenal ceiling.",
            "Velocity, breaking-ball shape, and strikeout-pressure indicators create a classic open-market asymmetry: ugly surface risk hiding impact stuff.",
            "Provision as a controlled upside arm. The window closes quickly if command stabilizes for even one clean start.",
            "≤25%",
            "72H",
            "Arm",
            "Command Discount",
            "Stash",
            "High",
        ),
        asset(
            "AJ Smith-Shawver",
            "ATL",
            "SP",
            12,
            "SPECULATIVE STASH",
            "stash",
            "PROMOTION + ROLE LEVERAGE",
            "Public ownership remains suppressed because the active role path is cloudy, but the skill set can reprice the moment opportunity clears.",
            "This is a role-lag play: prospect pedigree, power arsenal, and rotation fragility create a fast-moving acquisition window.",
            "Track aggressively in deeper formats. The correct move is often before the official role confirmation.",
            "≤15%",
            "96H",
            "Arm",
            "Role Lag",
            "Track",
            "Med",
        ),
        asset(
            "Cade Horton",
            "CHC",
            "SP",
            23,
            "ADD WHERE AVAILABLE",
            "",
            "CALL-UP ATTENTION GAP",
            "The market is partially aware, but shallow-league availability can still exist where managers are slow to react to promotion windows.",
            "Pitchability plus bat-missing indicators give the profile immediate fantasy relevance if the innings path holds.",
            "Claim where still exposed. This is no longer a hidden asset in sharper rooms, but some public markets will lag.",
            "≤25%",
            "48H",
            "Arm",
            "Promotion Lag",
            "Claim",
            "Med",
        ),
        asset(
            "Quinn Priester",
            "MIL",
            "SP",
            9,
            "DEEP LEAGUE TRACK",
            "monitor",
            "ENVIRONMENT CHANGE WATCH",
            "The market still prices the prior version of the pitcher. Context and development environment can matter before surface ratios catch up.",
            "Improved usage, pitch mix, and organizational context create a low-cost observation point for deeper leagues.",
            "Do not overpay. Track as a cheap arm whose price can jump if the new environment unlocks command or shape gains.",
            "≤10%",
            "96H",
            "Arm",
            "Context Lag",
            "Track",
            "Med",
        ),
        asset(
            "Zebby Matthews",
            "MIN",
            "SP",
            7,
            "WATCHLIST PROVISION",
            "monitor",
            "COMMAND-FIRST ASCENT",
            "Low public ownership reflects limited mainstream urgency, not a lack of usable fantasy signal.",
            "Strike-throwing stability and role proximity make this a pre-market tracking asset before the call-up headline arrives.",
            "Provision to watchlist now. The profile is more valuable as an early tracking asset than as a late public chase.",
            "≤10%",
            "7D",
            "Arm",
            "Headline Lag",
            "Track",
            "Low",
        ),
        asset(
            "Dylan Crews",
            "WSH",
            "OF",
            24,
            "BAT STASH",
            "stash",
            "PEDIGREE DISCOUNT WINDOW",
            "The public market may hesitate if the first box-score wave is uneven, creating a temporary discount on talent.",
            "Underlying bat speed, approach quality, and role runway can matter before the slash line stabilizes.",
            "Stash where available. The name value can reprice fast once production syncs with opportunity.",
            "≤25%",
            "7D",
            "Bat",
            "Box-Score Lag",
            "Stash",
            "Med",
        ),
        asset(
            "Coby Mayo",
            "BAL",
            "3B/1B",
            16,
            "POWER-PATH STASH",
            "stash",
            "ROLE BLOCK DISCOUNT",
            "The market discounts the bat because the roster fit is messy. That is exactly where power can remain underpriced.",
            "Damage-zone contact and raw power create a fast repricing risk if playing time opens suddenly.",
            "Provision as a power upside stash. The correct entry is before the lineup card makes the role obvious.",
            "≤20%",
            "7D",
            "Bat",
            "Role Block",
            "Stash",
            "Med",
        ),
        asset(
            "Kyle Manzardo",
            "CLE",
            "1B",
            22,
            "MONITOR DAMAGE WINDOW",
            "monitor",
            "PLATOON + POWER SIGNAL",
            "Casual formats may still treat him as a limited role bat, keeping the price manageable.",
            "Approach quality and left-handed damage potential make him a useful watch when role clarity improves.",
            "Track in formats where power scarcity matters. Move faster if lineup position or playing-time share improves.",
            "≤25%",
            "72H",
            "Bat",
            "Role Lag",
            "Track",
            "Med",
        ),
        asset(
            "Rece Hinds",
            "CIN",
            "OF",
            8,
            "HIGH-VOLATILITY WATCH",
            "monitor",
            "POWER OUTLIER PROFILE",
            "The market sees swing-and-miss risk first. That keeps a massive power outcome discounted.",
            "Barrel impact and raw damage potential create asymmetric upside if contact stabilizes even briefly.",
            "Track only in deeper or power-starved formats. This is not a safe asset, but it is a volatility weapon.",
            "≤10%",
            "96H",
            "Bat",
            "Contact Discount",
            "Track",
            "High",
        ),
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
    deployment_counts = {}
    visibility_counts = {}

    for row in assets or []:
        command = str(row.get("command") or "UNKNOWN").strip() or "UNKNOWN"
        command_counts[command] = command_counts.get(command, 0) + 1

        deployment = str(row.get("deployment_state") or "UNKNOWN").strip() or "UNKNOWN"
        deployment_counts[deployment] = deployment_counts.get(deployment, 0) + 1

        visibility = str(row.get("visibility_state") or "UNKNOWN").strip() or "UNKNOWN"
        visibility_counts[visibility] = visibility_counts.get(visibility, 0) + 1

    status_payload = build_report_status(
        "waiver_wire",
        build_success=True,
        threshold_minutes=240,
        build_started_at=build_started_at,
        build_finished_at=build_finished_at,
        source_updated_at=build_finished_at,
        section_counts={
            "waiver_assets": asset_count,
            "primary_assets": visibility_counts.get("primary", 0),
            "frozen_assets": visibility_counts.get("deployment_locked", 0) + visibility_counts.get("suppressed", 0),
            "command_groups": len(command_counts),
            "deployment_clear": deployment_counts.get("DEPLOYMENT_CLEAR", 0),
            "deployment_locked": deployment_counts.get("DEPLOYMENT_LOCKED", 0),
            "watchlist_only": deployment_counts.get("WATCHLIST_ONLY", 0),
            "surveillance_only": deployment_counts.get("SURVEILLANCE_ONLY", 0),
            "eligibility_unverified": deployment_counts.get("ELIGIBILITY_UNVERIFIED", 0),
        },
        degraded=degraded,
        notes=[
            f"Waiver Wire Open Market built with {asset_count} total assets; "
            f"{visibility_counts.get('primary', 0)} primary and "
            f"{visibility_counts.get('deployment_locked', 0) + visibility_counts.get('suppressed', 0)} frozen."
        ],
    )

    status_payload["command_counts"] = command_counts
    status_payload["deployment_counts"] = deployment_counts
    status_payload["visibility_counts"] = visibility_counts
    status_payload["hardening_notes"] = [
        "Signal detection is separated from deployment eligibility.",
        "Deployment-locked players are removed from the primary Waiver Wire board.",
        "Frozen Signals section preserves surveillance value without implying roster actionability.",
        "Manual operational override layer is active until live MLB status ingestion is wired.",
    ]
    status_payload["mode"] = "static_editable_v1_hardened"

    WAIVER_WIRE_STATUS_PATH.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote Waiver Wire status -> {WAIVER_WIRE_STATUS_PATH}")


def render() -> None:
    build_started_at = datetime.now(timezone.utc).isoformat()
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    assets = build_assets()

    assets = [apply_operational_status(row) for row in assets]

    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    primary_assets = [
        row for row in assets
        if row.get("deployment_state") not in {"DEPLOYMENT_LOCKED", "SUPPRESS"}
    ]
    frozen_assets = [
        row for row in assets
        if row.get("deployment_state") in {"DEPLOYMENT_LOCKED", "SUPPRESS"}
    ]

    template_env = Environment(loader=FileSystemLoader(str(TEMPLATE_PATH.parent)))
    template = template_env.get_template(TEMPLATE_PATH.name)
    shell_nav = Template(SHELL_NAV_PATH.read_text(encoding="utf-8")).render(active_nav="waiver_wire")

    HTML_PATH.write_text(
        template.render(
            assets=primary_assets,
            frozen_assets=frozen_assets,
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
        "mode": "static_editable_v1_hardened",
        "assets": primary_assets,
        "frozen_assets": frozen_assets,
        "all_assets": assets,
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
