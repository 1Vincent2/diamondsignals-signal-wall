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
WAIVER_CANDIDATE_PATH = DIST_DIR / "waiver_candidates.json"

TEMPLATE_PATH = BASE_DIR / "templates" / "waiver_wire.html"
SHELL_STYLES_PATH = BASE_DIR / "templates" / "shell_styles.css"
SHELL_NAV_PATH = BASE_DIR / "templates" / "shell_nav.html"
SHELL_NAV_V2_PATH = BASE_DIR / "templates" / "shell_nav_v2.html"

CANONICAL_PLAYERS = load_canonical_player_universe()
CANONICAL_NAME_LOOKUP = build_name_lookup(CANONICAL_PLAYERS)

WAIVER_IDENTITY_OVERRIDES = {}


WAIVER_PIPELINE_MODE = "verified_dynamic_candidates_only_v1"
WAIVER_PIPELINE_LAYERS = [
    "candidate_pool_file_only",
    "verified_market_eligibility_required",
    "no_static_seed_fallback",
    "market_attention_candidate_file_enrichment",
    "physics_signal_candidate_file_enrichment",
    "role_opportunity_candidate_file_enrichment",
    "waiver_score_candidate_file_rank_v1",
]


def load_waiver_candidate_file() -> list[dict]:
    """
    Optional dynamic candidate rail.

    Expected future source:
    dist/waiver_candidates.json

    Supported shapes:
    - {"assets": [...]}
    - {"candidates": [...]}
    - [...]
    """
    if not WAIVER_CANDIDATE_PATH.exists():
        return []

    try:
        raw = json.loads(WAIVER_CANDIDATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]

    if isinstance(raw, dict):
        rows = raw.get("assets") or raw.get("candidates") or []
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]

    return []


def normalize_candidate_row(row: dict) -> dict:
    """
    Normalize a dynamic candidate into the existing Waiver card contract.
    Missing fields are filled conservatively so the surface remains publish-safe.
    """
    player_name = str(row.get("player_name") or row.get("name") or "").strip()
    team = str(row.get("team") or "").strip()
    position = str(row.get("position") or "").strip()
    rostered_pct = int(float(row.get("rostered_pct") or row.get("roster_pct") or 0))

    return {
        "player_name": player_name,
        "team": team,
        "position": position,
        "rostered_pct": rostered_pct,
        "command": row.get("command") or "WATCHLIST PROVISION",
        "command_class": row.get("command_class") or "monitor",
        "market_status": row.get("market_status") or "DYNAMIC CANDIDATE",
        "surface_profile": row.get("surface_profile") or "Candidate entered through the dynamic Waiver candidate rail.",
        "forensic_trigger": row.get("forensic_trigger") or "Awaiting full physics-layer attribution.",
        "verdict": row.get("verdict") or "Track until full market, role, and physics layers are attached.",
        "deployment_state": row.get("deployment_state") or "DEPLOYMENT_CLEAR",
        "deployment_label": row.get("deployment_label") or "DEPLOYMENT CLEAR",
        "operational_status": row.get("operational_status") or "ACTIVE",
        "status_reason": row.get("status_reason") or "Candidate file source; verify status before deployment.",
        "card_action": row.get("card_action") or "OPEN PERFORMANCE AUDIT",
        "visibility_state": row.get("visibility_state") or "primary",
        "status_source": row.get("status_source") or "candidate_file",
        "asset_type": row.get("asset_type") or "Open Market",
        "market_defect": row.get("market_defect") or "Market Lag",
        "command_metric": row.get("command_metric") or "Track",
        "risk": row.get("risk") or "Med",
        "ownership_gate": row.get("ownership_gate") or f"≤{rostered_pct}%" if rostered_pct else "Unverified",
        "signal_window": row.get("signal_window") or "72H",
        "player_id": str(row.get("player_id") or "").strip(),
        "audit_slug": row.get("audit_slug") or "",
        "_candidate_source": "waiver_candidate_file",
    }


def asset_from_candidate(row: dict) -> dict:
    candidate = normalize_candidate_row(row)
    asset = _build_asset(
        candidate["player_name"],
        candidate["team"],
        candidate["position"],
        candidate["rostered_pct"],
        candidate["command"],
        candidate["command_class"],
        candidate["market_status"],
        candidate["surface_profile"],
        candidate["forensic_trigger"],
        candidate["verdict"],
        candidate["ownership_gate"],
        candidate["signal_window"],
        candidate["asset_type"],
        candidate["market_defect"],
        candidate["command_metric"],
        candidate["risk"],
        player_id=candidate["player_id"],
        audit_slug=candidate["audit_slug"],
        deployment_state=candidate["deployment_state"],
        deployment_label=candidate["deployment_label"],
        operational_status=candidate["operational_status"],
        status_reason=candidate["status_reason"],
        card_action=candidate["card_action"],
        visibility_state=candidate["visibility_state"],
        status_source=candidate["status_source"],
    )
    asset["candidate_source"] = candidate["_candidate_source"]
    return asset


def build_candidate_pool() -> list[dict]:
    """
    Candidate source priority:
    1. Dynamic/generated candidate file: dist/waiver_candidates.json
    2. Empty standby state when no verified market-eligible candidates exist

    No static/pre-seeded Waiver assets are allowed in production output.
    """
    candidate_rows = load_waiver_candidate_file()
    if candidate_rows:
        return [asset_from_candidate(row) for row in candidate_rows]

    return []


def attach_market_attention_layer(assets: list[dict]) -> list[dict]:
    for asset in assets:
        asset["market_attention"] = {
            "source": "candidate_file_dynamic_pending_enrichment",
            "rostered_pct": asset.get("rostered_pct"),
            "ownership_gate": next(
                (m.get("value") for m in asset.get("metrics", []) if m.get("label") == "Ownership Gate"),
                None,
            ),
        }
    return assets


def attach_physics_signal_layer(assets: list[dict]) -> list[dict]:
    for asset in assets:
        asset["physics_signal"] = {
            "source": "candidate_file_dynamic_pending_enrichment",
            "forensic_trigger": asset.get("forensic_trigger"),
            "surface_profile": asset.get("surface_profile"),
        }
    return assets


def attach_role_opportunity_layer(assets: list[dict]) -> list[dict]:
    for asset in assets:
        asset["role_opportunity"] = {
            "source": "candidate_file_dynamic_pending_enrichment",
            "deployment_state": asset.get("deployment_state"),
            "operational_status": asset.get("operational_status"),
            "status_source": asset.get("status_source"),
        }
    return assets


def score_waiver_assets(assets: list[dict]) -> list[dict]:
    for index, asset in enumerate(assets, start=1):
        asset["waiver_score"] = asset.get("waiver_score") or max(1, 100 - index)
        asset["scoring_mode"] = "candidate_file_priority_rank"
    return assets


def build_waiver_pipeline_assets() -> list[dict]:
    assets = build_candidate_pool()
    assets = attach_market_attention_layer(assets)
    assets = attach_physics_signal_layer(assets)
    assets = attach_role_opportunity_layer(assets)
    assets = score_waiver_assets(assets)
    return assets







def _build_asset(
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
    status_source: str = "verified_candidate_file",
) -> dict:
    """
    Dynamic Waiver asset factory.

    This function is intentionally retained because verified upstream candidates
    still need normalization, canonical identity resolution, scout URLs, metrics,
    and action URLs before render.

    It must not contain or call static demo player fixtures.
    """
    identity = resolve_player_identity(
        player_id=player_id,
        player_name=player_name,
        team=team,
        players=CANONICAL_PLAYERS,
        name_lookup=CANONICAL_NAME_LOOKUP,
    )

    resolved_player_id = str(identity.get("player_id") or player_id or "").strip()
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
        "search_blob": " ".join([
            str(player_name),
            str(team),
            str(position),
            str(command),
            str(market_status),
            str(deployment_label),
            str(operational_status),
        ]).lower(),
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



# Static Waiver demo/player fixtures intentionally removed.
# Waiver Wire is verified-dynamic only:
#   market eligibility feed -> upstream signal candidate file -> Waiver render.
# No hardcoded player-name fallback is allowed.


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
        "No static/pre-seeded Waiver assets are rendered when verified candidates are unavailable.",
        "Manual operational override layer is active until live MLB status ingestion is wired.",
    ]
    status_payload["mode"] = WAIVER_PIPELINE_MODE
    status_payload["pipeline_layers"] = WAIVER_PIPELINE_LAYERS

    WAIVER_WIRE_STATUS_PATH.write_text(
        json.dumps(status_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote Waiver Wire status -> {WAIVER_WIRE_STATUS_PATH}")


def render() -> None:
    build_started_at = datetime.now(timezone.utc).isoformat()
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    assets = build_waiver_pipeline_assets()

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
    shell_nav_v2 = Template(SHELL_NAV_V2_PATH.read_text(encoding="utf-8")).render(active_nav="waiver_wire")

    HTML_PATH.write_text(
        template.render(
            assets=primary_assets,
            frozen_assets=frozen_assets,
            generated_at=generated_at,
            shell_styles=SHELL_STYLES_PATH.read_text(encoding="utf-8"),
            shell_nav_v2=shell_nav_v2,
            shell_nav=shell_nav,
        ),
        encoding="utf-8",
    )

    print(f"Wrote waiver wire surface -> {HTML_PATH}")
    print(f"Assets rendered: {len(assets)}")

    payload = {
        "generated_at": generated_at,
        "mode": WAIVER_PIPELINE_MODE,
        "pipeline_layers": WAIVER_PIPELINE_LAYERS,
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
