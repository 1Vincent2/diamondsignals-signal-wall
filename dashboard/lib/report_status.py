from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def age_minutes_from_now(value: Optional[str]) -> Optional[float]:
    dt = parse_iso_timestamp(value)
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    return round((now - dt).total_seconds() / 60.0, 2)


def is_fresh(timestamp_iso: Optional[str], threshold_minutes: int) -> bool:
    age = age_minutes_from_now(timestamp_iso)
    if age is None:
        return False
    return age <= threshold_minutes


def derive_state(
    *,
    build_success: bool,
    used_fallback: bool = False,
    stale: bool = False,
    degraded: bool = False,
) -> str:
    if not build_success:
        return "failed"
    if used_fallback:
        return "stale"
    if degraded or stale:
        return "degraded"
    return "fresh"


def build_report_status(
    report_id: str,
    *,
    build_success: bool,
    threshold_minutes: int,
    build_started_at: Optional[str] = None,
    build_finished_at: Optional[str] = None,
    source_updated_at: Optional[str] = None,
    section_counts: Optional[Dict[str, int]] = None,
    used_fallback: bool = False,
    degraded: bool = False,
    errors: Optional[list[str]] = None,
    notes: Optional[list[str]] = None,
    mode: Optional[str] = None,
    pipeline_layers: Optional[list[str]] = None,
    hardening_notes: Optional[list[str]] = None,
) -> Dict[str, Any]:
    source_age_minutes = age_minutes_from_now(source_updated_at)
    stale = False
    if source_updated_at:
        stale = not is_fresh(source_updated_at, threshold_minutes)

    state = derive_state(
        build_success=build_success,
        used_fallback=used_fallback,
        stale=stale,
        degraded=degraded,
    )

    status: Dict[str, Any] = {
        "report_id": report_id,
        "state": state,
        "build_success": build_success,
        "used_fallback": used_fallback,
        "degraded": degraded,
        "threshold_minutes": threshold_minutes,
        "build_started_at": build_started_at,
        "build_finished_at": build_finished_at,
        "source_updated_at": source_updated_at,
        "source_age_minutes": source_age_minutes,
        "section_counts": section_counts or {},
        "errors": errors or [],
        "notes": notes or [],
        "generated_at": utc_now_iso(),
    }

    if mode is not None:
        status["mode"] = mode
    if pipeline_layers is not None:
        status["pipeline_layers"] = pipeline_layers
    if hardening_notes is not None:
        status["hardening_notes"] = hardening_notes

    return status
