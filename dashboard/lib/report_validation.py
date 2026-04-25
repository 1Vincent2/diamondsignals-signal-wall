from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def validate_required_sections(
    report_name: str,
    sections: Dict[str, Any],
    required_sections: Iterable[str],
) -> Dict[str, Any]:
    missing = [name for name in required_sections if name not in sections]
    return {
        "ok": len(missing) == 0,
        "report_name": report_name,
        "missing_sections": missing,
        "message": "" if not missing else f"Missing required sections: {', '.join(missing)}",
    }


def validate_min_rows(
    section_name: str,
    rows: Optional[int],
    min_rows: int,
) -> Dict[str, Any]:
    actual = 0 if rows is None else int(rows)
    ok = actual >= min_rows
    return {
        "ok": ok,
        "section_name": section_name,
        "actual_rows": actual,
        "min_rows": min_rows,
        "message": "" if ok else f"{section_name} has {actual} rows; requires at least {min_rows}",
    }


def validate_non_null_fields(
    record_count: int,
    field_non_null_counts: Dict[str, int],
    required_fields: Iterable[str],
) -> Dict[str, Any]:
    failed = []
    for field in required_fields:
        if field_non_null_counts.get(field, 0) <= 0:
            failed.append(field)

    return {
        "ok": len(failed) == 0,
        "record_count": record_count,
        "failed_fields": failed,
        "message": "" if not failed else f"Required fields all null or missing: {', '.join(failed)}",
    }


def build_validation_report(
    report_name: str,
    checks: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    checks = list(checks)
    failures = [check for check in checks if not check.get("ok", False)]
    return {
        "report_name": report_name,
        "ok": len(failures) == 0,
        "checks": checks,
        "failure_count": len(failures),
        "messages": [check.get("message", "") for check in failures if check.get("message")],
    }
