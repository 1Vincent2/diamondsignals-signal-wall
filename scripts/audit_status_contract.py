#!/usr/bin/env python3

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "dashboard/report_inventory.json"

REQUIRED_STATUS_KEYS = {
    "report_id",
    "state",
    "build_success",
    "generated_at",
    "source_age_minutes",
    "threshold_minutes",
    "used_fallback",
    "errors",
}

VALID_STATES = {
    "fresh",
    "stale",
    "empty",
    "error",
    "degraded",
}

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))

def parse_datetime(value):
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.replace("Z", "+00:00")

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None

def main() -> None:
    print("--- DiamondSignals status contract audit ---")

    if not INVENTORY_PATH.exists():
        print(f"FAIL: missing inventory file: {rel(INVENTORY_PATH)}")
        print("\nFINAL_STATUS: FAIL_STATUS_CONTRACT")
        sys.exit(1)

    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    reports = payload.get("reports") or []

    problems = []
    seen_status_paths = set()

    for report in reports:
        report_id = report.get("report_id") or "UNKNOWN_REPORT"
        status_output = report.get("status_output")

        print(f"\n--- {report_id} ---")

        if not status_output:
            problems.append(f"{report_id}: missing status_output declaration")
            print("FAIL: status_output not declared")
            continue

        status_path = ROOT / status_output
        seen_status_paths.add(rel(status_path))

        if not status_path.exists():
            problems.append(f"{report_id}: missing status file: {rel(status_path)}")
            print(f"FAIL: missing status file: {rel(status_path)}")
            continue

        print(f"OK: status_file: {rel(status_path)}")

        try:
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"{report_id}: status file is not valid JSON: {exc}")
            print(f"FAIL: invalid JSON: {exc}")
            continue

        missing_keys = sorted(REQUIRED_STATUS_KEYS - set(status_payload.keys()))
        if missing_keys:
            problems.append(f"{report_id}: missing required status keys: {missing_keys}")
            print(f"FAIL: missing_required_keys: {missing_keys}")
        else:
            print("OK: required status keys present")

        actual_report_id = status_payload.get("report_id")
        if actual_report_id != report_id:
            problems.append(
                f"{report_id}: status report_id mismatch: inventory={report_id} status={actual_report_id}"
            )
            print(f"FAIL: report_id mismatch: {actual_report_id}")
        else:
            print(f"OK: report_id: {actual_report_id}")

        state = status_payload.get("state")
        print(f"state: {state}")
        if state not in VALID_STATES:
            problems.append(f"{report_id}: invalid state value: {state}")

        build_success = status_payload.get("build_success")
        print(f"build_success: {build_success}")
        if not isinstance(build_success, bool):
            problems.append(
                f"{report_id}: build_success must be boolean, got {type(build_success).__name__}"
            )

        generated_at = status_payload.get("generated_at")
        print(f"generated_at: {generated_at}")
        if not parse_datetime(generated_at):
            problems.append(
                f"{report_id}: generated_at is missing or not ISO-parseable: {generated_at}"
            )

        for numeric_key in ["source_age_minutes", "threshold_minutes"]:
            value = status_payload.get(numeric_key)
            print(f"{numeric_key}: {value}")
            if not isinstance(value, (int, float)):
                problems.append(
                    f"{report_id}: {numeric_key} must be numeric, got {type(value).__name__}"
                )

        used_fallback = status_payload.get("used_fallback")
        print(f"used_fallback: {used_fallback}")
        if not isinstance(used_fallback, bool):
            problems.append(
                f"{report_id}: used_fallback must be boolean, got {type(used_fallback).__name__}"
            )

        errors = status_payload.get("errors")
        print(f"errors_count: {len(errors) if isinstance(errors, list) else 'INVALID'}")
        if not isinstance(errors, list):
            problems.append(f"{report_id}: errors must be a list")

        optional_keys = sorted(set(status_payload.keys()) - REQUIRED_STATUS_KEYS)
        print(f"optional_keys: {optional_keys}")

    status_dir = ROOT / "dist/status"
    actual_status_files = sorted(rel(p) for p in status_dir.glob("*.json")) if status_dir.exists() else []
    orphan_status_files = sorted(set(actual_status_files) - seen_status_paths)

    print("\n--- orphan status file check ---")
    if orphan_status_files:
        for path in orphan_status_files:
            print(f"WARN: status file exists but is not declared in report inventory: {path}")
    else:
        print("OK: no orphan status files")

    print("\n--- summary ---")
    print(f"reports_checked: {len(reports)}")
    print(f"status_files_seen: {len(seen_status_paths)}")
    print(f"orphan_status_files: {len(orphan_status_files)}")
    print(f"status_contract_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_STATUS_CONTRACT")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_STATUS_CONTRACT")

if __name__ == "__main__":
    main()
