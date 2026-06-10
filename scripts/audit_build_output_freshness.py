#!/usr/bin/env python3

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "dashboard/report_inventory.json"

MAX_HTML_TO_STATUS_DRIFT_SECONDS = 300
MAX_JSON_TO_STATUS_DRIFT_SECONDS = 300
MAX_FALLBACK_TO_SOURCE_DRIFT_SECONDS = 300

def mtime(path: Path) -> float:
    return path.stat().st_mtime

def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))

def main() -> None:
    print("--- DiamondSignals build output freshness audit ---")

    if not INVENTORY_PATH.exists():
        print(f"FAIL: missing inventory file: {rel(INVENTORY_PATH)}")
        print("\nFINAL_STATUS: FAIL_BUILD_OUTPUT_FRESHNESS")
        sys.exit(1)

    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    reports = payload.get("reports") or []

    problems = []

    for report in reports:
        report_id = report.get("report_id") or "UNKNOWN_REPORT"

        output_html = report.get("output_html")
        output_json = report.get("output_json")
        status_output = report.get("status_output")
        fallback = report.get("fallback_artifact")

        print(f"\n--- {report_id} ---")

        paths = {
            "output_html": ROOT / output_html if output_html else None,
            "output_json": ROOT / output_json if output_json else None,
            "status_output": ROOT / status_output if status_output else None,
            "fallback_artifact": ROOT / fallback if fallback else None,
        }

        for label, path in paths.items():
            if path is None:
                print(f"WARN: {label}: not declared")
                continue

            if not path.exists():
                problems.append(f"{report_id}: {label} missing: {rel(path)}")
                print(f"FAIL: {label}: {rel(path)} missing")
                continue

            print(f"OK: {label}: {rel(path)}")
            print(f"  size_bytes: {path.stat().st_size}")
            print(f"  mtime_utc: {iso_mtime(path)}")

            if path.stat().st_size <= 0:
                problems.append(f"{report_id}: {label} is empty: {rel(path)}")

        status_path = paths["status_output"]

        if status_path and status_path.exists():
            for label in ["output_html", "output_json"]:
                path = paths[label]
                if path and path.exists():
                    drift = abs(mtime(path) - mtime(status_path))
                    print(f"{label}_to_status_drift_seconds: {drift:.2f}")

                    max_drift = (
                        MAX_HTML_TO_STATUS_DRIFT_SECONDS
                        if label == "output_html"
                        else MAX_JSON_TO_STATUS_DRIFT_SECONDS
                    )

                    if drift > max_drift:
                        print(
                            f"WARN: {label} and status_output mtime drift exceeds soft threshold: "
                            f"{drift:.2f}s > {max_drift}s"
                        )

        fallback_path = paths["fallback_artifact"]

        if fallback_path and fallback_path.exists():
            source = None

            if fallback_path.suffix.lower() == ".html":
                source = paths["output_html"]
            elif fallback_path.suffix.lower() == ".json":
                source = paths["output_json"]

            if source and source.exists():
                drift = abs(mtime(fallback_path) - mtime(source))
                print(f"fallback_to_source_drift_seconds: {drift:.2f}")

                if drift > MAX_FALLBACK_TO_SOURCE_DRIFT_SECONDS:
                    problems.append(
                        f"{report_id}: fallback artifact is stale vs source: "
                        f"{drift:.2f}s > {MAX_FALLBACK_TO_SOURCE_DRIFT_SECONDS}s"
                    )

    print("\n--- summary ---")
    print(f"reports_checked: {len(reports)}")
    print(f"build_output_freshness_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_BUILD_OUTPUT_FRESHNESS")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_BUILD_OUTPUT_FRESHNESS")

if __name__ == "__main__":
    main()