#!/usr/bin/env python3
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "dashboard/report_inventory.json"

def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

def main() -> None:
    print("--- DiamondSignals fallback snapshot seeder ---")

    if not INVENTORY_PATH.exists():
        raise SystemExit(f"FAIL: missing inventory file: {rel(INVENTORY_PATH)}")

    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    reports = payload.get("reports") or []

    if not reports:
        raise SystemExit("FAIL: report inventory has no reports")

    seeded = []
    existing = []
    skipped = []
    problems = []

    for report in reports:
        report_id = report.get("report_id") or "UNKNOWN_REPORT"
        fallback = report.get("fallback_artifact")
        output_html = report.get("output_html")
        output_json = report.get("output_json")

        if not fallback:
            skipped.append((report_id, "no fallback_artifact declared"))
            continue

        fallback_path = ROOT / fallback

        if fallback_path.exists():
            existing.append((report_id, rel(fallback_path)))
            continue

        source = None

        if fallback_path.suffix.lower() == ".html" and output_html:
            candidate = ROOT / output_html
            if candidate.exists():
                source = candidate

        if source is None and fallback_path.suffix.lower() == ".json" and output_json:
            candidate = ROOT / output_json
            if candidate.exists():
                source = candidate

        if source is None and output_html:
            candidate = ROOT / output_html
            if candidate.exists():
                source = candidate

        if source is None and output_json:
            candidate = ROOT / output_json
            if candidate.exists():
                source = candidate

        if source is None:
            problems.append(
                f"{report_id}: fallback {fallback} missing and no valid output_html/output_json source exists"
            )
            continue

        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, fallback_path)
        seeded.append((report_id, rel(source), rel(fallback_path)))

    print("\n--- seeded fallback snapshots ---")
    for report_id, source, fallback in seeded:
        print(f"SEEDED: {report_id}: {source} -> {fallback}")

    print("\n--- already present ---")
    for report_id, fallback in existing:
        print(f"OK: {report_id}: {fallback}")

    print("\n--- skipped ---")
    for report_id, reason in skipped:
        print(f"SKIP: {report_id}: {reason}")

    print("\n--- summary ---")
    print(f"reports_checked: {len(reports)}")
    print(f"seeded_count: {len(seeded)}")
    print(f"existing_count: {len(existing)}")
    print(f"skipped_count: {len(skipped)}")
    print(f"fallback_snapshot_seed_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_SEED_FALLBACK_SNAPSHOTS")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_SEED_FALLBACK_SNAPSHOTS")

if __name__ == "__main__":
    main()
