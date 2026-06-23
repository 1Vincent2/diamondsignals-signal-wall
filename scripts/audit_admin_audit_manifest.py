#!/usr/bin/env python3

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dist/status/admin-audit-manifest.json"

REQUIRED_TOP_KEYS = {
    "manifest_id",
    "generated_at",
    "repo",
    "head",
    "green_baseline",
    "tier4",
    "reports",
    "admin_readiness",
}

REQUIRED_BASELINE_AUDITS = {
    "scripts/audit_report_inventory_truth.py",
    "scripts/audit_build_entrypoint_coverage.py",
    "scripts/audit_build_output_freshness.py",
    "scripts/audit_status_contract.py",
    "scripts/audit_canonical_player_identity_contract.py",
    "scripts/audit_refinement_queue_closure.py",
    "scripts/audit_green_baseline_registry.py",
}


def main():
    print("--- DiamondSignals admin audit manifest contract audit ---")
    problems = []

    if not MANIFEST.exists():
        problems.append("admin audit manifest does not exist")
        print(f"FAIL: missing {MANIFEST.relative_to(ROOT)}")
    else:
        print(f"OK: manifest exists: {MANIFEST.relative_to(ROOT)}")

    if problems:
        print("\nFINAL_STATUS: FAIL_ADMIN_AUDIT_MANIFEST")
        sys.exit(1)

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    missing = sorted(REQUIRED_TOP_KEYS - set(payload.keys()))
    if missing:
        problems.append(f"missing top-level keys: {missing}")
        print(f"FAIL: missing top-level keys: {missing}")
    else:
        print("OK: required top-level keys present")

    manifest_id = payload.get("manifest_id")
    print(f"manifest_id: {manifest_id}")
    if manifest_id != "admin_audit_manifest_v1":
        problems.append(f"unexpected manifest_id: {manifest_id}")

    head = payload.get("head", {})
    print(f"head_short_sha: {head.get('short_sha')}")
    print(f"head_subject: {head.get('subject')}")
    if not head.get("sha") or not head.get("short_sha"):
        problems.append("manifest head sha fields are missing")

    baseline = payload.get("green_baseline", {})
    invocations = set(baseline.get("audit_invocations", []))
    print(f"audit_invocation_count: {baseline.get('audit_invocation_count')}")

    if baseline.get("runner") != "scripts/run_green_baseline_audit.sh":
        problems.append("green baseline runner mismatch")

    for rel in sorted(REQUIRED_BASELINE_AUDITS):
        if rel not in invocations:
            problems.append(f"required audit missing from manifest invocations: {rel}")
            print(f"FAIL: missing invocation: {rel}")
        else:
            print(f"OK: manifest invocation: {rel}")

    reports = payload.get("reports", {})
    inventory_count = reports.get("inventory_count")
    report_ids = reports.get("report_ids", [])
    print(f"inventory_count: {inventory_count}")
    print(f"report_ids_count: {len(report_ids)}")
    if inventory_count != len(report_ids):
        problems.append("inventory_count does not match report_ids length")
    if inventory_count != 10:
        problems.append(f"expected 10 inventory reports, got {inventory_count}")

    admin = payload.get("admin_readiness", {})
    for key in [
        "green_baseline_button_ready",
        "audit_registry_ready",
        "status_manifest_ready",
        "safe_to_begin_admin_ui_integration_next",
    ]:
        value = admin.get(key)
        print(f"{key}: {value}")
        if value is not True:
            problems.append(f"admin readiness flag must be true: {key}")

    if admin.get("admin_ui_implemented") is not True:
        problems.append("admin_ui_implemented must be true now that the Admin System manifest panel is built")

    print("\n--- summary ---")
    print(f"admin_audit_manifest_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_ADMIN_AUDIT_MANIFEST")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_ADMIN_AUDIT_MANIFEST")


if __name__ == "__main__":
    main()
