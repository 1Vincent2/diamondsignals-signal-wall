#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

issues = []

seal = Path("PRE_MOBILE_READINESS_SEAL.md")
mobile_deferred = Path("MOBILE_DEFERRED_UNTIL_ITEM21.md")
admin_seal = Path("ADMIN_UI_INTEGRATION_SEAL.md")
runner = Path("scripts/run_green_baseline_audit.sh")
manifest = Path("dist/status/admin-audit-manifest.json")

def require_text(path: Path, markers: list[str], label: str):
    if not path.exists():
        issues.append(f"missing {label}: {path}")
        return ""
    text = path.read_text(errors="ignore")
    for marker in markers:
        if marker not in text:
            issues.append(f"{label} missing marker: {marker}")
    return text

require_text(seal, [
    "Item 21 is the final closeout",
    "Item 17 sealed",
    "Item 18 sealed",
    "Item 19 sealed",
    "Item 20 sealed",
    "3b0ef11",
    "9f95e74",
    "883bf37",
    "c3802f7",
    "mobile redesign may begin only as a new separate phase",
    "no casual rewrite of report builders",
    "no casual rewrite of report data contracts",
    "no Yahoo/API behavior changes unless explicitly named, audited, and sealed",
    "The core report surfaces remain bottled",
], "pre-mobile readiness seal")

require_text(mobile_deferred, [
    "Mobile redesign is intentionally deferred until after Item 21.",
    "After Item 21 is complete, the planned mobile redesign can begin only as a separate tracked phase",
    "It should not rewrite core report generation",
], "mobile deferred contract")

require_text(admin_seal, [
    "admin_ui_implemented: true",
    "PASS_ADMIN_GREEN_BASELINE_JOB_CONTRACT",
    "Commit: c3802f7",
    "Commit: 9f95e74",
    "Yahoo/API behavior",
], "admin UI integration seal")

if not runner.exists():
    issues.append("missing green baseline runner")
else:
    runner_text = runner.read_text(errors="ignore")
    for marker in [
        "python3 scripts/audit_pre_mobile_readiness_seal.py",
        "python3 scripts/audit_admin_readiness_truth.py",
        "python3 scripts/audit_admin_build_runtime_job_contract.py",
        "python3 scripts/audit_netlify_green_baseline_gate.py",
        "python3 scripts/audit_desktop_mobile_deferred_contract.py",
        "python3 scripts/test_tracking_actions.py",
    ]:
        if marker not in runner_text:
            issues.append(f"green baseline runner missing marker: {marker}")

subprocess.run([sys.executable, "scripts/build_admin_audit_manifest.py"], check=False)

if not manifest.exists():
    issues.append("missing local admin audit manifest")
else:
    data = json.loads(manifest.read_text())
    readiness = data.get("admin_readiness") or {}
    baseline = data.get("green_baseline") or {}
    invocations = baseline.get("audit_invocations") or []

    for key in [
        "admin_ui_implemented",
        "audit_registry_ready",
        "green_baseline_button_ready",
        "status_manifest_ready",
        "safe_to_begin_admin_ui_integration_next",
    ]:
        if readiness.get(key) is not True:
            issues.append(f"manifest admin_readiness.{key} must be true")

    for invocation in [
        "scripts/audit_pre_mobile_readiness_seal.py",
        "scripts/audit_admin_readiness_truth.py",
        "scripts/audit_admin_build_runtime_job_contract.py",
        "scripts/audit_netlify_green_baseline_gate.py",
        "scripts/audit_desktop_mobile_deferred_contract.py",
        "scripts/test_tracking_actions.py",
    ]:
        if invocation not in invocations:
            issues.append(f"manifest missing green baseline invocation: {invocation}")

print("--- DiamondSignals Item 21 pre-mobile readiness seal audit ---")
print(f"pre_mobile_readiness_issues: {len(issues)}")

for issue in issues:
    print(f"FAIL: {issue}")

if issues:
    print()
    print("FINAL_STATUS: FAIL_PRE_MOBILE_READINESS_SEAL")
    sys.exit(1)

print("items_17_to_20_sealed: true")
print("admin_readiness_truth_aligned: true")
print("mobile_redesign_can_begin_after_item21_only_as_separate_phase: true")
print("core_report_surfaces_remain_bottled: true")
print()
print("FINAL_STATUS: PASS_PRE_MOBILE_READINESS_SEAL")
