#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

issues = []

seal = Path("ADMIN_UI_INTEGRATION_SEAL.md")
manifest_path = Path("dist/status/admin-audit-manifest.json")
registry_path = Path("scripts/audit_green_baseline_registry.py")

if not seal.exists():
    issues.append("missing ADMIN_UI_INTEGRATION_SEAL.md")
else:
    seal_text = seal.read_text(errors="ignore")
    for marker in [
        "admin_ui_implemented: true",
        "PASS_ADMIN_GREEN_BASELINE_JOB_CONTRACT",
        "Commit: c3802f7",
        "Commit: 9f95e74",
        "audit_admin_build_runtime_job_contract.py visible",
        "audit_netlify_green_baseline_gate.py visible",
        "does not alter report builders",
        "mobile layout",
        "Yahoo/API behavior",
    ]:
        if marker not in seal_text:
            issues.append(f"admin UI seal missing marker: {marker}")

subprocess.run([sys.executable, "scripts/build_admin_audit_manifest.py"], check=False)

if not manifest_path.exists():
    issues.append("missing local admin audit manifest")
else:
    data = json.loads(manifest_path.read_text())
    readiness = data.get("admin_readiness") or {}
    baseline = data.get("green_baseline") or {}
    invocations = baseline.get("audit_invocations") or []

    expected_true = [
        "admin_ui_implemented",
        "audit_registry_ready",
        "green_baseline_button_ready",
        "status_manifest_ready",
        "safe_to_begin_admin_ui_integration_next",
    ]

    for key in expected_true:
        if readiness.get(key) is not True:
            issues.append(f"manifest admin_readiness.{key} must be true")

    for invocation in [
        "scripts/audit_netlify_green_baseline_gate.py",
        "scripts/audit_admin_build_runtime_job_contract.py",
        "scripts/audit_no_stale_live_v2_artifact.py",
    ]:
        if invocation not in invocations:
            issues.append(f"manifest missing baseline invocation: {invocation}")

registry_result = subprocess.run(
    [sys.executable, "scripts/audit_green_baseline_registry.py"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(registry_result.stdout)

if registry_result.returncode != 0:
    issues.append("green baseline registry audit failed")

if "admin_ui_implemented: true" not in registry_result.stdout:
    issues.append("green baseline registry audit must report admin_ui_implemented: true")

print("--- DiamondSignals admin readiness truth audit ---")
print(f"admin_readiness_truth_issues: {len(issues)}")

for issue in issues:
    print(f"FAIL: {issue}")

if issues:
    print()
    print("FINAL_STATUS: FAIL_ADMIN_READINESS_TRUTH")
    sys.exit(1)

print("admin_ui_seal_present: true")
print("manifest_admin_readiness_true: true")
print("registry_admin_ui_implemented_true: true")
print()
print("FINAL_STATUS: PASS_ADMIN_READINESS_TRUTH")
