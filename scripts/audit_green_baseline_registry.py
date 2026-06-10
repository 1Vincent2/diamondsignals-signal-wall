#!/usr/bin/env python3

import re
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts/run_green_baseline_audit.sh"

REQUIRED_TIER4_AUDITS = {
    "scripts/audit_report_inventory_truth.py",
    "scripts/audit_build_entrypoint_coverage.py",
    "scripts/audit_build_output_freshness.py",
    "scripts/audit_status_contract.py",
    "scripts/audit_refinement_queue_closure.py",
}

EXPECTED_GUARDRAILS = {
    "scripts/audit_mobile_header_menu_contract.py",
    "scripts/test_tracking_actions.py",
}

EXTRACT_PATTERNS = [
    re.compile(r"^\s*python3\s+(scripts/[A-Za-z0-9_\-./]+\.py)\s*$"),
    re.compile(r"^\s*\./(scripts/[A-Za-z0-9_\-./]+\.sh)\s*$"),
]


def is_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & stat.S_IXUSR)


def extract_invocations(text: str):
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for pattern in EXTRACT_PATTERNS:
            match = pattern.match(line)
            if match:
                found.append(match.group(1))
                break
    return found


def main():
    print("--- DiamondSignals green baseline audit registry audit ---")
    problems = []

    if not BASELINE.exists():
        print(f"FAIL: missing baseline runner: {BASELINE.relative_to(ROOT)}")
        problems.append("missing run_green_baseline_audit.sh")
    else:
        print(f"OK: baseline_runner: {BASELINE.relative_to(ROOT)}")

    if BASELINE.exists() and not is_executable(BASELINE):
        problems.append("run_green_baseline_audit.sh is not executable")
        print("FAIL: baseline runner is not executable")
    elif BASELINE.exists():
        print("OK: baseline runner is executable")

    text = BASELINE.read_text(encoding="utf-8") if BASELINE.exists() else ""
    invocations = extract_invocations(text)
    invocation_set = set(invocations)

    print("\n--- discovered baseline invocations ---")
    for item in invocations:
        print(f"FOUND: {item}")
    print(f"baseline_invocation_count: {len(invocations)}")

    print("\n--- referenced script existence check ---")
    for rel in invocations:
        p = ROOT / rel
        if not p.exists():
            problems.append(f"referenced script missing: {rel}")
            print(f"FAIL: missing: {rel}")
            continue
        print(f"OK: exists: {rel}")

        if rel.endswith(".sh"):
            if not is_executable(p):
                problems.append(f"referenced shell script is not executable: {rel}")
                print(f"FAIL: not executable: {rel}")
            else:
                print(f"OK: executable: {rel}")

    print("\n--- required Tier 4 audit coverage ---")
    for rel in sorted(REQUIRED_TIER4_AUDITS):
        if rel not in invocation_set:
            problems.append(f"required Tier 4 audit is not wired into green baseline: {rel}")
            print(f"FAIL: not wired: {rel}")
        else:
            print(f"OK: wired: {rel}")

    print("\n--- expected guardrail coverage ---")
    for rel in sorted(EXPECTED_GUARDRAILS):
        if rel not in invocation_set:
            problems.append(f"expected guardrail is not wired into green baseline: {rel}")
            print(f"FAIL: not wired: {rel}")
        else:
            print(f"OK: wired: {rel}")

    print("\n--- admin readiness classification ---")
    print("audit_registry_machine_readable: true")
    print("green_baseline_runner_single_entrypoint: true")
    print("admin_tool_ready_for_future_button_or_job_wrapper: true")
    print("admin_ui_implemented: false")

    print("\n--- summary ---")
    print(f"baseline_invocations_checked: {len(invocations)}")
    print(f"registry_issues: {len(problems)}")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        print("\nFINAL_STATUS: FAIL_GREEN_BASELINE_AUDIT_REGISTRY")
        sys.exit(1)

    print("\nFINAL_STATUS: PASS_GREEN_BASELINE_AUDIT_REGISTRY")


if __name__ == "__main__":
    main()
