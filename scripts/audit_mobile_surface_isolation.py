#!/usr/bin/env python3
"""
DiamondSignals Mobile Surface Isolation Audit

Purpose:
Confirm mobile surface overhaul work remains quarantined and does not
accidentally alter desktop report files, shared desktop styles, shared nav,
or production desktop shell files.

This audit is intentionally conservative.
"""

from pathlib import Path
import subprocess
import sys

ALLOWED_PREFIXES = (
    "dashboard/templates/mobile/",
    "dashboard/static/mobile/",
    "docs/mobile-overhaul/",
    "scripts/audit_mobile_surface_isolation.py",
)

HIGH_RISK_PATTERNS = (
    "dashboard/templates/components/",
    "dashboard/templates/ledger_styles.css",
    "dashboard/templates/shell_styles.css",
    "dashboard/templates/report_styles.css",
    "dashboard/templates/nav",
    "dashboard/build_dashboard.py",
    "dashboard/build_all.py",
    "dist/",
)

def run_git_status():
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.splitlines()

def normalize_status_path(line):
    # git status --short format: XY path
    if len(line) < 4:
        return ""
    return line[3:].strip()

def main():
    lines = run_git_status()

    if not lines:
        print("MOBILE_ISOLATION_AUDIT: clean working tree")
        print("FINAL_STATUS: PASS_MOBILE_SURFACE_ISOLATION")
        return 0

    violations = []
    warnings = []

    for line in lines:
        path = normalize_status_path(line)

        if not path:
            continue

        allowed = path.startswith(ALLOWED_PREFIXES)

        if not allowed:
            violations.append((line, "Path is outside quarantined mobile/doc/audit areas"))

        for pattern in HIGH_RISK_PATTERNS:
            if path.startswith(pattern) or path == pattern.rstrip("/"):
                warnings.append((line, f"High-risk path touched: {pattern}"))

    print("MOBILE_ISOLATION_AUDIT: changed files")
    for line in lines:
        print(f"  {line}")

    if warnings:
        print()
        print("MOBILE_ISOLATION_AUDIT: high-risk warnings")
        for line, reason in warnings:
            print(f"  WARN {line} :: {reason}")

    if violations:
        print()
        print("MOBILE_ISOLATION_AUDIT: violations")
        for line, reason in violations:
            print(f"  FAIL {line} :: {reason}")
        print("FINAL_STATUS: FAIL_MOBILE_SURFACE_ISOLATION")
        return 1

    print()
    print("MOBILE_ISOLATION_AUDIT: no violations")
    print("FINAL_STATUS: PASS_MOBILE_SURFACE_ISOLATION")
    return 0

if __name__ == "__main__":
    sys.exit(main())
